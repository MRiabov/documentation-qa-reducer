#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval.py

Evaluate a trained model (full or PEFT adapter) against a held-out dataset, using the
same configuration style as training via config.yaml. Supports both 'fixer' and
'degrader' modes and both 'good_only' and 'structured' targets.
"""

from __future__ import annotations
import argparse, csv, json, logging, os, re
from typing import List, Tuple, Optional, Dict
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import yaml
from tqdm import tqdm


# -----------------------------
# Config loading (mirror train_qLora.py)
# -----------------------------
CONFIG_PATH = "config.yaml"


def _load_config(path: str) -> dict:
    if not os.path.exists(path) or yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            return data
    except Exception:
        return {}


_CFG = _load_config(CONFIG_PATH)

LOGGER = logging.getLogger("eval")

BASE_MODEL = _CFG.get("base_model", "meta-llama/Llama-2-7b")
TINY_DEBUG_MODEL = _CFG.get("tiny_debug_model", "sshleifer/tiny-gpt2")

# Prompt templates and behavior
PROMPT_TEMPLATE = _CFG.get(
    "prompt_template",
    (
        "You are a documentation fixer. Identify the most degraded span in the input and propose a clear fix.\n"
        "{context_block}"
        "Bad:\n{bad}\n\n"
        "Respond EXACTLY with:\n"
        "Span: [START:<token_idx>][END:<token_idx>]\n"
        "Suggestion: <improved sentence or snippet>\n"
        "Rationale: <one-sentence explanation>\n"
    ),
)
TRAIN_DIRECTION = _CFG.get("train_direction", "fixer")  # 'fixer' or 'degrader'
DEGRADER_PROMPT_TEMPLATE = _CFG.get(
    "degrader_prompt_template",
    (
        "You are a text degrader. Introduce grammatical errors and reduce readability while preserving the original meaning.\n"
        "{context_block}"
        "Good:\n{good}\n\n"
        "Respond ONLY with the degraded text.\n"
    ),
)

# Optional generation length; re-use comet.gen_max_new_tokens if present
_COMET = _CFG.get("comet", {}) if isinstance(_CFG.get("comet", {}), dict) else {}
COMET_GEN_MAX_NEW_TOKENS = int(_COMET.get("gen_max_new_tokens", 128))

SPAN_RE = re.compile(r"Span:\s*\[START:(\d+)\]\[END:(\d+)\]", re.I)
SUG_RE = re.compile(r"Suggestion:\s*(.+)", re.I)
RAT_RE = re.compile(r"Rationale:\s*(.+)", re.I)


def set_seed_all(s: int) -> None:
    torch.manual_seed(s)
    try:
        torch.cuda.manual_seed_all(s)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        import random
        import numpy as np

        random.seed(s)
        np.random.seed(s)
    except Exception:
        pass


def build_prompt(bad: str, ctx: Optional[str]) -> str:
    ctxblk = f"Context:\n{ctx}\n\n" if ctx else ""
    return PROMPT_TEMPLATE.format(context_block=ctxblk, bad=bad)


def build_prompt_degrader(good: str, ctx: Optional[str]) -> str:
    ctxblk = f"Context:\n{ctx}\n\n" if ctx else ""
    return DEGRADER_PROMPT_TEMPLATE.format(context_block=ctxblk, good=good)


def parse_output(txt: str) -> Tuple[Optional[int], Optional[int], str, str]:
    s = SPAN_RE.search(txt)
    sug = SUG_RE.search(txt)
    rat = RAT_RE.search(txt)
    si = int(s.group(1)) if s else None
    ei = int(s.group(2)) if s else None
    return (
        si,
        ei,
        (sug.group(1).strip() if sug else ""),
        (rat.group(1).strip() if rat else ""),
    )


def span_prf1(ps: int, pe: int, gs: int, ge: int) -> Tuple[float, float, float]:
    if pe <= ps or ge <= gs:
        return (0.0, 0.0, 0.0)
    P = set(range(ps, pe))
    G = set(range(gs, ge))
    I = len(P & G)
    if I == 0:
        return (0.0, 0.0, 0.0)
    prec = I / len(P)
    rec = I / len(G)
    f1 = 0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return (prec, rec, f1)


def bleu_rouge_fallback(
    preds: List[str], refs: List[str]
) -> Tuple[List[float], List[float]]:
    def bleu1(p, r):
        pt, rt = p.split(), r.split()
        if not pt:
            return 0.0
        rc = {}
        [rc.__setitem__(t, rc.get(t, 0) + 1) for t in rt]
        ov = 0
        for t in pt:
            if rc.get(t, 0) > 0:
                ov += 1
                rc[t] -= 1
        return ov / len(pt)

    def rougeL(p, r):
        pt, rt = p.split(), r.split()
        if not pt or not rt:
            return 0.0
        dp = [[0] * (len(rt) + 1) for _ in range(len(pt) + 1)]
        for i in range(1, len(pt) + 1):
            for j in range(1, len(rt) + 1):
                dp[i][j] = (
                    dp[i - 1][j - 1] + 1
                    if pt[i - 1] == rt[j - 1]
                    else max(dp[i - 1][j], dp[i][j - 1])
                )
        lcs = dp[-1][-1]
        prec = lcs / len(pt)
        rec = lcs / len(rt)
        return 0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    return [bleu1(p, r) for p, r in zip(preds, refs)], [
        rougeL(p, r) for p, r in zip(preds, refs)
    ]


def compute_bleu_rouge(
    preds: List[str], refs: List[str]
) -> Tuple[List[float], List[float]]:
    try:
        import evaluate  # type: ignore

        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        # Fallback to per-example approximations for brevity
        return bleu_rouge_fallback(preds, refs)
    except Exception:
        return bleu_rouge_fallback(preds, refs)


def embed_cosine(preds: List[str], refs: List[str]) -> List[float]:
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore

        m = SentenceTransformer("all-MiniLM-L6-v2")
        ep = m.encode(preds, convert_to_tensor=True)
        er = m.encode(refs, convert_to_tensor=True)
        sims = torch.nn.functional.cosine_similarity(ep, er).cpu().tolist()
        return [float(s) for s in sims]
    except Exception:
        # Simple bag-of-words cosine
        sims = []
        for p, r in zip(preds, refs):
            wc = {}
            for t in p.split():
                wc[t] = wc.get(t, 0) + 1
            vr = {}
            for t in r.split():
                vr[t] = vr.get(t, 0) + 1
            keys = set(wc) | set(vr)
            import math

            dp = sum(wc.get(k, 0) * vr.get(k, 0) for k in keys)
            np = math.sqrt(sum(v * v for v in wc.values()))
            nr = math.sqrt(sum(v * v for v in vr.values()))
            sims.append(0.0 if np * nr == 0 else dp / (np * nr))
        return sims


def load_model(
    model_path_or_name: str,
    base_model: str,
    debug_tiny: bool,
    device_map: str = "auto",
    quantization: str = "auto",
    attn_implementation: Optional[str] = None,
):
    """
    Load a tokenizer and model. If model_path_or_name is a PEFT adapter dir, attach it
    to the specified base model (or the base in the adapter config). Otherwise, load as
    a full model. Supports BitsAndBytes 4/8-bit quantization like train_qLora.py.
    """

    def _load_base(name: str):
        if debug_tiny:
            return AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.float32,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
        if quantization == "bnb_4bit":
            q = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
            return AutoModelForCausalLM.from_pretrained(
                name,
                quantization_config=q,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
        if quantization == "bnb_8bit":
            q = BitsAndBytesConfig(load_in_8bit=True)
            return AutoModelForCausalLM.from_pretrained(
                name,
                quantization_config=q,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
        # auto/none
        return AutoModelForCausalLM.from_pretrained(
            name,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

    # Try PEFT adapter first
    try:
        cfg = PeftConfig.from_pretrained(model_path_or_name)
        base = cfg.base_model_name_or_path or base_model
        base = TINY_DEBUG_MODEL if debug_tiny else base
        tok = AutoTokenizer.from_pretrained(base, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = (
                tok.eos_token if getattr(tok, "eos_token", None) else "[PAD]"
            )
        # For decoder-only models, left padding aligns sequence ends for efficient batched generation
        tok.padding_side = "left"
        mdl = _load_base(base)
        mdl = PeftModel.from_pretrained(mdl, model_path_or_name)
        return tok, mdl
    except Exception:
        # Load as a full model dir or model name
        name = TINY_DEBUG_MODEL if debug_tiny else model_path_or_name
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = (
                tok.eos_token if getattr(tok, "eos_token", None) else "[PAD]"
            )
        tok.padding_side = "left"
        mdl = _load_base(name)
        return tok, mdl


def main():
    # Only allow lightweight CLI overrides; main config comes from config.yaml
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to PEFT adapter dir or full model. Defaults to resume_from_checkpoint or save_dir from config.yaml.",
    )
    ap.add_argument("--out", type=str, default="eval_results.csv")
    args_cli = ap.parse_args()

    class Args:
        pass

    args = Args()
    # Data/config
    args.model = args_cli.model or _CFG.get(
        "resume_from_checkpoint", _CFG.get("save_dir", "./lora_out")
    )
    args.base_model = BASE_MODEL
    args.dataset = str(_CFG.get("dataset", "sample_dataset.jsonl"))
    args.parquet_path = _CFG.get("parquet_path", None)
    args.hf_dataset = _CFG.get("hf_dataset", None)
    args.hf_split = _CFG.get("hf_split", "train")
    args.coedit_task = _CFG.get("coedit_task", None)
    args.out = args_cli.out

    # Runtime
    args.batch = int(_CFG.get("batch_size", 8))
    args.device_map = str(_CFG.get("device_map", "auto"))
    args.debug_tiny = bool(_CFG.get("debug_tiny", False))
    args.seed = int(_CFG.get("seed", 42))
    args.log_level = _CFG.get("log_level", "INFO")
    args.quantization = str(_CFG.get("quantization", "auto"))
    args.max_seq_len = int(_CFG.get("max_seq_len", 512))
    args.attn_implementation = _CFG.get("attn_implementation", None)
    args.train_target_format = _CFG.get("train_target_format", "good_only")
    args.max_generated = int(_CFG.get("max_generated", 0))

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    set_seed_all(args.seed)

    tok, model = load_model(
        args.model,
        args.base_model,
        args.debug_tiny,
        device_map=args.device_map,
        quantization=args.quantization,
        attn_implementation=args.attn_implementation,
    )

    # Load dataset (mirror train_qLora.py field mapping)
    if args.parquet_path:
        ds = load_dataset("parquet", data_files=args.parquet_path, split="train")

        def _prep_row_px(ex):
            bad = ex.get("bad", "")
            good = ex.get("good", "")
            instr = ex.get("instruction", None)
            task = ex.get("task", None)
            ctx_parts = []
            if instr:
                ctx_parts.append(f"Instruction: {instr}")
            if task:
                ctx_parts.append(f"Task: {task}")
            ctx = "\n".join(ctx_parts) if ctx_parts else None
            return {
                "bad": bad,
                "good": good,
                "context": ctx,
                "span_token_start": None,
                "span_token_end": None,
                "id": ex.get("id", None),
            }

        ds = ds.map(_prep_row_px)
        keep_cols = [
            c
            for c in [
                "bad",
                "good",
                "context",
                "span_token_start",
                "span_token_end",
                "id",
            ]
            if c in ds.column_names
        ]
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
    elif args.hf_dataset:
        ds = load_dataset(args.hf_dataset, split=args.hf_split)

        def _prep_row(ex):
            bad = ex.get("src", "")
            good = ex.get("tgt", "")
            task = ex.get("task", None)
            if args.coedit_task is not None and task != args.coedit_task:
                return {
                    "bad": None,
                    "good": None,
                    "context": None,
                    "span_token_start": None,
                    "span_token_end": None,
                    "id": None,
                }
            return {
                "bad": bad,
                "good": good,
                "context": (f"Task: {task}" if task else None),
                "span_token_start": None,
                "span_token_end": None,
                "id": ex.get("_id", None)
                if isinstance(ex.get("_id", None), (str, int))
                else None,
            }

        ds = ds.map(_prep_row)
        ds = ds.filter(lambda ex: ex["bad"] is not None and ex["good"] is not None)
        keep_cols = [
            c
            for c in [
                "bad",
                "good",
                "context",
                "span_token_start",
                "span_token_end",
                "id",
            ]
            if c in ds.column_names
        ]
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
    else:
        ds = load_dataset("json", data_files=args.dataset, split="train")

    # Optionally cap the number of evaluated samples for quicker runs
    if args.max_generated and args.max_generated > 0:
        take = min(int(args.max_generated), len(ds))
        ds = ds.select(range(take))
        LOGGER.info(f"Limiting evaluation to {take} examples (max_generated).")

    results = []
    model.eval()
    gen_max_new_tokens = COMET_GEN_MAX_NEW_TOKENS
    with torch.no_grad():
        for i in tqdm(
            range(0, len(ds), args.batch),
            total=(len(ds) + args.batch - 1) // args.batch,
            desc="Evaluating",
        ):
            start, end = i, min(i + args.batch, len(ds))
            # Convert to list of dict rows; note that ds[i:j] returns column-wise dict-of-lists
            batch_rows = [ds[j] for j in range(start, end)]
            if TRAIN_DIRECTION == "degrader":
                prompts = [
                    build_prompt_degrader(b.get("good", ""), b.get("context"))
                    for b in batch_rows
                ]
            else:
                prompts = [
                    build_prompt(b.get("bad", ""), b.get("context")) for b in batch_rows
                ]

            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            gen = model.generate(
                **enc,
                max_new_tokens=gen_max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.eos_token_id,
            )
            outs = tok.batch_decode(gen, skip_special_tokens=True)
            # Extract the part after prompt to reduce false matches
            txts = []
            for p, full in zip(prompts, outs):
                idx = full.find(p)
                txts.append(full[idx + len(p) :] if idx != -1 else full)

            for j, (ex, txt) in enumerate(zip(batch_rows, txts)):
                if TRAIN_DIRECTION == "degrader":
                    # No structured target; prediction is the whole generated text
                    sug = txt.strip()
                    ps = pe = None
                    gold_text = ex.get("bad", "")
                    span_f1 = 0.0
                else:
                    if _CFG.get("train_target_format", "good_only") == "structured":
                        ps, pe, sug, _ = parse_output(txt)
                    else:
                        ps, pe = None, None
                        sug = txt.strip()
                    gs = ex.get("span_token_start", None)
                    ge = ex.get("span_token_end", None)
                    try:
                        gs_i = int(gs) if gs is not None else 0
                        ge_i = int(ge) if ge is not None else 0
                    except Exception:
                        gs_i, ge_i = 0, 0
                    p_, r_, span_f1 = span_prf1(ps or 0, pe or 0, gs_i, ge_i)
                    gold_text = ex.get("good", "")

                results.append(
                    {
                        "id": ex.get("id", f"row_{i + j}"),
                        "bad": ex.get("bad", ""),
                        "model_suggestion": sug,
                        "gold": gold_text,
                        "bleu": 0.0,
                        "rougeL": 0.0,
                        "embed_sim": 0.0,
                        "span_pred_start": (ps if ps is not None else -1)
                        if TRAIN_DIRECTION != "degrader"
                        else -1,
                        "span_pred_end": (pe if pe is not None else -1)
                        if TRAIN_DIRECTION != "degrader"
                        else -1,
                        "span_f1": span_f1 if TRAIN_DIRECTION != "degrader" else 0.0,
                    }
                )

    preds = [r["model_suggestion"] for r in results]
    refs = [r["gold"] for r in results]
    bleu, rouge = compute_bleu_rouge(preds, refs)
    sims = embed_cosine(preds, refs)
    for r, b, rg, s in zip(results, bleu, rouge, sims):
        r["bleu"] = b
        r["rougeL"] = rg
        r["embed_sim"] = s
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["id"])
        w.writeheader()
        [w.writerow(r) for r in results]
    LOGGER.info(f"Wrote {len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
