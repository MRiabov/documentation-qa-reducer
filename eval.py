#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, json, logging, os, re
from typing import List, Tuple, Optional, Dict
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

LOGGER = logging.getLogger("eval")
BASE_MODEL = "meta-llama/Llama-2-7b"
TINY_DEBUG_MODEL = "sshleifer/tiny-gpt2"
PROMPT_TEMPLATE = (
    "You are a documentation fixer. Identify the most degraded span in the input and propose a clear fix.\n"
    "{context_block}Bad:\n{bad}\n\nRespond EXACTLY with:\n"
    "Span: [START:<token_idx>][END:<token_idx>]\nSuggestion: <improved sentence or snippet>\nRationale: <one-sentence explanation>\n"
)
SPAN_RE = re.compile(r"Span:\s*\[START:(\d+)\]\[END:(\d+)\]", re.I)
SUG_RE = re.compile(r"Suggestion:\s*(.+)", re.I)
RAT_RE = re.compile(r"Rationale:\s*(.+)", re.I)


def set_seed(s: int) -> None:
    torch.manual_seed(s)
    try:
        torch.cuda.manual_seed_all(s)  # type: ignore
    except Exception:
        pass


def build_prompt(bad: str, ctx: Optional[str]) -> str:
    ctxblk = f"Context:\n{ctx}\n\n" if ctx else ""
    return PROMPT_TEMPLATE.format(context_block=ctxblk, bad=bad)


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


def load_model(model_dir: str, base_model: str, debug_tiny: bool, device: str):
    # If model_dir is a PEFT adapter dir, attach to base model
    try:
        cfg = PeftConfig.from_pretrained(model_dir)
        base = cfg.base_model_name_or_path or base_model
        base = TINY_DEBUG_MODEL if debug_tiny else base
        tok = AutoTokenizer.from_pretrained(base, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = (
                tok.eos_token if getattr(tok, "eos_token", None) else "[PAD]"
            )
        mdl = AutoModelForCausalLM.from_pretrained(base, device_map=device)
        mdl = PeftModel.from_pretrained(mdl, model_dir)
        return tok, mdl
    except Exception:
        # Try load as full model directory
        name = TINY_DEBUG_MODEL if debug_tiny else model_dir
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = (
                tok.eos_token if getattr(tok, "eos_token", None) else "[PAD]"
            )
        mdl = AutoModelForCausalLM.from_pretrained(name, device_map=device)
        return tok, mdl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./lora_out")
    ap.add_argument("--base_model", type=str, default=BASE_MODEL)
    ap.add_argument("--dataset", type=str, default="sample_dataset.jsonl")
    ap.add_argument("--out", type=str, default="eval_results.csv")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--debug_tiny", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    set_seed(args.seed)
    tok, model = load_model(
        args.model, args.base_model, args.debug_tiny, args.device_map
    )
    ds = load_dataset("json", data_files=args.dataset, split="train")
    results = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ds), args.batch):
            batch = ds[i : i + args.batch]
            prompts = [build_prompt(b["bad"], b.get("context")) for b in batch]
            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            gen = model.generate(
                **enc,
                max_new_tokens=128,
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
            for ex, txt in zip(batch, txts):
                ps, pe, sug, rat = parse_output(txt)
                gs, ge = (
                    int(ex.get("span_token_start", 0)),
                    int(ex.get("span_token_end", 0)),
                )
                p, r, f1 = span_prf1(ps or 0, pe or 0, gs, ge)
                results.append(
                    {
                        "id": ex["id"],
                        "bad": ex["bad"],
                        "model_suggestion": sug,
                        "gold": ex["good"],
                        "bleu": 0.0,
                        "rougeL": 0.0,
                        "embed_sim": 0.0,
                        "span_pred_start": ps if ps is not None else -1,
                        "span_pred_end": pe if pe is not None else -1,
                        "span_f1": f1,
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
