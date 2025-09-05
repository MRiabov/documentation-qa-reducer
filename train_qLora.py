#!/usr/bin/env python
# Installation (run once):
#   pip install -U "transformers>=4.32" accelerate bitsandbytes peft datasets safetensors sentence_transformers evaluate

"""
train_qLora.py

Primary repo purpose: degrade high-quality developer documentation into lower-quality variants to synthesize a "bad" corpus from a "good" corpus.

This script is an optional consumer of that dataset. It shows how to train a 7B causal/instruction model with 4-bit quantization + LoRA adapters (QLoRA style) on the paired (bad, good) data.
By default, it trains a "fixer" model that maps bad -> good (i.e., rewrites degraded inputs back to high quality). If you want to train a degrader model instead, prepare batches externally using `good` as input and `bad` as target, without changing this file's logic.

Example runs:
  # Debug tiny quick test:
  python train_qLora.py --dataset sample_dataset.jsonl --debug_tiny --per_device_batch_size 1 --max_steps 1

  # With accelerate:
  accelerate launch train_qLora.py --dataset dataset.jsonl --save_dir ./lora_out --device_map auto

Notes:
- Default BASE_MODEL is meta-llama/Llama-2-7b. Use --debug_tiny to switch to a tiny model for CPU-only demos.
- Targets default to training a fixer (bad -> good). Optional --train_target_format "structured" supervises the exact eval format.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
)
from trl import SFTTrainer, SFTConfig
import yaml


# -----------------------------
# Defaults / Hyperparams (from config.yaml if present)
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

BASE_MODEL = _CFG.get("base_model", "meta-llama/Llama-2-7b")
TINY_DEBUG_MODEL = _CFG.get("tiny_debug_model", "sshleifer/tiny-gpt2")

_LORA = _CFG.get("lora", {}) if isinstance(_CFG.get("lora", {}), dict) else {}
LORA_R = int(_LORA.get("r", 8))
LORA_ALPHA = int(_LORA.get("alpha", 32))
LORA_DROPOUT = float(_LORA.get("dropout", 0.05))

LEARNING_RATE = float(_CFG.get("learning_rate", 2e-4))
BATCH_SIZE = int(_CFG.get("batch_size", 4))
GRAD_ACCUM = int(_CFG.get("grad_accum", 16))
MAX_SEQ_LEN = int(_CFG.get("max_seq_len", 512))
EPOCHS = int(_CFG.get("epochs", 3))
SAVE_DIR = str(_CFG.get("save_dir", "./lora_out"))
SEED = int(_CFG.get("seed", 42))

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

LOGGER = logging.getLogger("train_qLoRA")


# -----------------------------
# Utilities
# -----------------------------
def set_seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def guess_lora_target_modules(model) -> List[str]:
    """
    Try to infer good LoRA target modules from model architecture.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", "").lower()
    if "llama" in model_type or "mistral" in model_type or "falcon" in model_type:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    if "gpt" in model_type:
        # GPT-2-ish
        return ["c_attn", "c_proj", "q_attn"]  # minimal set; extend as needed
    # Generic fallback: apply LoRA to all Linear layers named with 'proj' or 'fc'
    target_names = set()
    for name, module in model.named_modules():
        if any(
            k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "proj", "fc"]
        ):
            target_names.add(name.split(".")[-1])
    # Use short unique layer names
    if not target_names:
        target_names = {"dense", "out_proj"}
    return sorted(set(target_names))


def build_prompt(bad: str, context: Optional[str]) -> str:
    context_block = f"Context:\n{context}\n\n" if context else ""
    return PROMPT_TEMPLATE.format(context_block=context_block, bad=bad)


def build_target_text(
    good: str,
    span_start: Optional[int],
    span_end: Optional[int],
    mode: str = "good_only",  # or "structured"
) -> str:
    """
    Per spec, default target is just the 'good' text.
    Optionally, 'structured' supervises the exact output format used in eval.
    """
    if mode == "good_only":
        return good
    # structured target for better alignment with eval format
    s = span_start if span_start is not None else 0
    e = span_end if span_end is not None else s + 1
    return (
        f"Span: [START:{int(s)}][END:{int(e)}]\n"
        f"Suggestion: {good}\n"
        f"Rationale: Improves clarity and correctness.\n"
    )


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if getattr(tok, "eos_token", None) else "[PAD]"
    return tok


def load_model(model_name: str, debug_tiny: bool, device_map: str = "auto", quantization: str = "auto"):
    if debug_tiny:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=device_map
        )
        return model
    # Quantization modes:
    # - auto: do not pass BitsAndBytes; allow model's own quantization config (e.g., BitNet) to load
    # - bnb_4bit: use BitsAndBytes 4-bit quantization
    # - none: no quantization arguments (may require large VRAM)
    if quantization == "bnb_4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=device_map,
        )
    # auto or none both avoid BitsAndBytes; for 'none' we simply don't pass quantization args
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Train QLoRA adapters on degraded docs dataset."
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="sample_dataset.jsonl",
        help="Path to JSONL dataset.",
    )
    ap.add_argument(
        "--save_dir", type=str, default=SAVE_DIR, help="Where to save LoRA adapters."
    )
    ap.add_argument(
        "--base_model", type=str, default=BASE_MODEL, help="Base model to load."
    )
    ap.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Device map for model, e.g., "auto" or "cpu".',
    )
    ap.add_argument(
        "--quantization",
        type=str,
        default="auto",
        choices=["auto", "bnb_4bit", "none"],
        help="Quantization mode: 'auto' loads model as-is (useful for pre-quantized models like BitNet); 'bnb_4bit' forces BitsAndBytes 4-bit; 'none' disables quantization args.",
    )
    ap.add_argument("--per_device_batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=GRAD_ACCUM)
    ap.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lora_r", type=int, default=LORA_R)
    ap.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    ap.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    ap.add_argument("--resume_from_checkpoint", type=str, default=None)
    ap.add_argument(
        "--debug_tiny",
        action="store_true",
        help=f"Use tiny model ({TINY_DEBUG_MODEL}) for quick tests.",
    )
    ap.add_argument(
        "--train_target_format",
        type=str,
        default="good_only",
        choices=["good_only", "structured"],
    )
    ap.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override number of training steps (e.g., 1 for test).",
    )
    ap.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    # Local Parquet dataset support
    ap.add_argument(
        "--parquet_path",
        type=str,
        default=None,
        help="Optional path to a Parquet file produced by coedit_preprocessing.py. If provided, overrides --hf_dataset/--dataset.",
    )
    # Hugging Face dataset integration (e.g., 'grammarly/coedit')
    ap.add_argument(
        "--hf_dataset",
        type=str,
        default=None,
        help="Optional Hugging Face dataset name to load, e.g., 'grammarly/coedit'. If provided, --dataset (JSONL) is ignored.",
    )
    ap.add_argument(
        "--hf_split",
        type=str,
        default="train",
        help="Split name for HF dataset (e.g., train, validation, test).",
    )
    ap.add_argument(
        "--coedit_task",
        type=str,
        default=None,
        help="Optional filter for CoEdIT task field (e.g., 'gec'). If set, only rows with matching 'task' are used.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    set_seed_all(args.seed)

    model_name = TINY_DEBUG_MODEL if args.debug_tiny else args.base_model
    tokenizer = load_tokenizer(model_name)
    model = load_model(
        model_name, debug_tiny=args.debug_tiny, device_map=args.device_map, quantization=args.quantization
    )

    # Inject LoRA adapters
    target_modules = guess_lora_target_modules(model)
    LOGGER.info(f"Using LoRA target modules: {target_modules}")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Let TRL's SFTTrainer apply PEFT adapters via peft_config

    # Load dataset
    if args.parquet_path:
        # Expect columns: id, task, instruction, bad, good
        data = load_dataset("parquet", data_files=args.parquet_path, split="train")
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
            }
        data = data.map(_prep_row_px)
        keep_cols = [c for c in ["bad", "good", "context", "span_token_start", "span_token_end"] if c in data.column_names]
        data = data.remove_columns([c for c in data.column_names if c not in keep_cols])
    elif args.hf_dataset:
        data = load_dataset(args.hf_dataset, split=args.hf_split)
        # Map CoEdIT-style fields to our expected schema
        def _prep_row(ex):
            # CoEdIT fields: '_id', 'task', 'src', 'tgt'
            # Use src as degraded input (bad), tgt as target (good)
            bad = ex.get("src", "")
            good = ex.get("tgt", "")
            task = ex.get("task", None)
            # Optional filter by task
            if args.coedit_task is not None and task != args.coedit_task:
                return {
                    "bad": None,
                    "good": None,
                    "context": None,
                    "span_token_start": None,
                    "span_token_end": None,
                }
            return {
                "bad": bad,
                "good": good,
                # Provide lightweight context to preserve task info if present
                "context": (f"Task: {task}" if task else None),
                # CoEdIT doesn't provide spans; leave None so training doesn't use them in 'good_only' mode
                "span_token_start": None,
                "span_token_end": None,
            }
        data = data.map(_prep_row)
        # Drop rows filtered out
        data = data.filter(lambda ex: ex["bad"] is not None and ex["good"] is not None)
        # Keep only necessary columns to reduce memory
        keep_cols = [c for c in ["bad", "good", "context", "span_token_start", "span_token_end"] if c in data.column_names]
        data = data.remove_columns([c for c in data.column_names if c not in keep_cols])
    else:
        data = load_dataset("json", data_files=args.dataset, split="train")

    # Prepare dataset for SFTTrainer in prompt-completion format
    def _to_prompt_completion(ex):
        bad = ex.get("bad", "")
        good = ex.get("good", "")
        ctx = ex.get("context", None)
        s = ex.get("span_token_start", None)
        e = ex.get("span_token_end", None)
        prompt = build_prompt(bad=bad, context=ctx)
        completion = build_target_text(good=good, span_start=s, span_end=e, mode=args.train_target_format)
        return {"prompt": prompt, "completion": completion}

    pc_ds = data.map(_to_prompt_completion, remove_columns=[c for c in data.column_names if c not in ["prompt", "completion"]])

    # 90/10 split
    split = pc_ds.train_test_split(test_size=0.1, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]

    total_steps = args.max_steps if args.max_steps > 0 else None
    sft_config = SFTConfig(
        output_dir=args.save_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=max(1, args.per_device_batch_size),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=0 if total_steps else args.epochs,
        max_steps=total_steps if total_steps else -1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=100,
        report_to=[],
        fp16=not args.debug_tiny,
        bf16=False,
        seed=args.seed,
        max_length=args.max_seq_len,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    os.makedirs(args.save_dir, exist_ok=True)
    # Save resulting PEFT adapters and tokenizer
    trainer.model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    LOGGER.info(f"Saved LoRA adapters to {args.save_dir}")

    # Example: merge LoRA for full-model export (optional; commented)
    # merged = model.merge_and_unload()
    # merged.save_pretrained(os.path.join(args.save_dir, "merged_full_model"))


if __name__ == "__main__":
    main()
