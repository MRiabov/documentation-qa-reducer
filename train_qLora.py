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
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)

# -----------------------------
# Defaults / Hyperparams
# -----------------------------
BASE_MODEL = "meta-llama/Llama-2-7b"
TINY_DEBUG_MODEL = "sshleifer/tiny-gpt2"

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 16
MAX_SEQ_LEN = 512
EPOCHS = 3
SAVE_DIR = "./lora_out"
SEED = 42

PROMPT_TEMPLATE = (
    # This template is for the fixer task (bad -> good). The core repo still focuses on creating the bad data.
    "You are a documentation fixer. Identify the most degraded span in the input and propose a clear fix.\n"
    "{context_block}"
    "Bad:\n{bad}\n\n"
    "Respond EXACTLY with:\n"
    "Span: [START:<token_idx>][END:<token_idx>]\n"
    "Suggestion: <improved sentence or snippet>\n"
    "Rationale: <one-sentence explanation>\n"
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


def load_model(model_name: str, debug_tiny: bool, device_map: str = "auto"):
    if debug_tiny:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=device_map
        )
        return model
    # 4-bit QLoRA-style load
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        device_map=device_map,
    )
    return model


def tokenize_examples(
    examples: Dict[str, List],
    tokenizer: AutoTokenizer,
    max_len: int,
    target_mode: str,
) -> Dict[str, List[List[int]]]:
    """
    Convert raw fields into tokenized inputs and labels for Causal LM with masked input loss.
    """
    bad_list = examples["bad"]
    good_list = examples["good"]
    ctx_list = examples.get("context", [None] * len(bad_list))
    s_list = examples.get("span_token_start", [None] * len(bad_list))
    e_list = examples.get("span_token_end", [None] * len(bad_list))

    input_ids_batch: List[List[int]] = []
    labels_batch: List[List[int]] = []
    attention_batch: List[List[int]] = []

    eos_id = tokenizer.eos_token_id

    for bad, good, ctx, s, e in zip(bad_list, good_list, ctx_list, s_list, e_list):
        input_text = build_prompt(bad=bad, context=ctx)
        target_text = build_target_text(
            good=good, span_start=s, span_end=e, mode=target_mode
        )

        in_ids = tokenizer.encode(input_text, add_special_tokens=False)
        tgt_ids = tokenizer.encode(target_text, add_special_tokens=False)

        # Concat: input + eos + target
        full_ids = in_ids + ([eos_id] if eos_id is not None else []) + tgt_ids
        # Create labels masking input portion
        n_input = len(in_ids) + (1 if eos_id is not None else 0)
        labels = [-100] * n_input + tgt_ids

        # Truncate from the left if too long (preserve the target tail)
        if len(full_ids) > max_len:
            cut = len(full_ids) - max_len
            # We must adjust labels accordingly
            full_ids = full_ids[cut:]
            if cut >= n_input:
                # All masked part removed; shift cut over labels
                labels = labels[cut - n_input :]
            else:
                labels = [-100] * (n_input - cut) + labels[n_input:]

        attn = [1] * len(full_ids)
        input_ids_batch.append(full_ids)
        labels_batch.append(labels)
        attention_batch.append(attn)

    return {
        "input_ids": input_ids_batch,
        "labels": labels_batch,
        "attention_mask": attention_batch,
    }


@dataclass
class SimpleDataCollator:
    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids, attn, labels = [], [], []
        for x in features:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad_len)
            attn.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [self.label_pad_id] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


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
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    set_seed_all(args.seed)

    model_name = TINY_DEBUG_MODEL if args.debug_tiny else args.base_model
    tokenizer = load_tokenizer(model_name)
    model = load_model(
        model_name, debug_tiny=args.debug_tiny, device_map=args.device_map
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
    model = get_peft_model(model, peft_config)

    # Load dataset
    data = load_dataset("json", data_files=args.dataset, split="train")

    # Tokenize
    def _map_fn(batch):
        return tokenize_examples(
            batch,
            tokenizer=tokenizer,
            max_len=args.max_seq_len,
            target_mode=args.train_target_format,
        )

    tokenized = data.map(_map_fn, batched=True, remove_columns=data.column_names)

    # 90/10 split
    split = tokenized.train_test_split(test_size=0.1, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]

    collator = SimpleDataCollator(pad_token_id=tokenizer.pad_token_id)

    total_steps = args.max_steps if args.max_steps > 0 else None
    training_args = TrainingArguments(
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
        evaluation_strategy="steps",
        eval_steps=100,
        report_to=[],
        fp16=not args.debug_tiny,
        bf16=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    LOGGER.info(f"Saved LoRA adapters to {args.save_dir}")

    # Example: merge LoRA for full-model export (optional; commented)
    # merged = model.merge_and_unload()
    # merged.save_pretrained(os.path.join(args.save_dir, "merged_full_model"))


if __name__ == "__main__":
    main()
