# Documentation QA Reducer (QLoRA/PEFT)

## Purpose

This repository’s primary purpose is to degrade high-quality developer documentation into lower-quality variants to create a synthetic dataset of “bad” docs from “good” docs. The training and evaluation scripts included here are optional consumers of that dataset (e.g., to train a fixer model that rewrites bad -> good). No executable behavior in the code has changed; this update clarifies intent and usage only.

Pipeline:

1) Prepare data
   - Create samples and build dataset:
     - python data_prep.py --run-tests
     - or: python data_prep.py --input sample_docs --output dataset.jsonl --n-per-doc 3 --seed 42 --debug_tiny
2) Train (QLoRA)
   - CPU/tiny quick test (1 step):
     - python train_qLora.py --dataset sample_dataset.jsonl --debug_tiny --per_device_batch_size 1 --max_steps 1
   - With accelerate (GPU):
     - accelerate launch train_qLora.py --dataset dataset.jsonl --save_dir ./lora_out --device_map auto
3) Evaluate
   - python eval.py --model ./lora_out --dataset sample_dataset.jsonl --out eval_results.csv --batch 8 --debug_tiny

## Using Hugging Face datasets (e.g., grammarly/coedit)

You can train/evaluate directly from the public dataset without converting to JSONL first:

- Train on CoEdIT (all tasks):
  - python train_qLora.py --hf_dataset grammarly/coedit --hf_split train --save_dir ./lora_out

- Train on a specific CoEdIT task (e.g., grammatical error correction `gec`):
  - python train_qLora.py --hf_dataset grammarly/coedit --hf_split train --coedit_task gec --save_dir ./lora_out

- Evaluate a trained fixer on CoEdIT:
  - python eval.py --model ./lora_out --hf_dataset grammarly/coedit --hf_split validation --out eval_results.csv --batch 8

Notes:

- Field mapping for CoEdIT: `src` -> `bad` (degraded input), `tgt` -> `good` (target), `task` -> `context` (as `Task: <task>`).
- CoEdIT does not provide token spans; training defaults to the `good_only` target format. You can still use `--train_target_format structured` if desired; span indices will fall back to simple defaults.

Defaults:

- BASE_MODEL: meta-llama/Llama-2-7b (use --debug_tiny for sshleifer/tiny-gpt2)
- QLoRA: 4-bit quant + LoRA (r=8, alpha=32, dropout=0.05), lr=2e-4, seq=512, epochs=3

Hardware:

- Recommended: 1x A100 80GB (or 40GB with grad-accum + lower batch). Use --debug_tiny for quick CPU demos.

Safety:

- When training a fixer model, review generated suggestions before applying. Avoid hallucinated code edits.

Notes:

- The dataset produced by data_prep.py consists of paired records where `good` is the original text and `bad` is the degraded version (created by rule-based degraders and an optional LLM-style stub). This repo is responsible for that degradation process.
- If you want to train a degrader model (i.e., to generate bad from good), keep the code as-is and swap the fields externally when preparing batches (use `good` as input and `bad` as target). The provided training script, by default, trains a fixer (bad -> good).

Install:

- pip install -U "transformers>=4.32" accelerate bitsandbytes peft datasets safetensors sentence_transformers evaluate
