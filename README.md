# Documentation QA Reducer (QLoRA/PEFT)

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

Defaults:

- BASE_MODEL: meta-llama/Llama-2-7b (use --debug_tiny for sshleifer/tiny-gpt2)
- QLoRA: 4-bit quant + LoRA (r=8, alpha=32, dropout=0.05), lr=2e-4, seq=512, epochs=3

Hardware:

- Recommended: 1x A100 80GB (or 40GB with grad-accum + lower batch). Use --debug_tiny for quick CPU demos.

Safety:

- Review suggestions before applying to docs. Avoid hallucinated code edits.

Install:

- pip install -U "transformers>=4.32" accelerate bitsandbytes peft datasets safetensors sentence_transformers evaluate
