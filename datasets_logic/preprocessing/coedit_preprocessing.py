#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
coedit_preprocessing.py

Purpose
- Load the Hugging Face dataset "grammarly/coedit"
- Split the `src` field into `instruction` and `bad` by the first colon
  Example src: "Fix disfluencies in the sentence: An example is ..."
  -> instruction = "Fix disfluencies in the sentence"
  -> bad = "An example is ..."
- Keep `tgt` as `good`
- Optionally filter by CoEdIT `task` (e.g., `gec`)
- Save per-split outputs to Parquet for faster downstream use

Usage examples
  python datasets_logic/preprocessing/coedit_preprocessing.py \
    --output_dir data/coedit_parquet \
    --splits train,validation \
    --num_proc 8

  python datasets_logic/preprocessing/coedit_preprocessing.py \
    --output_dir data/coedit_parquet \
    --splits train \
    --coedit_task gec \
    --limit 20000 \
    --shuffle --seed 42 \
    --num_proc 8

Notes
- Requires `datasets` and `pyarrow` to be installed.
- Will skip unavailable splits.
- Output columns: [id, task, instruction, bad, good]
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Optional

from datasets import load_dataset, Dataset


LOGGER = logging.getLogger("coedit_preprocessing")
DEFAULT_SPLITS = ["train", "validation", "test"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Preprocess grammarly/coedit into Parquet")
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write parquet files (created if missing)",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated list of splits to process (default: train,validation,test)",
    )
    ap.add_argument(
        "--coedit_task",
        type=str,
        default=None,
        help="Optional task filter (e.g., 'gec'); if set, only rows with matching 'task' are kept",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="If >0, limit rows after optional shuffle",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle before taking --limit",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle",
    )
    ap.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes for parallel map/filter",
    )
    ap.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return ap.parse_args()


def _split_src_batched(batch: dict) -> dict:
    srcs: List[Optional[str]] = batch.get("src", [])
    instructions: List[Optional[str]] = []
    bads: List[Optional[str]] = []
    for s in srcs:
        if s is None:
            instructions.append(None)
            bads.append(None)
            continue
        left, sep, right = s.partition(":")
        if sep == "":  # no colon
            instructions.append(None)
            bads.append(s.strip())
        else:
            instructions.append(left.strip())
            bads.append(right.strip())
    return {"instruction": instructions, "bad": bads}


def process_split(
    split: str,
    out_dir: str,
    task_filter: Optional[str],
    limit: int,
    shuffle: bool,
    seed: int,
    num_proc: int,
) -> None:
    # Attempt to load split; skip if missing
    try:
        ds: Dataset = load_dataset("grammarly/coedit", split=split)
    except Exception as e:
        LOGGER.warning("Skipping split '%s' (load failed): %s", split, e)
        return

    n0 = len(ds)
    LOGGER.info("Loaded %s split with %d rows", split, n0)

    # Optional filter by CoEdIT task
    if task_filter:
        ds = ds.filter(
            lambda ex: ex.get("task", None) == task_filter, num_proc=num_proc
        )
        LOGGER.info("After task=='%s' filter: %d rows", task_filter, len(ds))

    # Split src -> instruction, bad
    ds = ds.map(_split_src_batched, batched=True, num_proc=num_proc)

    # Keep minimal columns for downstream
    keep_cols = [
        c for c in ["_id", "task", "instruction", "bad", "tgt"] if c in ds.column_names
    ]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

    # Rename for clarity: _id -> id, tgt -> good
    if "_id" in ds.column_names:
        ds = ds.rename_column("_id", "id")
    if "tgt" in ds.column_names:
        ds = ds.rename_column("tgt", "good")

    # Optional shuffle + limit
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if limit and limit > 0 and len(ds) > limit:
        ds = ds.select(range(limit))
        LOGGER.info("After limit=%d: %d rows", limit, len(ds))

    # Ensure output dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"coedit_{split}.parquet")

    # Save to Parquet
    LOGGER.info("Writing %d rows to %s", len(ds), out_path)
    ds.to_parquet(out_path)
    LOGGER.info("Done: %s", out_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        splits = DEFAULT_SPLITS

    for split in splits:
        process_split(
            split=split,
            out_dir=args.output_dir,
            task_filter=args.coedit_task,
            limit=args.limit,
            shuffle=args.shuffle,
            seed=args.seed,
            num_proc=args.num_proc,
        )


if __name__ == "__main__":
    main()
