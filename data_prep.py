#!/usr/bin/env python
"""
data_prep.py

Primary role: This repository degrades high-quality developer documentation into lower-quality variants to synthesize a "bad" corpus from a "good" corpus.
This module constructs the degraded text ("bad") from the original ("good"), and computes span alignment for where degradation occurs.

Input: a directory of "good" docs (JSONL, one document per line with {"id","text","source"}).
Output: a dataset of paired examples ("bad" and "good") with span alignment for the degraded area.

CLI:
  python data_prep.py --input good_docs_dir --output dataset.jsonl --n-per-doc 3 --seed 42 --use-llm-degrader False

Notes:
- Deterministic by default (seed all RNGs).
- Tokenization uses the target model tokenizer (BASE_MODEL). Use --debug_tiny to switch to a tiny model for offline runs.
- Span alignment uses difflib.SequenceMatcher on token sequences. We prefer the longest changed contiguous span.
- We store both token-span (start/end) and character-span (start/end) for the BAD text.
- An LLM-based degrader stub is provided via llm_degrader_stub(), which does not call external APIs.

Output JSONL record format:
{
  "id": str,
  "context": Optional[str],
  "bad": str,
  "good": str,
  "bad_tokens": List[str],
  "span_token_start": int,   # inclusive
  "span_token_end": int,     # exclusive
  "span_char_start": int,    # inclusive
  "span_char_end": int       # exclusive
}
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Optional heavy deps, import lazily
try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - only for environments without transformers
    AutoTokenizer = None  # type: ignore

# Defaults
BASE_MODEL = "meta-llama/Llama-2-7b"
TINY_DEBUG_MODEL = "sshleifer/tiny-gpt2"

LOGGER = logging.getLogger("data_prep")


# -----------------------------
# Seeding and Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    """Deterministically seed all RNGs we can."""
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
    except Exception:
        pass


def list_jsonl_files(input_dir: str) -> List[Path]:
    """List all .jsonl files in a directory."""
    p = Path(input_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Input directory not found or not a dir: {input_dir}")
    return sorted(
        [fp for fp in p.iterdir() if fp.is_file() and fp.suffix.lower() == ".jsonl"]
    )


def read_jsonl(fp: Path) -> Iterable[dict]:
    """Read JSONL yielding objects. Tolerant to blank lines."""
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception as e:
                LOGGER.warning(f"Skipping malformed line in {fp.name}: {e}")


def safe_ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Degraders
# -----------------------------
def grammar_noise(text: str, rng: random.Random) -> str:
    """
    Apply small grammar/typographical noise:
      - Random punctuation removal
      - Duplicate occasional words
      - Small typos such as character swaps or drops
    """
    if not text:
        return text

    # 1) Randomly remove some punctuation
    punct_to_remove = set(
        rng.sample(list(string.punctuation), k=min(3, len(string.punctuation)))
    )

    def rm_punct(ch: str) -> str:
        return "" if ch in punct_to_remove and rng.random() < 0.4 else ch

    text = "".join(rm_punct(c) for c in text)

    # 2) Token-level modifications
    tokens = text.split()
    new_tokens: List[str] = []
    for tok in tokens:
        # 2a) Duplicate words occasionally
        if rng.random() < 0.08:
            new_tokens.append(tok)
        new_tokens.append(tok)
        # 2b) Small typos: swap adjacent chars or drop a char
        if len(tok) > 3 and rng.random() < 0.08:
            idx = rng.randint(1, len(tok) - 2)
            if rng.random() < 0.5:
                # swap
                lst = list(tok)
                lst[idx], lst[idx + 1] = lst[idx + 1], lst[idx]
                new_tokens[-1] = "".join(lst)
            else:
                # drop
                lst = list(tok)
                del lst[idx]
                new_tokens[-1] = "".join(lst)

    return " ".join(new_tokens)


def remove_step(text: str, rng: random.Random) -> str:
    """
    Drop a sentence or step. Attempt to find procedural bullets or numbered steps.
    """
    if not text:
        return text

    # Split into sentences/steps
    step_delims = re.split(r"(\n- |\n\* |\n\d+\.\s+)", text)  # keep delimiters
    if len(step_delims) > 1:
        # Reconstruct steps by merging delimiter+content
        steps: List[str] = []
        i = 0
        while i < len(step_delims):
            if i + 1 < len(step_delims) and re.match(
                r"(\n- |\n\* |\n\d+\.\s+)", step_delims[i]
            ):
                steps.append(step_delims[i] + step_delims[i + 1])
                i += 2
            else:
                steps.append(step_delims[i])
                i += 1
        candidates = [s for s in steps if s.strip()]
        if not candidates:
            return text
        idx = rng.randrange(len(candidates))
        candidates.pop(idx)
        return "".join(candidates)

    # Fallback: sentence split by punctuation
    sents = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    if len(sents) <= 1:
        return text
    idx = rng.randrange(len(sents))
    sents.pop(idx)
    return " ".join(sents)


def terminology_swap(text: str, rng: random.Random) -> str:
    """
    Change key API names or terms to inconsistent variants.
    """
    if not text:
        return text
    swaps: Dict[str, List[str]] = {
        "API": ["Api", "api", "endpoint api"],
        "endpoint": ["endPoint", "URL", "route"],
        "client": ["consumer", "caller", "clnt"],
        "server": ["service", "srv"],
        "POST": ["Send", "Post", "HTTP-POST"],
        "GET": ["Fetch", "Get", "HTTP-GET"],
        "token": ["secret", "apiKey", "tk"],
        "JSON": ["Json", "json-format", "jsn"],
        "Python": ["py", "Py"],
        "LLaMA": ["llama", "LLama"],
        "install": ["setup", "add", "instll"],
        "connect": ["link", "attach", "cnnect"],
    }

    # Replace words in a case-insensitive manner but preserve original casing if possible.
    def repl(match: re.Match) -> str:
        word = match.group(0)
        key = None
        for k in swaps:
            if word.lower() == k.lower():
                key = k
                break
        if key is None:
            return word
        return rng.choice(swaps[key])

    pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, swaps.keys())) + r")\b", flags=re.IGNORECASE
    )
    return pattern.sub(repl, text)


def formatting_corrupt(text: str, rng: random.Random) -> str:
    """
    Break code blocks or remove code fences; damage indentation or language hints.
    """
    if not text:
        return text
    t = text
    # Remove closing fences or language hints with some probability
    t = re.sub(r"```[a-zA-Z]*", "```", t)  # drop language hint
    if rng.random() < 0.6:
        t = t.replace("```", "``")  # break fence
    # Break indentation randomly
    lines = t.splitlines()
    for i in range(len(lines)):
        if lines[i].startswith("    ") and rng.random() < 0.5:
            lines[i] = lines[i][2:]  # remove some indentation
        elif (lines[i].startswith("\t")) and rng.random() < 0.5:
            lines[i] = lines[i][1:]
    t = "\n".join(lines)
    return t


def verbosity_change(text: str, rng: random.Random) -> str:
    """
    Insert hedges/adjectives or remove clarity modifiers to change tone/clarity.
    """
    if not text:
        return text
    hedges = [
        "basically",
        "probably",
        "maybe",
        "somewhat",
        "kind of",
        "sort of",
        "in general",
    ]
    tokens = text.split()
    out: List[str] = []
    for tok in tokens:
        # Occasionally insert a hedge before a noun/verb-ish token (approx random)
        if rng.random() < 0.08:
            out.append(rng.choice(hedges))
        out.append(tok)
        # Occasionally remove "very", "exactly", "clearly"
        if (
            tok.lower() in {"very", "exactly", "clearly", "specifically"}
            and rng.random() < 0.7
        ):
            out.pop()  # remove it
    return " ".join(out)


def llm_degrader_stub(text: str, rng: random.Random) -> str:
    """
    Stub for an LLM-based degrader. No external calls.
    This mock applies a composed, stronger degradation to better simulate LLM edits.
    """
    t = terminology_swap(text, rng)
    t = grammar_noise(t, rng)
    if rng.random() < 0.5:
        t = remove_step(t, rng)
    return t


DEGRADERS: Dict[str, Callable[[str, random.Random], str]] = {
    "grammar_noise": grammar_noise,
    "remove_step": remove_step,
    "terminology_swap": terminology_swap,
    "formatting_corrupt": formatting_corrupt,
    "verbosity_change": verbosity_change,
    "llm_stub": llm_degrader_stub,  # included only if --use-llm-degrader
}


# -----------------------------
# Tokenization and Alignment
# -----------------------------
@dataclass
class TokenizationResult:
    tokens: List[str]
    offsets: List[Tuple[int, int]]  # (char_start, char_end)
    used_whitespace_fallback: bool


def whitespace_tokenize_with_offsets(text: str) -> TokenizationResult:
    """Whitespace tokenization with offsets as a robust fallback."""
    tokens: List[str] = []
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for m in re.finditer(r"\S+", text):
        start, end = m.start(), m.end()
        tokens.append(text[start:end])
        offsets.append((start, end))
        cursor = end
    return TokenizationResult(
        tokens=tokens, offsets=offsets, used_whitespace_fallback=True
    )


def hf_tokenize_with_offsets(text: str, tokenizer) -> TokenizationResult:
    """Use HF fast tokenizer with offsets; fallback to whitespace if unsupported."""
    if tokenizer is None:
        return whitespace_tokenize_with_offsets(text)
    try:
        # encodings for a single string
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = enc.get("offset_mapping", None)
        input_ids = enc.get("input_ids", None)
        if offsets is None or input_ids is None:
            raise ValueError("No offsets or input_ids from tokenizer; using fallback.")
        # Convert ids to tokens to align with offsets
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        # Some slow tokenizers may return None for offsets
        if offsets is None or any(o is None for o in offsets):
            raise ValueError("Offsets contain None; using fallback.")
        return TokenizationResult(
            tokens=list(tokens), offsets=list(offsets), used_whitespace_fallback=False
        )
    except Exception as e:
        LOGGER.debug(f"Fast offsets not available ({e}); using whitespace fallback.")
        return whitespace_tokenize_with_offsets(text)


def longest_changed_span_opcodes(
    bad_tokens: List[str], good_tokens: List[str]
) -> Tuple[int, int]:
    """
    Use difflib.SequenceMatcher to find the longest changed contiguous span (in BAD tokens).
    Returns (start_idx_in_bad, end_idx_in_bad) with end exclusive.
    For 'insert' ops (len=0 in bad), we expand to at least 1 token near the insertion point.
    """
    sm = difflib.SequenceMatcher(a=bad_tokens, b=good_tokens, autojunk=False)
    best_start, best_end, best_len = 0, 0, -1
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        span_len = max(0, i2 - i1)
        if span_len > best_len:
            best_len = span_len
            best_start, best_end = i1, i2

    # Handle pure insertions where no bad tokens changed (span_len == 0)
    if best_len <= 0:
        # Place a 1-token span at a reasonable neighbor
        pos = min(best_start, len(bad_tokens)) if best_start is not None else 0
        if len(bad_tokens) == 0:
            return 0, 0
        start = min(max(pos - 1, 0), len(bad_tokens) - 1)
        end = min(start + 1, len(bad_tokens))
        return start, end

    return best_start, best_end


def align_spans(
    bad_text: str,
    good_text: str,
    tokenizer,
) -> Tuple[List[str], int, int, int, int]:
    """
    Compute token-level and char-level span of the BAD text that differs from GOOD.
    Returns: (bad_tokens, span_token_start, span_token_end, span_char_start, span_char_end)
    """
    bad_tok = hf_tokenize_with_offsets(bad_text, tokenizer)
    good_tok = hf_tokenize_with_offsets(good_text, tokenizer)
    bad_tokens, bad_offsets = bad_tok.tokens, bad_tok.offsets
    good_tokens = good_tok.tokens

    if len(bad_tokens) == 0:
        # Degenerate; nothing to compare
        return [], 0, 0, 0, 0

    s, e = longest_changed_span_opcodes(bad_tokens, good_tokens)
    s = max(0, min(s, len(bad_tokens) - 1))
    e = max(s + 1, min(e if e > s else s + 1, len(bad_tokens)))

    # Convert to char-span using offsets (inclusive start, exclusive end)
    span_char_start = bad_offsets[s][0]
    span_char_end = bad_offsets[e - 1][1]

    return bad_tokens, s, e, span_char_start, span_char_end


# -----------------------------
# Core Pipeline
# -----------------------------
def load_tokenizer(base_model: str, debug_tiny: bool) -> Optional[object]:
    """Load tokenizer; in debug mode, use a tiny model tokenizer."""
    model_name = TINY_DEBUG_MODEL if debug_tiny else base_model
    if AutoTokenizer is None:
        LOGGER.warning(
            "transformers not available; using whitespace tokenization fallback."
        )
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Ensure pad token
        if tok.pad_token is None:
            tok.pad_token = (
                tok.eos_token if getattr(tok, "eos_token", None) else "[PAD]"
            )
        return tok
    except Exception as e:
        LOGGER.warning(
            f"Failed to load tokenizer {model_name}: {e}. Falling back to whitespace tokenization."
        )
        return None


def choose_degrader(rng: random.Random, use_llm: bool) -> str:
    choices = [
        "grammar_noise",
        "remove_step",
        "terminology_swap",
        "formatting_corrupt",
        "verbosity_change",
    ]
    if use_llm:
        choices.append("llm_stub")
    return rng.choice(choices)


def apply_degrader_once(text: str, degrader_name: str, rng: random.Random) -> str:
    fn = DEGRADERS[degrader_name]
    return fn(text, rng)


def generate_bad_from_good(
    good: str, rng: random.Random, use_llm: bool
) -> Tuple[str, str]:
    """
    Generate bad text and the name of the degrader used. Retry to ensure the output differs.
    """
    for _ in range(5):
        name = choose_degrader(rng, use_llm)
        bad = apply_degrader_once(good, name, rng)
        if bad and bad != good:
            return bad, name
    # As a last resort, force a tiny change
    return good + " maybe", "fallback_append"


def prepare_dataset(
    input_dir: str,
    output_path: str,
    n_per_doc: int,
    seed: int,
    use_llm_degrader: bool,
    base_model: str,
    debug_tiny: bool,
) -> None:
    set_seed(seed)
    safe_ensure_dir(output_path)

    tokenizer = load_tokenizer(base_model, debug_tiny)
    in_files = list_jsonl_files(input_dir)
    if not in_files:
        raise FileNotFoundError(f"No .jsonl files found in {input_dir}")

    num_written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for fp in in_files:
            for obj in read_jsonl(fp):
                doc_id = str(obj.get("id", f"{fp.stem}:{num_written}"))
                good_text = str(obj.get("text", "")).strip()
                source = obj.get("source", None)
                context = str(source).strip() if source else None
                if not good_text:
                    continue

                rng = random.Random(seed ^ hash(doc_id))
                for k in range(n_per_doc):
                    bad_text, degrader_name = generate_bad_from_good(
                        good_text, rng, use_llm_degrader
                    )
                    bad_tokens, s_tok, e_tok, s_chr, e_chr = align_spans(
                        bad_text, good_text, tokenizer
                    )
                    rec = {
                        "id": f"{doc_id}::ex{k}",
                        "context": context,
                        "bad": bad_text,
                        "good": good_text,
                        "bad_tokens": bad_tokens,
                        "span_token_start": int(s_tok),
                        "span_token_end": int(e_tok),
                        "span_char_start": int(s_chr),
                        "span_char_end": int(e_chr),
                        "meta": {
                            "degrader": degrader_name,
                            "tokenizer_fallback": tokenizer is None,
                        },
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    num_written += 1

    LOGGER.info(f"Wrote {num_written} examples to {output_path}")


# -----------------------------
# Minimal Unit Tests
# -----------------------------
SAMPLE_DIR = "sample_docs"


def _write_sample_good_docs() -> str:
    """
    Create sample_docs/ with 3 small JSONL files, one doc per file.
    """
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    samples = [
        {
            "path": os.path.join(SAMPLE_DIR, "doc1.jsonl"),
            "line": {
                "id": "sample-1",
                "source": "API Guide v1",
                "text": "To initialize the client, call Client().connect(endpoint, token) and verify the response is 200 OK.",
            },
        },
        {
            "path": os.path.join(SAMPLE_DIR, "doc2.jsonl"),
            "line": {
                "id": "sample-2",
                "source": "Setup Steps",
                "text": "Install the package with `pip install mylib` and then run `mylib --help` to view commands.",
            },
        },
        {
            "path": os.path.join(SAMPLE_DIR, "doc3.jsonl"),
            "line": {
                "id": "sample-3",
                "source": "Formatting",
                "text": "Use the following code:\n```python\nprint('ready')\n```\nThen deploy.",
            },
        },
    ]
    for s in samples:
        with open(s["path"], "w", encoding="utf-8") as f:
            f.write(json.dumps(s["line"], ensure_ascii=False) + "\n")
    return SAMPLE_DIR


def run_unit_tests() -> None:
    """
    Minimal tests:
    - Build dataset from 3 sample docs, 2 examples per doc.
    - Validate that records are written and spans are within bounds.
    """
    LOGGER.info("Running minimal unit tests...")
    in_dir = _write_sample_good_docs()
    out_path = "sample_dataset_from_tests.jsonl"
    prepare_dataset(
        input_dir=in_dir,
        output_path=out_path,
        n_per_doc=2,
        seed=42,
        use_llm_degrader=True,
        base_model=BASE_MODEL,
        debug_tiny=True,
    )
    # Validate
    n = 0
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert "bad" in rec and "good" in rec and rec["bad"], "Missing text fields"
            assert rec["span_token_end"] >= rec["span_token_start"], "Bad span order"
            assert (
                0 <= rec["span_char_start"] <= rec["span_char_end"] <= len(rec["bad"])
            ), "Char spans out of range"
            n += 1
    assert n == 6, f"Expected 6 examples, got {n}"
    LOGGER.info("Unit tests passed.")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare degraded documentation dataset with span alignment."
    )
    p.add_argument(
        "--input",
        type=str,
        required=False,
        default=SAMPLE_DIR,
        help="Input directory of JSONL files.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=False,
        default="dataset.jsonl",
        help="Output JSONL file.",
    )
    p.add_argument(
        "--n-per-doc",
        type=int,
        default=3,
        help="Number of degraded examples per good doc.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--use-llm-degrader",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="Enable LLM-based degrader stub.",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL,
        help="Base model name for tokenizer.",
    )
    p.add_argument(
        "--debug_tiny",
        action="store_true",
        help="Use tiny tokenizer for speed/offline.",
    )
    p.add_argument(
        "--run-tests",
        action="store_true",
        help="Run minimal unit tests on sample docs.",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    if args.run_tests:
        run_unit_tests()
        return

    if args.input == SAMPLE_DIR and not os.path.exists(SAMPLE_DIR):
        LOGGER.info("No input provided; creating sample docs...")
        _write_sample_good_docs()

    prepare_dataset(
        input_dir=args.input,
        output_path=args.output,
        n_per_doc=args.n_per_doc,
        seed=args.seed,
        use_llm_degrader=args.use_llm_degrader,
        base_model=args.base_model,
        debug_tiny=args.debug_tiny,
    )


if __name__ == "__main__":
    main()
