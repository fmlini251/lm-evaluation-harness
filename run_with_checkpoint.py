#!/usr/bin/env python3
"""Checkpoint/resume wrapper for lm_eval with BATCH-LEVEL checkpointing.

Supports:
  1. Single-GPU with resume on crash
  2. Multi-GPU range splitting: each GPU processes a batch range
  3. Merge mode: combine results from multiple ranges, detect gaps

Usage:
    # Single GPU (resume on crash):
    python run_with_checkpoint.py \
        --model ozaki-hf --model_args ... --tasks mmlu --batch_size 8

    # Multi-GPU (3-way split):
    # GPU 0:
    python run_with_checkpoint.py \
        --model ozaki-hf --model_args ... --tasks mmlu --batch_size 8 \
        --device cuda:0 --batch_range 1:300

    # GPU 1:
    python run_with_checkpoint.py \
        --model ozaki-hf --model_args ... --tasks mmlu --batch_size 8 \
        --device cuda:1 --batch_range 301:700

    # GPU 2:
    python run_with_checkpoint.py \
        --model ozaki-hf --model_args ... --tasks mmlu --batch_size 8 \
        --device cuda:2 --batch_range 701:1000

    # Merge (no GPU needed):
    python run_with_checkpoint.py \
        --model ozaki-hf --model_args ... --tasks mmlu --batch_size 8 \
        --merge

Resume key: (pretrained, dtype, rslt_type, s_lst, batch_size) must match.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

from lm_eval.batch_checkpoint import (
    BatchCheckpointManager,
    _hash_requests,
    load_all_batches,
    find_missing_batches,
)
from lm_eval.evaluator import simple_evaluate
from lm_eval.utils import handle_non_serializable, make_table, simple_parse_args_string

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("checkpoint_runner")


def _make_checkpoint_key(model_args: dict, tasks_name: str, batch_size: int | None) -> str:
    key_fields = {
        "pretrained": str(model_args.get("pretrained", "")),
        "dtype": str(model_args.get("dtype", "")),
        "rslt_type": str(model_args.get("rslt_type", "")),
        "s_lst": str(model_args.get("s_lst", "")),
        "batch_size": str(batch_size if batch_size is not None else ""),
        "tasks": tasks_name,
    }
    key_str = json.dumps(key_fields, sort_keys=True)
    short_hash = hashlib.sha256(key_str.encode()).hexdigest()[:12]

    pretrained_short = model_args.get("pretrained", "model").split("/")[-1]
    bs_str = f"_bs{batch_size}" if batch_size is not None else ""
    return f"ckpt_{pretrained_short}_{key_fields['rslt_type']}_s{key_fields['s_lst']}{bs_str}_{tasks_name}_{short_hash}"


def parse_batch_range(s: str) -> tuple[int, int]:
    """Parse 'start:end' string into (start, end) inclusive tuple."""
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid batch_range format '{s}', expected 'start:end' (e.g. '1:300')")
    return int(parts[0]), int(parts[1])


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run lm_eval with batch-level checkpoint/resume and multi-GPU range split"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_args", type=str, default="")
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None)
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--log_samples", action="store_true", default=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--use_cache", type=str, default=None)
    parser.add_argument("--cache_requests", type=str, default=None, choices=["true", "refresh", "delete"])
    parser.add_argument("--gen_kwargs", type=str, default=None)
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    parser.add_argument("--fewshot_as_multiturn", action="store_true", default=False)
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N batches (default: 1)")
    # Multi-GPU range split
    parser.add_argument("--batch_range", type=str, default=None,
                        help="Batch range to process, e.g. '1:300' (1-based, inclusive)")
    # Merge mode
    parser.add_argument("--merge", action="store_true", default=False,
                        help="Merge results from all batch checkpoint files and report gaps")
    return parser.parse_args()


def _scan_batches(ckpt_dir: Path) -> dict[str, dict[int, dict]]:
    """Scan all per-batch checkpoint files. Returns {method: {batch_idx: data}}."""
    from collections import defaultdict
    import pickle

    all_files = sorted(ckpt_dir.glob("batch_*.pkl"))
    if not all_files:
        return {}

    groups = defaultdict(dict)
    for path in all_files:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            name = path.stem
            method = name.split("_")[1]
            groups[method][data["batch_idx"]] = data
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    return dict(groups)


def _format_ranges(indices: list[int]) -> list[str]:
    """Group consecutive indices into 'start:end' strings."""
    if not indices:
        return []
    ranges = []
    start = end = indices[0]
    for m in indices[1:]:
        if m == end + 1:
            end = m
        else:
            ranges.append(f"{start}:{end}" if start != end else str(start))
            start = end = m
    ranges.append(f"{start}:{end}" if start != end else str(start))
    return ranges


def do_merge(ckpt_dir: Path, args):
    """Scan batch checkpoints, report gaps. If all present, run final evaluation."""
    logger.info(f"Merge mode: scanning {ckpt_dir}")

    groups = _scan_batches(ckpt_dir)
    if not groups:
        logger.error("No batch checkpoint files found!")
        return False

    has_gaps = False
    for method, batches in groups.items():
        max_idx = max(batches.keys())
        missing = find_missing_batches(batches, max_idx)

        logger.info(f"\n{'='*60}")
        logger.info(f"[{method}] {len(batches)}/{max_idx} batches found")

        if missing:
            has_gaps = True
            range_strs = _format_ranges(missing)
            logger.warning(f"  Missing {len(missing)} batches:")
            for r in range_strs:
                logger.warning(f"    --batch_range {r}")
        else:
            total_results = sum(len(b["batch_results"]) for b in batches.values())
            logger.info(f"  All batches present! ({total_results} total results)")

    if has_gaps:
        logger.info("\nFill missing ranges with --batch_range, then run --merge again.")
        return False
    else:
        logger.info("\nAll batches complete.")
        return True


def main():
    args = parse_args()

    model_args_dict = simple_parse_args_string(args.model_args) if args.model_args else {}
    tasks_name = ",".join(sorted(args.tasks))

    ckpt_key = _make_checkpoint_key(model_args_dict, tasks_name, args.batch_size)
    ckpt_dir = Path("checkpoints") / ckpt_key
    logger.info(f"Checkpoint directory: {ckpt_dir}")

    # Parse batch range
    batch_start = None
    batch_end = None
    if args.batch_range:
        batch_start, batch_end = parse_batch_range(args.batch_range)
        logger.info(f"Batch range: [{batch_start}, {batch_end}]")

    # Enable batch-level checkpointing
    BatchCheckpointManager.enable(
        checkpoint_dir=ckpt_dir,
        save_every=args.save_every,
        batch_size=args.batch_size,
        batch_start=batch_start,
        batch_end=batch_end,
    )

    # Merge mode: scan first, then run final evaluation if all batches present
    if args.merge:
        all_complete = do_merge(ckpt_dir, args)
        if not all_complete:
            return
        logger.info("Running final evaluation (all inference from cache, no GPU compute)...")
        # Reset range so all batches are loaded from cache
        BatchCheckpointManager._batch_start = None
        BatchCheckpointManager._batch_end = None

    # Build cache_requests kwargs
    cache_requests_kwargs = {}
    if args.cache_requests:
        from lm_eval.evaluator import request_caching_arg_to_dict
        cache_requests_kwargs = request_caching_arg_to_dict(args.cache_requests)

    # Run evaluation — all tasks together, same batch composition as vanilla lm_eval
    try:
        results = simple_evaluate(
            model=args.model,
            model_args=args.model_args,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=args.device,
            use_cache=args.use_cache,
            log_samples=args.log_samples,
            gen_kwargs=args.gen_kwargs,
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
            **cache_requests_kwargs,
        )
    except (ValueError, TypeError) as e:
        if batch_start is not None:
            # Range mode: post-processing fails because not all batches are present.
            # That's expected — batch files were saved, job done.
            logger.info(f"Range [{batch_start}, {batch_end}] batch files saved.")
            logger.info("Run --merge to check status, or run without --batch_range for final results.")
            return
        raise

    if results is not None:
        results_path = ckpt_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=handle_non_serializable, ensure_ascii=False)
        logger.info(f"Results saved: {results_path}")

        if args.output_path:
            out = Path(args.output_path)
            out.mkdir(parents=True, exist_ok=True)
            out_file = out / "results.json"
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2, default=handle_non_serializable, ensure_ascii=False)

        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))


if __name__ == "__main__":
    main()
