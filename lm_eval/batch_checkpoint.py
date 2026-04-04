"""Batch-level checkpoint/resume for lm_eval model inference.

Supports:
  1. Single-GPU resume: saves per-batch results, skips completed on restart
  2. Multi-GPU range split: each GPU processes a batch range, merge at end

Each batch is saved as an individual file: batch_{method}_{hash}_{idx:05d}.pkl
This enables independent range execution and easy merge/gap detection.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class BatchCheckpointManager:
    """Global singleton managing batch-level checkpoint state."""

    _enabled: bool = False
    _checkpoint_dir: Path | None = None
    _save_every: int = 1
    _batch_size: int | None = None
    # Range: only process batches in [batch_start, batch_end).
    # None means process all.
    _batch_start: int | None = None
    _batch_end: int | None = None

    @classmethod
    def enable(
        cls,
        checkpoint_dir: str | Path,
        save_every: int = 1,
        batch_size: int | None = None,
        batch_start: int | None = None,
        batch_end: int | None = None,
    ):
        cls._enabled = True
        cls._checkpoint_dir = Path(checkpoint_dir)
        cls._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cls._save_every = max(1, save_every)
        cls._batch_size = batch_size
        cls._batch_start = batch_start
        cls._batch_end = batch_end
        range_str = f", range=[{batch_start}, {batch_end})" if batch_start is not None else ""
        logger.info(
            f"Batch checkpoint enabled: dir={cls._checkpoint_dir}, "
            f"batch_size={cls._batch_size}{range_str}"
        )

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled

    @classmethod
    def has_range(cls) -> bool:
        return cls._batch_start is not None

    @classmethod
    def in_range(cls, batch_idx: int) -> bool:
        """Check if batch_idx is within the configured range (1-based)."""
        if cls._batch_start is None:
            return True
        if cls._batch_end is not None:
            return cls._batch_start <= batch_idx <= cls._batch_end
        return batch_idx >= cls._batch_start

    @classmethod
    def past_range(cls, batch_idx: int) -> bool:
        """Check if batch_idx is past the end of the configured range (1-based)."""
        if cls._batch_end is None:
            return False
        return batch_idx > cls._batch_end

    @classmethod
    def get_batch_path(cls, method_name: str, request_hash: str, batch_idx: int) -> Path:
        bs_tag = f"_bs{cls._batch_size}" if cls._batch_size is not None else ""
        return cls._checkpoint_dir / f"batch_{method_name}_{request_hash}{bs_tag}_{batch_idx:05d}.pkl"

    @classmethod
    def should_save(cls, batch_idx: int) -> bool:
        return cls._enabled and (batch_idx % cls._save_every == 0)


def _hash_requests(requests, method_name: str) -> str:
    """Create a deterministic hash from request arguments."""
    n = len(requests)
    if n == 0:
        return hashlib.sha256(method_name.encode()).hexdigest()[:16]

    parts = [method_name, str(n)]
    indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    for idx in indices:
        if idx < n:
            req = requests[idx]
            if hasattr(req, 'args'):
                parts.append(str(req.args)[:200])
            elif isinstance(req, tuple) and len(req) >= 2:
                parts.append(str(req[0])[:200])
            else:
                parts.append(str(req)[:200])

    key = "|".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def save_single_batch(method_name: str, request_hash: str, batch_idx: int,
                      batch_results: list, batch_reorder_indices: list):
    """Save one batch's results and reorder indices to an individual file."""
    if not BatchCheckpointManager.is_enabled():
        return
    if not BatchCheckpointManager.should_save(batch_idx):
        return

    path = BatchCheckpointManager.get_batch_path(method_name, request_hash, batch_idx)
    data = {
        "batch_idx": batch_idx,
        "batch_results": batch_results,
        "batch_reorder_indices": batch_reorder_indices,
    }
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.rename(path)
    logger.info(f"[checkpoint] {method_name}: saved batch {batch_idx} ({len(batch_results)} results)")


def batch_exists(method_name: str, request_hash: str, batch_idx: int) -> bool:
    """Check if a specific batch checkpoint file exists."""
    if not BatchCheckpointManager.is_enabled():
        return False
    path = BatchCheckpointManager.get_batch_path(method_name, request_hash, batch_idx)
    return path.exists()


def load_single_batch(method_name: str, request_hash: str, batch_idx: int) -> dict | None:
    """Load a single batch's checkpoint."""
    if not BatchCheckpointManager.is_enabled():
        return None
    path = BatchCheckpointManager.get_batch_path(method_name, request_hash, batch_idx)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"[checkpoint] Failed to load {path}: {e}")
        return None


def load_all_batches(method_name: str, request_hash: str) -> dict[int, dict]:
    """Load all existing batch checkpoints. Returns {batch_idx: data}."""
    if not BatchCheckpointManager.is_enabled():
        return {}

    bs_tag = f"_bs{BatchCheckpointManager._batch_size}" if BatchCheckpointManager._batch_size is not None else ""
    pattern = f"batch_{method_name}_{request_hash}{bs_tag}_*.pkl"
    result = {}
    for path in BatchCheckpointManager._checkpoint_dir.glob(pattern):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            result[data["batch_idx"]] = data
        except Exception as e:
            logger.warning(f"[checkpoint] Failed to load {path}: {e}")

    if result:
        logger.info(
            f"[checkpoint] {method_name}: loaded {len(result)} batch files "
            f"(batches {min(result)}..{max(result)})"
        )
    return result


def find_missing_batches(existing: dict[int, dict], total_batches: int) -> list[int]:
    """Find batch indices that are missing from the checkpoint set."""
    return sorted(set(range(1, total_batches + 1)) - set(existing.keys()))


def clear_all_batches(method_name: str, request_hash: str):
    """Remove all batch checkpoint files for a given method+hash."""
    if not BatchCheckpointManager.is_enabled():
        return

    bs_tag = f"_bs{BatchCheckpointManager._batch_size}" if BatchCheckpointManager._batch_size is not None else ""
    pattern = f"batch_{method_name}_{request_hash}{bs_tag}_*.pkl"
    count = 0
    for path in BatchCheckpointManager._checkpoint_dir.glob(pattern):
        path.unlink()
        count += 1
    if count:
        logger.info(f"[checkpoint] {method_name}: cleared {count} batch files")
