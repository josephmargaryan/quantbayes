# quantbayes/ball_dp/trace_setup.py

from __future__ import annotations

import dataclasses as dc
from typing import Literal, Optional, Sequence

import numpy as np

from .types import Record


@dc.dataclass(frozen=True)
class TargetedTraceBatch:
    target: Record
    batch_indices: np.ndarray
    target_index: int


def prepare_targeted_trace_batch(
    X: np.ndarray,
    y: np.ndarray,
    *,
    target_index: int,
    batch_size: int,
    seed: int = 0,
    batch_indices: Optional[Sequence[int]] = None,
) -> TargetedTraceBatch:
    """Choose a target record and an attacked batch containing it.

    This is an experiment-construction helper only. It does not train anything and
    it does not mutate shapes. The caller is responsible for ensuring X already has
    the shape expected by the attacked model.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    n_total = int(len(X))
    target_index = int(target_index)
    batch_size = int(batch_size)

    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")
    if target_index < 0 or target_index >= n_total:
        raise IndexError(
            f"target_index={target_index} is out of range for dataset size {n_total}."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if batch_size > n_total:
        raise ValueError(f"batch_size={batch_size} exceeds dataset size {n_total}.")

    if batch_indices is None:
        rng = np.random.default_rng(int(seed))
        pool = np.arange(n_total, dtype=np.int64)
        pool = pool[pool != target_index]
        others = rng.choice(pool, size=batch_size - 1, replace=False)
        idx = np.concatenate(
            [
                np.asarray([target_index], dtype=np.int64),
                np.asarray(others, dtype=np.int64),
            ]
        )
    else:
        idx = np.asarray(batch_indices, dtype=np.int64).reshape(-1)
        if idx.size != batch_size:
            raise ValueError(
                f"batch_indices has length {idx.size}, expected batch_size={batch_size}."
            )
        if np.any(idx < 0) or np.any(idx >= n_total):
            raise ValueError("batch_indices contains an out-of-range index.")
        if len(np.unique(idx)) != len(idx):
            raise ValueError("batch_indices must not contain duplicates.")
        if not np.any(idx == target_index):
            raise ValueError(
                f"batch_indices does not contain target_index={target_index}."
            )
        idx = np.concatenate(
            [
                np.asarray([target_index], dtype=np.int64),
                np.asarray(idx[idx != target_index], dtype=np.int64),
            ]
        )

    target = Record(
        features=np.asarray(X[target_index]),
        label=int(y[target_index]),
    )
    return TargetedTraceBatch(
        target=target,
        batch_indices=np.asarray(idx, dtype=np.int64),
        target_index=target_index,
    )


def make_target_inclusion_schedule(
    *,
    num_steps: int,
    batch_indices: Sequence[int],
    mode: Literal["none", "first", "all"] = "first",
) -> tuple[Optional[tuple[int, ...]], ...]:
    """Build the explicit fixed-batch schedule used for oracle trace stress tests.

    mode='none'
        use the natural DP-SGD minibatch process at every step

    mode='first'
        force the attacked batch only at the first step

    mode='all'
        force the attacked batch at every step
    """
    mode = str(mode).lower()
    if mode not in {"none", "first", "all"}:
        raise ValueError("mode must be one of {'none', 'first', 'all'}.")

    num_steps = int(num_steps)
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    batch_tuple = tuple(
        int(v) for v in np.asarray(batch_indices, dtype=np.int64).tolist()
    )

    if mode == "none":
        return tuple(None for _ in range(num_steps))
    if mode == "first":
        return tuple(batch_tuple if t == 0 else None for t in range(num_steps))
    return tuple(batch_tuple for _ in range(num_steps))
