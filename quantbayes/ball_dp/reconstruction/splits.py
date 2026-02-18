# quantbayes/ball_dp/reconstruction/splits.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class InformedSplit:
    """
    Split for informed-adversary reconstruction:

      D^-  = fixed_idx
      shadow targets = shadow_idx  (for RecoNN)
      eval targets   = target_idx
    """

    fixed_idx: np.ndarray
    shadow_idx: np.ndarray
    target_idx: np.ndarray
    per_class_counts: Dict[int, Dict[str, int]]


def _shuffle(idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).copy()
    rng.shuffle(idx)
    return idx


def make_informed_split(
    y: np.ndarray,
    *,
    n_fixed_per_class: int,
    n_shadow_per_class: int,
    n_target_per_class: int,
    num_classes: Optional[int] = None,
    seed: int = 0,
) -> InformedSplit:
    """
    Stratified disjoint split per class.

    If you only need convex (no RecoNN), set n_shadow_per_class=0.
    """
    rng = np.random.default_rng(int(seed))
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if y.size == 0:
        raise ValueError("y must be non-empty")

    K = int(num_classes) if num_classes is not None else int(np.max(y) + 1)

    fixed_all, shadow_all, target_all = [], [], []
    per_class_counts: Dict[int, Dict[str, int]] = {}

    need = int(n_fixed_per_class + n_shadow_per_class + n_target_per_class)
    if need <= 0:
        raise ValueError("Need positive total per-class split size.")

    for c in range(K):
        idx_c = np.where(y == c)[0]
        idx_c = _shuffle(idx_c, rng)
        if idx_c.size < need:
            raise ValueError(
                f"Class {c} has {idx_c.size} points but need {need} "
                f"(fixed={n_fixed_per_class}, shadow={n_shadow_per_class}, target={n_target_per_class})."
            )

        fixed = idx_c[:n_fixed_per_class]
        shadow = idx_c[n_fixed_per_class : n_fixed_per_class + n_shadow_per_class]
        target = idx_c[
            n_fixed_per_class
            + n_shadow_per_class : n_fixed_per_class
            + n_shadow_per_class
            + n_target_per_class
        ]

        fixed_all.append(fixed)
        shadow_all.append(shadow)
        target_all.append(target)

        per_class_counts[c] = {
            "fixed": int(fixed.size),
            "shadow": int(shadow.size),
            "target": int(target.size),
        }

    fixed_idx = _shuffle(np.concatenate(fixed_all), rng)
    shadow_idx = (
        _shuffle(np.concatenate(shadow_all), rng)
        if n_shadow_per_class > 0
        else np.zeros((0,), dtype=np.int64)
    )
    target_idx = _shuffle(np.concatenate(target_all), rng)

    return InformedSplit(
        fixed_idx=fixed_idx,
        shadow_idx=shadow_idx,
        target_idx=target_idx,
        per_class_counts=per_class_counts,
    )


if __name__ == "__main__":
    # Tiny sanity check
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    sp = make_informed_split(
        y, n_fixed_per_class=4, n_shadow_per_class=3, n_target_per_class=2, seed=0
    )
    print(
        "fixed/shadow/target sizes:",
        sp.fixed_idx.size,
        sp.shadow_idx.size,
        sp.target_idx.size,
    )
    print("per-class:", sp.per_class_counts)
    # ensure disjoint
    assert (
        len(set(sp.fixed_idx.tolist()).intersection(set(sp.target_idx.tolist()))) == 0
    )
    assert (
        len(set(sp.fixed_idx.tolist()).intersection(set(sp.shadow_idx.tolist()))) == 0
    )
    assert (
        len(set(sp.shadow_idx.tolist()).intersection(set(sp.target_idx.tolist()))) == 0
    )
    print("[OK] splits disjoint.")
