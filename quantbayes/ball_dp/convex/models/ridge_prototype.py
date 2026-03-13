from __future__ import annotations

import dataclasses as dc
from typing import Tuple

import numpy as np


@dc.dataclass
class PrototypeRelease:
    prototypes: np.ndarray
    counts: np.ndarray


def fit_ridge_prototypes(
    z: np.ndarray, y: np.ndarray, *, num_classes: int, lam: float
) -> PrototypeRelease:
    if lam <= 0.0:
        raise ValueError("lam must be > 0")
    z = np.asarray(z, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n_total = int(z.shape[0])
    d = int(z.shape[1])
    mus = np.zeros((num_classes, d), dtype=np.float32)
    counts = np.zeros((num_classes,), dtype=np.int64)
    for c in range(int(num_classes)):
        idx = np.where(y == c)[0]
        counts[c] = idx.size
        if idx.size == 0:
            continue
        s = z[idx].sum(axis=0)
        denom = 2.0 * float(idx.size) + float(lam) * float(n_total)
        mus[c] = (2.0 * s) / denom
    return PrototypeRelease(prototypes=mus, counts=counts)


def prototype_predict(release: PrototypeRelease, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    diffs = z[:, None, :] - release.prototypes[None, :, :]
    d2 = np.sum(diffs**2, axis=-1)
    return np.argmin(d2, axis=1).astype(np.int64)


def prototype_exact_ball_sensitivity(
    *, radius: float, lam: float, n_total: int
) -> float:
    """
    Exact global sensitivity under label-preserving ball adjacency.

    If the missing label is fixed, the denominator for the affected prototype row is
    `2 n_c + lam n`, where `n_c >= 1` because the target itself belongs to the class.
    The worst case is therefore `n_c = 1`, giving `2 / (2 + lam n)`.
    """
    return float((2.0 * radius) / (2.0 + float(lam) * float(n_total)))
