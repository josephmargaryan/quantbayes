# quantbayes/ball_dp/convex/models/ridge_prototype.py

from __future__ import annotations

import dataclasses as dc
from typing import Optional, Sequence, Tuple

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
    The worst case over all possible class-count vectors is therefore `n_c = 1`,
    giving `2 r / (2 + lam n)`.
    """
    return float((2.0 * radius) / (2.0 + float(lam) * float(n_total)))


def prototype_count_aware_ball_sensitivity(
    *, radius: float, lam: float, n_total: int, counts: Sequence[int]
) -> float:
    """Exact ridge-prototype sensitivity conditional on the public label counts.

    Under the label-preserving adjacency relation, labels and therefore class
    counts are invariant along every adjacency edge. If the count vector is treated
    as public/invariant metadata, Gaussian output perturbation may be calibrated on
    each connected component of the adjacency graph. For a fixed count vector,

        Delta(r; counts) = max_c 2 r / (2 n_c + lam n),

    where the maximum is over classes with n_c > 0. Empty classes cannot be the
    label of the modified record and are therefore ignored.
    """
    if radius < 0.0:
        raise ValueError("radius must be >= 0.")
    if lam <= 0.0:
        raise ValueError("lam must be > 0.")
    if n_total <= 0:
        raise ValueError("n_total must be positive.")

    counts_arr = np.asarray(tuple(int(v) for v in counts), dtype=np.int64)
    positive = counts_arr[counts_arr > 0]
    if positive.size == 0:
        raise ValueError("At least one class count must be positive.")
    n_min = int(np.min(positive))
    return float(
        (2.0 * float(radius)) / (2.0 * float(n_min) + float(lam) * float(n_total))
    )


def prototype_instance_ball_sensitivity(
    *, radius: float, lam: float, n_total: int, class_count: int
) -> float:
    """Exact ridge sensitivity for replacing one known-label record in a class.

    Here `class_count` is the full-data count for the affected class, including
    the hidden/modified record. This quantity is useful for attack diagnostics:
    the actual separation among candidate releases for common classes can be much
    smaller than the global worst-case sensitivity.
    """
    if class_count <= 0:
        raise ValueError("class_count must be positive.")
    return float(
        (2.0 * float(radius)) / (2.0 * float(class_count) + float(lam) * float(n_total))
    )


def prototype_known_label_inverse_noise_scale(
    *, sigma: float, lam: float, n_total: int, n_label_minus: int
) -> float:
    """Noise scale after noisily inverting the known-label ridge prototype.

    If the hidden record has label y and the adversary knows D^-, then

        W_y = ((2(n_y^-+1)+lam n)/2) * \tilde mu_y - S_y^- = Z + eta,

    with eta ~ N(0, tau_y^2 I) and

        tau_y = ((2(n_y^-+1)+lam n)/2) * sigma.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if n_total <= 0:
        raise ValueError("n_total must be positive.")
    if n_label_minus < 0:
        raise ValueError("n_label_minus must be >= 0.")
    alpha = 2.0 * (float(n_label_minus) + 1.0) + float(lam) * float(n_total)
    return float(0.5 * alpha * float(sigma))
