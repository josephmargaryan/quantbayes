# quantbayes/ball_dp/heads/prototypes.py
from __future__ import annotations

from typing import Tuple, Optional

import numpy as np


def fit_ridge_prototypes(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed-form ridge prototypes minimizing the MEAN objective:

      F(mu) = (1/n) * sum_i ||z_i - mu_{y_i}||^2  +  (lam/2) * sum_c ||mu_c||^2

    Solution per class c:
      mu_c = 2 * sum_{i:y=c} z_i / (2*n_c + lam*n)

    Returns:
      mus: (K,d)
      counts: (K,)
    """
    if lam < 0:
        raise ValueError("lam must be >= 0")
    y = y.astype(np.int64)
    K = int(num_classes)
    d = int(Z.shape[1])
    n_total = int(Z.shape[0])
    if n_total <= 0:
        raise ValueError("Z must have at least 1 row")

    mus = np.zeros((K, d), dtype=np.float32)
    counts = np.zeros((K,), dtype=np.int64)

    for c in range(K):
        idx = np.where(y == c)[0]
        counts[c] = idx.size
        if idx.size == 0:
            continue
        s = Z[idx].sum(axis=0)  # (d,)
        denom = 2.0 * float(idx.size) + float(lam) * float(n_total)
        mus[c] = (2.0 * s) / denom

    return mus, counts


def predict_nearest_prototype(Z: np.ndarray, mus: np.ndarray) -> np.ndarray:
    """
    Nearest prototype under squared Euclidean distance.
    """
    z2 = (Z * Z).sum(axis=1, keepdims=True)  # (N,1)
    m2 = (mus * mus).sum(axis=1, keepdims=True).T  # (1,K)
    D = z2 + m2 - 2.0 * (Z @ mus.T)  # (N,K)
    return D.argmin(axis=1)


def prototypes_sensitivity_l2(
    *, r: float, n_min: int, n_total: int, lam: float
) -> float:
    """
    Exact L2 sensitivity of the closed-form ridge prototypes under ball adjacency,
    under the MEAN objective from fit_ridge_prototypes().

    If a single embedding in class c changes by at most ||z - z'|| <= r, then:
      ||mu_c - mu_c'|| <= 2r / (2 n_c + lam*n_total)

    Worst case is the smallest class size n_min:
      Δ2 = 2r / (2 n_min + lam*n_total)
    """
    if n_min <= 0:
        raise ValueError("n_min must be >= 1")
    if n_total <= 0:
        raise ValueError("n_total must be >= 1")
    if r < 0:
        raise ValueError("r must be >= 0")
    if lam < 0:
        raise ValueError("lam must be >= 0")
    denom = 2.0 * float(n_min) + float(lam) * float(n_total)
    return (2.0 * float(r)) / denom


def ridge_prototypes_replacement_delta(
    *,
    z_old: np.ndarray,
    z_new: np.ndarray,
    y: int,
    counts: np.ndarray,
    lam: float,
    n_total: Optional[int] = None,
) -> np.ndarray:
    """
    Closed-form change in the prototype vector for class y under a single
    label-preserving replacement z_old -> z_new.

    For the MEAN objective used in fit_ridge_prototypes:
      mu_y = 2 * sum_{i:y} z_i / (2 n_y + lam * n_total)

    Replacement changes sum by (z_new - z_old), so:
      mu_y' - mu_y = 2 (z_new - z_old) / (2 n_y + lam * n_total)

    Returns:
      delta_mu: (d,) float32
    """
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if n_total is None:
        n_total = int(np.sum(counts))
    n_total = int(n_total)
    if n_total <= 0:
        raise ValueError("n_total must be >= 1")
    if lam < 0:
        raise ValueError("lam must be >= 0")

    c = int(y)
    n_y = int(counts[c])
    if n_y <= 0:
        raise ValueError(f"class y={c} has zero count")

    denom = 2.0 * float(n_y) + float(lam) * float(n_total)
    z_old = np.asarray(z_old, dtype=np.float32).reshape(-1)
    z_new = np.asarray(z_new, dtype=np.float32).reshape(-1)
    delta = (2.0 / denom) * (z_new - z_old)
    return delta.astype(np.float32, copy=False)


def ridge_prototypes_release_distance_l2(
    *,
    z_old: np.ndarray,
    z_new: np.ndarray,
    y: int,
    counts: np.ndarray,
    lam: float,
    n_total: Optional[int] = None,
) -> float:
    """
    L2 distance between the FULL released prototype matrix vec(mu) before vs after
    replacing one example in class y, under fit_ridge_prototypes().

    Only ONE row changes, so:
      ||vec(mu') - vec(mu)||_2 = ||mu_y' - mu_y||_2
                              = || 2(z_new - z_old) / (2 n_y + lam n_total) ||_2
    """
    delta = ridge_prototypes_replacement_delta(
        z_old=z_old,
        z_new=z_new,
        y=int(y),
        counts=counts,
        lam=float(lam),
        n_total=n_total,
    )
    return float(np.linalg.norm(delta))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ---- Toy data: K Gaussian clusters in R^d ----
    N = 500
    d = 8
    K = 4
    lam = 1.0

    # true (unknown) class centers
    true_centers = rng.normal(size=(K, d)).astype(np.float32)

    # labels in {0, ..., K-1}
    y = rng.integers(0, K, size=N, dtype=np.int64)

    # embeddings: center + noise
    Z = true_centers[y] + 0.5 * rng.normal(size=(N, d)).astype(np.float32)

    # ---- Fit ridge prototypes ----
    mus, counts = fit_ridge_prototypes(Z, y, num_classes=K, lam=lam)
    print("counts per class:", counts)
    print("mus shape:", mus.shape)  # (K, d)

    # ---- Predict by nearest prototype ----
    y_hat = predict_nearest_prototype(Z, mus)
    acc = float(np.mean(y_hat == y))
    print(f"train nearest-prototype accuracy: {acc:.3f}")

    # ---- Sensitivity (ball adjacency) ----
    # r is the max allowed change ||z - z'|| for one example under your adjacency notion
    r = 1.0
    nonzero = counts[counts > 0]
    if nonzero.size == 0:
        raise RuntimeError("All classes are empty (unexpected in this toy example).")

    n_min = int(nonzero.min())
    n_total = int(N)

    sens = prototypes_sensitivity_l2(r=r, n_min=n_min, n_total=n_total, lam=lam)
    print(f"prototypes L2 sensitivity (worst-case over classes): {sens:.6f}")

    # ---- Single label-preserving replacement: z_old -> z_new ----
    i = 0
    c = int(y[i])
    z_old = Z[i].copy()

    # Construct a bounded replacement with ||z_new - z_old|| <= r
    direction = rng.normal(size=(d,)).astype(np.float32)
    direction /= np.linalg.norm(direction) + 1e-12
    step = 0.8 * r  # keep margin below r
    z_new = z_old + step * direction
    print("||z_new - z_old||:", float(np.linalg.norm(z_new - z_old)))

    delta_mu = ridge_prototypes_replacement_delta(
        z_old=z_old,
        z_new=z_new,
        y=c,
        counts=counts,
        lam=lam,
        n_total=n_total,
    )
    print(f"||delta_mu|| (closed-form): {float(np.linalg.norm(delta_mu)):.6f}")

    # Verify delta_mu matches recomputing prototypes after replacement
    Z2 = Z.copy()
    Z2[i] = z_new
    mus2, counts2 = fit_ridge_prototypes(Z2, y, num_classes=K, lam=lam)

    # Only row c should change (up to floating error)
    row_diff = mus2[c] - mus[c]
    full_diff = mus2 - mus

    print(f"||mus2[c] - mus[c]|| (recompute): {float(np.linalg.norm(row_diff)):.6f}")
    print(f"||full mus2 - mus|| (recompute): {float(np.linalg.norm(full_diff)):.6f}")
    print(
        f"||row_diff - delta_mu|| (sanity error): {float(np.linalg.norm(row_diff - delta_mu)):.6e}"
    )

    # Release distance helper should equal the same row-diff norm
    rel_dist = ridge_prototypes_release_distance_l2(
        z_old=z_old,
        z_new=z_new,
        y=c,
        counts=counts,
        lam=lam,
        n_total=n_total,
    )
    print(f"release distance L2 (helper): {rel_dist:.6f}")
