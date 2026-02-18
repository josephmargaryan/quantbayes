# quantbayes/ball_dp/reconstruction/audit_identification.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class ProtoCandidateMeans:
    """
    Candidate means for vec(mu) under D^- ∪ {z_candidate}.
    """

    means: np.ndarray  # (M, K*d)
    cands: np.ndarray  # (M, d)
    y: int  # label


def candidate_set_within_radius(
    *,
    Z_pool: np.ndarray,
    y_pool: np.ndarray,
    z_center: np.ndarray,
    y_center: int,
    r: float,
    k_max: int = 50,
    include_center: bool = True,
) -> np.ndarray:
    """
    Candidate set C ⊂ Ball(z,r) from a pool (usually D^-), label-preserving.
    Returns up to k_max candidates (plus optional center).
    """
    Z_pool = np.asarray(Z_pool, dtype=np.float32)
    y_pool = np.asarray(y_pool, dtype=np.int64).reshape(-1)
    z_center = np.asarray(z_center, dtype=np.float32).reshape(-1)
    y_center = int(y_center)
    r = float(r)

    mask = y_pool == y_center
    Zc = Z_pool[mask]
    if Zc.size == 0:
        cands = z_center[None, :]
        return (
            cands if include_center else np.zeros((0, z_center.size), dtype=np.float32)
        )

    d = np.linalg.norm(Zc - z_center[None, :], axis=1)
    ok = d <= r
    Z_ok = Zc[ok]
    d_ok = d[ok]

    if Z_ok.shape[0] == 0:
        cands = z_center[None, :]
        return (
            cands if include_center else np.zeros((0, z_center.size), dtype=np.float32)
        )

    order = np.argsort(d_ok)
    Z_ok = Z_ok[order]
    if int(k_max) > 0:
        Z_ok = Z_ok[: int(k_max)]

    if include_center:
        return np.concatenate([z_center[None, :], Z_ok], axis=0).astype(np.float32)
    return Z_ok.astype(np.float32)


def _class_sums_counts(
    Z: np.ndarray, y: np.ndarray, K: int
) -> Tuple[np.ndarray, np.ndarray]:
    Z = np.asarray(Z, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    sums = np.zeros((K, Z.shape[1]), dtype=np.float64)
    counts = np.zeros((K,), dtype=np.int64)
    for c in range(K):
        idx = np.where(y == c)[0]
        counts[c] = int(idx.size)
        if idx.size:
            sums[c] = Z[idx].sum(axis=0, dtype=np.float64)
    return sums, counts


def proto_candidate_means_vec(
    *,
    Z_minus: np.ndarray,
    y_minus: np.ndarray,
    z_candidates: np.ndarray,
    y_label: int,
    num_classes: int,
    lam: float,
) -> ProtoCandidateMeans:
    """
    Build mean vectors for vec(mu(D^- ∪ {z'})) for each z' in candidates.
    Uses the n_total_full=|D^-|+1 denominators (constant across candidates).
    """
    Z_minus = np.asarray(Z_minus, dtype=np.float32)
    y_minus = np.asarray(y_minus, dtype=np.int64).reshape(-1)
    z_candidates = np.asarray(z_candidates, dtype=np.float32)
    y_label = int(y_label)
    K = int(num_classes)
    lam = float(lam)

    n_minus = int(Z_minus.shape[0])
    n_full = n_minus + 1

    sums, counts = _class_sums_counts(Z_minus, y_minus, K)

    # base prototypes with n_total_full denom
    mu_base = np.zeros((K, Z_minus.shape[1]), dtype=np.float64)
    for c in range(K):
        denom = 2.0 * float(counts[c]) + lam * float(n_full)
        if denom > 0:
            mu_base[c] = (2.0 * sums[c]) / denom

    # candidate-specific mu_y update
    d = Z_minus.shape[1]
    means = np.zeros((z_candidates.shape[0], K * d), dtype=np.float64)

    for i in range(z_candidates.shape[0]):
        mu = mu_base.copy()
        denom_y = 2.0 * float(counts[y_label] + 1) + lam * float(n_full)
        mu[y_label] = (
            2.0 * (sums[y_label] + z_candidates[i].astype(np.float64))
        ) / denom_y
        means[i] = mu.reshape(-1)

    return ProtoCandidateMeans(
        means=means.astype(np.float32),
        cands=z_candidates.astype(np.float32),
        y=y_label,
    )


def gaussian_ml_identify(
    *,
    y_obs: np.ndarray,  # (K*d,)
    means: np.ndarray,  # (M, K*d)
) -> int:
    """
    ML under equal-covariance Gaussians: pick candidate with smallest ||y - mean||^2.
    """
    y_obs = np.asarray(y_obs, dtype=np.float64).reshape(1, -1)
    means = np.asarray(means, dtype=np.float64)
    d2 = np.sum((means - y_obs) ** 2, axis=1)
    return int(np.argmin(d2))


def gaussian_rank_of_true(
    *,
    y_obs: np.ndarray,
    means: np.ndarray,
    true_index: int,
) -> int:
    """
    Rank (1=best) of true hypothesis by squared distance.
    """
    y_obs = np.asarray(y_obs, dtype=np.float64).reshape(1, -1)
    means = np.asarray(means, dtype=np.float64)
    d2 = np.sum((means - y_obs) ** 2, axis=1)
    order = np.argsort(d2)
    pos = int(np.where(order == int(true_index))[0][0])
    return pos + 1


def run_proto_identification_audit(
    *,
    Z_minus: np.ndarray,
    y_minus: np.ndarray,
    z_true: np.ndarray,
    y_true: int,
    Z_pool: np.ndarray,
    y_pool: np.ndarray,
    num_classes: int,
    lam: float,
    r: float,
    sigma: float,
    k_max: int = 50,
    n_trials: int = 200,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Multi-hypothesis identification audit for ridge prototypes + Gaussian output perturbation.

    Candidate set:
      C = {z_true} ∪ {within-label neighbors from pool within radius r} (up to k_max).

    For each trial:
      y_obs = mean(z_true) + N(0, sigma^2 I)
      predict = argmin_c ||y_obs - mean(c)||^2

    Returns:
      top1_acc, mean_rank, candidate_count, plus some diagnostics.
    """
    rng = np.random.default_rng(int(seed))
    z_true = np.asarray(z_true, dtype=np.float32).reshape(-1)
    y_true = int(y_true)

    cands = candidate_set_within_radius(
        Z_pool=np.asarray(Z_pool, dtype=np.float32),
        y_pool=np.asarray(y_pool, dtype=np.int64),
        z_center=z_true,
        y_center=y_true,
        r=float(r),
        k_max=int(k_max),
        include_center=True,
    )
    M = int(cands.shape[0])

    cand_means = proto_candidate_means_vec(
        Z_minus=Z_minus,
        y_minus=y_minus,
        z_candidates=cands,
        y_label=y_true,
        num_classes=int(num_classes),
        lam=float(lam),
    )
    means = cand_means.means  # (M, K*d)

    # true is always index 0 because include_center=True
    true_idx = 0
    mu_true = means[true_idx]

    correct = 0
    ranks: List[int] = []

    for _ in range(int(n_trials)):
        noise = rng.normal(0.0, float(sigma), size=mu_true.shape).astype(np.float32)
        y_obs = mu_true + noise
        pred = gaussian_ml_identify(y_obs=y_obs, means=means)
        correct += int(pred == true_idx)
        ranks.append(
            gaussian_rank_of_true(y_obs=y_obs, means=means, true_index=true_idx)
        )

    return {
        "top1_acc": float(correct) / float(max(1, n_trials)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "M_candidates": int(M),
        "r": float(r),
        "sigma": float(sigma),
        "n_trials": int(n_trials),
    }


if __name__ == "__main__":
    # Tiny synthetic check
    rng = np.random.default_rng(0)
    K, d = 3, 5

    # D^- (fixed)
    Zm = rng.normal(size=(90, d)).astype(np.float32)
    ym = rng.integers(low=0, high=K, size=(90,), dtype=np.int64)

    # true target
    z = rng.normal(size=(d,)).astype(np.float32)
    yc = int(rng.integers(low=0, high=K))

    # pool (use D^- itself)
    Z_pool, y_pool = Zm, ym

    lam = 0.1
    r = 1.0
    sigma = 0.05

    out = run_proto_identification_audit(
        Z_minus=Zm,
        y_minus=ym,
        z_true=z,
        y_true=yc,
        Z_pool=Z_pool,
        y_pool=y_pool,
        num_classes=K,
        lam=lam,
        r=r,
        sigma=sigma,
        k_max=20,
        n_trials=200,
        seed=0,
    )
    print(out)
    print("[OK] audit_identification prototype audit runs.")
