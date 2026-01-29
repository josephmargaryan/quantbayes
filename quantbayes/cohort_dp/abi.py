# quantbayes/cohort_dp/abi.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Protocol
import numpy as np

from .metrics import Metric


@dataclass(frozen=True)
class AdaptiveBallInfo:
    """
    Ball defined as all points within r(z), where r(z) is the distance to the k0-th nearest neighbor.
    """

    r_z: float
    pool: np.ndarray  # (N_pool,)
    pool_dists: np.ndarray  # (N_pool,)
    ball: np.ndarray  # (m,)
    ball_dists: np.ndarray  # (m,)


def compute_adaptive_ball(
    X: np.ndarray,
    metric: Metric,
    z: np.ndarray,
    k0: int,
    *,
    candidates: Optional[np.ndarray] = None,
    r_min: float = 1e-12,
    r_max: float = 1e12,
) -> AdaptiveBallInfo:
    """
    Computes the adaptive ball B(z) using the k0-th order statistic on distances to the candidate pool.

    This matches the draft definition:
      r(z) = distance to k0-th nearest neighbor
      B(z) = {i: dist(z, x_i) <= r(z)}

    Notes:
      - If candidates is provided, the ball is defined relative to that pool.
      - We clamp r(z) into [r_min, r_max] for numerical safety.
    """
    X = np.asarray(X, dtype=float)
    z = np.asarray(z, dtype=float).reshape(1, -1)
    n = int(X.shape[0])

    if k0 <= 0:
        raise ValueError("k0 must be >= 1.")

    if candidates is None:
        pool = np.arange(n, dtype=int)
    else:
        pool = np.asarray(candidates, dtype=int)
        if pool.ndim != 1 or pool.size == 0:
            raise ValueError("candidates must be a non-empty 1D array of indices.")

    d = metric.pairwise(z, X[pool]).reshape(-1)
    if d.size == 0:
        raise ValueError("Empty candidate pool.")

    k0_eff = int(max(1, min(k0, d.size)))
    kth = float(np.partition(d, kth=k0_eff - 1)[k0_eff - 1])
    r_z = float(np.clip(kth, r_min, r_max))

    mask = d <= r_z
    ball = pool[mask]
    ball_dists = d[mask]

    # Safety: ensure ball contains at least k0_eff points
    if ball.size < k0_eff:
        idx = np.argpartition(d, kth=k0_eff - 1)[:k0_eff]
        ball = pool[idx]
        ball_dists = d[idx]
        r_z = float(np.max(ball_dists))
        r_z = float(np.clip(r_z, r_min, r_max))

    return AdaptiveBallInfo(
        r_z=r_z,
        pool=pool.astype(int),
        pool_dists=d.astype(float),
        ball=ball.astype(int),
        ball_dists=ball_dists.astype(float),
    )


class HasSingleDrawPMF(Protocol):
    def single_draw_pmf(
        self, z: np.ndarray, *, candidates: Optional[np.ndarray] = None, k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (support_indices, probs) for the single-draw distribution p_z over indices.
        probs must sum to 1 over support_indices.
        """


def _pmf_to_dict(idxs: np.ndarray, probs: np.ndarray) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for i, p in zip(idxs.tolist(), probs.tolist()):
        ii = int(i)
        pp = float(p)
        if pp > 0:
            out[ii] = out.get(ii, 0.0) + pp
    return out


def estimate_single_draw_pmf_mc(
    retriever: Any,
    z: np.ndarray,
    *,
    candidates: Optional[np.ndarray] = None,
    n_samples: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo estimate of p_z for arbitrary retrievers via repeated k=1 calls.

    Returns a sparse PMF: (unique_indices, probs).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be >= 1.")
    rng = rng or np.random.default_rng(0)

    counts: Dict[int, int] = {}
    for _ in range(int(n_samples)):
        out = retriever.query(z=z, k=1, candidates=candidates)
        if out is None or len(out) == 0:
            continue
        idx = int(np.asarray(out, dtype=int).reshape(-1)[0])
        counts[idx] = counts.get(idx, 0) + 1

    if not counts:
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)

    idxs = np.array(sorted(counts.keys()), dtype=int)
    freqs = np.array([counts[i] for i in idxs.tolist()], dtype=float)
    probs = freqs / float(np.sum(freqs) + 1e-12)
    return idxs, probs


@dataclass(frozen=True)
class ABIMetrics:
    """
    Empirical or exact ABI metrics for a fixed query z.

    delta: mass outside B(z)
    eps_ball_uncond: log(max p_i / min p_i) over i in B(z) using the *unconditional* p_z
    eps_ball_cond: same but after conditioning on being inside B(z)
    """

    m: int
    r_z: float
    delta: float
    eps_ball_uncond: float
    eps_ball_cond: float
    pmax_in_ball_uncond: float


def compute_abi_metrics(
    X: np.ndarray,
    metric: Metric,
    retriever: Any,
    z: np.ndarray,
    *,
    k0: int,
    candidates: Optional[np.ndarray] = None,
    # If retriever doesn't have an exact single_draw_pmf, fall back to MC
    mc_samples: int = 2000,
    rng: Optional[np.random.Generator] = None,
    # Smoothing only used for MC-estimated pmfs to avoid zeros from finite sampling
    mc_smoothing: float = 1e-9,
    # Some retrievers (DP-ish ones) implicitly depend on k; for ABI we typically want k=1
    k_for_pmf: int = 1,
) -> ABIMetrics:
    """
    Compute (k0, delta, eps_ball) for a single query z.

    - Uses exact PMF if retriever exposes single_draw_pmf(...)
    - Otherwise estimates via Monte Carlo.
    """
    rng = rng or np.random.default_rng(0)
    ball_info = compute_adaptive_ball(X, metric, z, k0, candidates=candidates)
    ball = ball_info.ball
    m = int(ball.size)
    if m == 0:
        return ABIMetrics(
            m=0,
            r_z=float(ball_info.r_z),
            delta=1.0,
            eps_ball_uncond=float("inf"),
            eps_ball_cond=float("inf"),
            pmax_in_ball_uncond=0.0,
        )

    # 1) get pmf (exact if available)
    pmf_dict: Dict[int, float]
    if hasattr(retriever, "single_draw_pmf"):
        idxs, probs = retriever.single_draw_pmf(
            z, candidates=candidates, k=int(k_for_pmf)
        )
        pmf_dict = _pmf_to_dict(
            np.asarray(idxs, dtype=int), np.asarray(probs, dtype=float)
        )
    else:
        idxs, probs = estimate_single_draw_pmf_mc(
            retriever, z, candidates=candidates, n_samples=mc_samples, rng=rng
        )
        pmf_dict = _pmf_to_dict(idxs, probs)

    # 2) delta (mass outside ball)
    p_in = 0.0
    p_ball_uncond = np.zeros((m,), dtype=float)
    for t, i in enumerate(ball.tolist()):
        pi = float(pmf_dict.get(int(i), 0.0))
        p_ball_uncond[t] = pi
        p_in += pi
    delta = float(max(0.0, 1.0 - p_in))

    # 3) eps_ball (unconditional)
    if np.all(p_ball_uncond <= 0.0):
        eps_uncond = float("inf")
        pmax_uncond = 0.0
    else:
        # If PMF is MC-estimated, avoid zeros from finite sampling
        if not hasattr(retriever, "single_draw_pmf") and mc_smoothing > 0:
            p_ball_uncond = p_ball_uncond + float(mc_smoothing)

        pmax_uncond = float(np.max(p_ball_uncond))
        pmin_uncond = float(np.min(p_ball_uncond))
        eps_uncond = (
            float("inf")
            if pmin_uncond <= 0.0
            else float(np.log(pmax_uncond / pmin_uncond))
        )

    # 4) eps_ball (conditional on being in ball)
    if p_in <= 0.0:
        eps_cond = float("inf")
    else:
        p_ball_cond = p_ball_uncond / float(np.sum(p_ball_uncond) + 1e-12)
        pmax_c = float(np.max(p_ball_cond))
        pmin_c = float(np.min(p_ball_cond))
        eps_cond = float("inf") if pmin_c <= 0.0 else float(np.log(pmax_c / pmin_c))

    return ABIMetrics(
        m=m,
        r_z=float(ball_info.r_z),
        delta=float(delta),
        eps_ball_uncond=float(eps_uncond),
        eps_ball_cond=float(eps_cond),
        pmax_in_ball_uncond=float(pmax_uncond),
    )
