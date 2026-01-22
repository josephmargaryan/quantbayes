# cohort_dp/novel_mechanisms.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from .metrics import Metric


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits)
    exps = np.exp(logits - m)
    s = np.sum(exps)
    if s <= 0 or not np.isfinite(s):
        raise FloatingPointError("Softmax normalization failed.")
    return exps / s


def _adaptive_radius_and_ball(
    dists: np.ndarray,
    pool: np.ndarray,
    k0: int,
    r_min: float,
    r_max: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Given distances from query to pool points:
      - pick r_z as distance to the k0-th nearest neighbor (k0>=1),
      - form ball indices where dist <= r_z.

    Returns: (r_z, ball_indices, ball_dists)
    """
    n = int(dists.shape[0])
    if n == 0:
        raise ValueError("Empty candidate pool.")

    k0_eff = int(max(1, min(k0, n)))
    kth = float(np.partition(dists, kth=k0_eff - 1)[k0_eff - 1])
    r_z = float(np.clip(kth, r_min, r_max))

    mask = dists <= r_z
    ball = pool[mask]
    ball_dists = dists[mask]

    # Safety: ball should have at least k0_eff points (ties/precision can break it rarely)
    if ball.size < k0_eff:
        # Expand slightly by including the closest k0_eff points
        idx = np.argpartition(dists, kth=k0_eff - 1)[:k0_eff]
        ball = pool[idx]
        ball_dists = dists[idx]
        r_z = float(np.max(ball_dists))
        r_z = float(np.clip(r_z, r_min, r_max))

    return r_z, ball.astype(int), ball_dists.astype(float)


@dataclass
class AdaptiveBallUniformRetriever:
    """
    Novel mechanism #1 (goal-targeting):
      - Choose adaptive radius r(z) so ball contains >= k0 points.
      - Return k samples uniformly from the ball (without replacement when possible).

    Intuition:
      - Enforces "hide among at least k0 neighbors" by construction.
      - Reduces exact-ID collapse because the target is just one of many.
    """

    X: np.ndarray
    metric: Metric
    k0: int
    rng: np.random.Generator
    r_min: float = 1e-6
    r_max: float = 1e9
    eps_total: float = 0.0  # not used; kept for a unified interface

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        if self.k0 <= 0:
            raise ValueError("k0 must be >= 1.")
        if self.r_min <= 0:
            raise ValueError("r_min must be > 0.")
        if self.r_max <= self.r_min:
            raise ValueError("r_max must be > r_min.")

    def privacy_cost(self, k: int) -> float:
        # This is a "hide-in-ball" randomized mechanism; not standard DP accounting here.
        return float(self.eps_total)

    def query(
        self, z: np.ndarray, k: int, candidates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if k <= 0:
            raise ValueError("k must be >= 1.")

        z = np.asarray(z, dtype=float).reshape(1, -1)
        if candidates is None:
            pool = np.arange(self.X.shape[0], dtype=int)
        else:
            pool = np.asarray(candidates, dtype=int)
            if pool.ndim != 1 or pool.size == 0:
                raise ValueError("candidates must be a non-empty 1D index array.")

        dists = self.metric.pairwise(z, self.X[pool]).reshape(-1)

        k_ball = int(max(self.k0, k))
        _, ball, _ = _adaptive_radius_and_ball(
            dists, pool, k_ball, self.r_min, self.r_max
        )

        if ball.size == 0:
            ball = pool

        # If pool itself is smaller than k, can't sample k unique -> sample with replacement
        if pool.size < k:
            return self.rng.choice(pool, size=int(k), replace=True).astype(int)

        # Ensure ball has at least k; otherwise fall back to pool
        if ball.size < k:
            ball = pool

        return self.rng.choice(ball, size=int(k), replace=False).astype(int)


@dataclass
class AdaptiveBallExponentialRetriever:
    """
    Novel mechanism #2:
      - Adaptive radius r(z) so ball contains >= k0 points.
      - Within-ball Exponential Mechanism (EM), splitting eps_total over k draws.

    More utility than uniform-in-ball, but more risk of collapsing to the nearest neighbor.
    """

    X: np.ndarray
    metric: Metric
    k0: int
    eps_total: float
    rng: np.random.Generator
    r_min: float = 1e-6
    r_max: float = 1e9

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        if self.k0 <= 0:
            raise ValueError("k0 must be >= 1.")
        if self.eps_total <= 0:
            raise ValueError("eps_total must be > 0.")
        if self.r_min <= 0:
            raise ValueError("r_min must be > 0.")
        if self.r_max <= self.r_min:
            raise ValueError("r_max must be > r_min.")

    def privacy_cost(self, k: int) -> float:
        return float(self.eps_total)

    def query(
        self, z: np.ndarray, k: int, candidates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if k <= 0:
            raise ValueError("k must be >= 1.")
        z = np.asarray(z, dtype=float).reshape(1, -1)

        if candidates is None:
            pool = np.arange(self.X.shape[0], dtype=int)
        else:
            pool = np.asarray(candidates, dtype=int)
            if pool.ndim != 1 or pool.size == 0:
                raise ValueError("candidates must be a non-empty 1D index array.")

        # If pool too small, fall back to with-replacement sampling
        if pool.size < k:
            return self.rng.choice(pool, size=int(k), replace=True).astype(int)

        dists = self.metric.pairwise(z, self.X[pool]).reshape(-1)
        k_ball = int(max(self.k0, k))
        r_z, ball, ball_dists = _adaptive_radius_and_ball(
            dists, pool, k_ball, self.r_min, self.r_max
        )

        # Ensure ball supports k draws without replacement
        if ball.size < k:
            ball = pool
            ball_dists = dists

        eps_draw = self.eps_total / float(k)
        remaining = ball.copy()
        remaining_d = ball_dists.copy()

        chosen: List[int] = []
        for _ in range(min(k, remaining.size)):
            coef = eps_draw / (2.0 * max(r_z, self.r_min))
            logits = -coef * remaining_d
            p = _stable_softmax(logits)
            j = int(self.rng.choice(remaining.size, p=p))
            chosen.append(int(remaining[j]))
            remaining = np.delete(remaining, j)
            remaining_d = np.delete(remaining_d, j)

        return np.array(chosen, dtype=int)


@dataclass
class AdaptiveBallMixedRetriever:
    """
    Novel mechanism #3 (often best in practice):
      - Adaptive radius r(z) so ball contains >= k0 points.
      - Mix uniform-in-ball with exponential-in-ball to enforce diversity:

        p = mix_uniform * Uniform(ball) + (1-mix_uniform) * Softmax(-coef * dist)

    mix_uniform close to 1 -> more diversity (less exact collapse), lower utility.
    """

    X: np.ndarray
    metric: Metric
    k0: int
    eps_total: float
    mix_uniform: float
    rng: np.random.Generator
    r_min: float = 1e-6
    r_max: float = 1e9

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        if self.k0 <= 0:
            raise ValueError("k0 must be >= 1.")
        if self.eps_total <= 0:
            raise ValueError("eps_total must be > 0.")
        if not (0.0 <= self.mix_uniform <= 1.0):
            raise ValueError("mix_uniform must be in [0, 1].")

    def privacy_cost(self, k: int) -> float:
        return float(self.eps_total)

    def query(
        self, z: np.ndarray, k: int, candidates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if k <= 0:
            raise ValueError("k must be >= 1.")
        z = np.asarray(z, dtype=float).reshape(1, -1)

        if candidates is None:
            pool = np.arange(self.X.shape[0], dtype=int)
        else:
            pool = np.asarray(candidates, dtype=int)
            if pool.ndim != 1 or pool.size == 0:
                raise ValueError("candidates must be a non-empty 1D index array.")

        # If pool too small, fall back to with-replacement sampling
        if pool.size < k:
            return self.rng.choice(pool, size=int(k), replace=True).astype(int)

        dists = self.metric.pairwise(z, self.X[pool]).reshape(-1)
        k_ball = int(max(self.k0, k))
        r_z, ball, ball_dists = _adaptive_radius_and_ball(
            dists, pool, k_ball, self.r_min, self.r_max
        )

        if ball.size < k:
            ball = pool
            ball_dists = dists

        eps_draw = self.eps_total / float(k)
        remaining = ball.copy()
        remaining_d = ball_dists.copy()

        chosen: List[int] = []
        for _ in range(min(k, remaining.size)):
            coef = eps_draw / (2.0 * max(r_z, self.r_min))
            logits = -coef * remaining_d
            p_exp = _stable_softmax(logits)
            p_uni = np.full_like(p_exp, 1.0 / float(p_exp.size))
            p = self.mix_uniform * p_uni + (1.0 - self.mix_uniform) * p_exp
            p = p / np.sum(p)

            j = int(self.rng.choice(remaining.size, p=p))
            chosen.append(int(remaining[j]))
            remaining = np.delete(remaining, j)
            remaining_d = np.delete(remaining_d, j)

        return np.array(chosen, dtype=int)
