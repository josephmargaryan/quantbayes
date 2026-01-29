# quantbayes/cohort_dp/novel_mechanisms.py
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
        idx = np.argpartition(dists, kth=k0_eff - 1)[:k0_eff]
        ball = pool[idx]
        ball_dists = dists[idx]
        r_z = float(np.max(ball_dists))
        r_z = float(np.clip(r_z, r_min, r_max))

    return r_z, ball.astype(int), ball_dists.astype(float)


def _normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    s = float(np.sum(p))
    if s <= 0 or not np.isfinite(s):
        raise FloatingPointError("Probability normalization failed.")
    return p / s


def _abi_optimal_probs_two_level(d_sorted: np.ndarray, eps_ball: float) -> np.ndarray:
    """
    ABI-optimal distribution (expected-distance minimizing) under within-ball skew constraint:

      max_i log(p_i/p_j) <= eps_ball

    Over a fixed ball with distances sorted ascending: d_1 <= ... <= d_m,
    the optimum is a two-level distribution:
      - first t points get p_max
      - remaining m-t points get p_min
    for some t in {0,...,m}.

    We brute-force t (O(m)) which is cheap for typical k0 (tens to hundreds).
    """
    d_sorted = np.asarray(d_sorted, dtype=float).reshape(-1)
    m = int(d_sorted.size)
    if m <= 0:
        return np.zeros((0,), dtype=float)

    if eps_ball <= 0:
        return np.full((m,), 1.0 / float(m), dtype=float)

    E = float(np.exp(float(eps_ball)))
    prefix = np.concatenate([[0.0], np.cumsum(d_sorted)])  # prefix[t] = sum_{i< t} d_i
    total = float(prefix[-1])

    best_t = 0
    best_val = float("inf")

    for t in range(0, m + 1):
        denom = float(m + t * (E - 1.0))
        pmin = 1.0 / denom
        pmax = E / denom

        sum_close = float(prefix[t])
        sum_far = float(total - sum_close)
        expected = pmax * sum_close + pmin * sum_far

        if expected < best_val:
            best_val = expected
            best_t = t

    denom = float(m + best_t * (E - 1.0))
    pmin = 1.0 / denom
    pmax = E / denom

    p = np.full((m,), pmin, dtype=float)
    if best_t > 0:
        p[:best_t] = pmax
    return _normalize_probs(p)


@dataclass
class AdaptiveBallUniformRetriever:
    """
    AB-Uniform:
      - define ball B(z) using k0-th NN distance
      - sample uniformly from B(z)

    By default this matches the draft definition: the ball is defined by k0 only.
    If ball_includes_k=True, the ball is defined using max(k0, k) (older behavior).
    """

    X: np.ndarray
    metric: Metric
    k0: int
    rng: np.random.Generator
    r_min: float = 1e-6
    r_max: float = 1e9
    eps_total: float = 0.0  # not used for the distribution; kept for unified interface
    ball_includes_k: bool = False

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        if self.k0 <= 0:
            raise ValueError("k0 must be >= 1.")
        if self.r_min <= 0:
            raise ValueError("r_min must be > 0.")
        if self.r_max <= self.r_min:
            raise ValueError("r_max must be > r_min.")

    def privacy_cost(self, k: int) -> float:
        # ABI mechanism (not DP by default); keep eps_total as an accounting hook if you want.
        return float(self.eps_total)

    def _ball(
        self, z: np.ndarray, candidates: Optional[np.ndarray], k: int
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        z1 = np.asarray(z, dtype=float).reshape(1, -1)
        if candidates is None:
            pool = np.arange(self.X.shape[0], dtype=int)
        else:
            pool = np.asarray(candidates, dtype=int)
            if pool.ndim != 1 or pool.size == 0:
                raise ValueError("candidates must be a non-empty 1D index array.")
        dists = self.metric.pairwise(z1, self.X[pool]).reshape(-1)

        k_ball = int(max(self.k0, k)) if self.ball_includes_k else int(self.k0)
        r_z, ball, ball_dists = _adaptive_radius_and_ball(
            dists, pool, k_ball, self.r_min, self.r_max
        )
        return r_z, ball, ball_dists

    def single_draw_pmf(
        self, z: np.ndarray, *, candidates: Optional[np.ndarray] = None, k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, ball, _ = self._ball(z, candidates, k=int(k))
        m = int(ball.size)
        if m <= 0:
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)
        return ball.astype(int), np.full((m,), 1.0 / float(m), dtype=float)

    def query(
        self, z: np.ndarray, k: int, candidates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if k <= 0:
            raise ValueError("k must be >= 1.")
        _, ball, _ = self._ball(z, candidates, k=int(k))
        if ball.size == 0:
            # fallback: sample from pool if ball empty (should not happen unless pool empty)
            if candidates is None:
                pool = np.arange(self.X.shape[0], dtype=int)
            else:
                pool = np.asarray(candidates, dtype=int)
            return self.rng.choice(pool, size=int(k), replace=True).astype(int)

        # without replacement when possible, otherwise with replacement
        replace = bool(k > int(ball.size))
        return self.rng.choice(ball, size=int(k), replace=replace).astype(int)


@dataclass
class AdaptiveBallExponentialRetriever:
    """
    AB-Exp (within-ball exponential bias):

      p(i|z) ∝ exp(-gamma(z) * dist(z, x_i)),   i in B(z)

    Parameterization:
      - If gamma is provided, use it (draft-style).
      - Else, compute gamma(z) from eps_total using a DP-inspired scaling:
            gamma(z) = (eps_total / max(1,k)) / (2 * r(z))
        (this keeps older "eps_total per call" sweeps working).

    Note: This mechanism is primarily an ABI mechanism, not a record-level DP guarantee.
    """

    X: np.ndarray
    metric: Metric
    k0: int
    eps_total: float
    rng: np.random.Generator
    gamma: Optional[float] = None
    r_min: float = 1e-6
    r_max: float = 1e9
    ball_includes_k: bool = False

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        if self.k0 <= 0:
            raise ValueError("k0 must be >= 1.")
        if self.gamma is None and self.eps_total <= 0:
            raise ValueError("Provide eps_total > 0 or gamma > 0.")
        if self.gamma is not None and self.gamma <= 0:
            raise ValueError("gamma must be > 0.")
        if self.r_min <= 0:
            raise ValueError("r_min must be > 0.")
        if self.r_max <= self.r_min:
            raise ValueError("r_max must be > r_min.")

    def privacy_cost(self, k: int) -> float:
        return float(self.eps_total)

    def _ball_and_gamma(
        self, z: np.ndarray, candidates: Optional[np.ndarray], k: int
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        z1 = np.asarray(z, dtype=float).reshape(1, -1)
        if candidates is None:
            pool = np.arange(self.X.shape[0], dtype=int)
        else:
            pool = np.asarray(candidates, dtype=int)
            if pool.ndim != 1 or pool.size == 0:
                raise ValueError("candidates must be a non-empty 1D index array.")
        dists = self.metric.pairwise(z1, self.X[pool]).reshape(-1)

        k_ball = int(max(self.k0, k)) if self.ball_includes_k else int(self.k0)
        r_z, ball, ball_dists = _adaptive_radius_and_ball(
            dists, pool, k_ball, self.r_min, self.r_max
        )

        if self.gamma is not None:
            gamma_z = float(self.gamma)
        else:
            # DP-inspired scaling used in earlier codepaths
            gamma_z = float(
                (self.eps_total / float(max(1, k))) / (2.0 * max(r_z, self.r_min))
            )
        return r_z, ball, ball_dists, gamma_z

    def single_draw_pmf(
        self, z: np.ndarray, *, candidates: Optional[np.ndarray] = None, k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, ball, ball_dists, gamma_z = self._ball_and_gamma(z, candidates, k=int(k))
        m = int(ball.size)
        if m <= 0:
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)
        logits = -float(gamma_z) * ball_dists
        p = _stable_softmax(logits)
        return ball.astype(int), p.astype(float)

    def query(
        self, z: np.ndarray, k: int, candidates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if k <= 0:
            raise ValueError("k must be >= 1.")

        _, ball, p_dists, gamma_z = self._ball_and_gamma(z, candidates, k=int(k))
        if ball.size == 0:
            if candidates is None:
                pool = np.arange(self.X.shape[0], dtype=int)
            else:
                pool = np.asarray(candidates, dtype=int)
            return self.rng.choice(pool, size=int(k), replace=True).astype(int)

        logits = -float(gamma_z) * p_dists
        p = _stable_softmax(logits)

        replace = bool(k > int(ball.size))
        return self.rng.choice(ball, size=int(k), replace=replace, p=p).astype(int)


@dataclass
class AdaptiveBallMixedRetriever:
    """
    AB-Mix (draft-style):

      p(i|z) = lambda * Uniform(B(z)) + (1-lambda) * Softmax(-gamma(z)*dist),  i in B(z)

    Parameters:
      - mix_uniform == lambda in your draft
      - gamma:
          * if provided -> use it (draft-style, easiest to interpret)
          * else -> DP-inspired gamma(z) computed from eps_total similar to AB-Exp:
                gamma(z) = (eps_total / max(1,k)) / (2 * r(z))

    This makes it easy to either:
      (A) treat eps_total as a "sharpness" sweep knob (older behavior), or
      (B) treat gamma as the explicit sharpness knob (clean ABI story).
    """

    X: np.ndarray
    metric: Metric
    k0: int
    eps_total: float
    mix_uniform: float
    rng: np.random.Generator
    gamma: Optional[float] = None
    r_min: float = 1e-6
    r_max: float = 1e9
    ball_includes_k: bool = False

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        if self.k0 <= 0:
            raise ValueError("k0 must be >= 1.")
        if not (0.0 <= self.mix_uniform <= 1.0):
            raise ValueError("mix_uniform must be in [0, 1].")
        if self.gamma is None and self.eps_total <= 0:
            raise ValueError("Provide eps_total > 0 or gamma > 0.")
        if self.gamma is not None and self.gamma <= 0:
            raise ValueError("gamma must be > 0.")
        if self.r_min <= 0:
            raise ValueError("r_min must be > 0.")
        if self.r_max <= self.r_min:
            raise ValueError("r_max must be > r_min.")

    def privacy_cost(self, k: int) -> float:
        return float(self.eps_total)

    def _ball_and_gamma(
        self, z: np.ndarray, candidates: Optional[np.ndarray], k: int
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        z1 = np.asarray(z, dtype=float).reshape(1, -1)
        if candidates is None:
            pool = np.arange(self.X.shape[0], dtype=int)
        else:
            pool = np.asarray(candidates, dtype=int)
            if pool.ndim != 1 or pool.size == 0:
                raise ValueError("candidates must be a non-empty 1D index array.")
        dists = self.metric.pairwise(z1, self.X[pool]).reshape(-1)

        k_ball = int(max(self.k0, k)) if self.ball_includes_k else int(self.k0)
        r_z, ball, ball_dists = _adaptive_radius_and_ball(
            dists, pool, k_ball, self.r_min, self.r_max
        )

        if self.gamma is not None:
            gamma_z = float(self.gamma)
        else:
            gamma_z = float(
                (self.eps_total / float(max(1, k))) / (2.0 * max(r_z, self.r_min))
            )
        return r_z, ball, ball_dists, gamma_z

    def single_draw_pmf(
        self, z: np.ndarray, *, candidates: Optional[np.ndarray] = None, k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, ball, ball_dists, gamma_z = self._ball_and_gamma(z, candidates, k=int(k))
        m = int(ball.size)
        if m <= 0:
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)

        logits = -float(gamma_z) * ball_dists
        p_exp = _stable_softmax(logits)
        p_uni = np.full((m,), 1.0 / float(m), dtype=float)
        p = float(self.mix_uniform) * p_uni + (1.0 - float(self.mix_uniform)) * p_exp
        p = _normalize_probs(p)
        return ball.astype(int), p.astype(float)

    def query(
        self, z: np.ndarray, k: int, candidates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if k <= 0:
            raise ValueError("k must be >= 1.")

        ball, p = self.single_draw_pmf(z, candidates=candidates, k=int(k))
        if ball.size == 0:
            if candidates is None:
                pool = np.arange(self.X.shape[0], dtype=int)
            else:
                pool = np.asarray(candidates, dtype=int)
            return self.rng.choice(pool, size=int(k), replace=True).astype(int)

        replace = bool(k > int(ball.size))
        return self.rng.choice(ball, size=int(k), replace=replace, p=p).astype(int)


@dataclass
class AdaptiveBallOptimalSkewRetriever:
    """
    NEW: AB-Optimal (ABI-optimal inside-ball distribution)

    Mechanism:
      - Build ball B(z) using k0-th NN distance
      - Choose p(i|z) that (approximately) minimizes expected distance
        subject to within-ball skew <= eps_ball:
            max_{i,i' in B(z)} log(p_i/p_i') <= eps_ball

    This gives you a principled "middle point" between AB-Uniform (eps_ball=0) and
    near-deterministic (eps_ball large).

    This is a very strong candidate for the paper's "needle mover" because:
      - it has an explicit guarantee by construction
      - it has a clean optimization interpretation
    """

    X: np.ndarray
    metric: Metric
    k0: int
    eps_ball: float
    rng: np.random.Generator
    r_min: float = 1e-6
    r_max: float = 1e9
    ball_includes_k: bool = False

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        if self.k0 <= 0:
            raise ValueError("k0 must be >= 1.")
        if self.eps_ball < 0:
            raise ValueError("eps_ball must be >= 0.")
        if self.r_min <= 0:
            raise ValueError("r_min must be > 0.")
        if self.r_max <= self.r_min:
            raise ValueError("r_max must be > r_min.")

    def privacy_cost(self, k: int) -> float:
        # ABI mechanism (not DP). Keep cost at 0 unless you explicitly want accounting.
        return 0.0

    def _ball(
        self, z: np.ndarray, candidates: Optional[np.ndarray], k: int
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        z1 = np.asarray(z, dtype=float).reshape(1, -1)
        if candidates is None:
            pool = np.arange(self.X.shape[0], dtype=int)
        else:
            pool = np.asarray(candidates, dtype=int)
            if pool.ndim != 1 or pool.size == 0:
                raise ValueError("candidates must be a non-empty 1D index array.")
        dists = self.metric.pairwise(z1, self.X[pool]).reshape(-1)

        k_ball = int(max(self.k0, k)) if self.ball_includes_k else int(self.k0)
        r_z, ball, ball_dists = _adaptive_radius_and_ball(
            dists, pool, k_ball, self.r_min, self.r_max
        )
        return r_z, ball, ball_dists

    def single_draw_pmf(
        self, z: np.ndarray, *, candidates: Optional[np.ndarray] = None, k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, ball, ball_dists = self._ball(z, candidates, k=int(k))
        m = int(ball.size)
        if m <= 0:
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)

        order = np.argsort(ball_dists)
        ball_sorted = ball[order]
        d_sorted = ball_dists[order]

        p_sorted = _abi_optimal_probs_two_level(d_sorted, float(self.eps_ball))
        return ball_sorted.astype(int), p_sorted.astype(float)

    def query(
        self, z: np.ndarray, k: int, candidates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if k <= 0:
            raise ValueError("k must be >= 1.")

        ball, p = self.single_draw_pmf(z, candidates=candidates, k=int(k))
        if ball.size == 0:
            if candidates is None:
                pool = np.arange(self.X.shape[0], dtype=int)
            else:
                pool = np.asarray(candidates, dtype=int)
            return self.rng.choice(pool, size=int(k), replace=True).astype(int)

        replace = bool(k > int(ball.size))
        return self.rng.choice(ball, size=int(k), replace=replace, p=p).astype(int)
