# quantbayes/retrieval_dp/mechanisms.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np

from .metrics import ScoreType, dot_scores, neg_l2_scores, clip_l2_rows
from .sensitivity import score_sensitivity


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    scores: (B,m) returns idx: (B,k) sorted by descending score.
    """
    if scores.ndim != 2:
        raise ValueError("scores must be 2D (B,m).")
    B, m = scores.shape
    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1.")
    k_eff = min(k, m)

    idx_part = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
    row = np.arange(B)[:, None]
    idx_sorted = idx_part[row, np.argsort(-scores[row, idx_part], axis=1)]
    return idx_sorted.astype(int)


def gaussian_sigma_classic(delta_u: float, eps: float, delta: float) -> float:
    """
    Classic Gaussian mechanism calibration for releasing a vector with L2 sensitivity delta_u:
      sigma >= (Δ/ε) * sqrt(2 log(1.25/δ)).
    """
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1).")
    Delta = float(delta_u)
    return (Delta / float(eps)) * np.sqrt(2.0 * np.log(1.25 / float(delta)))


def gaussian_sigma(
    delta_u: float,
    eps: float,
    delta: float,
    *,
    method: Literal["classic", "analytic"] = "classic",
    tol: float = 1e-12,
) -> float:
    """
    Calibrate sigma for Gaussian noisy-score release with L2 sensitivity `delta_u`.

    - "classic": sigma >= (Δ/ε) * sqrt(2 log(1.25/δ))
    - "analytic": Balle & Wang (2018) analytic calibration (minimal sigma), using
      quantbayes.ball_dp.analytical_gaussian_mechanism.calibrate_analytic_gaussian
    """
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1).")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")
    if float(delta_u) < 0.0:
        raise ValueError("delta_u must be >= 0.")

    if method == "classic":
        return gaussian_sigma_classic(float(delta_u), float(eps), float(delta))

    if method == "analytic":
        from quantbayes.ball_dp.analytical_gaussian_mechanism import (
            calibrate_analytic_gaussian,
        )

        return calibrate_analytic_gaussian(
            epsilon=float(eps),
            delta=float(delta),
            GS=float(delta_u),
            tol=float(tol),
        )

    raise ValueError(f"Unknown method: {method!r}. Use 'classic' or 'analytic'.")


@dataclass
class NonPrivateTopKRetriever:
    """
    Deterministic top-k retrieval.
    """

    V: np.ndarray  # (m,d) corpus embeddings
    score: ScoreType = "dot"
    q_norm_bound: Optional[float] = (
        None  # optionally clip queries (recommended if score="dot")
    )

    def __post_init__(self) -> None:
        self.V = np.asarray(self.V, dtype=np.float32)

    def query_many(
        self, Q: np.ndarray, k: int, *, candidates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        Q = np.asarray(Q, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q[None, :]

        if self.q_norm_bound is not None:
            Q = clip_l2_rows(Q, float(self.q_norm_bound))

        if candidates is None:
            V = self.V
            base = None
        else:
            cand = np.asarray(candidates, dtype=int)
            if cand.ndim != 1 or cand.size == 0:
                raise ValueError("candidates must be a non-empty 1D array.")
            V = self.V[cand]
            base = cand

        scores = dot_scores(Q, V) if self.score == "dot" else neg_l2_scores(Q, V)
        idx_local = _topk_indices(scores, k)
        if base is None:
            return idx_local
        return base[idx_local]


@dataclass
class NoisyScoresTopKLaplaceRetriever:
    """
    Pure ε-DP (w.r.t. ball adjacency at radius r) by:
      1) compute full score vector u(C,·)
      2) add i.i.d. Laplace(0, b) noise to ALL scores with b = Δu/ε
      3) return top-k indices (post-processing)

    This is "release noisy score vector" (stronger than noisy-max), then post-process.
    """

    V: np.ndarray
    score: ScoreType
    r: float
    eps: float
    rng: np.random.Generator
    q_norm_bound: float = 1.0  # used only when score="dot"

    def __post_init__(self) -> None:
        self.V = np.asarray(self.V, dtype=np.float32)
        if self.eps <= 0:
            raise ValueError("eps must be > 0.")
        if self.r < 0:
            raise ValueError("r must be >= 0.")
        if self.q_norm_bound <= 0:
            raise ValueError("q_norm_bound must be > 0.")

    def query_many(
        self, Q: np.ndarray, k: int, *, candidates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        Q = np.asarray(Q, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q[None, :]

        # enforce query bound if dot scoring
        if self.score == "dot":
            Q = clip_l2_rows(Q, float(self.q_norm_bound))

        if candidates is None:
            V = self.V
            base = None
        else:
            cand = np.asarray(candidates, dtype=int)
            if cand.ndim != 1 or cand.size == 0:
                raise ValueError("candidates must be a non-empty 1D array.")
            V = self.V[cand]
            base = cand

        scores = dot_scores(Q, V) if self.score == "dot" else neg_l2_scores(Q, V)

        Delta_u = score_sensitivity(
            self.score, r=float(self.r), q_norm_bound=float(self.q_norm_bound)
        )
        b = float(Delta_u) / float(self.eps)

        noise = self.rng.laplace(loc=0.0, scale=b, size=scores.shape).astype(
            scores.dtype, copy=False
        )
        noisy = scores + noise

        idx_local = _topk_indices(noisy, k)
        if base is None:
            return idx_local
        return base[idx_local]


@dataclass
class NoisyScoresTopKGaussianRetriever:
    """
    (ε,δ)-DP (w.r.t. ball adjacency at radius r) by:
      - compute score vector u(C,·)
      - add i.i.d. N(0, sigma^2) noise to ALL scores with sigma calibrated from Δu
      - return top-k indices (post-processing)
    """

    V: np.ndarray
    score: ScoreType
    r: float
    eps: float
    delta: float
    rng: np.random.Generator
    q_norm_bound: float = 1.0  # used only when score="dot"

    # NEW:
    sigma_method: Literal["classic", "analytic"] = "classic"
    sigma_tol: float = 1e-12

    def __post_init__(self) -> None:
        self.V = np.asarray(self.V, dtype=np.float32)
        if self.eps <= 0:
            raise ValueError("eps must be > 0.")
        if not (0.0 < self.delta < 1.0):
            raise ValueError("delta must be in (0,1).")
        if self.r < 0:
            raise ValueError("r must be >= 0.")
        if self.q_norm_bound <= 0:
            raise ValueError("q_norm_bound must be > 0.")
        if self.sigma_tol <= 0:
            raise ValueError("sigma_tol must be > 0.")
        if self.sigma_method not in ("classic", "analytic"):
            raise ValueError("sigma_method must be 'classic' or 'analytic'.")

    def query_many(
        self, Q: np.ndarray, k: int, *, candidates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        Q = np.asarray(Q, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q[None, :]

        if self.score == "dot":
            Q = clip_l2_rows(Q, float(self.q_norm_bound))

        if candidates is None:
            V = self.V
            base = None
        else:
            cand = np.asarray(candidates, dtype=int)
            if cand.ndim != 1 or cand.size == 0:
                raise ValueError("candidates must be a non-empty 1D array.")
            V = self.V[cand]
            base = cand

        scores = dot_scores(Q, V) if self.score == "dot" else neg_l2_scores(Q, V)

        Delta_u = score_sensitivity(
            self.score, r=float(self.r), q_norm_bound=float(self.q_norm_bound)
        )
        sigma = gaussian_sigma(
            float(Delta_u),
            float(self.eps),
            float(self.delta),
            method=self.sigma_method,
            tol=float(self.sigma_tol),
        )

        noise = self.rng.normal(loc=0.0, scale=float(sigma), size=scores.shape).astype(
            scores.dtype, copy=False
        )
        noisy = scores + noise

        idx_local = _topk_indices(noisy, k)
        if base is None:
            return idx_local
        return base[idx_local]


@dataclass
class ExponentialMechanismTopKRetriever:
    """
    Exponential mechanism for top-k (sequential without replacement).

    - Uses eps_total per query call.
    - Splits eps_draw = eps_total / k (basic composition => eps_total overall).
    - Utility is the score (higher is better).
    """

    V: np.ndarray
    score: ScoreType
    r: float
    eps_total: float
    rng: np.random.Generator
    q_norm_bound: float = 1.0

    def __post_init__(self) -> None:
        self.V = np.asarray(self.V, dtype=np.float32)
        if self.eps_total <= 0:
            raise ValueError("eps_total must be > 0.")
        if self.r < 0:
            raise ValueError("r must be >= 0.")
        if self.q_norm_bound <= 0:
            raise ValueError("q_norm_bound must be > 0.")

    def _stable_softmax(self, logits: np.ndarray) -> np.ndarray:
        m = float(np.max(logits))
        exps = np.exp(logits - m)
        s = float(np.sum(exps))
        if s <= 0 or not np.isfinite(s):
            raise FloatingPointError("softmax normalization failed")
        return exps / s

    def query_many(
        self, Q: np.ndarray, k: int, *, candidates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        Q = np.asarray(Q, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q[None, :]

        if self.score == "dot":
            Q = clip_l2_rows(Q, float(self.q_norm_bound))

        if candidates is None:
            pool0 = np.arange(self.V.shape[0], dtype=int)
        else:
            pool0 = np.asarray(candidates, dtype=int)
            if pool0.ndim != 1 or pool0.size == 0:
                raise ValueError("candidates must be a non-empty 1D array.")

        B = Q.shape[0]
        out = np.zeros((B, min(int(k), int(pool0.size))), dtype=int)

        Delta_u = score_sensitivity(
            self.score, r=float(self.r), q_norm_bound=float(self.q_norm_bound)
        )
        eps_draw = float(self.eps_total) / float(max(1, int(k)))

        for b in range(B):
            pool = pool0.copy()
            chosen = []
            q = Q[b : b + 1]  # (1,d)

            for _ in range(int(k)):
                if pool.size == 0:
                    break
                Vp = self.V[pool]
                s = (
                    dot_scores(q, Vp).reshape(-1)
                    if self.score == "dot"
                    else neg_l2_scores(q, Vp).reshape(-1)
                )

                logits = (eps_draw / (2.0 * float(Delta_u))) * s
                p = self._stable_softmax(logits.astype(np.float64))
                j = int(self.rng.choice(pool.size, p=p))
                chosen_idx = int(pool[j])
                chosen.append(chosen_idx)
                pool = np.delete(pool, j)

            out[b, : len(chosen)] = np.asarray(chosen, dtype=int)

        return out
