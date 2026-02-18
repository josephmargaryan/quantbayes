# quantbayes/ball_dp/reconstruction/convex/gaussian_identifier.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..types import Candidate, DeterministicTrainer, IdentificationResult, Vectorizer


@dataclass
class GaussianOutputIdentifier:
    """
    ML identification for Gaussian output perturbation:

      release_vec = mean_vec(candidate) + N(0, sigma^2 I)

    Score(candidate) = -||release - mean||^2 / (2 sigma^2) + log prior(candidate)

    This is mechanism-agnostic: works for prototypes (closed form) and for convex ERM
    (trainer.fit) as long as you can compute the deterministic mean output f(D).
    """

    trainer: DeterministicTrainer
    vectorizer: Vectorizer
    sigma: float
    prior_log_prob: Optional[callable] = None
    cache: bool = True

    _cache: Dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def _key(self, cand: Candidate) -> str:
        # stable cache key: prefer explicit ID; else fall back to label+hash of bytes
        if "id" in cand.meta:
            return str(cand.meta["id"])
        rec = np.asarray(cand.record, dtype=np.float64).reshape(-1)
        return f"lab={cand.label}|h={hash(rec.tobytes())}"

    def identify(
        self,
        *,
        release_params: Any,
        X_minus: np.ndarray,
        y_minus: np.ndarray,
        candidates: Sequence[Candidate],
    ) -> IdentificationResult:
        sigma = float(self.sigma)
        if sigma <= 0:
            raise ValueError("sigma must be > 0 for Gaussian identification.")

        v_obs = self.vectorizer(release_params)
        v_obs = np.asarray(v_obs, dtype=np.float64).reshape(-1)

        scores = np.empty((len(candidates),), dtype=np.float64)

        for i, cand in enumerate(candidates):
            key = self._key(cand)
            if self.cache and key in self._cache:
                v_mean = self._cache[key]
            else:
                X = np.concatenate(
                    [X_minus, np.asarray(cand.record, dtype=np.float64).reshape(1, -1)],
                    axis=0,
                )
                if cand.label is None:
                    raise ValueError(
                        "Candidate label is None; provide labels for this identification setting."
                    )
                y = np.concatenate(
                    [y_minus, np.asarray([int(cand.label)], dtype=np.int64)], axis=0
                )

                mean_params = self.trainer.fit(X, y)
                v_mean = np.asarray(
                    self.vectorizer(mean_params), dtype=np.float64
                ).reshape(-1)

                if self.cache:
                    self._cache[key] = v_mean

            d2 = float(np.sum((v_obs - v_mean) ** 2))
            lp = (
                0.0 if self.prior_log_prob is None else float(self.prior_log_prob(cand))
            )
            scores[i] = -0.5 * d2 / (sigma * sigma) + lp

        ranked = np.argsort(-scores)
        best = int(ranked[0])
        return IdentificationResult(best_idx=best, scores=scores, ranked_idx=ranked)
