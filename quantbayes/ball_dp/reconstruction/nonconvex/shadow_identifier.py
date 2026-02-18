# quantbayes/ball_dp/reconstruction/nonconvex/shadow_identifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import numpy as np

from ..types import Candidate, IdentificationResult, StochasticTrainer, Vectorizer


ScoreMode = Literal["nearest_mean", "kde_logsumexp"]


@dataclass
class ShadowModelIdentifier:
    """
    Shadow-model candidate identification for stochastic/nonconvex training.

    For each candidate z' in C:
      train S shadows: theta_{z',s} = A(D^- U {z'}; seed_s)
    Then score candidates against the observed release vector v_obs.

    Two scoring options:
      - nearest_mean: compare v_obs to mean shadow vector
      - kde_logsumexp: log-sum-exp of RBF kernel over shadow vectors (nonparametric)
    """

    trainer: StochasticTrainer
    vectorizer: Vectorizer
    shadows_per_candidate: int = 8
    score_mode: ScoreMode = "nearest_mean"
    kde_bandwidth: Optional[float] = None  # if None, estimate from shadow variance
    include_state_in_vector: bool = False

    def _train_one(
        self,
        *,
        X_minus: np.ndarray,
        y_minus: np.ndarray,
        cand: Candidate,
        seed: int,
    ) -> np.ndarray:
        X = np.concatenate(
            [X_minus, np.asarray(cand.record, dtype=np.float64).reshape(1, -1)], axis=0
        )
        if cand.label is None:
            raise ValueError(
                "Candidate label is None; shadow identification needs labeled candidates."
            )
        y = np.concatenate(
            [y_minus, np.asarray([int(cand.label)], dtype=np.int64)], axis=0
        )

        params, state = self.trainer.fit(X, y, seed=int(seed))
        v = self.vectorizer(
            params, state=state if self.include_state_in_vector else None
        )
        return np.asarray(v, dtype=np.float64).reshape(-1)

    def identify(
        self,
        *,
        release_params: Any,
        release_state: Any | None,
        X_minus: np.ndarray,
        y_minus: np.ndarray,
        candidates: Sequence[Candidate],
        rng: np.random.Generator,
    ) -> IdentificationResult:
        v_obs = self.vectorizer(
            release_params,
            state=release_state if self.include_state_in_vector else None,
        )
        v_obs = np.asarray(v_obs, dtype=np.float64).reshape(-1)

        S = int(self.shadows_per_candidate)
        scores = np.empty((len(candidates),), dtype=np.float64)

        # optional detail dump for research use
        details: Dict[str, Any] = {"shadow_means": [], "shadow_vars": []}

        for i, cand in enumerate(candidates):
            shadows = []
            for s in range(S):
                seed = int(rng.integers(0, 2**31 - 1))
                shadows.append(
                    self._train_one(
                        X_minus=X_minus, y_minus=y_minus, cand=cand, seed=seed
                    )
                )
            V = np.stack(shadows, axis=0)  # (S, P)
            mu = V.mean(axis=0)
            var = V.var(axis=0).mean()

            details["shadow_means"].append(mu)
            details["shadow_vars"].append(var)

            if self.score_mode == "nearest_mean":
                d2 = float(np.sum((v_obs - mu) ** 2))
                scores[i] = -d2
            elif self.score_mode == "kde_logsumexp":
                h = (
                    float(self.kde_bandwidth)
                    if (self.kde_bandwidth is not None)
                    else float(np.sqrt(max(var, 1e-12)))
                )
                # logsumexp of -||v_obs - v||^2/(2h^2)
                d2s = np.sum((V - v_obs[None, :]) ** 2, axis=1)
                logs = -0.5 * d2s / (h * h)
                m = float(np.max(logs))
                scores[i] = m + float(np.log(np.sum(np.exp(logs - m)) + 1e-300))
            else:
                raise ValueError(f"Unknown score_mode={self.score_mode}")

        ranked = np.argsort(-scores)
        return IdentificationResult(
            best_idx=int(ranked[0]),
            scores=scores,
            ranked_idx=ranked,
            details=details,
        )
