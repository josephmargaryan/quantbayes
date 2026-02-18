# quantbayes/ball_dp/reconstruction/nonconvex/shadow_identifier.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence

import numpy as np

from ..types import Candidate, IdentificationResult, StochasticTrainer, Vectorizer

ScoreMode = Literal["nearest_mean", "kde_logsumexp"]


@dataclass
class ShadowModelIdentifier:
    """
    Shadow-model candidate identification for stochastic/nonconvex training.

    Adds optional disk caching of shadow vectors for research workflows.
    """

    trainer: StochasticTrainer
    vectorizer: Vectorizer
    shadows_per_candidate: int = 6
    score_mode: ScoreMode = "nearest_mean"
    kde_bandwidth: Optional[float] = None
    include_state_in_vector: bool = False

    cache_dir: Optional[str] = None  # if set, saves/loads shadow vectors as .npy

    def _cache_path(self, cand: Candidate, seed: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        cd = Path(self.cache_dir)
        cd.mkdir(parents=True, exist_ok=True)
        # prefer pool_index if available
        cid = cand.meta.get("pool_index", None)
        if cid is None:
            cid = cand.meta.get("id", None)
        if cid is None:
            # fallback: hash bytes
            rec = np.asarray(cand.record, dtype=np.float64).reshape(-1)
            cid = f"h{hash(rec.tobytes())}"
        return cd / f"cand_{cid}_seed_{int(seed)}.npy"

    def _train_one_vec(
        self,
        *,
        X_minus: np.ndarray,
        y_minus: np.ndarray,
        cand: Candidate,
        seed: int,
    ) -> np.ndarray:
        p = self._cache_path(cand, seed)
        if p is not None and p.exists():
            return np.load(str(p)).astype(np.float64, copy=False)

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
        v = np.asarray(v, dtype=np.float64).reshape(-1)

        if p is not None:
            np.save(str(p), v)

        return v

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

        details: Dict[str, Any] = {"shadow_means": [], "shadow_vars": []}

        for i, cand in enumerate(candidates):
            V = []
            for _ in range(S):
                seed = int(rng.integers(0, 2**31 - 1))
                V.append(
                    self._train_one_vec(
                        X_minus=X_minus, y_minus=y_minus, cand=cand, seed=seed
                    )
                )
            V = np.stack(V, axis=0)  # (S,P)

            mu = V.mean(axis=0)
            var = float(V.var(axis=0).mean())
            details["shadow_means"].append(mu)
            details["shadow_vars"].append(var)

            if self.score_mode == "nearest_mean":
                d2 = float(np.sum((v_obs - mu) ** 2))
                scores[i] = -d2
            else:
                h = (
                    float(self.kde_bandwidth)
                    if (self.kde_bandwidth is not None)
                    else float(np.sqrt(max(var, 1e-12)))
                )
                d2s = np.sum((V - v_obs[None, :]) ** 2, axis=1)
                logs = -0.5 * d2s / (h * h)
                m = float(np.max(logs))
                scores[i] = m + float(np.log(np.sum(np.exp(logs - m)) + 1e-300))

        ranked = np.argsort(-scores)
        return IdentificationResult(
            best_idx=int(ranked[0]),
            scores=scores,
            ranked_idx=ranked,
            details=details,
        )
