# cohort_dp/eval.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .metrics import Metric


def true_knn(metric: Metric, X: np.ndarray, z: np.ndarray, k: int) -> np.ndarray:
    z = np.asarray(z, dtype=float).reshape(1, -1)
    d = metric.pairwise(z, X).reshape(-1)
    k = min(k, d.shape[0])
    idx = np.argpartition(d, kth=k - 1)[:k]
    idx = idx[np.argsort(d[idx])]
    return idx.astype(int)


def precision_at_k(retrieved: np.ndarray, truth: np.ndarray) -> float:
    retrieved_set = set(map(int, retrieved.tolist()))
    truth_set = set(map(int, truth.tolist()))
    if len(retrieved_set) == 0:
        return 0.0
    return len(retrieved_set.intersection(truth_set)) / float(len(retrieved_set))


@dataclass
class FrequencyAttacker:
    """
    Toy attacker:
    - queries around a target with small Gaussian noise
    - repeats Q times
    - counts returned indices (optionally all k outputs)
    - predicts the most frequent index

    Optional:
      - session_id: if set, attacker uses the same authenticated session across queries
      - new_session_per_query: if True, attacker rotates session IDs (models attacker with many accounts)
    """

    query_noise_std: float
    Q: int
    k_attack: int
    rng: np.random.Generator
    count_all_returned: bool = True
    session_id: str | None = None
    new_session_per_query: bool = False

    def attack(self, api, X: np.ndarray, target_idx: int) -> int:
        x_t = X[target_idx]
        counts = {}

        for t in range(self.Q):
            z = x_t + self.rng.normal(
                loc=0.0, scale=self.query_noise_std, size=x_t.shape[0]
            )

            sid = None
            if self.session_id is not None:
                sid = (
                    f"{self.session_id}_{t}"
                    if self.new_session_per_query
                    else self.session_id
                )

            # compatible with APIs that may or may not accept session_id
            try:
                out = api.query(z=z, k=self.k_attack, session_id=sid)
            except TypeError:
                out = api.query(z=z, k=self.k_attack)

            out = out.tolist()
            if self.count_all_returned:
                for idx in out:
                    idx = int(idx)
                    counts[idx] = counts.get(idx, 0) + 1
            else:
                idx = int(out[0])
                counts[idx] = counts.get(idx, 0) + 1

        return max(counts.items(), key=lambda kv: kv[1])[0]
