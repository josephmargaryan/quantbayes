# cohort_dp/candidates.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from .metrics import Metric, L2Metric
from .clustering import kmeans_lloyd


@dataclass
class AllCandidates:
    n: int

    def candidates(self, z: np.ndarray) -> np.ndarray:
        return np.arange(self.n, dtype=int)


@dataclass
class E2LSHCandidates:
    """
    (A) E2LSH-style candidate generation for L2:
      h(x) = floor((A x + b) / w), where A ~ N(0, I), b ~ Uniform(0, w)
    Use L tables, each with K projections.

    NOTE (important for thesis rigor):
    This is a computational acceleration. If you RESTRICT the DP mechanism only to the candidate set,
    you must analyze privacy jointly (candidate selection can be data-dependent).
    For Month 1/2, keep AllCandidates to preserve the cleanest DP story.
    """

    X: np.ndarray
    L: int
    K: int
    w: float
    rng: np.random.Generator
    min_candidates: int = 200
    fallback_random: int = 500

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        n, d = self.X.shape
        if self.L <= 0 or self.K <= 0:
            raise ValueError("L and K must be positive.")
        if self.w <= 0:
            raise ValueError("w must be > 0.")
        self.A = self.rng.normal(size=(self.L, self.K, d))
        self.b = self.rng.uniform(low=0.0, high=self.w, size=(self.L, self.K))
        self.tables: List[Dict[Tuple[int, ...], List[int]]] = [
            dict() for _ in range(self.L)
        ]
        self._build_tables()

    def _hash(self, x: np.ndarray, l: int) -> Tuple[int, ...]:
        proj = (self.A[l] @ x) + self.b[l]  # (K,)
        code = np.floor(proj / self.w).astype(np.int32)
        return tuple(int(v) for v in code.tolist())

    def _build_tables(self) -> None:
        for i in range(self.X.shape[0]):
            x = self.X[i]
            for l in range(self.L):
                key = self._hash(x, l)
                self.tables[l].setdefault(key, []).append(i)

    def candidates(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(-1)
        cand_set = set()
        for l in range(self.L):
            key = self._hash(z, l)
            bucket = self.tables[l].get(key, [])
            cand_set.update(bucket)

        cand = np.fromiter(cand_set, dtype=int)
        # ensure some minimum size so retrieval isn't too brittle
        if cand.size < self.min_candidates:
            n = self.X.shape[0]
            extra = self.rng.choice(n, size=min(self.fallback_random, n), replace=False)
            cand = np.unique(np.concatenate([cand, extra]))
        return cand


@dataclass
class PrototypeCandidates:
    """
    (B) Prototype/cluster candidate generation.

    Build k-means clusters once; at query time:
      - find nearest `n_probe` prototypes
      - return union of members of those clusters

    NOTE (for thesis rigor):
    Same caveat as LSH: if you restrict the DP mechanism to candidates,
    analyze privacy jointly. For Month 1/2, use this mainly for speed experiments.
    """

    X: np.ndarray
    n_clusters: int
    n_iters: int
    n_probe: int
    rng: np.random.Generator
    metric: Optional[Metric] = None
    min_candidates: int = 200
    fallback_random: int = 500

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        self.metric = self.metric or L2Metric()
        self.centers, self.labels = kmeans_lloyd(
            self.X, self.n_clusters, self.n_iters, self.rng
        )

        # map cluster -> indices
        self.members: List[List[int]] = [[] for _ in range(self.n_clusters)]
        for i, c in enumerate(self.labels.tolist()):
            self.members[int(c)].append(i)

    def candidates(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(1, -1)
        d = self.metric.pairwise(z, self.centers).reshape(-1)
        p = min(self.n_probe, d.shape[0])
        proto_ids = np.argpartition(d, kth=p - 1)[:p]
        cand = np.concatenate(
            [np.array(self.members[int(cid)], dtype=int) for cid in proto_ids], axis=0
        )
        cand = np.unique(cand)

        if cand.size < self.min_candidates:
            n = self.X.shape[0]
            extra = self.rng.choice(n, size=min(self.fallback_random, n), replace=False)
            cand = np.unique(np.concatenate([cand, extra]))
        return cand
