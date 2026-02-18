# quantbayes/ball_dp/reconstruction/priors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .types import Array, Candidate, MetricBatchFn


def l2_metric_batch(center: Array, pool: Array) -> Array:
    """
    Vectorized L2 distance: returns distances from center to each row in pool.
    center: (d,)
    pool:   (N,d)
    """
    c = np.asarray(center).reshape(1, -1)
    P = np.asarray(pool)
    return np.linalg.norm(P - c, axis=1)


@dataclass
class PoolBallPrior:
    """
    Discrete "uniform over pool points within a ball" prior:

        Z ~ Uniform({ zbar in pool : dist(center, zbar) <= r })

    This exactly matches your intended:
      Uniform({ \bar{z} in Z : ||u - \bar{z}|| <= r })
    where 'pool' is a finite candidate universe (e.g. training set embeddings).

    If label_fixed is provided, candidates are label-preserving.
    """

    pool_X: Array  # (N,d)
    pool_y: Optional[Array]  # (N,) or None
    radius: float
    metric_batch: MetricBatchFn = l2_metric_batch
    label_fixed: Optional[int] = None

    def candidate_indices(self, *, center: Array) -> Array:
        d = self.metric_batch(center, self.pool_X)
        idx = np.where(d <= float(self.radius))[0]

        if self.pool_y is not None and self.label_fixed is not None:
            y = np.asarray(self.pool_y).reshape(-1)
            idx = idx[y[idx] == int(self.label_fixed)]

        return idx

    def sample(
        self,
        *,
        center: Array,
        m: int,
        rng: np.random.Generator,
    ) -> Sequence[Candidate]:
        idx = self.candidate_indices(center=center)
        if idx.size == 0:
            raise ValueError(
                "PoolBallPrior: no points found inside the ball. Increase radius or change pool/metric."
            )
        m = int(m)
        pick = rng.choice(idx, size=min(m, idx.size), replace=False)

        out = []
        for j in pick:
            rec = np.asarray(self.pool_X[j]).reshape(-1)
            lab = None if self.pool_y is None else int(self.pool_y[j])
            out.append(Candidate(record=rec, label=lab, meta={"pool_index": int(j)}))
        return out
