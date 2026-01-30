# quantbayes/retrieval_dp/api.py

"""
4) I want private top-k retrieval on my corpus embeddings
from quantbayes.retrieval_dp.api import make_topk_retriever

retr = make_topk_retriever(
    V=corpus_embeddings,
    mechanism="gaussian",
    score="neg_l2",
    r=policy.r_values[50.0],
    eps=1.0,
    delta=1e-5,
    sigma_method="analytic",
)

idx = retr.query_many(query_embeddings, k=10)
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from quantbayes.retrieval_dp.metrics import ScoreType
from quantbayes.retrieval_dp.mechanisms import (
    NonPrivateTopKRetriever,
    NoisyScoresTopKLaplaceRetriever,
    NoisyScoresTopKGaussianRetriever,
)


def make_topk_retriever(
    *,
    V: np.ndarray,
    score: ScoreType = "neg_l2",
    mechanism: Literal["non_private", "laplace", "gaussian"] = "gaussian",
    r: float = 0.0,
    eps: float = 1.0,
    delta: float = 1e-5,
    q_norm_bound: float = 1.0,
    sigma_method: Literal["classic", "analytic"] = "classic",
    rng: Optional[np.random.Generator] = None,
):
    """
    One-stop factory for retrieval.

    Users call:
      retr = make_topk_retriever(V=..., mechanism="gaussian", r=..., eps=..., delta=..., sigma_method="analytic")
      idx = retr.query_many(Q, k=10)

    No need to import internal mechanism classes.
    """
    V = np.asarray(V, dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng(0)

    if mechanism == "non_private":
        return NonPrivateTopKRetriever(V=V, score=score, q_norm_bound=q_norm_bound)

    if mechanism == "laplace":
        return NoisyScoresTopKLaplaceRetriever(
            V=V,
            score=score,
            r=float(r),
            eps=float(eps),
            rng=rng,
            q_norm_bound=float(q_norm_bound),
        )

    if mechanism == "gaussian":
        return NoisyScoresTopKGaussianRetriever(
            V=V,
            score=score,
            r=float(r),
            eps=float(eps),
            delta=float(delta),
            rng=rng,
            q_norm_bound=float(q_norm_bound),
            sigma_method=sigma_method,
        )

    raise ValueError(f"Unknown mechanism={mechanism!r}.")
