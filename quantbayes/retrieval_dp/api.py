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
        rng = np.random.default_rng()

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


if __name__ == "__main__":
    import numpy as np

    from quantbayes.retrieval_dp.eval import eval_retrieval_classifier_accuracy
    from quantbayes.retrieval_dp.metrics import l2_normalize_rows
    from quantbayes.retrieval_dp.sensitivity import bounded_replacement_radius

    def make_synthetic(
        *,
        m_db: int = 5000,
        m_query: int = 1000,
        d: int = 64,
        n_classes: int = 10,
        noise: float = 0.15,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)

        # class prototypes on the unit sphere
        centers = rng.normal(size=(n_classes, d)).astype(np.float32)
        centers = l2_normalize_rows(centers)

        # database embeddings
        y_db = rng.integers(0, n_classes, size=(m_db,), dtype=np.int64)
        V = centers[y_db] + noise * rng.normal(size=(m_db, d)).astype(np.float32)
        V = l2_normalize_rows(V)

        # query embeddings
        y_query = rng.integers(0, n_classes, size=(m_query,), dtype=np.int64)
        Q = centers[y_query] + noise * rng.normal(size=(m_query, d)).astype(np.float32)
        Q = l2_normalize_rows(Q)

        return V, y_db.astype(int), Q, y_query.astype(int)

    V, y_db, Q, y_query = make_synthetic()

    # If we L2-normalize, we have a public bound B=1.0 on ||v|| and ||q||.
    B = 1.0
    q_norm_bound = 1.0

    # Replacement adjacency worst-case: ||v - v'|| <= 2B
    r = bounded_replacement_radius(B)

    k = 5
    eps = 1.0
    delta = 1e-5

    # Non-private baseline
    retr_np = make_topk_retriever(
        V=V,
        mechanism="non_private",
        score="neg_l2",
        q_norm_bound=q_norm_bound,
    )
    acc_np = eval_retrieval_classifier_accuracy(
        retr_np, Q, y_query, y_db, k=k, batch_size=256, n_classes=int(y_db.max() + 1)
    )
    print(f"[non_private]  acc@{k} = {acc_np:.4f}")

    # DP Gaussian noisy-scores retriever
    retr_dp = make_topk_retriever(
        V=V,
        mechanism="gaussian",
        score="neg_l2",
        r=r,
        eps=eps,
        delta=delta,
        sigma_method="classic",  # switch to "analytic" if you want
        q_norm_bound=q_norm_bound,
        rng=np.random.default_rng(123),
    )
    acc_dp = eval_retrieval_classifier_accuracy(
        retr_dp, Q, y_query, y_db, k=k, batch_size=256, n_classes=int(y_db.max() + 1)
    )
    print(
        f"[gaussian DP]   eps={eps}, delta={delta}, r={r:.3f}  acc@{k} = {acc_dp:.4f}"
    )

    # Show raw retrieval output shape
    idx = retr_dp.query_many(Q[:3], k=k)
    print("idx.shape:", idx.shape)  # (3,k)
    print("idx[0]:", idx[0])
    print("labels of idx[0]:", y_db[idx[0]])
