# quantbayes/cohort_dp/experiments/exp_embeddings_quickstart.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from quantbayes.cohort_dp.metrics import L2Metric
from quantbayes.cohort_dp.baselines import NonPrivateKNNRetriever
from quantbayes.cohort_dp.novel_mechanisms import AdaptiveBallUniformRetriever
from quantbayes.cohort_dp.api import CohortDiscoveryAPI
from quantbayes.cohort_dp.accountant import PrivacyAccountant
from quantbayes.cohort_dp.eval import FrequencyAttacker


@dataclass
class Config:
    seed: int = 0
    d_embed: int = 25  # embedding dimension
    k_service: int = 10
    k0: int = 40
    Q: int = 50


def standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def majority_vote(labels: np.ndarray) -> int:
    labels = labels.astype(int)
    return int(np.argmax(np.bincount(labels)))


def make_random_projection_embeddings(
    X_raw: np.ndarray, d_embed: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Toy 'embedding model': z = xW where W is random Gaussian.
    This simulates a learned encoder but is deterministic given seed.
    """
    X_raw = standardize(X_raw)
    d_in = X_raw.shape[1]
    W = rng.normal(size=(d_in, d_embed)) / np.sqrt(d_embed)
    Z = X_raw @ W
    return standardize(Z)


def run(cfg: Config) -> None:
    rng = np.random.default_rng(cfg.seed)

    # ---- Load a real dataset (digits). Requires scikit-learn.
    try:
        from sklearn.datasets import load_digits
    except ImportError as e:
        raise RuntimeError(
            "This demo requires scikit-learn: pip install scikit-learn"
        ) from e

    digits = load_digits()
    X_raw = digits.data.astype(float)  # (n, 64) flattened 8x8 images
    y = digits.target.astype(int)  # labels 0..9

    # ---- Create embeddings (this is the key idea!)
    X_emb = make_random_projection_embeddings(X_raw, cfg.d_embed, rng)

    # ---- Split
    n = X_emb.shape[0]
    idx = rng.permutation(n)
    n_train = int(0.7 * n)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X_train, y_train = X_emb[train_idx], y[train_idx]
    X_test, y_test = X_emb[test_idx], y[test_idx]

    metric = L2Metric()

    # ---- Build cohort services
    knn = NonPrivateKNNRetriever(X=X_train, metric=metric)
    abu = AdaptiveBallUniformRetriever(
        X=X_train, metric=metric, k0=cfg.k0, rng=np.random.default_rng(cfg.seed + 123)
    )

    api_knn = CohortDiscoveryAPI(
        retriever=knn, accountant=PrivacyAccountant(eps_budget=1e9)
    )
    api_abu = CohortDiscoveryAPI(
        retriever=abu, accountant=PrivacyAccountant(eps_budget=1e9)
    )

    # ---- Retrieval-only classification (majority vote over retrieved labels)
    def eval_api(api) -> float:
        correct = 0
        for i in range(X_test.shape[0]):
            idxs = api.query(z=X_test[i], k=cfg.k_service)
            pred = majority_vote(y_train[np.asarray(idxs, dtype=int)])
            correct += int(pred == int(y_test[i]))
        return correct / float(X_test.shape[0])

    acc_knn = eval_api(api_knn)
    acc_abu = eval_api(api_abu)
    print(
        f"[Digits embeddings] Retrieval-only accuracy | NonPrivate kNN: {acc_knn:.3f} | AB-Uniform: {acc_abu:.3f}"
    )

    # ---- Simple reconstruction-style attack on TRAIN points
    targets = rng.choice(X_train.shape[0], size=60, replace=False)
    attacker = FrequencyAttacker(
        query_noise_std=0.05,
        Q=cfg.Q,
        k_attack=cfg.k_service,
        rng=np.random.default_rng(cfg.seed + 999),
        count_all_returned=True,
        session_id="attacker",
        new_session_per_query=False,
    )

    def attack_exact(api) -> float:
        hits = 0
        for t in targets:
            pred = int(attacker.attack(api, X_train, int(t)))
            hits += int(pred == int(t))
        return hits / float(len(targets))

    exact_knn = attack_exact(api_knn)
    exact_abu = attack_exact(api_abu)
    print(
        f"[Digits embeddings] Attack exact-ID | NonPrivate kNN: {exact_knn:.3f} | AB-Uniform: {exact_abu:.3f}"
    )


if __name__ == "__main__":
    run(Config())
