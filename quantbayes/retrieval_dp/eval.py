# quantbayes/retrieval_dp/eval.py
from __future__ import annotations

from typing import Callable, Optional, Tuple
import numpy as np


def majority_vote(labels: np.ndarray, n_classes: Optional[int] = None) -> int:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if labels.size == 0:
        return 0
    if n_classes is None:
        n_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=int(n_classes))
    return int(np.argmax(counts))


def predict_labels_from_indices(
    idx: np.ndarray, y_db: np.ndarray, *, n_classes: Optional[int] = None
) -> np.ndarray:
    """
    idx: (B,k), y_db: (m,) -> y_pred: (B,)
    """
    idx = np.asarray(idx, dtype=int)
    y_db = np.asarray(y_db, dtype=int)
    if idx.ndim != 2:
        raise ValueError("idx must be 2D (B,k).")
    B = idx.shape[0]
    if n_classes is None:
        n_classes = int(y_db.max()) + 1

    y_pred = np.zeros((B,), dtype=int)
    for i in range(B):
        y_pred[i] = majority_vote(y_db[idx[i]], n_classes=int(n_classes))
    return y_pred


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    return float(np.mean(y_true == y_pred))


def eval_retrieval_classifier_accuracy(
    retriever,
    X_query: np.ndarray,
    y_query: np.ndarray,
    y_db: np.ndarray,
    *,
    k: int,
    batch_size: int = 128,
    n_classes: Optional[int] = None,
) -> float:
    X_query = np.asarray(X_query, dtype=np.float32)
    y_query = np.asarray(y_query, dtype=int).reshape(-1)
    y_db = np.asarray(y_db, dtype=int).reshape(-1)

    if n_classes is None:
        n_classes = int(max(y_db.max(), y_query.max())) + 1

    preds = []
    for s in range(0, X_query.shape[0], int(batch_size)):
        e = min(s + int(batch_size), X_query.shape[0])
        Qb = X_query[s:e]
        idx = retriever.query_many(Qb, k=int(k))
        yb = predict_labels_from_indices(idx, y_db, n_classes=int(n_classes))
        preds.append(yb)

    y_pred = np.concatenate(preds, axis=0)
    return accuracy(y_query, y_pred)


def eval_accuracy_trials(
    make_retriever: Callable[[int], object],
    X_query: np.ndarray,
    y_query: np.ndarray,
    y_db: np.ndarray,
    *,
    k: int,
    trials: int,
    seed: int,
    batch_size: int = 128,
    n_classes: Optional[int] = None,
) -> Tuple[float, float]:
    accs = []
    for t in range(int(trials)):
        retriever = make_retriever(int(seed) + 10_000 * t)
        acc = eval_retrieval_classifier_accuracy(
            retriever,
            X_query,
            y_query,
            y_db,
            k=int(k),
            batch_size=int(batch_size),
            n_classes=n_classes,
        )
        accs.append(float(acc))
    return float(np.mean(accs)), float(np.std(accs))
