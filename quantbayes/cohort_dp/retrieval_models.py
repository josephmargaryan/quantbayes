# cohort_dp/retrieval_models.py
from __future__ import annotations
from typing import List, Optional
import numpy as np

from .api import CohortDiscoveryAPI
from .simple_models import accuracy, blend_probs


def label_histogram_probs(labels: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=n_classes).astype(float)
    s = float(np.sum(counts))
    if s <= 0:
        return np.ones((n_classes,), dtype=float) / float(n_classes)
    return counts / s


def retrieval_predict_proba(
    api: CohortDiscoveryAPI,
    X_query: np.ndarray,
    y_db: np.ndarray,
    n_classes: int,
    k: int,
    session_id: Optional[str] = None,
) -> np.ndarray:
    probs = np.zeros((X_query.shape[0], n_classes), dtype=float)
    for i in range(X_query.shape[0]):
        idx = api.query(z=X_query[i], k=k, session_id=session_id)
        probs[i] = label_histogram_probs(y_db[idx], n_classes)
    return probs


def retrieval_predict(
    api: CohortDiscoveryAPI,
    X_query: np.ndarray,
    y_db: np.ndarray,
    k: int,
    session_id: Optional[str] = None,
) -> np.ndarray:
    y_pred = np.zeros((X_query.shape[0],), dtype=int)
    for i in range(X_query.shape[0]):
        idx = api.query(z=X_query[i], k=k, session_id=session_id)
        labels = y_db[idx]
        y_pred[i] = int(np.argmax(np.bincount(labels.astype(int))))
    return y_pred


def tune_alpha(
    p_base: np.ndarray,
    p_retr: np.ndarray,
    y_true: np.ndarray,
    grid: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
) -> float:
    best_a = float(grid[0])
    best_acc = -1.0
    for a in grid:
        p = blend_probs(p_base, p_retr, alpha=float(a))
        y_pred = np.argmax(p, axis=1)
        acc = accuracy(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_a = float(a)
    return best_a
