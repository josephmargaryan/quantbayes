# quantbayes/ball_dp/metrics.py
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from .types import Record


class LabelPreservingL2Metric:
    def distance(self, a: Record, b: Record) -> float:
        if int(a.label) != int(b.label):
            return float("inf")
        return float(np.linalg.norm(np.ravel(a.features) - np.ravel(b.features), ord=2))


class PlainL2Metric:
    def distance(self, a: Record, b: Record) -> float:
        return float(np.linalg.norm(np.ravel(a.features) - np.ravel(b.features), ord=2))


def reconstruction_metrics(
    true_record: Record, pred_record: Optional[Record], eta_grid: Sequence[float]
) -> Dict[str, float]:
    if pred_record is None:
        out = {"distance": float("inf"), "label_correct": 0.0}
        for eta in eta_grid:
            out[f"success@{eta:g}"] = 0.0
        return out
    metric = LabelPreservingL2Metric()
    d = metric.distance(true_record, pred_record)
    out = {
        "distance": float(d),
        "label_correct": float(int(true_record.label == pred_record.label)),
        "feature_l2": float(
            np.linalg.norm(
                np.ravel(true_record.features) - np.ravel(pred_record.features), ord=2
            )
        ),
    }
    for eta in eta_grid:
        out[f"success@{eta:g}"] = float(d <= float(eta))
    return out


def accuracy_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    if logits.ndim == 1 or logits.shape[-1] == 1:
        preds = (np.ravel(logits) >= 0.0).astype(np.int64)
        return float(np.mean(preds == y_true.astype(np.int64)))
    preds = np.argmax(logits, axis=-1)
    return float(np.mean(preds.astype(np.int64) == y_true.astype(np.int64)))


def mean_squared_error(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    return float(np.mean((pred - target) ** 2))
