# quantbayes/ball_dp/metrics.py
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .types import Record


def _flatten_features(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def l2_normalize_features(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    flat = _flatten_features(x)
    norm = float(np.linalg.norm(flat, ord=2))
    if not np.isfinite(norm) or norm <= float(eps):
        return flat.copy()
    return (flat / norm).astype(np.float32, copy=False)


def cosine_similarity_features(
    a: np.ndarray, b: np.ndarray, eps: float = 1e-12
) -> float:
    a_flat = _flatten_features(a)
    b_flat = _flatten_features(b)
    if a_flat.shape != b_flat.shape:
        raise ValueError(
            f"a.shape={a_flat.shape} must match b.shape={b_flat.shape} after flattening."
        )
    a_norm = float(np.linalg.norm(a_flat, ord=2))
    b_norm = float(np.linalg.norm(b_flat, ord=2))
    denom = max(a_norm * b_norm, float(eps))
    cos = float(np.dot(a_flat, b_flat) / denom)
    return float(np.clip(cos, -1.0, 1.0))


def cosine_error_features(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(1.0 - cosine_similarity_features(a, b, eps=eps))


def angular_error_radians_features(
    a: np.ndarray, b: np.ndarray, eps: float = 1e-12
) -> float:
    cos = cosine_similarity_features(a, b, eps=eps)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


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
        out = {
            "distance": float("inf"),
            "label_correct": 0.0,
            "feature_l2": float("inf"),
            "feature_cosine": float("nan"),
            "feature_cosine_error": float("inf"),
            "feature_angle_rad": float("inf"),
            "feature_angle_deg": float("inf"),
            "true_feature_norm": float("nan"),
            "pred_feature_norm": float("nan"),
        }
        for eta in eta_grid:
            out[f"success@{eta:g}"] = 0.0
        return out

    metric = LabelPreservingL2Metric()
    d = metric.distance(true_record, pred_record)
    true_flat = _flatten_features(true_record.features)
    pred_flat = _flatten_features(pred_record.features)
    angle_rad = angular_error_radians_features(true_flat, pred_flat)
    out = {
        "distance": float(d),
        "label_correct": float(int(true_record.label == pred_record.label)),
        "feature_l2": float(np.linalg.norm(true_flat - pred_flat, ord=2)),
        "feature_cosine": float(cosine_similarity_features(true_flat, pred_flat)),
        "feature_cosine_error": float(cosine_error_features(true_flat, pred_flat)),
        "feature_angle_rad": float(angle_rad),
        "feature_angle_deg": float(np.degrees(angle_rad)),
        "true_feature_norm": float(np.linalg.norm(true_flat, ord=2)),
        "pred_feature_norm": float(np.linalg.norm(pred_flat, ord=2)),
    }
    for eta in eta_grid:
        out[f"success@{eta:g}"] = float(d <= float(eta))
    return out


def feature_reconstruction_metrics(
    true_features: np.ndarray,
    pred_features: Optional[np.ndarray],
    eta_grid: Sequence[float],
) -> Dict[str, float]:
    true_flat = _flatten_features(true_features)

    if pred_features is None:
        out = {
            "distance": float("inf"),
            "feature_l2": float("inf"),
            "feature_cosine": float("nan"),
            "feature_cosine_error": float("inf"),
            "feature_angle_rad": float("inf"),
            "feature_angle_deg": float("inf"),
            "true_feature_norm": float(np.linalg.norm(true_flat, ord=2)),
            "pred_feature_norm": float("nan"),
        }
        for eta in eta_grid:
            out[f"success@{eta:g}"] = 0.0
        return out

    pred_flat = _flatten_features(pred_features)
    if true_flat.shape != pred_flat.shape:
        raise ValueError(
            f"true_features.shape={true_flat.shape} must match pred_features.shape={pred_flat.shape} after flattening."
        )

    d = float(np.linalg.norm(true_flat - pred_flat, ord=2))
    angle_rad = angular_error_radians_features(true_flat, pred_flat)
    out = {
        "distance": float(d),
        "feature_l2": float(d),
        "feature_cosine": float(cosine_similarity_features(true_flat, pred_flat)),
        "feature_cosine_error": float(cosine_error_features(true_flat, pred_flat)),
        "feature_angle_rad": float(angle_rad),
        "feature_angle_deg": float(np.degrees(angle_rad)),
        "true_feature_norm": float(np.linalg.norm(true_flat, ord=2)),
        "pred_feature_norm": float(np.linalg.norm(pred_flat, ord=2)),
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
