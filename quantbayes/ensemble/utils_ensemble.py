# utils_ensemble.py

import numpy as np
from typing import Sequence

EPS = 1e-12


def align_proba_to_global(
    proba: np.ndarray, model_classes: Sequence, global_classes: Sequence
) -> np.ndarray:
    """
    Map a model's predict_proba output (columns ordered by model_classes)
    into an array with columns ordered by global_classes.
    """
    proba = np.asarray(proba, dtype=float)
    g = np.asarray(global_classes)
    m = np.asarray(model_classes)
    out = np.zeros((proba.shape[0], len(g)), dtype=float)
    for i, cls in enumerate(m):
        j = int(np.where(g == cls)[0][0])
        out[:, j] = proba[:, i]
    return out


def geometric_mean_ensemble(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    probs: (n_models, n_samples, n_classes)
    weights: (n_models,) nonnegative, sum=1.
    Returns weighted geometric mean over models (n_samples, n_classes).
    """
    probs = np.clip(probs, EPS, 1.0)  # avoid log(0)
    logp = np.log(probs)  # (M,N,C)
    wlog = (weights[:, None, None] * logp).sum(axis=0)
    out = np.exp(wlog)
    out /= out.sum(axis=1, keepdims=True)
    return out


def binary_logit_average(probs_pos: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    probs_pos: (n_models, n_samples) positive-class probabilities
    weights  : (n_models,) nonnegative, sum=1
    Returns combined probs (n_samples, 2) in fixed order [neg, pos].
    """
    p = np.clip(probs_pos, EPS, 1 - EPS)
    logits = np.log(p) - np.log(1 - p)  # logit
    z = (weights[:, None] * logits).sum(axis=0)
    p_pos = 1.0 / (1.0 + np.exp(-z))
    return np.vstack([1 - p_pos, p_pos]).T
