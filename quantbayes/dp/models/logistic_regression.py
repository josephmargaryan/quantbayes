from __future__ import annotations
import numpy as np
from typing import Tuple


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logistic_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gradient of average unregularized logistic loss:
      (1/n) sum log(1 + exp(-y w^T x))
    y ∈ {±1}.
    """
    z = y * (X @ w)
    s = np.empty_like(z)
    pos = z >= 0
    s[pos] = np.exp(-z[pos]) / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    s[~pos] = 1.0 / (1.0 + ez)
    g = -(y[:, None] * X) * s[:, None]
    return g.mean(axis=0)


def logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Average unregularized logistic loss:
      (1/n) sum log(1 + exp(-y w^T x))
    Use a numerically stable softplus via logaddexp.
    """
    z = -y * (X @ w)  # shape (n,)
    return float(np.logaddexp(0.0, z).mean())


def train_logreg_gd(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    steps: int = 2000,
    lr: float = 0.1,
) -> np.ndarray:
    """
    Non-private baseline GD for L2-regularized logistic regression.
    """
    d = X.shape[1]
    w = np.zeros(d, dtype=float)
    for _ in range(steps):
        g = logistic_grad(w, X, y) + lam * w
        w = w - lr * g
    return w
