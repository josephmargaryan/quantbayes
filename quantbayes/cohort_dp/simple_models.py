# cohort_dp/simple_models.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits)
    return exps / (np.sum(exps, axis=1, keepdims=True) + 1e-12)


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = y.astype(int)
    oh = np.zeros((y.shape[0], n_classes), dtype=float)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


@dataclass
class SoftmaxRegression:
    """
    Multiclass linear classifier:
      p(y=c|x) = softmax(xW + b)_c

    Trained with full-batch gradient descent.
    """

    n_classes: int
    lr: float = 0.2
    l2: float = 1e-3
    epochs: int = 400
    seed: int = 0

    W: np.ndarray | None = None
    b: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        C = self.n_classes

        rng = np.random.default_rng(self.seed)
        W = 0.01 * rng.normal(size=(d, C))
        b = np.zeros((C,), dtype=float)
        Y = _one_hot(y, C)

        for _ in range(self.epochs):
            logits = X @ W + b[None, :]
            P = _softmax(logits)

            # gradients
            G = (P - Y) / float(n)  # (n,C)
            gW = X.T @ G + self.l2 * W  # (d,C)
            gb = np.sum(G, axis=0)  # (C,)

            W -= self.lr * gW
            b -= self.lr * gb

        self.W, self.b = W, b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.W is None or self.b is None:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=float)
        return _softmax(X @ self.W + self.b[None, :])

    def predict(self, X: np.ndarray) -> np.ndarray:
        P = self.predict_proba(X)
        return np.argmax(P, axis=1).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(np.mean(y_true == y_pred))


def blend_probs(p_base: np.ndarray, p_retr: np.ndarray, alpha: float) -> np.ndarray:
    """
    alpha in [0,1], returns alpha*base + (1-alpha)*retrieval
    """
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1].")
    return alpha * p_base + (1.0 - alpha) * p_retr
