from __future__ import annotations
import numpy as np


def assert_2d(X: np.ndarray, name: str = "X") -> None:
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {X.shape}")


def assert_shapes(X: np.ndarray, y: np.ndarray) -> None:
    assert_2d(X, "X")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same number of rows: {X.shape[0]} vs {y.shape[0]}"
        )
