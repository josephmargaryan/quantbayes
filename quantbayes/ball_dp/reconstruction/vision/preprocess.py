# quantbayes/ball_dp/reconstruction/vision/preprocess.py
from __future__ import annotations

import math
import numpy as np


def clip01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.clip(x, 0.0, 1.0)


def flatten_chw(X: np.ndarray) -> np.ndarray:
    """
    (N,C,H,W) -> (N, C*H*W)
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 4:
        raise ValueError("Expected (N,C,H,W)")
    return X.reshape(X.shape[0], -1)


def unflatten_mnist(Xflat: np.ndarray) -> np.ndarray:
    """
    (N,784) -> (N,1,28,28)
    """
    Xflat = np.asarray(Xflat, dtype=np.float32)
    if Xflat.ndim != 2 or Xflat.shape[1] != 784:
        raise ValueError("Expected (N,784)")
    return Xflat.reshape(Xflat.shape[0], 1, 28, 28)


def unflatten_cifar10(Xflat: np.ndarray) -> np.ndarray:
    """
    (N,3072) -> (N,3,32,32)
    """
    Xflat = np.asarray(Xflat, dtype=np.float32)
    if Xflat.ndim != 2 or Xflat.shape[1] != 3072:
        raise ValueError("Expected (N,3072)")
    return Xflat.reshape(Xflat.shape[0], 3, 32, 32)


def pixel_l2_bound_unit_box(dim: int, max_val: float = 1.0) -> float:
    """
    Public L2 bound for x in [0,max_val]^dim:
      ||x||_2 <= sqrt(dim) * max_val
    This gives a DP-valid bounded replacement baseline r_std = 2B.
    """
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be >= 1")
    return float(math.sqrt(dim) * float(max_val))


if __name__ == "__main__":
    B_mnist = pixel_l2_bound_unit_box(784, 1.0)
    B_cifar = pixel_l2_bound_unit_box(3072, 1.0)
    print("B_mnist:", B_mnist, "B_cifar:", B_cifar)
    assert abs(B_mnist - 28.0) < 1e-6
    print("[OK] preprocess utilities.")
