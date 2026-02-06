from __future__ import annotations

from typing import Tuple
import numpy as np


def _to_01(x: np.ndarray) -> np.ndarray:
    # assumes x in [-1,1]
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def make_image_grid(
    x: np.ndarray,
    *,
    nrow: int,
) -> np.ndarray:
    """
    Create a square grid for MNIST-like images.

    x: (B,1,H,W) or (B,H,W)
    returns: (H*nrow, W*nrow)
    """
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]  # (B,H,W)
    if x.ndim != 3:
        raise ValueError(f"Expected (B,1,H,W) or (B,H,W). Got {x.shape}")

    b, h, w = x.shape
    need = nrow * nrow
    if b < need:
        raise ValueError(f"Need at least {need} images for nrow={nrow}, got {b}")

    x = x[:need]
    x = _to_01(x)

    x = x.reshape(nrow, nrow, h, w)
    rows = []
    for i in range(nrow):
        rows.append(np.concatenate([x[i, j] for j in range(nrow)], axis=-1))
    return np.concatenate(rows, axis=-2)
