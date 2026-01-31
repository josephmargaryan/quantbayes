from __future__ import annotations
import numpy as np
from typing import Optional
from ..utils.rng import make_rng


def gaussian_noise(shape, sigma: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Draw N(0, sigma^2 I) noise with given shape.
    """
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    rng = make_rng(seed)
    return rng.normal(loc=0.0, scale=sigma, size=shape)
