from __future__ import annotations
import numpy as np
from typing import Optional


def make_rng(seed: Optional[int] = None) -> np.random.RandomState:
    """Create a RandomState with optional seed."""
    return np.random.RandomState(seed)
