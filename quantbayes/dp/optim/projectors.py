from __future__ import annotations
import numpy as np
from typing import Optional


def project_l2_ball(w: np.ndarray, radius: Optional[float]) -> np.ndarray:
    """
    Project w onto the L2 ball of given radius. If radius is None, returns w.
    """
    if radius is None:
        return w
    if radius <= 0:
        raise ValueError("radius must be > 0")
    nrm = float(np.linalg.norm(w))
    if nrm <= radius:
        return w
    return (radius / (nrm + 1e-12)) * w
