# quantbayes/ball_dp/sensitivity.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ERMSensitivity:
    """
    Convenience container for a sensitivity bound.
    """

    delta_l2: float
    lz: float
    r: float
    lam: float
    n: int


def erm_sensitivity_l2(*, lz: float, r: float, lam: float, n: int) -> ERMSensitivity:
    """
    Theorem-style sensitivity bound for strongly convex ERM under Ball adjacency:

      Δ2 <= (L_z * r) / (lam * n)

    where:
      - lam is the strong convexity parameter induced by L2 regularization,
      - n is dataset size,
      - r is ball radius in the record metric,
      - L_z is Lipschitz-in-data constant of the per-example gradient.
    """
    if lam <= 0:
        raise ValueError("lam must be > 0")
    if n <= 0:
        raise ValueError("n must be >= 1")
    if r < 0:
        raise ValueError("r must be >= 0")
    if lz < 0:
        raise ValueError("lz must be >= 0")

    delta = (float(lz) * float(r)) / (float(lam) * float(n))
    return ERMSensitivity(
        delta_l2=float(delta), lz=float(lz), r=float(r), lam=float(lam), n=int(n)
    )
