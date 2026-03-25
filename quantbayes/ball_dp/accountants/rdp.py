# quantbayes/ball_dp/accountants/rdp.py
from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from ..types import DpCertificate, RdpCurve


def gaussian_rdp_curve(
    *,
    orders: Sequence[float],
    sensitivity: float,
    sigma: float,
    source: str,
    radius: float | None = None,
) -> RdpCurve:
    sensitivity = float(sensitivity)
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    eps = []
    for alpha in orders:
        alpha = float(alpha)
        if alpha <= 1.0:
            raise ValueError("RDP orders must exceed 1")
        eps.append(alpha * sensitivity * sensitivity / (2.0 * sigma * sigma))
    return RdpCurve(
        orders=tuple(float(a) for a in orders),
        epsilons=tuple(float(e) for e in eps),
        source=source,
        radius=radius,
    )


def compose_rdp_curves(
    curves: Sequence[RdpCurve], *, source: str, radius: float | None = None
) -> RdpCurve:
    if not curves:
        raise ValueError("Need at least one curve.")
    ref_orders = curves[0].orders
    total = np.zeros(len(ref_orders), dtype=np.float64)
    for curve in curves:
        if curve.orders != ref_orders:
            raise ValueError("All RDP curves must use the same order grid.")
        total += np.asarray(curve.epsilons, dtype=np.float64)
    return RdpCurve(
        orders=ref_orders, epsilons=tuple(total.tolist()), source=source, radius=radius
    )


def rdp_to_dp(
    curve: RdpCurve, *, delta: float, source: str | None = None
) -> DpCertificate:
    delta = float(delta)
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    best_eps = float("inf")
    best_alpha = None
    for alpha, eps_rdp in zip(curve.orders, curve.epsilons):
        alpha = float(alpha)
        eps_rdp = float(eps_rdp)
        candidate = eps_rdp + math.log(1.0 / delta) / (alpha - 1.0)
        if candidate < best_eps:
            best_eps = candidate
            best_alpha = alpha
    return DpCertificate(
        epsilon=float(best_eps),
        delta=delta,
        source=curve.source if source is None else source,
        radius=curve.radius,
        order_opt=best_alpha,
    )
