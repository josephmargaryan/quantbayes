from __future__ import annotations

from typing import Optional, Sequence

import math

from ..types import PriorFamily, RdpCurve, ReRoPoint, ReRoReport


def ball_pn_rdp_success_bound(
    curve: RdpCurve,
    *,
    kappa: float,
) -> tuple[float, Optional[float]]:
    """Optimized Ball-PN-RDP -> Ball-ReRo conversion.

    Returns the theorem-side bound
        inf_alpha min{1, exp(((alpha-1)/alpha) * (log kappa + eps_alpha))}
    together with the minimizing order when one improves over 1.
    """
    kappa = float(kappa)
    if kappa <= 0.0:
        return 0.0, None

    log_kappa = math.log(kappa)
    best = 1.0
    best_alpha = None
    for alpha, eps in zip(curve.orders, curve.epsilons):
        alpha = float(alpha)
        eps = float(eps)
        exponent = (alpha - 1.0) / alpha
        log_candidate = exponent * (log_kappa + eps)
        candidate = 1.0 if log_candidate >= 0.0 else math.exp(log_candidate)
        if candidate < best:
            best = float(candidate)
            best_alpha = float(alpha)
    return float(best), best_alpha


def compute_ball_pn_rero_report(
    curve: RdpCurve,
    prior: PriorFamily,
    eta_grid: Sequence[float],
    *,
    metadata: Optional[dict] = None,
) -> ReRoReport:
    """Compute observer-specific Ball-ReRo bounds from a Ball-PN-RDP curve."""
    points = []
    for eta in eta_grid:
        eta_f = float(eta)
        kappa = float(prior.kappa(eta_f))
        gamma, alpha_opt = ball_pn_rdp_success_bound(curve, kappa=kappa)
        points.append(
            ReRoPoint(
                eta=eta_f,
                kappa=kappa,
                gamma_ball=float(gamma),
                gamma_standard=None,
                alpha_opt_ball=alpha_opt,
                alpha_opt_standard=None,
            )
        )

    md = {
        "mode": "observer_specific_ball_pn_rdp",
        "rdp_source": str(curve.source),
        "radius": None if curve.radius is None else float(curve.radius),
        "orders": tuple(float(a) for a in curve.orders),
    }
    if metadata is not None:
        md.update(dict(metadata))

    return ReRoReport(mode="observer_specific_ball_pn_rdp", points=points, metadata=md)
