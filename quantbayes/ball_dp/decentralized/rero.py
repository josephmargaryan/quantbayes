from __future__ import annotations

from typing import Optional, Sequence

import math
from statistics import NormalDist

from ..types import PriorFamily, RdpCurve, ReRoPoint, ReRoReport


_NORMAL = NormalDist()


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


def direct_gaussian_rero_success_bound(
    *,
    kappa: float,
    transferred_sensitivity: float,
) -> float:
    """Exact Gaussian blow-up success bound for a linear Gaussian observer view.

    For equal-covariance Gaussian views whose whitened mean separation is at most
    ``transferred_sensitivity``, every test/reconstructor with Ball-local
    anti-concentration ``kappa`` succeeds with probability at most

        Phi(Phi^{-1}(kappa) + transferred_sensitivity).

    This is the direct Gaussian ReRo bound used in Paper 3.
    """
    kappa = float(kappa)
    c = float(transferred_sensitivity)
    if kappa <= 0.0:
        return 0.0
    if kappa >= 1.0:
        return 1.0
    if c < 0.0 or not math.isfinite(c):
        raise ValueError("transferred_sensitivity must be finite and nonnegative.")
    return float(min(1.0, max(0.0, _NORMAL.cdf(_NORMAL.inv_cdf(kappa) + c))))


def direct_gaussian_rero_success_bound_from_sensitivity_sq(
    *,
    kappa: float,
    sensitivity_sq: float,
) -> float:
    """Convenience wrapper for ``direct_gaussian_rero_success_bound``."""
    sensitivity_sq = float(sensitivity_sq)
    if sensitivity_sq < 0.0 or not math.isfinite(sensitivity_sq):
        raise ValueError("sensitivity_sq must be finite and nonnegative.")
    return direct_gaussian_rero_success_bound(
        kappa=float(kappa),
        transferred_sensitivity=math.sqrt(max(0.0, sensitivity_sq)),
    )


# Compatibility alias used by the Paper 3 official scripts.
def gaussian_direct_success_bound(
    *, transferred_sensitivity: float, kappa: float
) -> float:
    return direct_gaussian_rero_success_bound(
        kappa=float(kappa),
        transferred_sensitivity=float(transferred_sensitivity),
    )


def gaussian_direct_success_bound_from_sensitivity_sq(
    *, sensitivity_sq: float, kappa: float
) -> float:
    return direct_gaussian_rero_success_bound_from_sensitivity_sq(
        kappa=float(kappa),
        sensitivity_sq=float(sensitivity_sq),
    )
