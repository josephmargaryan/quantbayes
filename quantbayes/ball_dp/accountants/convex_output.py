# quantbayes/ball_dp/accountants/convex_output.py
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from .gaussian import epsilon_from_sigma, gaussian_sigma
from .rdp import gaussian_rdp_curve, rdp_to_dp
from ..types import DpCertificate, DualPrivacyLedger, PrivacyLedger


def build_convex_gaussian_ledgers(
    *,
    sigma: float,
    delta_ball: float,
    delta_std: Optional[float],
    radius: float,
    standard_radius: Optional[float],
    orders: Sequence[float],
    dp_delta: Optional[float],
    gaussian_method: str,
    gaussian_tol: float = 1e-12,
    extra_dp_deltas: Sequence[float] = (),
) -> DualPrivacyLedger:
    ball_rdp = gaussian_rdp_curve(
        orders=orders,
        sensitivity=delta_ball,
        sigma=sigma,
        source="ball_gaussian_output",
        radius=radius,
    )
    ball_dp_certs = []
    if dp_delta is not None:
        ball_eps = epsilon_from_sigma(
            sensitivity=delta_ball,
            sigma=sigma,
            delta=dp_delta,
            method=gaussian_method,
            tol=gaussian_tol,
        )
        ball_dp_certs.append(
            DpCertificate(
                epsilon=ball_eps,
                delta=dp_delta,
                source=f"gaussian_inverse:{gaussian_method}",
                radius=radius,
            )
        )
    for delt in extra_dp_deltas:
        cert = rdp_to_dp(ball_rdp, delta=delt, source="rdp_to_dp")
        ball_dp_certs.append(cert)
    ball = PrivacyLedger(
        mechanism="gaussian_output_perturbation",
        sigma=sigma,
        radius=radius,
        rdp_curve=ball_rdp,
        dp_certificates=ball_dp_certs,
    )

    if delta_std is None:
        std = PrivacyLedger(
            mechanism="gaussian_output_perturbation",
            sigma=sigma,
            radius=standard_radius,
        )
        std.notes.append(
            "Standard comparator unavailable because no standard radius / sensitivity was supplied."
        )
        return DualPrivacyLedger(ball=ball, standard=std)

    std_rdp = gaussian_rdp_curve(
        orders=orders,
        sensitivity=delta_std,
        sigma=sigma,
        source="standard_gaussian_output",
        radius=standard_radius,
    )
    std_dp_certs = []
    if dp_delta is not None:
        std_eps = epsilon_from_sigma(
            sensitivity=delta_std,
            sigma=sigma,
            delta=dp_delta,
            method=gaussian_method,
            tol=gaussian_tol,
        )
        std_dp_certs.append(
            DpCertificate(
                epsilon=std_eps,
                delta=dp_delta,
                source=f"gaussian_inverse:{gaussian_method}",
                radius=standard_radius,
            )
        )
    for delt in extra_dp_deltas:
        std_dp_certs.append(rdp_to_dp(std_rdp, delta=delt, source="rdp_to_dp"))
    std = PrivacyLedger(
        mechanism="gaussian_output_perturbation",
        sigma=sigma,
        radius=standard_radius,
        rdp_curve=std_rdp,
        dp_certificates=std_dp_certs,
    )
    return DualPrivacyLedger(ball=ball, standard=std)
