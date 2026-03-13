# quantbayes/ball_dp/accountants/subsampled_gaussian.py
from __future__ import annotations

import math
from typing import List, Sequence

from .rdp import RdpCurve, compose_rdp_curves, rdp_to_dp
from ..types import DpCertificate, DualPrivacyLedger, PrivacyLedger, StepPrivacyRecord


def _logsumexp(xs: Sequence[float]) -> float:
    xs = list(xs)
    m = max(xs)
    return float(m + math.log(sum(math.exp(x - m) for x in xs)))


def _log_comb(n: int, k: int) -> float:
    return float(math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0))


def _log_expm1(x: float) -> float:
    if x <= math.log(2.0):
        return float(math.log(math.expm1(x)))
    return float(x + math.log1p(-math.exp(-x)))


def fixed_size_subsampled_gaussian_rdp(
    *, alpha: int, sample_rate: float, sensitivity: float, noise_std: float
) -> float:
    alpha = int(alpha)
    q = float(sample_rate)
    delta = float(sensitivity)
    nu = float(noise_std)
    if alpha < 2:
        raise ValueError(
            "The fixed-size theorem-backed accountant requires integer alpha >= 2."
        )
    if not (0.0 <= q <= 1.0):
        raise ValueError("sample_rate must be in [0,1].")
    if delta < 0.0:
        raise ValueError("sensitivity must be >= 0.")
    if nu <= 0.0:
        raise ValueError("noise_std must be > 0.")
    if delta == 0.0 or q == 0.0:
        return 0.0

    logs = [0.0]  # the leading 1 inside the logarithm
    eps_g2 = (delta * delta) / (nu * nu)
    log_term2_a = math.log(4.0) + _log_expm1(eps_g2)
    log_term2_b = math.log(2.0) + eps_g2
    logs.append(2.0 * math.log(q) + _log_comb(alpha, 2) + min(log_term2_a, log_term2_b))
    for j in range(3, alpha + 1):
        eps_gj = j * delta * delta / (2.0 * nu * nu)
        log_coeff = (
            j * math.log(q) + _log_comb(alpha, j) + math.log(2.0) + (j - 1.0) * eps_gj
        )
        logs.append(log_coeff)
    return float((_logsumexp(logs)) / (alpha - 1.0))


def build_ball_sgd_rdp_ledgers(
    *,
    orders: Sequence[int],
    step_batch_sizes: Sequence[int],
    dataset_size: int,
    step_clip_norms: Sequence[float],
    step_noise_stds: Sequence[float],
    step_delta_ball: Sequence[float] | None,
    step_delta_std: Sequence[float] | None,
    radius: float | None,
    dp_delta: float | None = None,
) -> DualPrivacyLedger:
    if not (len(step_batch_sizes) == len(step_clip_norms) == len(step_noise_stds)):
        raise ValueError("Step schedules must have the same length.")

    n_steps = len(step_batch_sizes)
    if step_delta_ball is not None and len(step_delta_ball) != n_steps:
        raise ValueError(
            "step_delta_ball must have the same length as the step schedule."
        )
    if step_delta_std is not None and len(step_delta_std) != n_steps:
        raise ValueError(
            "step_delta_std must have the same length as the step schedule."
        )

    step_records: List[StepPrivacyRecord] = []
    ball_curves: List[RdpCurve] = []
    std_curves: List[RdpCurve] = []

    for t in range(n_steps):
        q = float(step_batch_sizes[t]) / float(dataset_size)

        ball_map = {}
        std_map = {}
        delta_ball_t = float("nan")
        delta_std_t = float("nan")

        if step_delta_ball is not None:
            delta_ball_t = float(step_delta_ball[t])
            for alpha in orders:
                eps_ball = fixed_size_subsampled_gaussian_rdp(
                    alpha=int(alpha),
                    sample_rate=q,
                    sensitivity=delta_ball_t,
                    noise_std=float(step_noise_stds[t]),
                )
                ball_map[float(alpha)] = float(eps_ball)

            ball_curves.append(
                RdpCurve(
                    orders=tuple(float(a) for a in orders),
                    epsilons=tuple(ball_map[float(a)] for a in orders),
                    source=f"ball_sgd_step_{t + 1}",
                    radius=radius,
                )
            )

        if step_delta_std is not None:
            delta_std_t = float(step_delta_std[t])
            for alpha in orders:
                eps_std = fixed_size_subsampled_gaussian_rdp(
                    alpha=int(alpha),
                    sample_rate=q,
                    sensitivity=delta_std_t,
                    noise_std=float(step_noise_stds[t]),
                )
                std_map[float(alpha)] = float(eps_std)

            std_curves.append(
                RdpCurve(
                    orders=tuple(float(a) for a in orders),
                    epsilons=tuple(std_map[float(a)] for a in orders),
                    source=f"std_sgd_step_{t + 1}",
                    radius=None,
                )
            )

        step_records.append(
            StepPrivacyRecord(
                step=t + 1,
                batch_size=int(step_batch_sizes[t]),
                sample_rate=q,
                clip_norm=float(step_clip_norms[t]),
                noise_std=float(step_noise_stds[t]),
                delta_ball=delta_ball_t,
                delta_std=delta_std_t,
                ball_rdp=ball_map,
                std_rdp=std_map,
            )
        )

    if ball_curves:
        ball_total = compose_rdp_curves(
            ball_curves, source="ball_sgd_total", radius=radius
        )
        ball = PrivacyLedger(
            mechanism="ball_sgd_rdp",
            radius=radius,
            rdp_curve=ball_total,
            step_records=step_records,
        )
        if dp_delta is not None:
            ball.dp_certificates.append(
                rdp_to_dp(ball_total, delta=dp_delta, source="rdp_to_dp")
            )
    else:
        ball = PrivacyLedger(
            mechanism="ball_sgd_rdp_unavailable",
            radius=radius,
            step_records=step_records,
            notes=[
                "Ball accounting unavailable because no Ball sensitivity schedule was supplied."
            ],
        )

    if std_curves:
        std_total = compose_rdp_curves(std_curves, source="std_sgd_total")
        std = PrivacyLedger(
            mechanism="std_sgd_rdp",
            rdp_curve=std_total,
            step_records=step_records,
        )
        if dp_delta is not None:
            std.dp_certificates.append(
                rdp_to_dp(std_total, delta=dp_delta, source="rdp_to_dp")
            )
    else:
        std = PrivacyLedger(
            mechanism="std_sgd_rdp_unavailable",
            step_records=step_records,
            notes=[
                "Standard accounting unavailable because no finite clipping-based sensitivity schedule was supplied."
            ],
        )

    return DualPrivacyLedger(ball=ball, standard=std)
