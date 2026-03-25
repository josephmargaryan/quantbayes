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


def _expand_schedule(value_or_seq, T: int, *, cast_fn=float):
    if isinstance(value_or_seq, (tuple, list)):
        seq = [cast_fn(v) for v in value_or_seq]
        if len(seq) != T:
            raise ValueError(f"Expected schedule of length {T}, got {len(seq)}")
        return seq
    return [cast_fn(value_or_seq) for _ in range(T)]


def _effective_noise_stds(
    clip_schedule: Sequence[float], noise_multiplier_schedule: Sequence[float]
) -> list[float]:
    out: list[float] = []
    for c, nm in zip(clip_schedule, noise_multiplier_schedule):
        c = float(c)
        nm = float(nm)
        if c < 0.0:
            raise ValueError("clip norms must be >= 0.")
        if nm < 0.0:
            raise ValueError("noise multipliers must be >= 0.")
        out.append(float(c * nm))
    return out


def step_delta_ball_from_schedule(
    *,
    clip_schedule: Sequence[float],
    lz: float | None,
    radius: float,
) -> list[float] | None:
    if lz is None:
        return None
    lz = float(lz)
    radius = float(radius)
    if lz < 0.0:
        raise ValueError("lz must be >= 0.")
    if radius < 0.0:
        raise ValueError("radius must be >= 0.")
    out = []
    for c in clip_schedule:
        c = float(c)
        out.append(float(min(lz * radius, 2.0 * c)))
    return out


def step_delta_standard_from_schedule(
    *,
    clip_schedule: Sequence[float],
) -> list[float]:
    out = []
    for c in clip_schedule:
        c = float(c)
        if not math.isfinite(c):
            raise ValueError(
                "Standard DP-SGD accounting requires finite clip norms at every step."
            )
        out.append(float(2.0 * c))
    return out


def account_ball_sgd_noise_multiplier(
    *,
    noise_multiplier: float,
    accounting_view: str,
    orders: Sequence[int],
    dataset_size: int,
    num_steps: int,
    batch_size: int | Sequence[int],
    clip_norm: float | Sequence[float],
    radius: float,
    lz: float | None,
    dp_delta: float,
) -> dict[str, object]:
    """Account privacy for a scalar noise multiplier without running training.

    This is the accountant-only path for the fixed-size subsampled Gaussian
    DP-SGD comparison used by the nonconvex Ball-vs-standard experiments.
    """
    accounting_view = str(accounting_view).lower()
    if accounting_view not in {"ball", "standard"}:
        raise ValueError("accounting_view must be one of {'ball', 'standard'}.")

    dataset_size = int(dataset_size)
    num_steps = int(num_steps)
    noise_multiplier = float(noise_multiplier)
    radius = float(radius)
    dp_delta = float(dp_delta)

    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if noise_multiplier < 0.0:
        raise ValueError("noise_multiplier must be >= 0.")
    if not (0.0 < dp_delta < 1.0):
        raise ValueError("dp_delta must be in (0,1).")

    step_batch_sizes = _expand_schedule(batch_size, num_steps, cast_fn=int)
    step_clip_norms = _expand_schedule(clip_norm, num_steps, cast_fn=float)
    step_noise_multipliers = _expand_schedule(
        noise_multiplier, num_steps, cast_fn=float
    )
    step_noise_stds = _effective_noise_stds(step_clip_norms, step_noise_multipliers)
    step_delta_ball = step_delta_ball_from_schedule(
        clip_schedule=step_clip_norms,
        lz=lz,
        radius=radius,
    )
    step_delta_std = step_delta_standard_from_schedule(clip_schedule=step_clip_norms)

    if accounting_view == "ball" and step_delta_ball is None:
        raise ValueError(
            "Ball accounting requires a theorem-backed lz so that step_delta_ball is available."
        )

    dual_ledger = build_ball_sgd_rdp_ledgers(
        orders=tuple(int(v) for v in orders),
        step_batch_sizes=step_batch_sizes,
        dataset_size=dataset_size,
        step_clip_norms=step_clip_norms,
        step_noise_stds=step_noise_stds,
        step_delta_ball=step_delta_ball,
        step_delta_std=step_delta_std,
        radius=radius,
        dp_delta=dp_delta,
    )

    ledger = dual_ledger.ball if accounting_view == "ball" else dual_ledger.standard
    if not ledger.dp_certificates:
        raise ValueError(
            f"No DP certificate available for accounting_view={accounting_view!r}."
        )

    epsilon = float(ledger.dp_certificates[0].epsilon)
    return {
        "accounting_view": accounting_view,
        "noise_multiplier": noise_multiplier,
        "epsilon": epsilon,
        "ledger": dual_ledger,
        "step_batch_sizes": list(map(int, step_batch_sizes)),
        "step_clip_norms": list(map(float, step_clip_norms)),
        "step_noise_stds": list(map(float, step_noise_stds)),
        "step_delta_ball": (
            None if step_delta_ball is None else list(map(float, step_delta_ball))
        ),
        "step_delta_std": list(map(float, step_delta_std)),
    }


def calibrate_ball_sgd_noise_multiplier(
    *,
    target_epsilon: float,
    accounting_view: str,
    orders: Sequence[int],
    dataset_size: int,
    num_steps: int,
    batch_size: int | Sequence[int],
    clip_norm: float | Sequence[float],
    radius: float,
    lz: float | None,
    dp_delta: float,
    lower: float = 1e-3,
    upper: float = 0.25,
    max_upper: float = 128.0,
    num_bisection_steps: int = 10,
) -> dict[str, object]:
    """Calibrate the minimum scalar noise multiplier using only the accountant.

    This is the recommended privacy-calibration path for the nonconvex Ball-vs-standard
    DP-SGD experiments: no training is needed during calibration.
    """
    target_epsilon = float(target_epsilon)
    lower = float(lower)
    upper = float(upper)
    max_upper = float(max_upper)
    num_bisection_steps = int(num_bisection_steps)

    if target_epsilon <= 0.0:
        raise ValueError("target_epsilon must be > 0.")
    if not (0.0 < lower < upper):
        raise ValueError("Require 0 < lower < upper.")
    if max_upper <= upper:
        raise ValueError("max_upper must exceed upper.")
    if num_bisection_steps < 0:
        raise ValueError("num_bisection_steps must be >= 0.")

    lo = lower
    hi = upper

    while True:
        summary_hi = account_ball_sgd_noise_multiplier(
            noise_multiplier=hi,
            accounting_view=accounting_view,
            orders=orders,
            dataset_size=dataset_size,
            num_steps=num_steps,
            batch_size=batch_size,
            clip_norm=clip_norm,
            radius=radius,
            lz=lz,
            dp_delta=dp_delta,
        )
        eps_hi = float(summary_hi["epsilon"])
        if eps_hi <= target_epsilon:
            best_summary = summary_hi
            break
        lo = hi
        hi *= 2.0
        if hi > max_upper:
            raise RuntimeError(
                "Failed to bracket a privacy-feasible noise multiplier. "
                f"Last tried upper={hi:.6g}, target_epsilon={target_epsilon:.6g}."
            )

    for _ in range(num_bisection_steps):
        mid = 0.5 * (lo + hi)
        summary_mid = account_ball_sgd_noise_multiplier(
            noise_multiplier=mid,
            accounting_view=accounting_view,
            orders=orders,
            dataset_size=dataset_size,
            num_steps=num_steps,
            batch_size=batch_size,
            clip_norm=clip_norm,
            radius=radius,
            lz=lz,
            dp_delta=dp_delta,
        )
        eps_mid = float(summary_mid["epsilon"])
        if eps_mid <= target_epsilon:
            hi = mid
            best_summary = summary_mid
        else:
            lo = mid

    out = dict(best_summary)
    out["noise_multiplier"] = float(hi)
    return out
