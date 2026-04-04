from __future__ import annotations

import dataclasses as dc
import math
from statistics import NormalDist
from typing import Literal, Sequence

from ..types import PriorFamily, ReRoPoint, ReRoReport, ReleaseArtifact

_STD_NORMAL = NormalDist()


@dc.dataclass(frozen=True)
class DirectStepProfile:
    """One-step direct profile for the Bernoulli-revealed Gaussian shift.

    This represents the theorem-side profile
        Psi_{gamma, c}(kappa)
    with
        gamma = sample_rate,
        c     = sensitivity / noise_std.
    """

    step: int
    sample_rate: float
    sensitivity: float
    noise_std: float
    c: float
    tau: float
    v: float
    kappa_left: float
    kappa_right: float

    def evaluate(self, kappa: float) -> float:
        return bernoulli_revealed_gaussian_test_profile(
            kappa=kappa,
            sample_rate=self.sample_rate,
            c=self.c,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "step": int(self.step),
            "sample_rate": float(self.sample_rate),
            "sensitivity": float(self.sensitivity),
            "noise_std": float(self.noise_std),
            "c": float(self.c),
            "tau": float(self.tau),
            "v": float(self.v),
            "kappa_left": float(self.kappa_left),
            "kappa_right": float(self.kappa_right),
        }


def gaussian_shift_test_profile(*, kappa: float, c: float) -> float:
    """Equal-covariance Gaussian blow-up profile G_c(kappa).

    Returns
        Phi(Phi^{-1}(kappa) + c)
    with boundary cases handled by continuity.
    """
    kappa = float(kappa)
    c = float(c)

    if not math.isfinite(c) or c < 0.0:
        raise ValueError("c must be finite and >= 0.")
    if kappa <= 0.0:
        return 0.0
    if kappa >= 1.0:
        return 1.0

    z = _STD_NORMAL.inv_cdf(kappa) + c
    return float(min(1.0, max(0.0, _STD_NORMAL.cdf(z))))


def bernoulli_revealed_gaussian_test_profile(
    *,
    kappa: float,
    sample_rate: float,
    c: float,
) -> float:
    """The theorem-side one-step direct profile Psi_{gamma,c}(kappa).

    Implements the piecewise closed form from
    "A direct Ball-ReRo theorem for Poisson Ball-SGD".
    """
    kappa = float(kappa)
    gamma = float(sample_rate)
    c = float(c)

    if not (0.0 <= gamma <= 1.0):
        raise ValueError("sample_rate must lie in [0, 1].")
    if not math.isfinite(c) or c < 0.0:
        raise ValueError("c must be finite and >= 0.")
    if kappa <= 0.0:
        return 0.0
    if kappa >= 1.0:
        return 1.0
    if gamma == 0.0 or c == 0.0:
        return float(min(1.0, max(0.0, kappa)))

    tau = float(_STD_NORMAL.cdf(-0.5 * c))
    v = float(2.0 * _STD_NORMAL.cdf(0.5 * c) - 1.0)
    left = float(gamma * tau)
    right = float(1.0 - gamma + left)

    if kappa <= left:
        inner = min(1.0, max(0.0, kappa / gamma))
        out = gamma * gaussian_shift_test_profile(kappa=inner, c=c)
        return float(min(1.0, max(0.0, out)))

    if kappa <= right:
        out = kappa + gamma * v
        return float(min(1.0, max(0.0, out)))

    inner = min(1.0, max(0.0, (kappa - (1.0 - gamma)) / gamma))
    out = (1.0 - gamma) + gamma * gaussian_shift_test_profile(kappa=inner, c=c)
    return float(min(1.0, max(0.0, out)))


def make_direct_step_profile(
    *,
    step: int,
    sample_rate: float,
    sensitivity: float,
    noise_std: float,
) -> DirectStepProfile:
    sample_rate = float(sample_rate)
    sensitivity = float(sensitivity)
    noise_std = float(noise_std)

    if not (0.0 <= sample_rate <= 1.0):
        raise ValueError("sample_rate must lie in [0, 1].")
    if not math.isfinite(sensitivity) or sensitivity < 0.0:
        raise ValueError("sensitivity must be finite and >= 0.")
    if not math.isfinite(noise_std) or noise_std <= 0.0:
        raise ValueError("noise_std must be finite and > 0.")

    c = float(sensitivity / noise_std)
    tau = float(_STD_NORMAL.cdf(-0.5 * c))
    v = float(2.0 * _STD_NORMAL.cdf(0.5 * c) - 1.0)
    kappa_left = float(sample_rate * tau)
    kappa_right = float(1.0 - sample_rate + kappa_left)

    return DirectStepProfile(
        step=int(step),
        sample_rate=sample_rate,
        sensitivity=sensitivity,
        noise_std=noise_std,
        c=c,
        tau=tau,
        v=v,
        kappa_left=kappa_left,
        kappa_right=kappa_right,
    )


def compose_direct_profiles(
    kappa: float,
    profiles: Sequence[DirectStepProfile],
) -> float:
    """Evaluate Gamma_1 o ... o Gamma_T at kappa.

    Since function composition applies the last function first, this evaluates
    the profiles in reverse time order:
        kappa -> Gamma_T -> ... -> Gamma_2 -> Gamma_1.
    """
    value = float(kappa)
    for profile in reversed(tuple(profiles)):
        value = profile.evaluate(value)
    return float(min(1.0, max(0.0, value)))


def _fixed_batch_schedule_count(release: ReleaseArtifact) -> int:
    cfg = dict(getattr(release, "training_config", {}) or {})
    extra = dict(getattr(release, "extra", {}) or {})

    if "fixed_batch_schedule_num_steps" in cfg:
        return int(cfg["fixed_batch_schedule_num_steps"])
    if "fixed_batch_schedule_num_steps" in extra:
        return int(extra["fixed_batch_schedule_num_steps"])

    present_flag = bool(
        cfg.get(
            "fixed_batch_schedule_present",
            extra.get("fixed_batch_schedule_present", False),
        )
    )
    return 1 if present_flag else 0


def extract_ball_sgd_direct_step_profiles(
    release: ReleaseArtifact,
    *,
    sensitivity_view: Literal["ball", "standard"] = "ball",
) -> list[DirectStepProfile]:
    """Extract theorem-valid one-step direct profiles from a release artifact.

    The theorem-backed conditions enforced here are:
      - actual Poisson subsampling during training,
      - no forced / fixed batch schedule,
      - fixed normalization of the noisy sum (either none or target batch size),
      - strictly positive Gaussian noise std at every step,
      - an available sensitivity schedule for the requested view.
    """
    cfg = dict(getattr(release, "training_config", {}) or {})

    batch_sampler = str(
        cfg.get("resolved_batch_sampler", cfg.get("batch_sampler", ""))
    ).lower()
    if batch_sampler != "poisson":
        raise ValueError(
            "mode='ball_sgd_direct' is theorem-backed only for releases trained "
            "with actual Poisson subsampling (batch_sampler='poisson')."
        )

    fixed_count = _fixed_batch_schedule_count(release)
    if fixed_count > 0:
        raise ValueError(
            "mode='ball_sgd_direct' is not theorem-backed when a fixed/forced "
            "batch schedule was used. Re-run the release with ordinary Poisson "
            "sampling and no fixed_batch_indices_schedule."
        )

    normalize_mode = str(cfg.get("normalize_noisy_sum_by", "batch_size")).lower()
    if normalize_mode not in {"batch_size", "none"}:
        raise ValueError(
            "mode='ball_sgd_direct' is theorem-backed only when the sanitized "
            "object is either the noisy sum or a fixed rescaling of it. "
            "normalize_noisy_sum_by='realized_batch_size' is not supported."
        )

    sample_rates = tuple(float(v) for v in cfg.get("sample_rates", ()))
    noise_stds = tuple(float(v) for v in cfg.get("effective_noise_stds", ()))
    if not sample_rates or not noise_stds:
        raise ValueError(
            "mode='ball_sgd_direct' requires per-step sample_rates and effective_noise_stds "
            "stored in the release artifact."
        )
    if len(sample_rates) != len(noise_stds):
        raise ValueError(
            "Release artifact is inconsistent: len(sample_rates) != len(effective_noise_stds)."
        )
    if any((not math.isfinite(q)) or q < 0.0 or q > 1.0 for q in sample_rates):
        raise ValueError("All per-step sample rates must be finite and lie in [0,1].")
    if any((not math.isfinite(nu)) or nu <= 0.0 for nu in noise_stds):
        raise ValueError(
            "mode='ball_sgd_direct' requires strictly positive effective Gaussian noise std "
            "at every step."
        )

    if sensitivity_view == "ball":
        sensitivities = release.sensitivity.step_delta_ball
        if sensitivities is None:
            raise ValueError(
                "mode='ball_sgd_direct' requires release.sensitivity.step_delta_ball."
            )
    elif sensitivity_view == "standard":
        sensitivities = release.sensitivity.step_delta_std
        if sensitivities is None:
            raise ValueError(
                "Standard direct comparator requested, but release.sensitivity.step_delta_std is missing."
            )
    else:
        raise ValueError("sensitivity_view must be one of {'ball', 'standard'}.")

    if len(sensitivities) != len(sample_rates):
        raise ValueError(
            "Release artifact is inconsistent: step sensitivity schedule length does not "
            "match the number of training steps."
        )

    profiles: list[DirectStepProfile] = []
    for step, (q, delta, nu) in enumerate(
        zip(sample_rates, sensitivities, noise_stds),
        start=1,
    ):
        profiles.append(
            make_direct_step_profile(
                step=step,
                sample_rate=float(q),
                sensitivity=float(delta),
                noise_std=float(nu),
            )
        )
    return profiles


def compute_ball_sgd_direct_rero_report(
    release: ReleaseArtifact,
    prior: PriorFamily,
    eta_grid: Sequence[float],
) -> ReRoReport:
    """Compute the direct Ball-ReRo bound for Poisson Ball-SGD final releases."""
    ball_profiles = extract_ball_sgd_direct_step_profiles(
        release,
        sensitivity_view="ball",
    )

    try:
        standard_profiles = extract_ball_sgd_direct_step_profiles(
            release,
            sensitivity_view="standard",
        )
    except Exception:
        standard_profiles = None

    points: list[ReRoPoint] = []
    for eta in eta_grid:
        eta_f = float(eta)
        kappa = float(prior.kappa(eta_f))
        gamma_ball = compose_direct_profiles(kappa, ball_profiles)
        gamma_standard = None
        if standard_profiles is not None:
            gamma_standard = compose_direct_profiles(kappa, standard_profiles)
        points.append(
            ReRoPoint(
                eta=eta_f,
                kappa=kappa,
                gamma_ball=float(gamma_ball),
                gamma_standard=(
                    None if gamma_standard is None else float(gamma_standard)
                ),
            )
        )

    cfg = dict(getattr(release, "training_config", {}) or {})
    metadata = {
        "radius": release.privacy.ball.radius,
        "release_kind": release.release_kind,
        "ball_mechanism": release.privacy.ball.mechanism,
        "batch_sampler": str(
            cfg.get("resolved_batch_sampler", cfg.get("batch_sampler", "unknown"))
        ),
        "normalize_noisy_sum_by": str(cfg.get("normalize_noisy_sum_by", "batch_size")),
        "fixed_batch_schedule_present": bool(_fixed_batch_schedule_count(release) > 0),
        "num_steps": int(len(ball_profiles)),
        "composition": "Gamma_1 o ... o Gamma_T",
        "application_order": "step_T_to_step_1",
        "final_release_view": "postprocessing_of_poisson_sgd_transcript",
        "standard_comparator_note": (
            "gamma_standard uses the same direct-profile composition with the "
            "clipping-only sensitivity schedule step_delta_std = 2 C_t when that "
            "schedule is available in the release artifact."
        ),
        "ball_step_profiles": [profile.as_dict() for profile in ball_profiles],
        "standard_step_profiles": (
            None
            if standard_profiles is None
            else [profile.as_dict() for profile in standard_profiles]
        ),
    }
    return ReRoReport(
        mode="ball_sgd_direct",
        points=points,
        metadata=metadata,
    )


def direct_profile_step_summary(
    *,
    sample_rate: float,
    sensitivity: float | None,
    noise_std: float | None,
) -> dict[str, float | None]:
    """Convenience summary for printing / tabulating one-step direct-profile inputs."""
    if sensitivity is None or noise_std is None:
        return {
            "direct_c": None,
            "direct_tau": None,
            "direct_v": None,
            "direct_kappa_left": None,
            "direct_kappa_right": None,
        }
    if not math.isfinite(float(noise_std)) or float(noise_std) <= 0.0:
        return {
            "direct_c": None,
            "direct_tau": None,
            "direct_v": None,
            "direct_kappa_left": None,
            "direct_kappa_right": None,
        }

    profile = make_direct_step_profile(
        step=0,
        sample_rate=float(sample_rate),
        sensitivity=float(sensitivity),
        noise_std=float(noise_std),
    )
    return {
        "direct_c": float(profile.c),
        "direct_tau": float(profile.tau),
        "direct_v": float(profile.v),
        "direct_kappa_left": float(profile.kappa_left),
        "direct_kappa_right": float(profile.kappa_right),
    }
