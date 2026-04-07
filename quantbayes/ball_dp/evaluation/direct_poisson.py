from __future__ import annotations

import dataclasses as dc
import functools
import math
from statistics import NormalDist
from typing import Literal, Sequence

import numpy as np

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


@dc.dataclass(frozen=True)
class LatentMixtureStepProfile:
    r"""One-step latent-mixture profile for Poisson Ball-SGD.

    This represents the theorem-side profile
        \widetilde{Gamma}_{gamma, c}(kappa)
      = (1 - gamma) * kappa + gamma * G_c(kappa)
    where G_c is the Gaussian shift profile.
    """

    step: int
    sample_rate: float
    sensitivity: float
    noise_std: float
    c: float

    def evaluate(self, kappa: float) -> float:
        return latent_mixture_gaussian_test_profile(
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
        }


@dc.dataclass(frozen=True)
class DiscretePrivacyLossGrid:
    """Discrete approximation to a privacy-loss distribution.

    The support is the uniform grid
        {(offset_index + j) * step : j = 0, ..., m - 1}
    with q_mass and p_mass storing the approximate distribution of the total
    privacy-loss random variable under the reference null and alternative,
    respectively.
    """

    step: float
    offset_index: int
    q_mass: np.ndarray
    p_mass: np.ndarray

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.step)) or float(self.step) <= 0.0:
            raise ValueError("step must be finite and > 0.")
        if self.q_mass.ndim != 1 or self.p_mass.ndim != 1:
            raise ValueError("q_mass and p_mass must be one-dimensional arrays.")
        if self.q_mass.shape != self.p_mass.shape:
            raise ValueError("q_mass and p_mass must have the same shape.")
        if self.q_mass.size == 0:
            raise ValueError("DiscretePrivacyLossGrid requires a non-empty support.")

    @property
    def size(self) -> int:
        return int(self.q_mass.size)

    @property
    def loss_min(self) -> float:
        return float(self.offset_index * self.step)

    @property
    def loss_max(self) -> float:
        return float((self.offset_index + self.size - 1) * self.step)

    def blowup(self, kappa: float) -> float:
        """Approximate B_kappa(P, Q) from the discretized privacy-loss law.

        The privacy-loss support is already sorted in increasing order, so the
        Neyman--Pearson optimal event is approximated by an upper tail of the
        discretized support, with linear interpolation within the threshold bin.
        """
        kappa = float(kappa)
        if kappa <= 0.0:
            return 0.0
        if kappa >= 1.0:
            return 1.0

        q = np.asarray(self.q_mass, dtype=np.float64)
        p = np.asarray(self.p_mass, dtype=np.float64)
        q_sum = float(np.sum(q))
        p_sum = float(np.sum(p))
        if q_sum <= 0.0 or p_sum <= 0.0:
            raise ValueError(
                "DiscretePrivacyLossGrid mass arrays must have positive total mass."
            )
        q = q / q_sum
        p = p / p_sum

        q_desc = q[::-1]
        p_desc = p[::-1]
        cum_q = np.cumsum(q_desc)
        cum_p = np.cumsum(p_desc)

        idx = int(np.searchsorted(cum_q, kappa, side="left"))
        if idx >= q_desc.size:
            return 1.0

        prev_q = 0.0 if idx == 0 else float(cum_q[idx - 1])
        prev_p = 0.0 if idx == 0 else float(cum_p[idx - 1])
        q_bin = float(q_desc[idx])
        p_bin = float(p_desc[idx])

        if q_bin <= 0.0:
            return float(min(1.0, max(0.0, cum_p[idx])))

        theta = (kappa - prev_q) / q_bin
        theta = float(min(1.0, max(0.0, theta)))
        value = prev_p + theta * p_bin
        return float(min(1.0, max(0.0, value)))

    def as_summary(self) -> dict[str, float | int]:
        return {
            "grid_step": float(self.step),
            "offset_index": int(self.offset_index),
            "support_size": int(self.size),
            "loss_min": float(self.loss_min),
            "loss_max": float(self.loss_max),
        }


@functools.lru_cache(maxsize=None)
def _standard_normal_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    if int(order) <= 0:
        raise ValueError("quadrature order must be positive.")
    nodes, weights = np.polynomial.hermite.hermgauss(int(order))
    z = np.sqrt(2.0) * np.asarray(nodes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64) / math.sqrt(math.pi)
    return z, w


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


def latent_mixture_gaussian_test_profile(
    *,
    kappa: float,
    sample_rate: float,
    c: float,
) -> float:
    """One-step latent-mixture profile for Poisson Ball-SGD.

    For the one-dimensional pair
        mu = (1-gamma) N(0,1) + gamma N(c,1),
        nu = N(0,1),
    the likelihood ratio is monotone in the observation, so the optimal level-kappa
    test is an upper tail under nu. This yields the closed form
        (1-gamma) * kappa + gamma * G_c(kappa).
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

    value = (1.0 - gamma) * kappa + gamma * gaussian_shift_test_profile(
        kappa=kappa, c=c
    )
    return float(min(1.0, max(0.0, value)))


def bernoulli_revealed_gaussian_test_profile(
    *,
    kappa: float,
    sample_rate: float,
    c: float,
) -> float:
    """The theorem-side one-step direct profile Psi_{gamma,c}(kappa).

    Implements the piecewise closed form for the revealed-inclusion-bit upper bound.
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


def make_latent_mixture_step_profile(
    *,
    step: int,
    sample_rate: float,
    sensitivity: float,
    noise_std: float,
) -> LatentMixtureStepProfile:
    sample_rate = float(sample_rate)
    sensitivity = float(sensitivity)
    noise_std = float(noise_std)

    if not (0.0 <= sample_rate <= 1.0):
        raise ValueError("sample_rate must lie in [0, 1].")
    if not math.isfinite(sensitivity) or sensitivity < 0.0:
        raise ValueError("sensitivity must be finite and >= 0.")
    if not math.isfinite(noise_std) or noise_std <= 0.0:
        raise ValueError("noise_std must be finite and > 0.")

    return LatentMixtureStepProfile(
        step=int(step),
        sample_rate=sample_rate,
        sensitivity=sensitivity,
        noise_std=noise_std,
        c=float(sensitivity / noise_std),
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


def compose_latent_mixture_profiles(
    kappa: float,
    profiles: Sequence[LatentMixtureStepProfile],
) -> float:
    """Evaluate the per-step latent-mixture composition upper bound.

    This is the same adaptive composition pattern as the direct revealed-bit bound,
    but each one-step profile is the tighter latent-mixture profile.
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


def _extract_ball_sgd_step_inputs(
    release: ReleaseArtifact,
    *,
    sensitivity_view: Literal["ball", "standard"],
    mode_name: str,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], dict[str, object]]:
    """Extract theorem-valid one-step inputs from a release artifact."""
    cfg = dict(getattr(release, "training_config", {}) or {})

    batch_sampler = str(
        cfg.get("resolved_batch_sampler", cfg.get("batch_sampler", ""))
    ).lower()
    if batch_sampler != "poisson":
        raise ValueError(
            f"mode='{mode_name}' is theorem-backed only for releases trained "
            "with actual Poisson subsampling (batch_sampler='poisson')."
        )

    fixed_count = _fixed_batch_schedule_count(release)
    if fixed_count > 0:
        raise ValueError(
            f"mode='{mode_name}' is not theorem-backed when a fixed/forced "
            "batch schedule was used. Re-run the release with ordinary Poisson "
            "sampling and no fixed_batch_indices_schedule."
        )

    normalize_mode = str(cfg.get("normalize_noisy_sum_by", "batch_size")).lower()
    if normalize_mode not in {"batch_size", "none"}:
        raise ValueError(
            f"mode='{mode_name}' is theorem-backed only when the sanitized "
            "object is either the noisy sum or a fixed rescaling of it. "
            "normalize_noisy_sum_by='realized_batch_size' is not supported."
        )

    sample_rates = tuple(float(v) for v in cfg.get("sample_rates", ()))
    noise_stds = tuple(float(v) for v in cfg.get("effective_noise_stds", ()))
    if not sample_rates or not noise_stds:
        raise ValueError(
            f"mode='{mode_name}' requires per-step sample_rates and effective_noise_stds "
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
            f"mode='{mode_name}' requires strictly positive effective Gaussian noise std "
            "at every step."
        )

    if sensitivity_view == "ball":
        sensitivities = release.sensitivity.step_delta_ball
        if sensitivities is None:
            raise ValueError(
                f"mode='{mode_name}' requires release.sensitivity.step_delta_ball."
            )
    elif sensitivity_view == "standard":
        sensitivities = release.sensitivity.step_delta_std
        if sensitivities is None:
            raise ValueError(
                "Standard comparator requested, but release.sensitivity.step_delta_std is missing."
            )
    else:
        raise ValueError("sensitivity_view must be one of {'ball', 'standard'}.")

    if len(sensitivities) != len(sample_rates):
        raise ValueError(
            "Release artifact is inconsistent: step sensitivity schedule length does not "
            "match the number of training steps."
        )

    sensitivities_tuple = tuple(float(v) for v in sensitivities)
    return sample_rates, sensitivities_tuple, noise_stds, cfg


def extract_ball_sgd_direct_step_profiles(
    release: ReleaseArtifact,
    *,
    sensitivity_view: Literal["ball", "standard"] = "ball",
) -> list[DirectStepProfile]:
    sample_rates, sensitivities, noise_stds, _ = _extract_ball_sgd_step_inputs(
        release,
        sensitivity_view=sensitivity_view,
        mode_name="ball_sgd_direct",
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


def extract_ball_sgd_latent_step_profiles(
    release: ReleaseArtifact,
    *,
    sensitivity_view: Literal["ball", "standard"] = "ball",
    mode_name: str = "ball_sgd_mix_comp",
) -> list[LatentMixtureStepProfile]:
    sample_rates, sensitivities, noise_stds, _ = _extract_ball_sgd_step_inputs(
        release,
        sensitivity_view=sensitivity_view,
        mode_name=mode_name,
    )

    profiles: list[LatentMixtureStepProfile] = []
    for step, (q, delta, nu) in enumerate(
        zip(sample_rates, sensitivities, noise_stds),
        start=1,
    ):
        profiles.append(
            make_latent_mixture_step_profile(
                step=step,
                sample_rate=float(q),
                sensitivity=float(delta),
                noise_std=float(nu),
            )
        )
    return profiles


def _common_direct_metadata(
    release: ReleaseArtifact, cfg: dict[str, object]
) -> dict[str, object]:
    return {
        "radius": release.privacy.ball.radius,
        "release_kind": release.release_kind,
        "ball_mechanism": release.privacy.ball.mechanism,
        "batch_sampler": str(
            cfg.get("resolved_batch_sampler", cfg.get("batch_sampler", "unknown"))
        ),
        "normalize_noisy_sum_by": str(cfg.get("normalize_noisy_sum_by", "batch_size")),
        "fixed_batch_schedule_present": bool(_fixed_batch_schedule_count(release) > 0),
    }


def compute_ball_sgd_direct_rero_report(
    release: ReleaseArtifact,
    prior: PriorFamily,
    eta_grid: Sequence[float],
) -> ReRoReport:
    """Compute the revealed-bit direct Ball-ReRo bound for Poisson Ball-SGD."""
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
        **_common_direct_metadata(release, cfg),
        "num_steps": int(len(ball_profiles)),
        "composition": "Gamma_1 o ... o Gamma_T",
        "application_order": "step_T_to_step_1",
        "final_release_view": "postprocessing_of_poisson_sgd_transcript",
        "profile_family": "revealed_inclusion_bit",
        "theorem_backed_for_raw_transcript": True,
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


def compute_ball_sgd_mixture_composed_rero_report(
    release: ReleaseArtifact,
    prior: PriorFamily,
    eta_grid: Sequence[float],
) -> ReRoReport:
    r"""Compute the latent-mixture per-step composition reference quantity.

    This evaluates the adaptive composition of the one-step hidden-mixture profiles
    \widetilde{Gamma}_t(kappa) = (1-gamma_t)kappa + gamma_t G_{c_t}(kappa).
    In the replacement-style Ball setting this is a useful centered/reference object,
    but it is not, by itself, a theorem-backed bound for the raw transcript unless an
    additional domination argument is supplied.
    """
    ball_profiles = extract_ball_sgd_latent_step_profiles(
        release,
        sensitivity_view="ball",
        mode_name="ball_sgd_mix_comp",
    )

    try:
        standard_profiles = extract_ball_sgd_latent_step_profiles(
            release,
            sensitivity_view="standard",
            mode_name="ball_sgd_mix_comp",
        )
    except Exception:
        standard_profiles = None

    points: list[ReRoPoint] = []
    for eta in eta_grid:
        eta_f = float(eta)
        kappa = float(prior.kappa(eta_f))
        gamma_ball = compose_latent_mixture_profiles(kappa, ball_profiles)
        gamma_standard = None
        if standard_profiles is not None:
            gamma_standard = compose_latent_mixture_profiles(kappa, standard_profiles)
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
        **_common_direct_metadata(release, cfg),
        "num_steps": int(len(ball_profiles)),
        "composition": "widetilde_Gamma_1 o ... o widetilde_Gamma_T",
        "application_order": "step_T_to_step_1",
        "final_release_view": "postprocessing_of_poisson_sgd_transcript",
        "profile_family": "latent_mixture_per_step",
        "ball_step_profiles": [profile.as_dict() for profile in ball_profiles],
        "standard_step_profiles": (
            None
            if standard_profiles is None
            else [profile.as_dict() for profile in standard_profiles]
        ),
        "status": "centered_reference_quantity_only",
        "theorem_backed_for_raw_transcript": False,
        "dominance_note": (
            "This per-step hidden-mixture composition is a useful centered/reference "
            "quantity, but for the raw replacement-style transcript it should be treated "
            "as exploratory unless you separately prove the missing domination step."
        ),
    }
    return ReRoReport(
        mode="ball_sgd_mix_comp",
        points=points,
        metadata=metadata,
    )


def _one_step_latent_mixture_privacy_loss_nodes(
    *,
    sample_rate: float,
    c: float,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gamma = float(sample_rate)
    c = float(c)
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("sample_rate must lie in [0,1].")
    if not math.isfinite(c) or c < 0.0:
        raise ValueError("c must be finite and >= 0.")

    if gamma == 0.0 or c == 0.0:
        losses = np.asarray([0.0], dtype=np.float64)
        q_mass = np.asarray([1.0], dtype=np.float64)
        p_mass = np.asarray([1.0], dtype=np.float64)
        return losses, q_mass, p_mass

    z, w = _standard_normal_quadrature(int(quadrature_order))
    if gamma == 1.0:
        losses = c * z - 0.5 * c * c
    else:
        log1m = math.log1p(-gamma)
        logg = math.log(gamma)
        losses = np.logaddexp(log1m, logg + c * z - 0.5 * c * c)
    q_mass = np.asarray(w, dtype=np.float64)
    p_mass = np.exp(losses) * q_mass
    q_mass = q_mass / float(np.sum(q_mass))
    p_mass = p_mass / float(np.sum(p_mass))
    return np.asarray(losses, dtype=np.float64), q_mass, p_mass


def _adaptive_privacy_loss_grid_step_from_losses(
    losses_collection: Sequence[np.ndarray],
    *,
    min_grid_step: float,
    target_bins: int,
) -> float:
    total_width = 0.0
    for losses in losses_collection:
        if losses.size == 0:
            continue
        total_width += float(np.max(losses) - np.min(losses))
    if total_width <= 0.0:
        return float(min_grid_step)
    return float(max(min_grid_step, total_width / float(max(1, target_bins))))


def _discretize_loss_nodes(
    *,
    losses: np.ndarray,
    q_weights: np.ndarray,
    p_weights: np.ndarray,
    grid_step: float,
) -> DiscretePrivacyLossGrid:
    losses = np.asarray(losses, dtype=np.float64)
    q_weights = np.asarray(q_weights, dtype=np.float64)
    p_weights = np.asarray(p_weights, dtype=np.float64)
    if losses.size == 1:
        return DiscretePrivacyLossGrid(
            step=float(grid_step),
            offset_index=int(round(float(losses[0]) / float(grid_step))),
            q_mass=np.asarray([1.0], dtype=np.float64),
            p_mass=np.asarray([1.0], dtype=np.float64),
        )

    lo_idx = int(math.floor(float(np.min(losses)) / float(grid_step))) - 1
    hi_idx = int(math.ceil(float(np.max(losses)) / float(grid_step))) + 1
    size = int(hi_idx - lo_idx + 1)

    q_mass = np.zeros(size, dtype=np.float64)
    p_mass = np.zeros(size, dtype=np.float64)

    idx = np.rint(losses / float(grid_step)).astype(np.int64) - int(lo_idx)
    idx = np.clip(idx, 0, size - 1)
    np.add.at(q_mass, idx, q_weights)
    np.add.at(p_mass, idx, p_weights)

    q_mass = q_mass / float(np.sum(q_mass))
    p_mass = p_mass / float(np.sum(p_mass))
    return DiscretePrivacyLossGrid(
        step=float(grid_step),
        offset_index=int(lo_idx),
        q_mass=q_mass,
        p_mass=p_mass,
    )


def _fft_convolve_nonnegative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out_size = int(a.size + b.size - 1)
    if out_size <= 256:
        out = np.convolve(a, b)
    else:
        fft_size = 1 << (out_size - 1).bit_length()
        out = np.fft.irfft(
            np.fft.rfft(a, n=fft_size) * np.fft.rfft(b, n=fft_size),
            n=fft_size,
        )[:out_size]
    out = np.maximum(out, 0.0)
    total = float(np.sum(out))
    if total <= 0.0:
        raise ValueError("Convolution produced non-positive total mass.")
    return out / total


def _trim_pld_support(
    grid: DiscretePrivacyLossGrid,
    *,
    trim_tolerance: float,
) -> DiscretePrivacyLossGrid:
    tol = float(trim_tolerance)
    if tol <= 0.0:
        return grid
    mass = np.asarray(grid.q_mass + grid.p_mass, dtype=np.float64)
    keep = np.flatnonzero(mass > tol)
    if keep.size == 0:
        return grid
    lo = int(keep[0])
    hi = int(keep[-1]) + 1
    q = np.asarray(grid.q_mass[lo:hi], dtype=np.float64)
    p = np.asarray(grid.p_mass[lo:hi], dtype=np.float64)
    q = q / float(np.sum(q))
    p = p / float(np.sum(p))
    return DiscretePrivacyLossGrid(
        step=float(grid.step),
        offset_index=int(grid.offset_index + lo),
        q_mass=q,
        p_mass=p,
    )


def _compose_discrete_pld_grids(
    grids: Sequence[DiscretePrivacyLossGrid],
    *,
    trim_tolerance: float,
) -> DiscretePrivacyLossGrid:
    grids = tuple(grids)
    if not grids:
        raise ValueError("At least one grid is required.")
    current = grids[0]
    for nxt in grids[1:]:
        q = _fft_convolve_nonnegative(current.q_mass, nxt.q_mass)
        p = _fft_convolve_nonnegative(current.p_mass, nxt.p_mass)
        current = DiscretePrivacyLossGrid(
            step=float(current.step),
            offset_index=int(current.offset_index + nxt.offset_index),
            q_mass=q,
            p_mass=p,
        )
        current = _trim_pld_support(current, trim_tolerance=float(trim_tolerance))
    return current


def _one_step_revealed_privacy_loss_nodes(
    *,
    sample_rate: float,
    c: float,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gamma = float(sample_rate)
    c = float(c)
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("sample_rate must lie in [0,1].")
    if not math.isfinite(c) or c < 0.0:
        raise ValueError("c must be finite and >= 0.")

    if gamma == 0.0 or c == 0.0:
        losses = np.asarray([0.0], dtype=np.float64)
        q_mass = np.asarray([1.0], dtype=np.float64)
        p_mass = np.asarray([1.0], dtype=np.float64)
        return losses, q_mass, p_mass

    z, w = _standard_normal_quadrature(int(quadrature_order))
    branch_losses = c * z - 0.5 * c * c
    q_mass = np.concatenate(
        [
            np.asarray([1.0 - gamma], dtype=np.float64),
            gamma * np.asarray(w, dtype=np.float64),
        ]
    )
    p_mass = np.concatenate(
        [
            np.asarray([1.0 - gamma], dtype=np.float64),
            gamma * np.exp(branch_losses) * np.asarray(w, dtype=np.float64),
        ]
    )
    losses = np.concatenate(
        [
            np.asarray([0.0], dtype=np.float64),
            np.asarray(branch_losses, dtype=np.float64),
        ]
    )
    q_mass = q_mass / float(np.sum(q_mass))
    p_mass = p_mass / float(np.sum(p_mass))
    return losses, q_mass, p_mass


def build_product_revealed_privacy_loss_grid(
    profiles: Sequence[DirectStepProfile],
    *,
    quadrature_order: int = 80,
    min_grid_step: float = 1e-3,
    target_bins: int = 16384,
    trim_tolerance: float = 1e-15,
) -> DiscretePrivacyLossGrid:
    profiles = tuple(profiles)
    if not profiles:
        raise ValueError("At least one profile is required.")

    node_triples = [
        _one_step_revealed_privacy_loss_nodes(
            sample_rate=profile.sample_rate,
            c=profile.c,
            quadrature_order=int(quadrature_order),
        )
        for profile in profiles
    ]
    grid_step = _adaptive_privacy_loss_grid_step_from_losses(
        [losses for losses, _, _ in node_triples],
        min_grid_step=float(min_grid_step),
        target_bins=int(target_bins),
    )
    grids = [
        _discretize_loss_nodes(
            losses=losses,
            q_weights=q_weights,
            p_weights=p_weights,
            grid_step=grid_step,
        )
        for losses, q_weights, p_weights in node_triples
    ]
    return _compose_discrete_pld_grids(grids, trim_tolerance=float(trim_tolerance))


def build_product_mixture_privacy_loss_grid(
    profiles: Sequence[LatentMixtureStepProfile],
    *,
    quadrature_order: int = 80,
    min_grid_step: float = 1e-3,
    target_bins: int = 16384,
    trim_tolerance: float = 1e-15,
) -> DiscretePrivacyLossGrid:
    profiles = tuple(profiles)
    if not profiles:
        raise ValueError("At least one profile is required.")

    node_triples = [
        _one_step_latent_mixture_privacy_loss_nodes(
            sample_rate=profile.sample_rate,
            c=profile.c,
            quadrature_order=int(quadrature_order),
        )
        for profile in profiles
    ]
    grid_step = _adaptive_privacy_loss_grid_step_from_losses(
        [losses for losses, _, _ in node_triples],
        min_grid_step=float(min_grid_step),
        target_bins=int(target_bins),
    )
    grids = [
        _discretize_loss_nodes(
            losses=losses,
            q_weights=q_weights,
            p_weights=p_weights,
            grid_step=grid_step,
        )
        for losses, q_weights, p_weights in node_triples
    ]
    return _compose_discrete_pld_grids(grids, trim_tolerance=float(trim_tolerance))


def compute_ball_sgd_global_mixture_rero_report(
    release: ReleaseArtifact,
    prior: PriorFamily,
    eta_grid: Sequence[float],
    *,
    mode_label: str = "ball_sgd_hayes",
    quadrature_order: int = 80,
    min_grid_step: float = 1e-3,
    target_bins: int = 16384,
    trim_tolerance: float = 1e-15,
) -> ReRoReport:
    r"""Compute the global revealed-inclusion direct Ball-ReRo quantity numerically.

    For the raw replacement-style Poisson Ball-SGD transcript, the theorem-backed
    dominating pair is the product of the one-step *revealed-inclusion* reference
    pairs. The theorem quantity is exact; the implementation below numerically
    evaluates that product/reference object by discretizing its privacy-loss
    distribution via Gauss--Hermite quadrature and FFT convolution.
    """
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

    ball_grid = None
    if len(ball_profiles) >= 2:
        ball_grid = build_product_revealed_privacy_loss_grid(
            ball_profiles,
            quadrature_order=int(quadrature_order),
            min_grid_step=float(min_grid_step),
            target_bins=int(target_bins),
            trim_tolerance=float(trim_tolerance),
        )

    standard_grid = None
    if standard_profiles is not None and len(standard_profiles) >= 2:
        standard_grid = build_product_revealed_privacy_loss_grid(
            standard_profiles,
            quadrature_order=int(quadrature_order),
            min_grid_step=float(min_grid_step),
            target_bins=int(target_bins),
            trim_tolerance=float(trim_tolerance),
        )

    points: list[ReRoPoint] = []
    for eta in eta_grid:
        eta_f = float(eta)
        kappa = float(prior.kappa(eta_f))

        if ball_grid is None:
            gamma_ball = ball_profiles[0].evaluate(kappa)
        else:
            gamma_ball = ball_grid.blowup(kappa)

        gamma_standard = None
        if standard_profiles is not None:
            if standard_grid is None:
                gamma_standard = standard_profiles[0].evaluate(kappa)
            else:
                gamma_standard = standard_grid.blowup(kappa)

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
        **_common_direct_metadata(release, cfg),
        "num_steps": int(len(ball_profiles)),
        "profile_family": "global_product_revealed_inclusion",
        "theorem_backed_for_raw_transcript": True,
        "representation": (
            "tradeoff" if str(mode_label) == "ball_sgd_kaissis" else "blowup"
        ),
        "final_release_view": "postprocessing_of_revealed_inclusion_reference_transcript",
        "numerical_method": (
            "exact_closed_form_when_num_steps=1; otherwise discretized_privacy_loss_"
            "composition_via_gauss_hermite_quadrature_and_fft_convolution"
        ),
        "ball_step_profiles": [profile.as_dict() for profile in ball_profiles],
        "standard_step_profiles": (
            None
            if standard_profiles is None
            else [profile.as_dict() for profile in standard_profiles]
        ),
        "ball_privacy_loss_grid": (
            None if ball_grid is None else ball_grid.as_summary()
        ),
        "standard_privacy_loss_grid": (
            None if standard_grid is None else standard_grid.as_summary()
        ),
        "theorem_note": (
            "This global mode evaluates the exact product/reference quantity for the "
            "revealed-inclusion dominating pair. It is theorem-backed for the raw "
            "replacement-style transcript, unlike hidden-mixture references that "
            "would require an additional domination argument."
        ),
        "standard_comparator_note": (
            "gamma_standard uses the same global revealed-inclusion product/reference "
            "quantity with the clipping-only sensitivity schedule step_delta_std = 2 C_t "
            "when that schedule is available in the release artifact."
        ),
    }
    return ReRoReport(
        mode=str(mode_label),
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
            "latent_mix_c": None,
        }
    if not math.isfinite(float(noise_std)) or float(noise_std) <= 0.0:
        return {
            "direct_c": None,
            "direct_tau": None,
            "direct_v": None,
            "direct_kappa_left": None,
            "direct_kappa_right": None,
            "latent_mix_c": None,
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
        "latent_mix_c": float(profile.c),
    }
