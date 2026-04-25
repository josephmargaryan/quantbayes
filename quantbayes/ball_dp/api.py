# quantbayes/ball_dp/api.py

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Dict, Literal
import dataclasses as dc

import math
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from .config import (
    BallSGDConfig,
    ConvexOptimizationConfig,
    ConvexReleaseConfig,
    GaussianCalibrationConfig,
)
from .accountants.subsampled_gaussian import (
    account_ball_sgd_noise_multiplier as _account_ball_sgd_noise_multiplier,
    calibrate_ball_sgd_noise_multiplier as _calibrate_ball_sgd_noise_multiplier,
)
from .metrics import accuracy_from_logits, feature_reconstruction_metrics
from .convex.attacks import run_convex_attack
from .convex.releases import (
    run_convex_ball_erm_dp,
    run_convex_ball_erm_rdp,
    run_convex_noiseless_erm_release,
)
from .evaluation.rero import (
    EmpiricalBallPrior,
    FiniteExactIdentificationPrior,
    UniformL2BallPrior,
    compute_ball_rero_report,
    summarize_attack_trials as _summarize_attack_trials,
)
from .evaluation.direct_poisson import direct_profile_step_summary
from .nonconvex.per_example import ExampleLossFn, PredictFn, default_predict_fn
from .nonconvex.sgd import (
    run_ball_sgd_dp,
    run_ball_sgd_rdp,
    run_noiseless_sgd_release,
    run_standard_sgd_dp,
    run_standard_sgd_rdp,
)
from .types import (
    ArrayDataset,
    AttackResult,
    Record,
    ReRoReport,
    ReleaseArtifact,
)
from .informed import (
    InformedAttackData,
    prepare_informed_attack_data,
    train_released_model,
    build_attack_corpus,
    train_reconstructor,
)
from .attacks.model_based import run_model_based_attack
from .trace_setup import (
    TargetedTraceBatch,
    prepare_targeted_trace_batch,
    make_target_inclusion_schedule,
)
from .attacks.ball_policy import (
    BallTraceMapAttackConfig,
    run_ball_trace_finite_prior_attack,
    run_ball_trace_map_attack,
)
from .attacks.gradient_based import (
    DPSGDTrace,
    DPSGDTraceRecorder,
    TraceOptimizationAttackConfig,
    run_prior_aware_trace_attack,
    run_trace_optimization_attack,
    subtract_known_batch_gradients,
)
from .attacks.spear import (
    SpearAttackConfig,
    SpearAttackResult,
    default_noisy_spear_config,
    run_spear_batch_attack,
    run_spear_model_batch_attack,
    run_spear_trace_step_attack,
)
from .attacks.ball_priors import (
    TruncatedGaussianBallAttackPrior,
    UniformBallAttackPrior,
)
from .attacks.ball_policy import (
    BallTraceMapAttackConfig,
    run_ball_trace_map_attack,
)
from .convex.ball_output_attacks import (
    BallOutputMapAttackConfig,
    run_convex_ball_output_finite_prior_attack,
    run_convex_ball_output_map_attack,
)
from .convex.finite_prior_diagnostics import (
    compute_convex_finite_prior_diagnostics,
)

from .plots import plot_attack_result, plot_batch_reconstruction_grid
from .radius_selection import (
    summarize_embedding_ball_radii as _summarize_embedding_ball_radii,
    select_ball_radius as _select_ball_radius,
)

__all__ = [
    "fit_convex",
    "attack_convex",
    "make_uniform_ball_prior",
    "make_finite_identification_prior",
    "summarize_embedding_ball_radii",
    "select_ball_radius",
    "ball_rero",
    "fit_ball_sgd",
    "make_trace_metadata_from_release",
    "summarize_attack_trials",
    "InformedAttackData",
    "prepare_informed_attack_data",
    "train_released_model",
    "build_attack_corpus",
    "train_reconstructor",
    "run_model_based_attack",
    "attack_nonconvex_prior_aware_trace",
    "make_uniform_ball_attack_prior",
    "make_truncated_gaussian_ball_attack_prior",
    "attack_convex_ball_output",
    "attack_convex_ball_output_finite_prior",
    "attack_nonconvex_ball_trace_finite_prior",
    "BallTraceMapAttackConfig",
    "BallOutputMapAttackConfig",
    "diagnose_convex_ball_output_finite_prior",
]

ConvexModelFamily = str
BallPrivacyKind = str


def _as_dataset(X: np.ndarray, y: np.ndarray, *, name: str) -> ArrayDataset:
    return ArrayDataset(np.asarray(X), np.asarray(y), name=name)


def _normalize_convex_model_family(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "softmax_logreg": "softmax_logistic",
        "softmax_logistic": "softmax_logistic",
        "multiclass_logreg": "softmax_logistic",
        "multiclass_logistic": "softmax_logistic",
        "binary_logreg": "binary_logistic",
        "binary_logistic": "binary_logistic",
        "logistic": "binary_logistic",
        "ridge_prototype": "ridge_prototype",
        "ridgeprototype": "ridge_prototype",
        "ridge_prototypes": "ridge_prototype",
        "svm": "squared_hinge",
        "squared_hinge": "squared_hinge",
        "linear_svm": "squared_hinge",
    }
    if key not in aliases:
        raise ValueError(
            "Unsupported convex model_family. Supported aliases map to one of "
            "{'softmax_logistic', 'binary_logistic', 'squared_hinge', 'ridge_prototype'}."
        )
    return aliases[key]


def _normalize_privacy_kind(kind: str) -> str:
    key = str(kind).strip().lower().replace("-", "_")
    aliases = {
        "ball_dp": "ball_dp",
        "dp": "ball_dp",
        "ball_rdp": "ball_rdp",
        "rdp": "ball_rdp",
        "standard_dp": "standard_dp",
        "standard_rdp": "standard_rdp",
        "noiseless": "noiseless",
        "none": "noiseless",
    }
    if key not in aliases:
        raise ValueError(
            "Unsupported privacy kind. Supported values include "
            "'ball_dp', 'ball_rdp', 'standard_dp', 'standard_rdp', 'noiseless'."
        )
    return aliases[key]


def fit_convex(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_family: ConvexModelFamily,
    radius: float,
    lam: float,
    X_eval: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
    privacy: str = "ball_dp",
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    sigma: Optional[float] = None,
    embedding_bound: Optional[float] = None,
    num_classes: Optional[int] = None,
    lz: Optional[float] = None,
    standard_radius: Optional[float] = None,
    ridge_sensitivity_mode: Literal["global", "count_aware"] = "global",
    gaussian_method: str = "analytic",
    gaussian_tol: float = 1e-12,
    orders: Sequence[float] = (2, 3, 4, 5, 8, 16, 32, 64, 128),
    dp_deltas_for_rdp: Sequence[float] = (1e-6,),
    solver: str = "lbfgs_fullbatch",
    max_iter: int = 500,
    learning_rate: float = 1e-1,
    grad_tol: Optional[float] = 1e-8,
    param_tol: Optional[float] = 1e-10,
    objective_tol: Optional[float] = 1e-12,
    early_stop: bool = True,
    stop_rule: Literal["any", "all", "grad_only"] = "any",
    min_iter: int = 0,
    line_search_steps: int = 15,
    seed: int = 0,
) -> ReleaseArtifact:
    fam = _normalize_convex_model_family(model_family)
    privacy_kind = _normalize_privacy_kind(privacy)

    if solver not in {"lbfgs_fullbatch", "gd_fullbatch"}:
        raise ValueError(
            "fit_convex exposes only {'lbfgs_fullbatch', 'gd_fullbatch'} in the "
            "stable public API."
        )

    if stop_rule not in {"any", "all", "grad_only"}:
        raise ValueError("stop_rule must be one of {'any', 'all', 'grad_only'}.")

    train_ds = _as_dataset(X_train, y_train, name="train")
    eval_ds = None
    if X_eval is not None and y_eval is not None:
        eval_ds = _as_dataset(X_eval, y_eval, name="eval")

    opt_cfg = ConvexOptimizationConfig(
        solver=solver,
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        grad_tol=None if grad_tol is None else float(grad_tol),
        param_tol=None if param_tol is None else float(param_tol),
        objective_tol=None if objective_tol is None else float(objective_tol),
        early_stop=bool(early_stop),
        stop_rule=str(stop_rule),
        min_iter=int(min_iter),
        certify_approximation=True,
        approximation_mode="optimality_residual",
        theorem_backed_only=True,
        line_search_steps=int(line_search_steps),
    )
    gauss_cfg = GaussianCalibrationConfig(
        method=str(gaussian_method),
        tol=float(gaussian_tol),
    )

    cfg = ConvexReleaseConfig(
        model_family=fam,
        radius=float(radius),
        lam=float(lam),
        gaussian=gauss_cfg,
        optimization=opt_cfg,
        epsilon=None if epsilon is None else float(epsilon),
        delta=None if delta is None else float(delta),
        sigma=None if sigma is None else float(sigma),
        orders=tuple(float(v) for v in orders),
        dp_deltas_for_rdp=tuple(float(v) for v in dp_deltas_for_rdp),
        embedding_bound=None if embedding_bound is None else float(embedding_bound),
        standard_radius=None if standard_radius is None else float(standard_radius),
        num_classes=None if num_classes is None else int(num_classes),
        lz_mode="provided" if lz is not None else "glm_bound",
        provided_lz=None if lz is None else float(lz),
        use_exact_sensitivity_if_available=True,
        ridge_sensitivity_mode=str(ridge_sensitivity_mode),
        seed=int(seed),
        store_nonprivate_reference=False,
    )

    if privacy_kind == "ball_dp":
        return run_convex_ball_erm_dp(train_ds, cfg, eval_dataset=eval_ds)
    if privacy_kind == "ball_rdp":
        return run_convex_ball_erm_rdp(train_ds, cfg, eval_dataset=eval_ds)
    if privacy_kind == "noiseless":
        return run_convex_noiseless_erm_release(train_ds, cfg, eval_dataset=eval_ds)

    raise ValueError(
        "The convex public API supports 'ball_dp', 'ball_rdp', and 'noiseless'. "
        "For the usual replacement-adjacent comparator, set radius=2*embedding_bound "
        "when that is the intended threat model."
    )


def attack_convex(
    release: ReleaseArtifact,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    target_index: int,
    known_label: Optional[int] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
) -> tuple[AttackResult, ArrayDataset, Any]:
    ds = _as_dataset(X_train, y_train, name="train")
    d_minus, target = ds.remove_index(int(target_index))
    side_info = None if known_label is None else {"known_label": int(known_label)}
    attack = run_convex_attack(
        release,
        d_minus,
        side_info=side_info,
        true_record=target,
        eta_grid=tuple(float(v) for v in eta_grid),
    )
    return attack, d_minus, target


def attack_convex_ball_output(
    release: ReleaseArtifact,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    target_index: int,
    prior,
    cfg: Optional[BallOutputMapAttackConfig] = None,
    known_label: Optional[int] = None,
    label_space: Optional[Sequence[int]] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
    feature_shape: Optional[Sequence[int]] = None,
    out_path: Optional[str] = None,
) -> tuple[AttackResult, ArrayDataset, Record]:
    """Targeted Ball-constrained MAP attack for Gaussian output perturbation."""
    ds = _as_dataset(X_train, y_train, name="train")
    d_minus, target = ds.remove_index(int(target_index))

    attack = run_convex_ball_output_map_attack(
        release,
        d_minus,
        prior=prior,
        cfg=cfg,
        known_label=known_label,
        label_space=label_space,
        true_record=target,
        eta_grid=tuple(float(v) for v in eta_grid),
    )
    attack.diagnostics["target_index"] = int(target_index)

    if out_path is not None and attack.z_hat is not None:
        plot_attack_result(
            attack,
            target,
            feature_shape=feature_shape,
            out_path=out_path,
        )

    return attack, d_minus, target


def make_uniform_ball_prior(
    *, center: np.ndarray, radius: float, dimension: Optional[int] = None
) -> UniformL2BallPrior:
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    dim = center.size if dimension is None else int(dimension)
    return UniformL2BallPrior(center=center, radius=float(radius), dimension=dim)


def make_uniform_ball_attack_prior(
    *,
    center: np.ndarray,
    radius: float,
) -> UniformBallAttackPrior:
    return UniformBallAttackPrior(
        center=np.asarray(center, dtype=np.float32),
        radius=float(radius),
    )


def make_truncated_gaussian_ball_attack_prior(
    *,
    center: np.ndarray,
    radius: float,
    sigma: float,
) -> TruncatedGaussianBallAttackPrior:
    return TruncatedGaussianBallAttackPrior(
        center=np.asarray(center, dtype=np.float32),
        radius=float(radius),
        sigma=float(sigma),
    )


def make_empirical_ball_prior(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    label: Optional[int] = None,
    max_samples: Optional[int] = None,
    safe_kappa_mode: str = "raise",
) -> EmpiricalBallPrior:
    X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    if y is not None and label is not None:
        y = np.asarray(y, dtype=np.int64)
        X = X[y == int(label)]
    if max_samples is not None and len(X) > int(max_samples):
        X = X[: int(max_samples)]
    return EmpiricalBallPrior(X, safe_kappa_mode=safe_kappa_mode)


def ball_rero(
    release: ReleaseArtifact,
    *,
    prior,
    eta_grid: Sequence[float],
    mode: str = "auto",
    out_path: Optional[str] = None,
) -> ReRoReport:
    return compute_ball_rero_report(
        release,
        prior,
        eta_grid=tuple(float(v) for v in eta_grid),
        mode=str(mode),
        out_path=out_path,
    )


def get_public_curve_history(release: ReleaseArtifact) -> list[dict[str, Any]]:
    history = release.extra.get("public_curve_history", [])
    return [dict(row) for row in history]


def get_operator_norm_history(release: ReleaseArtifact) -> list[dict[str, Any]]:
    history = release.extra.get("operator_norm_history", [])
    return [dict(row) for row in history]


def get_release_step_table(release: ReleaseArtifact) -> list[dict[str, Any]]:
    cfg = dict(release.training_config)
    sens = release.sensitivity
    extra = dict(release.extra)

    batch_sizes = list(cfg.get("batch_sizes", ()))
    sample_rates = list(cfg.get("sample_rates", ()))
    clip_norms = list(cfg.get("clip_norms", ()))
    noise_multipliers = list(cfg.get("noise_multipliers", ()))
    effective_noise_stds = list(cfg.get("effective_noise_stds", ()))
    step_delta_ball = [] if sens.step_delta_ball is None else list(sens.step_delta_ball)
    step_delta_std = [] if sens.step_delta_std is None else list(sens.step_delta_std)
    ratios = list(extra.get("ball_to_standard_sensitivity_ratio_by_step", []) or [])
    rho_by_step = list(extra.get("rho_by_step", []) or [])

    T = max(
        len(batch_sizes),
        len(sample_rates),
        len(clip_norms),
        len(noise_multipliers),
        len(effective_noise_stds),
        len(step_delta_ball),
        len(step_delta_std),
        len(ratios),
        len(rho_by_step),
    )

    rows: list[dict[str, Any]] = []
    for t in range(T):
        batch_size_t = None if t >= len(batch_sizes) else int(batch_sizes[t])
        effective_noise_std_t = (
            None if t >= len(effective_noise_stds) else float(effective_noise_stds[t])
        )
        normalize_mode = str(cfg.get("normalize_noisy_sum_by", "batch_size"))
        if normalize_mode == "none":
            stored_private_object = "noisy_sum"
            noise_std_on_stored_object = effective_noise_std_t
        elif normalize_mode == "batch_size":
            stored_private_object = "noisy_mean"
            noise_std_on_stored_object = (
                None
                if effective_noise_std_t is None or batch_size_t in {None, 0}
                else float(effective_noise_std_t) / float(batch_size_t)
            )
        else:
            stored_private_object = "noisy_mean"
            noise_std_on_stored_object = None

        sample_rate_t = None if t >= len(sample_rates) else float(sample_rates[t])
        delta_ball_t = None if t >= len(step_delta_ball) else float(step_delta_ball[t])
        delta_standard_t = (
            None if t >= len(step_delta_std) else float(step_delta_std[t])
        )

        if sample_rate_t is None:
            direct_ball = {
                "direct_c": None,
                "direct_tau": None,
                "direct_v": None,
                "direct_kappa_left": None,
                "direct_kappa_right": None,
            }
            direct_standard = dict(direct_ball)
        else:
            direct_ball = direct_profile_step_summary(
                sample_rate=float(sample_rate_t),
                sensitivity=delta_ball_t,
                noise_std=effective_noise_std_t,
            )
            direct_standard = direct_profile_step_summary(
                sample_rate=float(sample_rate_t),
                sensitivity=delta_standard_t,
                noise_std=effective_noise_std_t,
            )

        row = {
            "step": int(t + 1),
            "batch_sampler": cfg.get(
                "resolved_batch_sampler", cfg.get("batch_sampler")
            ),
            "accountant_subsampling": cfg.get(
                "resolved_accountant_subsampling",
                cfg.get("accountant_subsampling"),
            ),
            "batch_size": batch_size_t,
            "target_batch_size": batch_size_t,
            "sample_rate": sample_rate_t,
            "clip_norm": None if t >= len(clip_norms) else float(clip_norms[t]),
            "noise_multiplier": (
                None if t >= len(noise_multipliers) else float(noise_multipliers[t])
            ),
            "effective_noise_std": effective_noise_std_t,
            "stored_private_object": stored_private_object,
            "noise_std_on_stored_object": noise_std_on_stored_object,
            "delta_ball": delta_ball_t,
            "delta_standard": delta_standard_t,
            "ball_to_standard_ratio": None if t >= len(ratios) else ratios[t],
            "rho": None if t >= len(rho_by_step) else rho_by_step[t],
            "direct_c_ball": direct_ball["direct_c"],
            "direct_tau_ball": direct_ball["direct_tau"],
            "direct_kappa_left_ball": direct_ball["direct_kappa_left"],
            "direct_kappa_right_ball": direct_ball["direct_kappa_right"],
            "direct_c_standard": direct_standard["direct_c"],
            "direct_tau_standard": direct_standard["direct_tau"],
            "direct_kappa_left_standard": direct_standard["direct_kappa_left"],
            "direct_kappa_right_standard": direct_standard["direct_kappa_right"],
        }
        rows.append(row)
    return rows


def account_ball_sgd_noise_multiplier(
    *,
    dataset_size: int,
    radius: float,
    lz: Optional[float],
    num_steps: int,
    batch_size: int | Sequence[int] = 128,
    clip_norm: float | Sequence[float] = 1.0,
    noise_multiplier: float = 1.0,
    delta: float = 1e-5,
    privacy: BallPrivacyKind = "ball_rdp",
    batch_sampler: str = "shuffle",
    accountant_subsampling: str = "auto",
    orders: Sequence[int] = (2, 3, 4, 5, 8, 16, 32, 64, 128),
) -> dict[str, Any]:
    """Account privacy for a scalar DP-SGD noise multiplier without training."""
    privacy_kind = _normalize_privacy_kind(privacy)
    if privacy_kind not in {"ball_rdp", "ball_dp", "standard_rdp", "standard_dp"}:
        raise ValueError(
            "account_ball_sgd_noise_multiplier supports only "
            "{'ball_rdp', 'ball_dp', 'standard_rdp', 'standard_dp'}."
        )

    accounting_view = "ball" if privacy_kind.startswith("ball") else "standard"
    return _account_ball_sgd_noise_multiplier(
        noise_multiplier=float(noise_multiplier),
        accounting_view=accounting_view,
        orders=tuple(int(v) for v in orders),
        dataset_size=int(dataset_size),
        num_steps=int(num_steps),
        batch_size=batch_size,
        clip_norm=clip_norm,
        radius=float(radius),
        lz=None if lz is None else float(lz),
        dp_delta=float(delta),
        batch_sampler=str(batch_sampler),
        accountant_subsampling=str(accountant_subsampling),
    )


def calibrate_ball_sgd_noise_multiplier(
    *,
    dataset_size: int,
    radius: float,
    lz: Optional[float],
    num_steps: int,
    batch_size: int | Sequence[int] = 128,
    clip_norm: float | Sequence[float] = 1.0,
    target_epsilon: float,
    delta: float,
    privacy: BallPrivacyKind = "ball_rdp",
    batch_sampler: str = "shuffle",
    accountant_subsampling: str = "auto",
    orders: Sequence[int] = (2, 3, 4, 5, 8, 16, 32, 64, 128),
    lower: float = 1e-3,
    upper: float = 0.25,
    max_upper: float = 128.0,
    num_bisection_steps: int = 10,
) -> dict[str, Any]:
    """Calibrate the minimum scalar DP-SGD noise multiplier using only the accountant."""
    privacy_kind = _normalize_privacy_kind(privacy)
    if privacy_kind not in {"ball_rdp", "ball_dp", "standard_rdp", "standard_dp"}:
        raise ValueError(
            "calibrate_ball_sgd_noise_multiplier supports only "
            "{'ball_rdp', 'ball_dp', 'standard_rdp', 'standard_dp'}."
        )

    accounting_view = "ball" if privacy_kind.startswith("ball") else "standard"
    return _calibrate_ball_sgd_noise_multiplier(
        target_epsilon=float(target_epsilon),
        accounting_view=accounting_view,
        orders=tuple(int(v) for v in orders),
        dataset_size=int(dataset_size),
        num_steps=int(num_steps),
        batch_size=batch_size,
        clip_norm=clip_norm,
        radius=float(radius),
        lz=None if lz is None else float(lz),
        dp_delta=float(delta),
        batch_sampler=str(batch_sampler),
        accountant_subsampling=str(accountant_subsampling),
        lower=float(lower),
        upper=float(upper),
        max_upper=float(max_upper),
        num_bisection_steps=int(num_bisection_steps),
    )


def extract_privacy_epsilon(
    release: ReleaseArtifact,
    *,
    accounting_view: str = "primary",
) -> float:
    view = str(accounting_view).lower()
    if view == "primary":
        view = str(release.attack_metadata.get("primary_privacy_view", "ball")).lower()

    if view not in {"ball", "standard"}:
        raise ValueError(
            "accounting_view must be one of {'primary', 'ball', 'standard'}."
        )

    ledger = release.privacy.ball if view == "ball" else release.privacy.standard
    if not ledger.dp_certificates:
        raise ValueError(
            f"Release does not carry a DP certificate for accounting_view={view!r}."
        )
    return float(ledger.dp_certificates[0].epsilon)


def calibrate_privacy_parameter(
    make_release: Callable[[float], ReleaseArtifact],
    *,
    target_epsilon: float,
    accounting_view: str = "primary",
    lower: float = 1e-3,
    upper: float = 0.25,
    max_upper: float = 128.0,
    num_bisection_steps: int = 10,
) -> tuple[float, ReleaseArtifact]:
    """Calibrate a monotone scalar parameter (typically a noise multiplier) so that
    the resulting release satisfies epsilon <= target_epsilon.

    Assumes the parameter is privacy-improving when increased, e.g. larger Gaussian
    noise multiplier => smaller epsilon.
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
        release_hi = make_release(hi)
        eps_hi = extract_privacy_epsilon(release_hi, accounting_view=accounting_view)
        if eps_hi <= target_epsilon:
            best_release = release_hi
            break
        lo = hi
        hi *= 2.0
        if hi > max_upper:
            raise RuntimeError(
                "Failed to bracket a privacy-feasible parameter value. "
                f"Last tried upper={hi:.6g}, target_epsilon={target_epsilon:.6g}."
            )

    for _ in range(num_bisection_steps):
        mid = 0.5 * (lo + hi)
        release_mid = make_release(mid)
        eps_mid = extract_privacy_epsilon(release_mid, accounting_view=accounting_view)
        if eps_mid <= target_epsilon:
            hi = mid
            best_release = release_mid
        else:
            lo = mid

    return float(hi), best_release


def evaluate_release_classifier(
    release: ReleaseArtifact,
    X: np.ndarray,
    y: np.ndarray,
    *,
    key: Optional[jax.Array] = None,
    batch_size: int = 1024,
    state: Any = None,
) -> Dict[str, float]:
    """Evaluate a nonconvex release carrying an Equinox classifier model."""
    if key is None:
        key = jr.PRNGKey(0)

    model = release.payload
    if not callable(model):
        raise ValueError(
            "evaluate_release_classifier expects release.payload to be a callable "
            "model, as produced by the nonconvex Ball-SGD / standard DP-SGD path."
        )

    model_eval = eqx.nn.inference_mode(model, value=True)
    if state is None:
        state = release.extra.get("model_state", None)

    Xj = jnp.asarray(X, dtype=jnp.float32)
    y_np = np.asarray(y)

    n = int(Xj.shape[0])
    if n == 0:
        return {"accuracy": float("nan"), "n_eval": 0.0}

    num_batches = math.ceil(n / int(batch_size))
    keys = jr.split(key, num_batches)
    logits_chunks = []

    for i in range(num_batches):
        lo = i * int(batch_size)
        hi = min((i + 1) * int(batch_size), n)
        xb = Xj[lo:hi]
        batch_keys = jr.split(keys[i], xb.shape[0])

        logits = jax.vmap(
            lambda x_one, k_one: model_eval(x_one, key=k_one, state=state)[0],
            in_axes=(0, 0),
        )(xb, batch_keys)
        logits_chunks.append(np.asarray(logits))

    logits_all = np.concatenate(logits_chunks, axis=0)

    y_eval = np.asarray(y_np, dtype=np.int64)
    if logits_all.ndim == 1 or (logits_all.ndim > 1 and logits_all.shape[-1] == 1):
        uniq = set(np.unique(y_eval).tolist())
        if uniq.issubset({-1, 1}):
            y_eval = (y_eval > 0).astype(np.int64)

    acc = accuracy_from_logits(logits_all, y_eval)
    return {
        "accuracy": float(acc),
        "n_eval": float(n),
    }


def fit_ball_sgd(
    model: Any,
    optimizer: optax.GradientTransformation,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    radius: float = 1.0,
    lz: Optional[float] = None,
    X_eval: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
    privacy: BallPrivacyKind = "ball_rdp",
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    num_steps: int = 1000,
    batch_size: int | Sequence[int] = 128,
    batch_sampler: str = "shuffle",
    accountant_subsampling: str = "auto",
    clip_norm: float | Sequence[float] = 1.0,
    noise_multiplier: float | Sequence[float] = 1.0,
    orders: Sequence[int] = (2, 3, 4, 5, 8, 16, 32, 64, 128),
    loss_name: str = "softmax_cross_entropy",
    state: Any = None,
    loss_fn: Optional[ExampleLossFn] = None,
    predict_fn: Optional[PredictFn] = default_predict_fn,
    parameter_regularizer: Optional[Callable[[Any, Any], jax.Array]] = None,
    normalize_noisy_sum_by: str = "batch_size",
    frobenius_reg_strength: float = 0.0,
    spectral_reg_strength: float = 0.0,
    spectral_reg_kwargs: Optional[dict[str, Any]] = None,
    record_operator_norms: bool = False,
    operator_norms_every: int = 250,
    operator_norm_kwargs: Optional[dict[str, Any]] = None,
    checkpoint_selection: str = "last",
    eval_every: int = 250,
    eval_batch_size: int = 1024,
    warn_if_ball_equals_standard: bool = True,
    fixed_batch_indices_schedule: Optional[Sequence[Optional[Sequence[int]]]] = None,
    seed: int = 0,
    key: Optional[jax.Array] = None,
    return_debug_history: bool = False,
    trace_recorder=None,
    param_projector: Optional[Callable[[Any], Any]] = None,
):
    privacy_kind = _normalize_privacy_kind(privacy)

    if key is None:
        key = jr.PRNGKey(int(seed))

    train_ds = _as_dataset(X_train, y_train, name="train")
    eval_ds = None
    if X_eval is not None and y_eval is not None:
        eval_ds = _as_dataset(X_eval, y_eval, name="eval")

    cfg = BallSGDConfig(
        radius=float(radius),
        lz=None if lz is None else float(lz),
        num_steps=int(num_steps),
        batch_sizes=(
            batch_size
            if isinstance(batch_size, int)
            else tuple(int(v) for v in batch_size)
        ),
        batch_sampler=str(batch_sampler),
        accountant_subsampling=str(accountant_subsampling),
        clip_norms=(
            clip_norm
            if isinstance(clip_norm, (int, float))
            else tuple(float(v) for v in clip_norm)
        ),
        noise_multipliers=(
            noise_multiplier
            if isinstance(noise_multiplier, (int, float))
            else tuple(float(v) for v in noise_multiplier)
        ),
        orders=tuple(int(v) for v in orders),
        epsilon=None if epsilon is None else float(epsilon),
        delta=None if delta is None else float(delta),
        loss_name=str(loss_name),
        normalize_noisy_sum_by=str(normalize_noisy_sum_by),
        fixed_batch_indices_schedule=(
            None
            if fixed_batch_indices_schedule is None
            else tuple(
                None if step_idx is None else tuple(int(v) for v in step_idx)
                for step_idx in fixed_batch_indices_schedule
            )
        ),
        frobenius_reg_strength=float(frobenius_reg_strength),
        spectral_reg_strength=float(spectral_reg_strength),
        spectral_reg_kwargs=dict(spectral_reg_kwargs or {}),
        eval_every=int(eval_every),
        eval_batch_size=int(eval_batch_size),
        checkpoint_selection=str(checkpoint_selection),
        record_operator_norms=bool(record_operator_norms),
        operator_norms_every=int(operator_norms_every),
        operator_norm_kwargs=dict(operator_norm_kwargs or {}),
        warn_if_ball_equals_standard=bool(warn_if_ball_equals_standard),
        seed=int(seed),
    )

    kwargs = dict(
        dataset=train_ds,
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        state=state,
        public_eval_dataset=eval_ds,
        loss_fn=loss_fn,
        predict_fn=predict_fn,
        parameter_regularizer=parameter_regularizer,
        key=key,
        return_debug_history=bool(return_debug_history),
        trace_recorder=trace_recorder,
        param_projector=param_projector,
    )

    if privacy_kind == "ball_dp":
        return run_ball_sgd_dp(**kwargs)
    if privacy_kind == "ball_rdp":
        return run_ball_sgd_rdp(**kwargs)
    if privacy_kind == "standard_dp":
        return run_standard_sgd_dp(**kwargs)
    if privacy_kind == "standard_rdp":
        return run_standard_sgd_rdp(**kwargs)
    if privacy_kind == "noiseless":
        return run_noiseless_sgd_release(**kwargs)

    raise ValueError(f"Unsupported privacy kind: {privacy!r}")


def summarize_embedding_ball_radii(
    X: np.ndarray,
    y: np.ndarray,
    *,
    quantiles: Sequence[float] = (0.5, 0.8, 0.9, 0.95, 0.99, 1.0),
    max_exact_pairs: int = 250_000,
    max_sampled_pairs: int = 100_000,
    seed: int = 0,
) -> dict[str, Any]:
    return _summarize_embedding_ball_radii(
        X,
        y,
        quantiles=quantiles,
        max_exact_pairs=int(max_exact_pairs),
        max_sampled_pairs=int(max_sampled_pairs),
        seed=int(seed),
    )


def select_ball_radius(
    report: dict[str, Any],
    *,
    strategy: str = "max_labelwise_quantile",
    quantile: float = 0.95,
    allow_observed_max: bool = False,
) -> float:
    return _select_ball_radius(
        report,
        strategy=str(strategy),
        quantile=float(quantile),
        allow_observed_max=bool(allow_observed_max),
    )


def make_finite_identification_prior(
    X: np.ndarray,
    *,
    weights: Optional[Sequence[float]] = None,
) -> FiniteExactIdentificationPrior:
    X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    return FiniteExactIdentificationPrior(X, weights=weights)


def diagnose_convex_ball_output_finite_prior(
    release: ReleaseArtifact,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    target_index: int,
    X_candidates: np.ndarray,
    y_candidates: np.ndarray,
    prior_weights: Optional[Sequence[float]] = None,
    known_label: Optional[int] = None,
    center_features: Optional[np.ndarray] = None,
    center_label: Optional[int] = None,
) -> dict[str, float | int | str | bool | None]:
    """Compute theorem-aligned diagnostics for a finite-prior Gaussian attack.

    The finite-prior MAP attack is already Bayes-optimal for the supplied support.
    These diagnostics explain how hard that support is: they report model-space
    candidate separation, the direct finite-Gaussian upper bound using the actual
    candidate means, and ridge-specific inversion noise quantities when available.
    """
    ds = _as_dataset(X_train, y_train, name="train")
    d_minus, _ = ds.remove_index(int(target_index))

    Xc = np.asarray(X_candidates)
    yc = np.asarray(y_candidates)
    if len(Xc) != len(yc):
        raise ValueError("X_candidates and y_candidates must have the same length.")
    prior_records = [
        Record(features=np.asarray(Xc[i]), label=int(yc[i]))
        for i in range(int(len(Xc)))
    ]

    center_record = None
    if center_features is not None:
        if center_label is None:
            if known_label is None:
                raise ValueError(
                    "center_label or known_label is required when center_features is provided."
                )
            center_label = int(known_label)
        center_record = Record(
            features=np.asarray(center_features), label=int(center_label)
        )

    return compute_convex_finite_prior_diagnostics(
        release,
        d_minus,
        prior_records=prior_records,
        prior_weights=prior_weights,
        known_label=known_label,
        center_record=center_record,
    )


def make_trace_metadata_from_release(
    release: ReleaseArtifact,
    *,
    target_index: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build theorem-aligned trace metadata from a release artifact.

    This helper packages the run-level information needed by the exact Ball-trace
    attacks, including the per-step Poisson probabilities when they are present in
    the release configuration.
    """
    cfg = dict(release.training_config)
    meta: dict[str, Any] = {
        "dataset_size": int(release.dataset_metadata.get("n_total", 0)),
        "feature_shape": tuple(release.dataset_metadata.get("feature_shape", ())),
        "sample_rates": tuple(float(v) for v in cfg.get("sample_rates", ())),
        "batch_sampler": cfg.get(
            "resolved_batch_sampler",
            cfg.get("batch_sampler", "unknown"),
        ),
        "normalize_noisy_sum_by": cfg.get("normalize_noisy_sum_by", "batch_size"),
        "reduction": (
            "sum"
            if str(cfg.get("normalize_noisy_sum_by", "batch_size")) == "none"
            else "mean"
        ),
    }
    sample_rates = tuple(float(v) for v in cfg.get("sample_rates", ()))
    if sample_rates:
        if len(set(sample_rates)) == 1:
            meta["sample_rate"] = float(sample_rates[0])
    if target_index is not None:
        meta["target_index"] = int(target_index)
    if extra is not None:
        meta.update(dict(extra))
    return meta


def summarize_attack_trials(
    attack_results: Sequence[AttackResult],
    *,
    eta_grid: Optional[Sequence[float]] = None,
    oblivious_kappa: Optional[float] = None,
) -> dict[str, float]:
    return _summarize_attack_trials(
        attack_results,
        eta_grid=eta_grid,
        oblivious_kappa=oblivious_kappa,
    )


def attack_convex_ball_output_finite_prior(
    release: ReleaseArtifact,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    target_index: int,
    X_candidates: np.ndarray,
    y_candidates: np.ndarray,
    prior_weights: Optional[Sequence[float]] = None,
    known_label: Optional[int] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
    out_path: Optional[str] = None,
) -> tuple[AttackResult, ArrayDataset, Record]:
    ds = _as_dataset(X_train, y_train, name="train")
    d_minus, target = ds.remove_index(int(target_index))

    Xc = np.asarray(X_candidates)
    yc = np.asarray(y_candidates)
    if len(Xc) != len(yc):
        raise ValueError("X_candidates and y_candidates must have the same length.")
    prior_records = [
        Record(features=np.asarray(Xc[i]), label=int(yc[i]))
        for i in range(int(len(Xc)))
    ]

    attack = run_convex_ball_output_finite_prior_attack(
        release,
        d_minus,
        prior_records=prior_records,
        prior_weights=prior_weights,
        known_label=known_label,
        true_record=target,
        eta_grid=tuple(float(v) for v in eta_grid),
    )
    attack.diagnostics["target_index"] = int(target_index)

    if out_path is not None and attack.z_hat is not None:
        plot_attack_result(
            attack,
            target,
            feature_shape=ds.feature_shape,
            out_path=out_path,
        )

    return attack, d_minus, target


def attack_nonconvex_ball_trace_finite_prior(
    trace: DPSGDTrace,
    X_candidates: np.ndarray,
    y_candidates: np.ndarray,
    *,
    prior_weights: Optional[Sequence[float]] = None,
    cfg: Optional[BallTraceMapAttackConfig] = None,
    target_index: Optional[int] = None,
    loss_name: Optional[str] = None,
    loss_fn: Optional[ExampleLossFn] = None,
    known_label: Optional[int] = None,
    true_record: Optional[Record] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
) -> AttackResult:
    Xc = np.asarray(X_candidates)
    yc = np.asarray(y_candidates)
    if len(Xc) != len(yc):
        raise ValueError("X_candidates and y_candidates must have the same length.")
    prior_records = [
        Record(features=np.asarray(Xc[i]), label=int(yc[i]))
        for i in range(int(len(Xc)))
    ]
    return run_ball_trace_finite_prior_attack(
        trace,
        prior_records,
        prior_weights=prior_weights,
        cfg=cfg,
        target_index=target_index,
        loss_name=loss_name,
        loss_fn=loss_fn,
        known_label=known_label,
        true_record=true_record,
        eta_grid=tuple(float(v) for v in eta_grid),
    )
