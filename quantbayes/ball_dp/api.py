from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import jax
import jax.random as jr
import numpy as np
import optax

from .config import (
    BallSGDConfig,
    ConvexOptimizationConfig,
    ConvexReleaseConfig,
    GaussianCalibrationConfig,
    ReconstructorTrainingConfig,
    ShadowCorpusConfig,
)
from .convex.attacks import run_convex_attack
from .convex.releases import (
    run_convex_ball_erm_dp,
    run_convex_ball_erm_rdp,
    run_convex_noiseless_erm_release,
)
from .evaluation.rero import (
    EmpiricalBallPrior,
    UniformL2BallPrior,
    compute_ball_rero_report,
)
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
    AttackFeatureMap,
    AttackResult,
    Record,
    RecordCodec,
    ReconstructorArtifact,
    ReRoReport,
    ReleaseArtifact,
    ShadowCorpus,
)
from .attacks.model_based import (
    build_shadow_corpus,
    make_attack_feature_map,
    run_model_based_attack,
    train_shadow_reconstructor,
)
from .attacks.gradient_based import (
    DPSGDTrace,
    DPSGDTraceRecorder,
    TraceOptimizationAttackConfig,
    run_prior_aware_trace_attack,
    run_trace_optimization_attack,
    subtract_known_batch_gradients,
)

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
    gaussian_method: str = "analytic",
    gaussian_tol: float = 1e-12,
    orders: Sequence[float] = (2, 3, 4, 5, 8, 16, 32, 64, 128),
    dp_deltas_for_rdp: Sequence[float] = (1e-6,),
    solver: str = "lbfgs_fullbatch",
    max_iter: int = 500,
    learning_rate: float = 1e-1,
    grad_tol: float = 1e-8,
    param_tol: float = 1e-10,
    objective_tol: float = 1e-12,
    line_search_steps: int = 15,
    seed: int = 0,
) -> ReleaseArtifact:
    fam = _normalize_convex_model_family(model_family)
    privacy_kind = _normalize_privacy_kind(privacy)

    if solver not in {"lbfgs_fullbatch", "gd_fullbatch"}:
        raise ValueError(
            "fit_convex exposes only {'lbfgs_fullbatch', 'gd_fullbatch'} in the "
            "stable public API. If you need experimental external solvers, call the "
            "low-level convex release functions directly."
        )

    train_ds = _as_dataset(X_train, y_train, name="train")
    eval_ds = None
    if X_eval is not None and y_eval is not None:
        eval_ds = _as_dataset(X_eval, y_eval, name="eval")

    opt_cfg = ConvexOptimizationConfig(
        solver=solver,
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        grad_tol=float(grad_tol),
        param_tol=float(param_tol),
        objective_tol=float(objective_tol),
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


def make_uniform_ball_prior(
    *, center: np.ndarray, radius: float, dimension: Optional[int] = None
) -> UniformL2BallPrior:
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    dim = center.size if dimension is None else int(dimension)
    return UniformL2BallPrior(center=center, radius=float(radius), dimension=dim)


def make_empirical_ball_prior(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    label: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> EmpiricalBallPrior:
    X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    if y is not None and label is not None:
        y = np.asarray(y, dtype=np.int64)
        X = X[y == int(label)]
    if max_samples is not None and len(X) > int(max_samples):
        X = X[: int(max_samples)]
    return EmpiricalBallPrior(X)


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
    clip_norms = list(cfg.get("clip_norms", ()))
    noise_multipliers = list(cfg.get("noise_multipliers", ()))
    effective_noise_stds = list(cfg.get("effective_noise_stds", ()))
    step_delta_ball = [] if sens.step_delta_ball is None else list(sens.step_delta_ball)
    step_delta_std = [] if sens.step_delta_std is None else list(sens.step_delta_std)
    ratios = list(extra.get("ball_to_standard_sensitivity_ratio_by_step", []) or [])
    rho_by_step = list(extra.get("rho_by_step", []) or [])

    T = max(
        len(batch_sizes),
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
        row = {
            "step": int(t + 1),
            "batch_size": None if t >= len(batch_sizes) else int(batch_sizes[t]),
            "clip_norm": None if t >= len(clip_norms) else float(clip_norms[t]),
            "noise_multiplier": (
                None if t >= len(noise_multipliers) else float(noise_multipliers[t])
            ),
            "effective_noise_std": (
                None
                if t >= len(effective_noise_stds)
                else float(effective_noise_stds[t])
            ),
            "delta_ball": (
                None if t >= len(step_delta_ball) else float(step_delta_ball[t])
            ),
            "delta_standard": (
                None if t >= len(step_delta_std) else float(step_delta_std[t])
            ),
            "ball_to_standard_ratio": None if t >= len(ratios) else ratios[t],
            "rho": None if t >= len(rho_by_step) else rho_by_step[t],
        }
        rows.append(row)
    return rows


def fit_ball_sgd(
    model: Any,
    optimizer: optax.GradientTransformation,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    radius: float,
    lz: Optional[float],
    X_eval: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
    privacy: BallPrivacyKind = "ball_rdp",
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    num_steps: int = 1000,
    batch_size: int | Sequence[int] = 128,
    clip_norm: float | Sequence[float] = 1.0,
    noise_multiplier: float | Sequence[float] = 1.0,
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
    seed: int = 0,
    key: Optional[jax.Array] = None,
    return_debug_history: bool = False,
    trace_recorder=None,
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
        epsilon=None if epsilon is None else float(epsilon),
        delta=None if delta is None else float(delta),
        loss_name=str(loss_name),
        normalize_noisy_sum_by=str(normalize_noisy_sum_by),
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


def build_nonconvex_shadow_corpus(
    *,
    d_minus: ArrayDataset,
    shadow_targets: Sequence[Record],
    victim_train_fn,
    feature_map: AttackFeatureMap,
    cfg: ShadowCorpusConfig,
    record_codec: Optional[RecordCodec] = None,
    seed_policy: str = "vary",
    fixed_seed: Optional[int] = None,
) -> ShadowCorpus:
    return build_shadow_corpus(
        d_minus=d_minus,
        shadow_targets=shadow_targets,
        victim_train_fn=victim_train_fn,
        feature_map=feature_map,
        cfg=cfg,
        record_codec=record_codec,
        seed_policy=seed_policy,
        fixed_seed=fixed_seed,
    )


def fit_shadow_reconstructor(
    corpus: ShadowCorpus,
    cfg: ReconstructorTrainingConfig,
) -> ReconstructorArtifact:
    return train_shadow_reconstructor(corpus, cfg)


def attack_nonconvex_model_based(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    *,
    reconstructor: ReconstructorArtifact,
    feature_map: AttackFeatureMap,
    true_record: Optional[Record] = None,
    known_label: Optional[int] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
    ball_center: Optional[np.ndarray] = None,
    ball_radius: Optional[float] = None,
    box_bounds: Optional[tuple[float, float]] = None,
) -> AttackResult:
    return run_model_based_attack(
        release,
        d_minus,
        reconstructor=reconstructor,
        feature_map=feature_map,
        true_record=true_record,
        known_label=known_label,
        eta_grid=eta_grid,
        ball_center=ball_center,
        ball_radius=ball_radius,
        box_bounds=box_bounds,
    )


def attack_nonconvex_prior_aware_trace(
    trace: DPSGDTrace,
    prior_records: Sequence[Record],
    *,
    dataset: Optional[ArrayDataset] = None,
    target_index: Optional[int] = None,
    **kwargs,
) -> AttackResult:
    if dataset is not None and target_index is not None:
        trace = subtract_known_batch_gradients(
            trace,
            dataset,
            target_index=int(target_index),
        )
    return run_prior_aware_trace_attack(
        trace,
        prior_records,
        **kwargs,
    )


def attack_nonconvex_trace_optimization(
    trace: DPSGDTrace,
    *,
    cfg: Optional[TraceOptimizationAttackConfig] = None,
    dataset: Optional[ArrayDataset] = None,
    target_index: Optional[int] = None,
    loss_name: Optional[str] = None,
    loss_fn: Optional[ExampleLossFn] = None,
    feature_shape: Optional[Sequence[int]] = None,
    known_label: Optional[int] = None,
    label_space: Optional[Sequence[int]] = None,
    true_record: Optional[Record] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
) -> AttackResult:
    return run_trace_optimization_attack(
        trace,
        cfg=cfg,
        dataset=dataset,
        target_index=target_index,
        loss_name=loss_name,
        loss_fn=loss_fn,
        feature_shape=feature_shape,
        known_label=known_label,
        label_space=label_space,
        true_record=true_record,
        eta_grid=eta_grid,
    )
