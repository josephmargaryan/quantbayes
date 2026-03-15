# quantbayes/ball_dp/api.py

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Dict, Literal

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
    ReconstructorTrainingConfig,
    ShadowCorpusConfig,
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

from .attacks.spear import (
    SpearAttackConfig,
    SpearAttackResult,
    run_spear_model_batch_attack,
    run_spear_trace_step_attack,
)
from .plots import plot_attack_result, plot_batch_reconstruction_grid

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


def _prepare_attack_arrays(
    X: np.ndarray,
    *,
    flatten_inputs: bool,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    X_np = np.asarray(X)
    if X_np.ndim < 2:
        raise ValueError("X must have shape (n, ...).")
    feature_shape = tuple(int(v) for v in X_np.shape[1:])
    if flatten_inputs:
        X_attack = np.asarray(X_np.reshape(len(X_np), -1), dtype=np.float64)
    else:
        X_attack = np.asarray(X_np, dtype=np.float64)
    return X_np, X_attack, feature_shape


def _resolve_target_batch_indices(
    *,
    n_total: int,
    target_index: int,
    batch_size: int,
    seed: int,
    batch_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    target_index = int(target_index)
    batch_size = int(batch_size)

    if target_index < 0 or target_index >= int(n_total):
        raise IndexError(
            f"target_index={target_index} is out of range for dataset size {n_total}."
        )
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2 for SPEAR.")
    if batch_size > int(n_total):
        raise ValueError(f"batch_size={batch_size} exceeds dataset size {n_total}.")

    if batch_indices is not None:
        idx = np.asarray(batch_indices, dtype=np.int64).reshape(-1)
        if idx.size != batch_size:
            raise ValueError(
                f"Provided batch_indices has length {idx.size}, expected batch_size={batch_size}."
            )
        if np.any(idx < 0) or np.any(idx >= int(n_total)):
            raise ValueError("batch_indices contains an out-of-range index.")
        if len(np.unique(idx)) != len(idx):
            raise ValueError("batch_indices must not contain duplicates.")
        if not np.any(idx == target_index):
            raise ValueError(
                f"Provided batch_indices does not contain target_index={target_index}."
            )
        others = idx[idx != target_index]
        return np.concatenate(
            [
                np.asarray([target_index], dtype=np.int64),
                np.asarray(others, dtype=np.int64),
            ]
        )

    rng = np.random.default_rng(int(seed))
    pool = np.arange(int(n_total), dtype=np.int64)
    pool = pool[pool != target_index]
    others = rng.choice(pool, size=int(batch_size) - 1, replace=False)
    return np.concatenate(
        [
            np.asarray([target_index], dtype=np.int64),
            np.asarray(others, dtype=np.int64),
        ]
    )


def _spear_batch_reconstruction(
    batch_result: SpearAttackResult,
) -> Optional[np.ndarray]:
    recon = (
        batch_result.x_hat_aligned
        if batch_result.x_hat_aligned is not None
        else batch_result.x_hat
    )
    return None if recon is None else np.asarray(recon, dtype=np.float32)


def _spear_target_attack_result(
    batch_result: SpearAttackResult,
    *,
    true_features_flat: np.ndarray,
    eta_grid: Sequence[float],
    attack_family: str,
) -> AttackResult:
    recon_batch = _spear_batch_reconstruction(batch_result)
    z_hat = None
    if recon_batch is not None and recon_batch.shape[0] >= 1:
        z_hat = np.asarray(recon_batch[0], dtype=np.float32)

    metrics = feature_reconstruction_metrics(
        np.asarray(true_features_flat, dtype=np.float32).reshape(-1),
        None if z_hat is None else np.asarray(z_hat, dtype=np.float32).reshape(-1),
        eta_grid=eta_grid,
    )
    for k, v in batch_result.metrics.items():
        if np.isscalar(v):
            metrics[f"batch_{k}"] = float(v)

    diagnostics = dict(batch_result.diagnostics)
    diagnostics["target_position_in_batch"] = 0

    return AttackResult(
        attack_family=str(attack_family),
        z_hat=None if z_hat is None else np.asarray(z_hat, dtype=np.float32),
        y_hat=None,
        status=str(batch_result.status),
        diagnostics=diagnostics,
        metrics=metrics,
    )


def fit_targeted_ball_sgd_trace(
    model: Any,
    optimizer: optax.GradientTransformation,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    target_index: int,
    radius: float,
    lz: Optional[float],
    X_eval: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
    privacy: BallPrivacyKind = "ball_rdp",
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    num_steps: int = 1,
    batch_size: int = 8,
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.0,
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
    batch_indices: Optional[Sequence[int]] = None,
    fixed_target_steps: Literal["first", "all"] = "first",
    flatten_inputs: bool = True,
    seed: int = 0,
    key: Optional[jax.Array] = None,
) -> tuple[ReleaseArtifact, DPSGDTrace, Record, np.ndarray]:
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive.")

    X_orig, X_attack, feature_shape = _prepare_attack_arrays(
        X_train, flatten_inputs=bool(flatten_inputs)
    )
    y_np = np.asarray(y_train)
    if len(X_attack) != len(y_np):
        raise ValueError("X_train and y_train must have the same length.")

    selected_batch_indices = _resolve_target_batch_indices(
        n_total=len(X_attack),
        target_index=int(target_index),
        batch_size=int(batch_size),
        seed=int(seed),
        batch_indices=batch_indices,
    )

    mode = str(fixed_target_steps).lower()
    if mode not in {"first", "all"}:
        raise ValueError("fixed_target_steps must be one of {'first', 'all'}.")

    fixed_schedule: list[Optional[tuple[int, ...]]] = []
    batch_tuple = tuple(int(v) for v in selected_batch_indices.tolist())
    for t in range(int(num_steps)):
        if mode == "all" or t == 0:
            fixed_schedule.append(batch_tuple)
        else:
            fixed_schedule.append(None)

    recorder = DPSGDTraceRecorder(
        capture_every=1,
        keep_models=True,
        keep_batch_indices=True,
    )

    X_eval_attack = None
    if X_eval is not None:
        _, X_eval_attack, _ = _prepare_attack_arrays(
            X_eval, flatten_inputs=bool(flatten_inputs)
        )

    release = fit_ball_sgd(
        model,
        optimizer,
        X_attack,
        y_np,
        radius=float(radius),
        lz=None if lz is None else float(lz),
        X_eval=X_eval_attack,
        y_eval=None if y_eval is None else np.asarray(y_eval),
        privacy=str(privacy),
        epsilon=None if epsilon is None else float(epsilon),
        delta=None if delta is None else float(delta),
        num_steps=int(num_steps),
        batch_size=int(batch_size),
        clip_norm=float(clip_norm),
        noise_multiplier=float(noise_multiplier),
        loss_name=str(loss_name),
        state=state,
        loss_fn=loss_fn,
        predict_fn=predict_fn,
        parameter_regularizer=parameter_regularizer,
        normalize_noisy_sum_by=str(normalize_noisy_sum_by),
        frobenius_reg_strength=float(frobenius_reg_strength),
        spectral_reg_strength=float(spectral_reg_strength),
        spectral_reg_kwargs=dict(spectral_reg_kwargs or {}),
        record_operator_norms=bool(record_operator_norms),
        operator_norms_every=int(operator_norms_every),
        operator_norm_kwargs=dict(operator_norm_kwargs or {}),
        checkpoint_selection=str(checkpoint_selection),
        eval_every=int(eval_every),
        eval_batch_size=int(eval_batch_size),
        warn_if_ball_equals_standard=bool(warn_if_ball_equals_standard),
        fixed_batch_indices_schedule=tuple(fixed_schedule),
        seed=int(seed),
        key=key,
        trace_recorder=recorder,
    )

    trace = recorder.to_trace(
        state=state,
        loss_name=str(loss_name),
        reduction="mean",
        metadata={
            "dataset_size": int(len(X_attack)),
            "sample_rate": float(batch_size) / float(len(X_attack)),
            "feature_shape": tuple(int(v) for v in feature_shape),
            "target_index": int(target_index),
            "target_position": 0,
            "selected_batch_indices": selected_batch_indices.astype(np.int64).tolist(),
            "privacy": str(privacy),
        },
    )
    if not trace.steps:
        raise RuntimeError("Trace recorder did not capture any step.")

    true_record = Record(
        features=np.asarray(X_orig[int(target_index)]),
        label=int(y_np[int(target_index)]),
    )
    return (
        release,
        trace,
        true_record,
        np.asarray(selected_batch_indices, dtype=np.int64),
    )


def attack_nonconvex_spear_exact_batch(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    target_index: int,
    batch_size: int = 8,
    batch_indices: Optional[Sequence[int]] = None,
    flatten_inputs: bool = True,
    feature_shape: Optional[Sequence[int]] = None,
    layer_path: Sequence[Any] = ("layers", 0),
    loss_name: str = "softmax_cross_entropy",
    state: Any = None,
    reduction: str = "mean",
    max_samples: int = 20_000,
    false_rejection_rate: float = 1e-5,
    zero_tol: float = 1e-7,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
    seed: int = 0,
    pair_out_path: Optional[str] = None,
    grid_out_path: Optional[str] = None,
) -> tuple[AttackResult, SpearAttackResult, np.ndarray]:
    X_orig, X_attack, feature_shape_default = _prepare_attack_arrays(
        X, flatten_inputs=bool(flatten_inputs)
    )
    y_np = np.asarray(y)
    feature_shape_use = (
        tuple(int(v) for v in feature_shape)
        if feature_shape is not None
        else feature_shape_default
    )

    batch_idx = _resolve_target_batch_indices(
        n_total=len(X_attack),
        target_index=int(target_index),
        batch_size=int(batch_size),
        seed=int(seed),
        batch_indices=batch_indices,
    )

    xb_attack = np.asarray(X_attack[batch_idx], dtype=np.float64)
    yb = np.asarray(y_np[batch_idx])

    batch_result = run_spear_model_batch_attack(
        model,
        xb_attack,
        yb,
        layer_path=layer_path,
        loss_name=str(loss_name),
        state=state,
        reduction=str(reduction),
        cfg=SpearAttackConfig(
            max_samples=int(max_samples),
            batch_size=int(batch_size),
            false_rejection_rate=float(false_rejection_rate),
            zero_tol=float(zero_tol),
            random_seed=int(seed),
            greedy_swap_rule="best_improvement",
            noisy_mode=False,
        ),
        true_batch=xb_attack,
        eta_grid=eta_grid,
        seed=int(seed),
    )

    target_attack = _spear_target_attack_result(
        batch_result,
        true_features_flat=xb_attack[0],
        eta_grid=eta_grid,
        attack_family="spear_exact_batch_target",
    )
    target_attack.diagnostics["batch_indices"] = np.asarray(batch_idx, dtype=np.int64)
    target_attack.diagnostics["target_index"] = int(target_index)

    true_record = Record(
        features=np.asarray(X_orig[int(target_index)]),
        label=int(y_np[int(target_index)]),
    )

    if pair_out_path is not None and target_attack.z_hat is not None:
        plot_attack_result(
            target_attack,
            true_record,
            feature_shape=feature_shape_use,
            out_path=pair_out_path,
        )

    recon_batch = _spear_batch_reconstruction(batch_result)
    if grid_out_path is not None and recon_batch is not None:
        plot_batch_reconstruction_grid(
            np.asarray(X_orig[batch_idx]),
            np.asarray(recon_batch),
            feature_shape=feature_shape_use,
            title=f"{batch_result.attack_family} ({batch_result.status})",
            out_path=grid_out_path,
        )

    return target_attack, batch_result, np.asarray(batch_idx, dtype=np.int64)


def attack_nonconvex_spear_private_step(
    model: Any,
    optimizer: optax.GradientTransformation,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    target_index: int,
    radius: float,
    lz: Optional[float],
    X_eval: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
    privacy: BallPrivacyKind = "ball_rdp",
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    batch_size: int = 8,
    batch_indices: Optional[Sequence[int]] = None,
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    loss_name: str = "softmax_cross_entropy",
    state: Any = None,
    loss_fn: Optional[ExampleLossFn] = None,
    predict_fn: Optional[PredictFn] = default_predict_fn,
    parameter_regularizer: Optional[Callable[[Any, Any], jax.Array]] = None,
    normalize_noisy_sum_by: str = "batch_size",
    flatten_inputs: bool = True,
    feature_shape: Optional[Sequence[int]] = None,
    layer_path: Sequence[Any] = ("layers", 0),
    max_samples: int = 20_000,
    false_rejection_rate: float = 1e-5,
    zero_tol_exact: float = 1e-7,
    zero_tol_noisy: float = 1e-4,
    noisy_gamma_target: float = 0.98,
    noisy_submatrix_rows: Optional[int] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
    seed: int = 0,
    pair_out_path: Optional[str] = None,
    grid_out_path: Optional[str] = None,
) -> tuple[AttackResult, SpearAttackResult, ReleaseArtifact, DPSGDTrace, np.ndarray]:
    release, trace, true_record, batch_idx = fit_targeted_ball_sgd_trace(
        model,
        optimizer,
        X_train,
        y_train,
        target_index=int(target_index),
        radius=float(radius),
        lz=None if lz is None else float(lz),
        X_eval=X_eval,
        y_eval=y_eval,
        privacy=str(privacy),
        epsilon=None if epsilon is None else float(epsilon),
        delta=None if delta is None else float(delta),
        num_steps=1,
        batch_size=int(batch_size),
        clip_norm=float(clip_norm),
        noise_multiplier=float(noise_multiplier),
        loss_name=str(loss_name),
        state=state,
        loss_fn=loss_fn,
        predict_fn=predict_fn,
        parameter_regularizer=parameter_regularizer,
        normalize_noisy_sum_by=str(normalize_noisy_sum_by),
        batch_indices=batch_indices,
        fixed_target_steps="first",
        flatten_inputs=bool(flatten_inputs),
        seed=int(seed),
    )

    step = trace.steps[0]

    X_orig, X_attack, feature_shape_default = _prepare_attack_arrays(
        X_train, flatten_inputs=bool(flatten_inputs)
    )
    y_np = np.asarray(y_train)
    feature_shape_use = (
        tuple(int(v) for v in feature_shape)
        if feature_shape is not None
        else feature_shape_default
    )

    xb_attack = np.asarray(X_attack[batch_idx], dtype=np.float64)

    is_noisy = float(getattr(step, "effective_noise_std", 0.0)) > 0.0
    cfg = SpearAttackConfig(
        max_samples=int(max_samples),
        batch_size=int(batch_size),
        false_rejection_rate=float(false_rejection_rate),
        zero_tol=float(zero_tol_noisy if is_noisy else zero_tol_exact),
        random_seed=int(seed),
        greedy_swap_rule="best_improvement",
        noisy_mode=bool(is_noisy),
        noisy_gamma_target=(float(noisy_gamma_target) if is_noisy else None),
        noisy_submatrix_rows=(
            None if noisy_submatrix_rows is None else int(noisy_submatrix_rows)
        ),
    )

    batch_result = run_spear_trace_step_attack(
        step,
        layer_path=layer_path,
        cfg=cfg,
        true_batch=xb_attack,
        eta_grid=eta_grid,
    )

    target_attack = _spear_target_attack_result(
        batch_result,
        true_features_flat=xb_attack[0],
        eta_grid=eta_grid,
        attack_family="spear_private_step_target",
    )
    target_attack.diagnostics["batch_indices"] = np.asarray(batch_idx, dtype=np.int64)
    target_attack.diagnostics["target_index"] = int(target_index)
    target_attack.diagnostics["release_kind"] = str(release.release_kind)
    target_attack.diagnostics["primary_privacy_view"] = str(
        release.attack_metadata.get("primary_privacy_view", "ball")
    )

    if pair_out_path is not None and target_attack.z_hat is not None:
        plot_attack_result(
            target_attack,
            true_record,
            feature_shape=feature_shape_use,
            out_path=pair_out_path,
        )

    recon_batch = _spear_batch_reconstruction(batch_result)
    if grid_out_path is not None and recon_batch is not None:
        plot_batch_reconstruction_grid(
            np.asarray(X_orig[batch_idx]),
            np.asarray(recon_batch),
            feature_shape=feature_shape_use,
            title=f"{batch_result.attack_family} ({batch_result.status})",
            out_path=grid_out_path,
        )

    return (
        target_attack,
        batch_result,
        release,
        trace,
        np.asarray(batch_idx, dtype=np.int64),
    )


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
