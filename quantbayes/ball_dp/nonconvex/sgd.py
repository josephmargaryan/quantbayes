# quantbayes/ball_dp/nonconvex/sgd.py
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from ..accountants.subsampled_gaussian import build_ball_sgd_rdp_ledgers
from ..config import BallSGDConfig, NonconvexReleaseConfig
from ..metrics import accuracy_from_logits
from ..types import (
    ArrayDataset,
    DualPrivacyLedger,
    PrivacyLedger,
    ReleaseArtifact,
    SensitivityMetadata,
)
from .per_example import (
    ExampleLossFn,
    PredictFn,
    add_gaussian_noise,
    clip_and_aggregate_per_example_grads,
    default_predict_fn,
    make_batched_predict_fn,
    make_parameter_regularizer_grad_fn,
    make_per_example_grad_fn,
    partition_model,
    resolve_loss_fn,
    tree_add,
    tree_scalar_mul,
)
from quantbayes.stochax.utils.regularizers import (
    collect_operator_norms,
    global_frobenius_penalty,
    global_spectral_norm_penalty,
    summarize_operator_norms,
)


def _contains_state_index(pytree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(
        pytree,
        is_leaf=lambda x: type(x).__name__ == "StateIndex",
    )
    return any(type(leaf).__name__ == "StateIndex" for leaf in leaves)


def _assert_state_is_supported(model: Any, state: Any) -> None:
    del state
    if _contains_state_index(model):
        raise ValueError(
            "Mutable Equinox state carried by StateIndex leaves is not supported in the "
            "Ball-SGD core trainer. In particular, training-mode BatchNorm and built-in "
            "stateful SpectralNorm wrappers should be frozen or removed. Use stateless "
            "layers, Dropout, LayerNorm/GroupNorm/RMSNorm, or custom read-only state."
        )


def _expand_schedule(value_or_seq, T: int, *, cast_fn=float):
    if isinstance(value_or_seq, (tuple, list)):
        seq = [cast_fn(v) for v in value_or_seq]
        if len(seq) != T:
            raise ValueError(f"Expected schedule of length {T}, got {len(seq)}")
        return seq
    return [cast_fn(value_or_seq) for _ in range(T)]


def _binary_targets01_np(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    uniq = set(np.unique(y).tolist())
    if uniq.issubset({0, 1}):
        return y.astype(np.int64)
    if uniq.issubset({-1, 1}):
        return (y > 0).astype(np.int64)
    raise ValueError(
        "binary_logistic requires labels in {0,1} or {-1,+1}. "
        f"Got unique labels={sorted(uniq)}."
    )


def _labels_for_accuracy(y: np.ndarray, cfg: BallSGDConfig) -> np.ndarray:
    if (
        str(getattr(cfg, "loss_name", "softmax_cross_entropy")).lower()
        == "binary_logistic"
    ):
        return _binary_targets01_np(y)
    return np.asarray(y, dtype=np.int64)


def _validate_dataset_labels_for_loss(
    dataset: ArrayDataset, cfg: BallSGDConfig
) -> None:
    loss_name = str(getattr(cfg, "loss_name", "softmax_cross_entropy")).lower()
    if loss_name == "binary_logistic":
        _binary_targets01_np(np.asarray(dataset.y))


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


def _step_delta_ball(
    cfg: BallSGDConfig, clip_schedule: Sequence[float]
) -> list[float] | None:
    if cfg.lz is None:
        return None
    lz = float(cfg.lz)
    if lz < 0.0:
        raise ValueError("cfg.lz must be >= 0.")
    radius = float(cfg.radius)
    if radius < 0.0:
        raise ValueError("cfg.radius must be >= 0.")
    out = []
    for c in clip_schedule:
        c = float(c)
        out.append(float(min(lz * radius, 2.0 * c)))
    return out


def _step_delta_standard(clip_schedule: Sequence[float]) -> list[float]:
    out = []
    for c in clip_schedule:
        c = float(c)
        if not math.isfinite(c):
            raise ValueError(
                "Standard DP-SGD accounting requires finite clip norms at every step."
            )
        out.append(float(2.0 * c))
    return out


def _regime_summary(
    lz: float | None,
    clip_norms: Sequence[float],
    step_delta_ball: Sequence[float] | None,
    step_delta_std: Sequence[float] | None,
) -> dict[str, float | bool | None]:
    finite_clips = [float(c) for c in clip_norms if math.isfinite(float(c))]
    min_clip = min(finite_clips) if finite_clips else None

    critical_radius = None
    if lz is not None and lz > 0.0 and min_clip is not None:
        critical_radius = float(2.0 * min_clip / lz)

    sat_frac = None
    sat_all = False
    if step_delta_ball is not None and step_delta_std is not None:
        sat_flags = [
            abs(float(a) - float(b)) <= 1e-12
            for a, b in zip(step_delta_ball, step_delta_std)
        ]
        sat_frac = float(sum(sat_flags) / len(sat_flags)) if sat_flags else None
        sat_all = bool(all(sat_flags)) if sat_flags else False

    return {
        "critical_radius_for_min_clip": critical_radius,
        "ball_standard_saturation_fraction": sat_frac,
        "ball_equals_standard_at_all_steps": sat_all,
    }


def _build_parameter_regularizer(
    cfg: BallSGDConfig,
    *,
    parameter_regularizer: Optional[Callable[[Any, Any], jnp.ndarray]],
) -> Optional[Callable[[Any, Any], jnp.ndarray]]:
    frob = float(getattr(cfg, "frobenius_reg_strength", 0.0))
    spec = float(getattr(cfg, "spectral_reg_strength", 0.0))
    spectral_kwargs = dict(getattr(cfg, "spectral_reg_kwargs", {}) or {})

    if parameter_regularizer is None and frob == 0.0 and spec == 0.0:
        return None

    def reg(model: Any, state: Any) -> jnp.ndarray:
        val = jnp.asarray(0.0, dtype=jnp.float32)
        if frob != 0.0:
            val = val + jnp.asarray(frob, dtype=val.dtype) * global_frobenius_penalty(
                model
            )
        if spec != 0.0:
            val = val + jnp.asarray(
                spec, dtype=val.dtype
            ) * global_spectral_norm_penalty(
                model,
                **spectral_kwargs,
            )
        if parameter_regularizer is not None:
            val = val + jnp.asarray(
                parameter_regularizer(model, state), dtype=val.dtype
            )
        return val

    return reg


def _make_train_step(
    *,
    optimizer: optax.GradientTransformation,
    per_example_grad_fn,
    regularizer_grad_fn,
    normalize_noisy_sum_by: str,
    param_projector=None,
):
    @eqx.filter_jit
    def _step(
        params: Any,
        opt_state: Any,
        xb: jnp.ndarray,
        yb: jnp.ndarray,
        key: jax.Array,
        clip_norm: jax.Array,
        noise_multiplier: jax.Array,
    ):
        grad_key, noise_key = jr.split(key)
        per_example_grads = per_example_grad_fn(params, xb, yb, grad_key)
        summed_clipped, norms, clip_frac = clip_and_aggregate_per_example_grads(
            per_example_grads,
            clip_norm,
        )

        noise_std = jnp.asarray(
            noise_multiplier,
            dtype=jnp.asarray(clip_norm).dtype,
        ) * jnp.asarray(clip_norm)
        noisy_sum = add_gaussian_noise(summed_clipped, noise_std, noise_key)

        if normalize_noisy_sum_by == "batch_size":
            sanitized_grad = tree_scalar_mul(
                noisy_sum,
                1.0 / jnp.asarray(xb.shape[0], dtype=jnp.float32),
            )
        elif normalize_noisy_sum_by == "none":
            sanitized_grad = noisy_sum
        else:
            raise ValueError(
                "normalize_noisy_sum_by must be one of {'batch_size', 'none'}."
            )

        if regularizer_grad_fn is not None:
            reg_grad = regularizer_grad_fn(params)
            total_grad = tree_add(sanitized_grad, reg_grad)
        else:
            total_grad = sanitized_grad

        updates, opt_state = optimizer.update(total_grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        if param_projector is not None:
            params = param_projector(params)

        mean_norm = jnp.mean(norms)
        max_norm = jnp.max(norms)
        return params, opt_state, mean_norm, max_norm, clip_frac, sanitized_grad

    return _step


def _batched_example_losses(
    model: Any,
    state: Any,
    xb: jnp.ndarray,
    yb: jnp.ndarray,
    key: jax.Array,
    loss_fn: ExampleLossFn,
) -> jnp.ndarray:
    keys = jr.split(key, xb.shape[0])
    return jax.vmap(lambda x, y, k: loss_fn(model, state, x, y, k), in_axes=(0, 0, 0))(
        xb, yb, keys
    )


def _dataset_loss_and_accuracy(
    *,
    model: Any,
    state: Any,
    dataset: ArrayDataset,
    loss_fn: ExampleLossFn,
    predict_fn: Optional[PredictFn],
    cfg: BallSGDConfig,
    batch_size: int,
    key: jax.Array,
) -> tuple[float, Optional[float]]:
    model_eval = eqx.nn.inference_mode(model, value=True)
    x = jnp.asarray(dataset.X, dtype=jnp.float32)
    y = jnp.asarray(dataset.y)

    n = int(x.shape[0])
    if n == 0:
        return float("nan"), None

    num_batches = math.ceil(n / int(batch_size))
    keys = jr.split(key, num_batches)

    total_loss = 0.0
    logits_chunks = []

    for i in range(num_batches):
        lo = i * int(batch_size)
        hi = min((i + 1) * int(batch_size), n)
        xb = x[lo:hi]
        yb = y[lo:hi]
        batch_key = keys[i]

        losses = _batched_example_losses(
            model_eval,
            state,
            xb,
            yb,
            batch_key,
            loss_fn,
        )
        total_loss += float(jnp.sum(losses))

        if predict_fn is not None:
            pred_keys = jr.split(batch_key, xb.shape[0])
            logits = jax.vmap(
                lambda x_one, k_one: predict_fn(model_eval, state, x_one, k_one),
                in_axes=(0, 0),
            )(xb, pred_keys)
            logits_chunks.append(np.asarray(logits))

    loss = total_loss / float(n)

    if predict_fn is None:
        return float(loss), None

    logits_all = np.concatenate(logits_chunks, axis=0)
    acc = float(
        accuracy_from_logits(
            logits_all,
            _labels_for_accuracy(np.asarray(dataset.y), cfg),
        )
    )
    return float(loss), float(acc)


def _empty_dual_ledger(radius: float) -> DualPrivacyLedger:
    note = "No Gaussian noise added; this is a nonprivate clipped-SGD baseline."
    return DualPrivacyLedger(
        ball=PrivacyLedger(
            mechanism="noiseless_nonprivate",
            radius=float(radius),
            notes=[note],
        ),
        standard=PrivacyLedger(
            mechanism="noiseless_nonprivate",
            notes=[note],
        ),
    )


def _verify_dp_target(
    artifact: ReleaseArtifact,
    *,
    target_epsilon: float,
    target_delta: float,
    accounting_view: str,
) -> ReleaseArtifact:
    if accounting_view not in {"ball", "standard"}:
        raise ValueError(f"Unknown accounting_view={accounting_view!r}.")
    ledger = (
        artifact.privacy.ball
        if accounting_view == "ball"
        else artifact.privacy.standard
    )
    if not ledger.dp_certificates:
        raise RuntimeError(
            f"Expected an RDP->DP certificate on the {accounting_view} ledger because cfg.delta was supplied."
        )
    achieved = float(ledger.dp_certificates[0].epsilon)
    target = float(target_epsilon)
    if achieved > target + 1e-12:
        raise ValueError(
            f"Configured noise schedule does not meet requested epsilon={target:.12g} "
            f"under {accounting_view} accounting; achieved epsilon={achieved:.12g} "
            f"at delta={target_delta:.12g}."
        )
    artifact.attack_metadata[f"{accounting_view}_dp_target_epsilon"] = target
    artifact.attack_metadata[f"achieved_{accounting_view}_epsilon"] = achieved
    artifact.attack_metadata["dp_target_verified"] = True
    artifact.attack_metadata["primary_privacy_view"] = accounting_view
    return artifact


def _set_primary_privacy_view(
    artifact: ReleaseArtifact, *, release_kind: str, primary_view: str
) -> ReleaseArtifact:
    artifact.release_kind = release_kind
    artifact.attack_metadata["primary_privacy_view"] = primary_view
    if primary_view == "standard":
        artifact.attack_metadata["note"] = (
            "This release uses the same clipped Gaussian SGD mechanism as the Ball view. "
            "The standard replacement-adjacent ledger is treated as the primary certificate."
        )
    return artifact


def _make_release_artifact(
    *,
    cfg: BallSGDConfig,
    model: Any,
    state: Any,
    optimizer: optax.GradientTransformation,
    dual_ledger: DualPrivacyLedger,
    lz_source: str,
    dataset: ArrayDataset,
    public_eval_dataset: Optional[ArrayDataset],
    step_delta_ball: Sequence[float] | None,
    step_delta_std: Sequence[float] | None,
    batch_schedule: Sequence[int],
    clip_schedule: Sequence[float],
    noise_multiplier_schedule: Sequence[float],
    effective_noise_stds: Sequence[float],
    public_curve_history: Sequence[dict[str, Any]],
    operator_norm_history: Sequence[dict[str, Any]],
    utility_metrics: dict[str, float],
    checkpoint_selection: str,
    selected_step: int,
) -> ReleaseArtifact:
    regime = _regime_summary(cfg.lz, clip_schedule, step_delta_ball, step_delta_std)

    ball_raw_sensitivity = None if cfg.lz is None else float(cfg.lz) * float(cfg.radius)

    rho_by_step: list[float | None] = []
    if ball_raw_sensitivity is not None:
        for c in clip_schedule:
            c = float(c)
            if math.isfinite(c) and c > 0.0:
                rho_by_step.append(float(ball_raw_sensitivity / (2.0 * c)))
            else:
                rho_by_step.append(None)

    sensitivity_ratio_by_step: list[float | None] | None = None
    if step_delta_ball is not None and step_delta_std is not None:
        sensitivity_ratio_by_step = []
        for a, b in zip(step_delta_ball, step_delta_std):
            a = float(a)
            b = float(b)
            if b <= 0.0:
                sensitivity_ratio_by_step.append(None)
            else:
                sensitivity_ratio_by_step.append(float(a / b))

    finite_ratio = (
        []
        if sensitivity_ratio_by_step is None
        else [
            float(v)
            for v in sensitivity_ratio_by_step
            if v is not None and math.isfinite(float(v))
        ]
    )

    if step_delta_ball is None and step_delta_std is None:
        ball_regime = "none"
    elif step_delta_std is None:
        ball_regime = "ball_only_no_standard_comparator"
    elif finite_ratio and all(abs(v - 1.0) <= 1e-12 for v in finite_ratio):
        ball_regime = "saturated_no_advantage"
    elif finite_ratio and all(v < 1.0 - 1e-12 for v in finite_ratio):
        ball_regime = "strict_improvement_all_finite_steps"
    elif finite_ratio and any(v < 1.0 - 1e-12 for v in finite_ratio):
        ball_regime = "mixed"
    else:
        ball_regime = "unknown"

    num_classes_meta = (
        2
        if str(getattr(cfg, "loss_name", "softmax_cross_entropy")).lower()
        == "binary_logistic"
        else int(dataset.num_classes)
    )

    training_config = {
        "radius": float(cfg.radius),
        "lz": None if cfg.lz is None else float(cfg.lz),
        "num_steps": int(cfg.num_steps),
        "batch_sizes": tuple(int(v) for v in batch_schedule),
        "clip_norms": tuple(float(v) for v in clip_schedule),
        "noise_multipliers": tuple(float(v) for v in noise_multiplier_schedule),
        "effective_noise_stds": tuple(float(v) for v in effective_noise_stds),
        "orders": tuple(int(v) for v in cfg.orders),
        "loss_name": str(cfg.loss_name),
        "normalize_noisy_sum_by": str(cfg.normalize_noisy_sum_by),
        "frobenius_reg_strength": float(getattr(cfg, "frobenius_reg_strength", 0.0)),
        "spectral_reg_strength": float(getattr(cfg, "spectral_reg_strength", 0.0)),
        "spectral_reg_kwargs": dict(getattr(cfg, "spectral_reg_kwargs", {}) or {}),
        "record_operator_norms": bool(getattr(cfg, "record_operator_norms", False)),
        "operator_norms_every": int(getattr(cfg, "operator_norms_every", 250)),
        "operator_norm_kwargs": dict(getattr(cfg, "operator_norm_kwargs", {}) or {}),
        "store_full_operator_norm_history": bool(
            getattr(cfg, "store_full_operator_norm_history", False)
        ),
        "checkpoint_selection": str(checkpoint_selection),
        "optimizer": type(optimizer).__name__,
        "seed": int(getattr(cfg, "seed", 0)),
    }

    attack_metadata = {
        "theorem_backed_direct_attack_available": False,
        "shadow_attack_recommended": False,
        "primary_privacy_view": "ball",
        "selected_checkpoint_step": int(selected_step),
    }

    extra = {
        "public_curve_history": list(public_curve_history),
        "operator_norm_history": list(operator_norm_history),
        "ball_raw_sensitivity": ball_raw_sensitivity,
        "rho_by_step": list(rho_by_step),
        "ball_to_standard_sensitivity_ratio_by_step": (
            None
            if sensitivity_ratio_by_step is None
            else list(sensitivity_ratio_by_step)
        ),
        "ball_regime": ball_regime,
        "critical_radius_for_min_clip": regime["critical_radius_for_min_clip"],
        "ball_standard_saturation_fraction": regime[
            "ball_standard_saturation_fraction"
        ],
        "ball_equals_standard_at_all_steps": bool(
            regime["ball_equals_standard_at_all_steps"]
        ),
        "selected_checkpoint_step": int(selected_step),
        # Backward-compatible place to store the (read-only) Equinox state.
        "model_state": state,
    }

    return ReleaseArtifact(
        release_kind="nonconvex_ball_sgd_rdp",
        payload=model,
        model_family="nonconvex",
        architecture="custom",
        training_config=training_config,
        privacy=dual_ledger,
        sensitivity=SensitivityMetadata(
            lz=None if cfg.lz is None else float(cfg.lz),
            lz_source=lz_source,
            radius=float(cfg.radius),
            delta_ball=None if step_delta_ball is None else float(max(step_delta_ball)),
            delta_std=None if step_delta_std is None else float(max(step_delta_std)),
            exact_vs_upper="per_step_min{Lz*r,2C}",
            step_delta_ball=(
                None if step_delta_ball is None else list(map(float, step_delta_ball))
            ),
            step_delta_std=(
                None if step_delta_std is None else list(map(float, step_delta_std))
            ),
        ),
        optimization=None,
        attack_metadata=attack_metadata,
        dataset_metadata={
            "n_total": len(dataset),
            "feature_shape": dataset.feature_shape,
            "num_classes": num_classes_meta,
            "label_values": tuple(
                int(v)
                for v in np.unique(np.asarray(dataset.y, dtype=np.int64)).tolist()
            ),
            "public_eval_n": (
                None if public_eval_dataset is None else int(len(public_eval_dataset))
            ),
        },
        utility_metrics=dict(utility_metrics),
        extra=extra,
    )


def _run_training(
    dataset: ArrayDataset,
    cfg: BallSGDConfig,
    *,
    model: Any,
    optimizer: optax.GradientTransformation,
    state: Any = None,
    public_eval_dataset: Optional[ArrayDataset] = None,
    loss_fn: Optional[ExampleLossFn] = None,
    predict_fn: Optional[PredictFn] = default_predict_fn,
    parameter_regularizer: Optional[Callable[[Any, Any], jnp.ndarray]] = None,
    key: Optional[jax.Array] = None,
    accounting_view: str = "ball",  # 'ball' | 'standard' | 'none'
    force_noiseless: bool = False,
    return_debug_history: bool = False,
    trace_recorder=None,
    param_projector: Optional[Callable[[Any], Any]] = None,
):
    if accounting_view not in {"ball", "standard", "none"}:
        raise ValueError("accounting_view must be one of {'ball', 'standard', 'none'}.")

    if model is None:
        raise ValueError(
            "You must explicitly provide model=... . Internal model builders were removed."
        )
    if optimizer is None:
        raise ValueError("You must explicitly provide optimizer=... .")

    _assert_state_is_supported(model, state)
    _validate_dataset_labels_for_loss(dataset, cfg)
    if public_eval_dataset is not None:
        _validate_dataset_labels_for_loss(public_eval_dataset, cfg)

    if key is None:
        key = jr.PRNGKey(int(getattr(cfg, "seed", 0)))

    resolved_loss_fn = resolve_loss_fn(cfg.loss_name) if loss_fn is None else loss_fn

    n_total = int(len(dataset))
    if n_total <= 0:
        raise ValueError("Training dataset must be non-empty.")

    T = int(cfg.num_steps)
    if T <= 0:
        raise ValueError("cfg.num_steps must be positive.")

    batch_schedule = _expand_schedule(cfg.batch_sizes, T, cast_fn=int)
    clip_schedule = _expand_schedule(cfg.clip_norms, T, cast_fn=float)
    noise_multiplier_schedule = (
        [0.0 for _ in range(T)]
        if force_noiseless
        else _expand_schedule(cfg.noise_multipliers, T, cast_fn=float)
    )
    effective_noise_stds = _effective_noise_stds(
        clip_schedule, noise_multiplier_schedule
    )

    if any(m <= 0 for m in batch_schedule):
        raise ValueError("All batch sizes must be positive.")
    if any(m > n_total for m in batch_schedule):
        raise ValueError(
            f"Batch size schedule contains values larger than dataset size n={n_total}."
        )

    if accounting_view in {"ball", "standard"} and not force_noiseless:
        if any(float(nm) <= 0.0 for nm in noise_multiplier_schedule):
            raise ValueError(
                "Private releases require strictly positive noise multipliers at every step."
            )

    step_delta_std = _step_delta_standard(clip_schedule)
    step_delta_ball = _step_delta_ball(cfg, clip_schedule)

    if accounting_view == "ball" and step_delta_ball is None:
        raise ValueError(
            "Ball accounting requires cfg.lz to be supplied explicitly. "
            "Automatic L_z derivation has been removed."
        )

    regime = _regime_summary(cfg.lz, clip_schedule, step_delta_ball, step_delta_std)
    if regime["ball_equals_standard_at_all_steps"] and bool(
        getattr(cfg, "warn_if_ball_equals_standard", True)
    ):
        crit = regime["critical_radius_for_min_clip"]
        crit_txt = "unknown" if crit is None else f"{float(crit):.6g}"
        print(
            "[ball_dp warning] Ball sensitivity saturates at the standard clipped sensitivity for the whole schedule: "
            "min(L_z * r, 2 C_t) = 2 C_t at every step. "
            f"critical_radius_for_min_clip={crit_txt}, configured_radius={float(cfg.radius):.6g}. "
            "So Ball accounting gives no advantage over standard replacement-adjacent accounting for this run.",
            flush=True,
        )

    params, static = partition_model(model)
    opt_state = optimizer.init(params)

    built_regularizer = _build_parameter_regularizer(
        cfg,
        parameter_regularizer=parameter_regularizer,
    )
    per_example_grad_fn = make_per_example_grad_fn(
        static=static,
        state=state,
        loss_fn=resolved_loss_fn,
    )
    regularizer_grad_fn = make_parameter_regularizer_grad_fn(
        static=static,
        state=state,
        parameter_regularizer=built_regularizer,
    )
    train_step = _make_train_step(
        optimizer=optimizer,
        per_example_grad_fn=per_example_grad_fn,
        regularizer_grad_fn=regularizer_grad_fn,
        normalize_noisy_sum_by=str(cfg.normalize_noisy_sum_by),
        param_projector=param_projector,
    )

    eval_predict_fn = None if predict_fn is None else predict_fn
    eval_batch_size = int(getattr(cfg, "eval_batch_size", 1024))
    eval_every = max(1, int(getattr(cfg, "eval_every", 250)))
    record_operator_norms = bool(getattr(cfg, "record_operator_norms", False))
    operator_norms_every = max(1, int(getattr(cfg, "operator_norms_every", eval_every)))
    operator_norm_kwargs = dict(getattr(cfg, "operator_norm_kwargs", {}) or {})
    store_full_operator_norm_history = bool(
        getattr(cfg, "store_full_operator_norm_history", False)
    )

    checkpoint_selection = str(getattr(cfg, "checkpoint_selection", "last")).lower()
    if checkpoint_selection not in {
        "last",
        "best_public_eval_loss",
        "best_public_eval_accuracy",
    }:
        raise ValueError(
            "cfg.checkpoint_selection must be one of "
            "{'last', 'best_public_eval_loss', 'best_public_eval_accuracy'}."
        )
    if checkpoint_selection != "last" and public_eval_dataset is None:
        raise ValueError(
            "Public-eval checkpoint selection requires public_eval_dataset=... ."
        )

    x_train = jnp.asarray(dataset.X, dtype=jnp.float32)
    y_train = jnp.asarray(dataset.y)

    public_curve_history: list[dict[str, Any]] = []
    operator_norm_history: list[dict[str, Any]] = []
    debug_history: list[dict[str, Any]] = []

    best_params = params
    if checkpoint_selection == "best_public_eval_accuracy":
        best_score = -float("inf")
    else:
        best_score = float("inf")
    best_step = 0

    train_key = key
    for t in range(T):
        train_key, sample_key, step_key = jr.split(train_key, 3)

        m_t = int(batch_schedule[t])
        idx = jr.choice(sample_key, n_total, shape=(m_t,), replace=False)
        xb = x_train[idx]
        yb = y_train[idx]

        model_before_step = (
            eqx.combine(params, static) if trace_recorder is not None else None
        )

        params, opt_state, mean_norm_j, max_norm_j, clip_frac_j, sanitized_grad = (
            train_step(
                params,
                opt_state,
                xb,
                yb,
                step_key,
                jnp.asarray(clip_schedule[t], dtype=jnp.float32),
                jnp.asarray(noise_multiplier_schedule[t], dtype=jnp.float32),
            )
        )

        if trace_recorder is not None:
            # `sanitized_grad` is what gets stored in the trace.
            # If we normalize the noisy sum by batch size, the stored Gaussian noise
            # standard deviation must be scaled the same way.
            observed_noise_std = float(effective_noise_stds[t])
            if str(cfg.normalize_noisy_sum_by) == "batch_size":
                observed_noise_std /= float(m_t)
            elif str(cfg.normalize_noisy_sum_by) == "none":
                observed_noise_std = float(effective_noise_stds[t])
            else:
                raise ValueError(
                    "normalize_noisy_sum_by must be one of {'batch_size', 'none'}."
                )

            trace_recorder(
                step=int(t + 1),
                model_before=model_before_step,
                observed_private_gradient=sanitized_grad,
                batch_indices=np.asarray(idx),
                clip_norm=float(clip_schedule[t]),
                noise_multiplier=float(noise_multiplier_schedule[t]),
                effective_noise_std=float(observed_noise_std),
            )

        current_model = eqx.combine(params, static)

        if return_debug_history:
            debug_row = {
                "step": int(t + 1),
                "batch_size": int(m_t),
                "clip_norm": float(clip_schedule[t]),
                "noise_multiplier": float(noise_multiplier_schedule[t]),
                "effective_noise_std": float(effective_noise_stds[t]),
                "mean_per_example_grad_norm": float(mean_norm_j),
                "max_per_example_grad_norm": float(max_norm_j),
                "clip_fraction": float(clip_frac_j),
            }
            debug_history.append(debug_row)

        should_eval_public = (t % eval_every == 0) or (t == T - 1)
        should_record_norms = record_operator_norms and (
            (t % operator_norms_every == 0) or (t == T - 1)
        )

        public_row: dict[str, Any] | None = None
        if should_eval_public:
            public_row = {
                "step": int(t + 1),
                "batch_size": int(m_t),
                "clip_norm": float(clip_schedule[t]),
                "noise_multiplier": float(noise_multiplier_schedule[t]),
                "effective_noise_std": float(effective_noise_stds[t]),
            }
            if public_eval_dataset is not None:
                train_key, eval_key = jr.split(train_key)
                eval_loss, eval_acc = _dataset_loss_and_accuracy(
                    model=current_model,
                    state=state,
                    dataset=public_eval_dataset,
                    loss_fn=resolved_loss_fn,
                    predict_fn=eval_predict_fn,
                    cfg=cfg,
                    batch_size=eval_batch_size,
                    key=eval_key,
                )
                public_row["public_eval_loss"] = float(eval_loss)
                if eval_acc is not None:
                    public_row["public_eval_accuracy"] = float(eval_acc)

                if checkpoint_selection == "best_public_eval_loss":
                    score = float(eval_loss)
                    if score < best_score:
                        best_score = score
                        best_step = int(t + 1)
                        best_params = params
                elif checkpoint_selection == "best_public_eval_accuracy":
                    if eval_acc is None:
                        raise ValueError(
                            "best_public_eval_accuracy requires predict_fn to produce logits."
                        )
                    score = float(eval_acc)
                    if score > best_score:
                        best_score = score
                        best_step = int(t + 1)
                        best_params = params

            public_curve_history.append(public_row)

        if should_record_norms:
            rows = collect_operator_norms(current_model, **operator_norm_kwargs)
            if rows:
                entry = {
                    "step": int(t + 1),
                    "summary": summarize_operator_norms(rows),
                }
                if store_full_operator_norm_history:
                    entry["layers"] = rows
                operator_norm_history.append(entry)
                if public_row is not None:
                    public_row["operator_norm_summary"] = dict(entry["summary"])

        if public_row is not None:
            print(
                "step "
                f"{t + 1}/{T} | "
                f"clip={clip_schedule[t]:.4g} | "
                f"noise_multiplier={noise_multiplier_schedule[t]:.4g}",
                flush=True,
            )

    if checkpoint_selection == "last":
        final_params = params
        best_step = int(T)
    else:
        final_params = best_params

    final_model = eqx.combine(final_params, static)

    utility_metrics: dict[str, float] = {}
    if public_eval_dataset is not None:
        train_key, final_eval_key = jr.split(train_key)
        final_eval_loss, final_eval_acc = _dataset_loss_and_accuracy(
            model=final_model,
            state=state,
            dataset=public_eval_dataset,
            loss_fn=resolved_loss_fn,
            predict_fn=eval_predict_fn,
            cfg=cfg,
            batch_size=eval_batch_size,
            key=final_eval_key,
        )
        utility_metrics["public_eval_loss"] = float(final_eval_loss)
        if final_eval_acc is not None:
            utility_metrics["accuracy"] = float(final_eval_acc)
            utility_metrics["public_eval_accuracy"] = float(final_eval_acc)

    if force_noiseless:
        dual_ledger = _empty_dual_ledger(float(cfg.radius))
    else:
        dual_ledger = build_ball_sgd_rdp_ledgers(
            orders=cfg.orders,
            step_batch_sizes=batch_schedule,
            dataset_size=n_total,
            step_clip_norms=clip_schedule,
            step_noise_stds=effective_noise_stds,
            step_delta_ball=step_delta_ball,
            step_delta_std=step_delta_std,
            radius=float(cfg.radius),
            dp_delta=cfg.delta,
        )

    artifact = _make_release_artifact(
        cfg=cfg,
        model=final_model,
        state=state,
        optimizer=optimizer,
        dual_ledger=dual_ledger,
        lz_source="provided" if cfg.lz is not None else "none",
        dataset=dataset,
        public_eval_dataset=public_eval_dataset,
        step_delta_ball=step_delta_ball,
        step_delta_std=step_delta_std,
        batch_schedule=batch_schedule,
        clip_schedule=clip_schedule,
        noise_multiplier_schedule=noise_multiplier_schedule,
        effective_noise_stds=effective_noise_stds,
        public_curve_history=public_curve_history,
        operator_norm_history=operator_norm_history,
        utility_metrics=utility_metrics,
        checkpoint_selection=checkpoint_selection,
        selected_step=best_step,
    )

    if accounting_view == "standard":
        artifact = _set_primary_privacy_view(
            artifact,
            release_kind="nonconvex_standard_sgd_rdp",
            primary_view="standard",
        )
    elif accounting_view == "none":
        artifact = _set_primary_privacy_view(
            artifact,
            release_kind="nonconvex_noiseless_sgd",
            primary_view="none",
        )
    else:
        artifact = _set_primary_privacy_view(
            artifact,
            release_kind="nonconvex_ball_sgd_rdp",
            primary_view="ball",
        )

    if return_debug_history:
        return artifact, {
            "step_history": debug_history,
            "warning": (
                "This debug history contains raw train-time statistics computed directly "
                "from private per-example gradients. Treat it as internal-only and do not "
                "publish it without separate privatization/accounting."
            ),
        }

    return artifact


def run_ball_sgd_rdp(
    dataset: ArrayDataset,
    cfg: NonconvexReleaseConfig,
    *,
    model: Any,
    optimizer: optax.GradientTransformation,
    state: Any = None,
    public_eval_dataset: Optional[ArrayDataset] = None,
    loss_fn: Optional[ExampleLossFn] = None,
    predict_fn: Optional[PredictFn] = default_predict_fn,
    parameter_regularizer: Optional[Callable[[Any, Any], jnp.ndarray]] = None,
    key: Optional[jax.Array] = None,
    return_debug_history: bool = False,
    trace_recorder=None,
    param_projector: Optional[Callable[[Any], Any]] = None,
):
    return _run_training(
        dataset,
        cfg,
        model=model,
        optimizer=optimizer,
        state=state,
        public_eval_dataset=public_eval_dataset,
        loss_fn=loss_fn,
        predict_fn=predict_fn,
        parameter_regularizer=parameter_regularizer,
        key=key,
        accounting_view="ball",
        force_noiseless=False,
        return_debug_history=return_debug_history,
        trace_recorder=trace_recorder,
        param_projector=param_projector,
    )


def run_ball_sgd_dp(
    dataset: ArrayDataset,
    cfg: NonconvexReleaseConfig,
    *,
    model: Any,
    optimizer: optax.GradientTransformation,
    state: Any = None,
    public_eval_dataset: Optional[ArrayDataset] = None,
    loss_fn: Optional[ExampleLossFn] = None,
    predict_fn: Optional[PredictFn] = default_predict_fn,
    parameter_regularizer: Optional[Callable[[Any, Any], jnp.ndarray]] = None,
    key: Optional[jax.Array] = None,
    return_debug_history: bool = False,
    trace_recorder=None,
    param_projector: Optional[Callable[[Any], Any]] = None,
):
    if cfg.epsilon is None or cfg.delta is None:
        raise ValueError("Ball-SGD-DP requires both cfg.epsilon and cfg.delta.")

    out = run_ball_sgd_rdp(
        dataset,
        cfg,
        model=model,
        optimizer=optimizer,
        state=state,
        public_eval_dataset=public_eval_dataset,
        loss_fn=loss_fn,
        predict_fn=predict_fn,
        parameter_regularizer=parameter_regularizer,
        key=key,
        return_debug_history=return_debug_history,
        trace_recorder=trace_recorder,
        param_projector=param_projector,
    )
    if return_debug_history:
        artifact, debug = out
    else:
        artifact = out

    artifact = _set_primary_privacy_view(
        artifact,
        release_kind="nonconvex_ball_sgd_dp",
        primary_view="ball",
    )
    artifact = _verify_dp_target(
        artifact,
        target_epsilon=float(cfg.epsilon),
        target_delta=float(cfg.delta),
        accounting_view="ball",
    )

    if return_debug_history:
        return artifact, debug
    return artifact


def run_standard_sgd_rdp(
    dataset: ArrayDataset,
    cfg: NonconvexReleaseConfig,
    *,
    model: Any,
    optimizer: optax.GradientTransformation,
    state: Any = None,
    public_eval_dataset: Optional[ArrayDataset] = None,
    loss_fn: Optional[ExampleLossFn] = None,
    predict_fn: Optional[PredictFn] = default_predict_fn,
    parameter_regularizer: Optional[Callable[[Any, Any], jnp.ndarray]] = None,
    key: Optional[jax.Array] = None,
    return_debug_history: bool = False,
    trace_recorder=None,
    param_projector: Optional[Callable[[Any], Any]] = None,
):
    return _run_training(
        dataset,
        cfg,
        model=model,
        optimizer=optimizer,
        state=state,
        public_eval_dataset=public_eval_dataset,
        loss_fn=loss_fn,
        predict_fn=predict_fn,
        parameter_regularizer=parameter_regularizer,
        key=key,
        accounting_view="standard",
        force_noiseless=False,
        return_debug_history=return_debug_history,
        trace_recorder=trace_recorder,
        param_projector=param_projector,
    )


def run_standard_sgd_dp(
    dataset: ArrayDataset,
    cfg: NonconvexReleaseConfig,
    *,
    model: Any,
    optimizer: optax.GradientTransformation,
    state: Any = None,
    public_eval_dataset: Optional[ArrayDataset] = None,
    loss_fn: Optional[ExampleLossFn] = None,
    predict_fn: Optional[PredictFn] = default_predict_fn,
    parameter_regularizer: Optional[Callable[[Any, Any], jnp.ndarray]] = None,
    key: Optional[jax.Array] = None,
    return_debug_history: bool = False,
    trace_recorder=None,
    param_projector: Optional[Callable[[Any], Any]] = None,
):
    if cfg.epsilon is None or cfg.delta is None:
        raise ValueError("Standard DP-SGD requires both cfg.epsilon and cfg.delta.")

    out = run_standard_sgd_rdp(
        dataset,
        cfg,
        model=model,
        optimizer=optimizer,
        state=state,
        public_eval_dataset=public_eval_dataset,
        loss_fn=loss_fn,
        predict_fn=predict_fn,
        parameter_regularizer=parameter_regularizer,
        key=key,
        return_debug_history=return_debug_history,
        trace_recorder=trace_recorder,
        param_projector=param_projector,
    )
    if return_debug_history:
        artifact, debug = out
    else:
        artifact = out

    artifact = _set_primary_privacy_view(
        artifact,
        release_kind="nonconvex_standard_sgd_dp",
        primary_view="standard",
    )
    artifact = _verify_dp_target(
        artifact,
        target_epsilon=float(cfg.epsilon),
        target_delta=float(cfg.delta),
        accounting_view="standard",
    )

    if return_debug_history:
        return artifact, debug
    return artifact


def run_noiseless_sgd_release(
    dataset: ArrayDataset,
    cfg: NonconvexReleaseConfig,
    *,
    model: Any,
    optimizer: optax.GradientTransformation,
    state: Any = None,
    public_eval_dataset: Optional[ArrayDataset] = None,
    loss_fn: Optional[ExampleLossFn] = None,
    predict_fn: Optional[PredictFn] = default_predict_fn,
    parameter_regularizer: Optional[Callable[[Any, Any], jnp.ndarray]] = None,
    key: Optional[jax.Array] = None,
    return_debug_history: bool = False,
    trace_recorder=None,
    param_projector: Optional[Callable[[Any], Any]] = None,
):
    return _run_training(
        dataset,
        cfg,
        model=model,
        optimizer=optimizer,
        state=state,
        public_eval_dataset=public_eval_dataset,
        loss_fn=loss_fn,
        predict_fn=predict_fn,
        parameter_regularizer=parameter_regularizer,
        key=key,
        accounting_view="none",
        force_noiseless=True,
        return_debug_history=return_debug_history,
        trace_recorder=trace_recorder,
        param_projector=param_projector,
    )


# Backward-compatible aliases.
run_nonconvex_ball_sgd_rdp = run_ball_sgd_rdp
run_nonconvex_ball_sgd_dp = run_ball_sgd_dp
run_nonconvex_standard_sgd_rdp = run_standard_sgd_rdp
run_nonconvex_standard_sgd_dp = run_standard_sgd_dp
run_nonconvex_noiseless_sgd_release = run_noiseless_sgd_release
