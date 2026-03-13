# quantbayes/ball_dp/attacks/gradient_based.py

from __future__ import annotations

import dataclasses as dc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from ..metrics import reconstruction_metrics
from ..nonconvex.per_example import (
    ExampleLossFn,
    combine_model,
    partition_model,
    resolve_loss_fn,
)
from ..types import ArrayDataset, AttackResult, Record

BatchStatRegularizer = Callable[[Any, Any, jnp.ndarray, jax.Array], jnp.ndarray]


@dc.dataclass
class GradientObservation:
    """Observed gradient for a gradient-inversion attack.

    Notes
    -----
    This object is intentionally agnostic about how the gradient was obtained. The caller is
    responsible for ensuring that ``gradient`` corresponds to the attack objective they want to
    invert, e.g. a single-example gradient, or a single-target equivalent residual after subtracting
    all known batch contributions.
    """

    model: Any
    gradient: Any
    state: Any = None
    batch_size: int = 1
    reduction: Literal["mean", "sum"] = "mean"
    metadata: Dict[str, Any] = dc.field(default_factory=dict)


@dc.dataclass
class DPSGDTraceStep:
    step: int
    model_before: Any
    observed_private_gradient: Any
    batch_indices: np.ndarray
    clip_norm: float
    noise_multiplier: float
    effective_noise_std: float


@dc.dataclass
class DPSGDTrace:
    steps: List[DPSGDTraceStep]
    state: Any = None
    loss_name: str = "softmax_cross_entropy"
    reduction: Literal["mean", "sum"] = "mean"
    metadata: Dict[str, Any] = dc.field(default_factory=dict)


class DPSGDTraceRecorder:
    """Optional recorder for sanitized DP-SGD per-step gradients.

    The final model is post-processing of the full transcript of privatized updates. Recording and
    keeping this transcript changes the adversary's observation model, but not the accounting target
    if your accountant already treats the full transcript as the mechanism output.
    """

    def __init__(
        self,
        *,
        capture_every: int = 1,
        keep_models: bool = True,
        keep_batch_indices: bool = True,
    ):
        self.capture_every = max(1, int(capture_every))
        self.keep_models = bool(keep_models)
        self.keep_batch_indices = bool(keep_batch_indices)
        self.steps: list[DPSGDTraceStep] = []

    def __call__(
        self,
        *,
        step: int,
        model_before: Any,
        observed_private_gradient: Any,
        batch_indices: np.ndarray,
        clip_norm: float,
        noise_multiplier: float,
        effective_noise_std: float,
    ) -> None:
        if int(step) % self.capture_every != 0:
            return
        self.steps.append(
            DPSGDTraceStep(
                step=int(step),
                model_before=model_before if self.keep_models else None,
                observed_private_gradient=observed_private_gradient,
                batch_indices=(
                    np.asarray(batch_indices, dtype=np.int64).copy()
                    if self.keep_batch_indices
                    else np.zeros((0,), dtype=np.int64)
                ),
                clip_norm=float(clip_norm),
                noise_multiplier=float(noise_multiplier),
                effective_noise_std=float(effective_noise_std),
            )
        )

    def to_trace(
        self,
        *,
        state: Any = None,
        loss_name: str = "softmax_cross_entropy",
        reduction: Literal["mean", "sum"] = "mean",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DPSGDTrace:
        return DPSGDTrace(
            steps=list(self.steps),
            state=state,
            loss_name=str(loss_name),
            reduction=str(reduction),
            metadata={} if metadata is None else dict(metadata),
        )


def _tree_leaves_strict(tree: Any) -> list[Any]:
    return list(jax.tree_util.tree_leaves(tree))


def _assert_same_tree_structure(a: Any, b: Any) -> None:
    sa = jax.tree_util.tree_structure(a)
    sb = jax.tree_util.tree_structure(b)
    if sa != sb:
        raise ValueError("Tree structures do not match.")


def _tree_add(a: Any, b: Any) -> Any:
    _assert_same_tree_structure(a, b)
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_sub(a: Any, b: Any) -> Any:
    _assert_same_tree_structure(a, b)
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def _tree_scalar_mul(tree: Any, scalar: float | jnp.ndarray) -> Any:
    scalar = jnp.asarray(scalar)
    return jax.tree_util.tree_map(
        lambda x: x * scalar.astype(jnp.asarray(x).dtype), tree
    )


def _tree_zeros_like(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_dot(a: Any, b: Any) -> jnp.ndarray:
    _assert_same_tree_structure(a, b)
    out = jnp.asarray(0.0, dtype=jnp.float32)
    for x, y in zip(_tree_leaves_strict(a), _tree_leaves_strict(b)):
        x = jnp.asarray(x).reshape(-1)
        y = jnp.asarray(y).reshape(-1)
        out = out + jnp.sum(x * y)
    return out


def _tree_l2_sq(tree: Any) -> jnp.ndarray:
    out = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in _tree_leaves_strict(tree):
        arr = jnp.asarray(leaf)
        out = out + jnp.sum(arr * arr)
    return out


def _tree_sq_distance(a: Any, b: Any) -> jnp.ndarray:
    return _tree_l2_sq(_tree_sub(a, b))


def _tree_batch_l2_norms(tree: Any) -> jnp.ndarray:
    leaves = _tree_leaves_strict(tree)
    if not leaves:
        return jnp.zeros((0,), dtype=jnp.float32)
    sq = None
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        flat = arr.reshape((arr.shape[0], -1))
        term = jnp.sum(flat * flat, axis=1)
        sq = term if sq is None else (sq + term)
    return jnp.sqrt(jnp.maximum(sq, jnp.asarray(0.0, dtype=sq.dtype)))


def _clip_single_grad(grad_tree: Any, clip_norm: float) -> Any:
    c = float(clip_norm)
    if not np.isfinite(c):
        return grad_tree
    norm = jnp.sqrt(_tree_l2_sq(grad_tree))
    scale = jnp.minimum(
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(c, dtype=jnp.float32)
        / jnp.maximum(norm, jnp.asarray(1e-12, dtype=jnp.float32)),
    )
    return _tree_scalar_mul(grad_tree, scale)


def _clip_per_example_grad_tree(
    per_example_grads: Any, clip_norm: float
) -> Tuple[Any, jnp.ndarray]:
    norms = _tree_batch_l2_norms(per_example_grads)
    c = float(clip_norm)
    if not np.isfinite(c):
        return per_example_grads, norms
    scale = jnp.minimum(
        jnp.asarray(1.0, dtype=norms.dtype),
        jnp.asarray(c, dtype=norms.dtype)
        / jnp.maximum(norms, jnp.asarray(1e-12, dtype=norms.dtype)),
    )

    def _scale_leaf(g):
        arr = jnp.asarray(g)
        shape = (arr.shape[0],) + (1,) * (arr.ndim - 1)
        return arr * scale.reshape(shape).astype(arr.dtype)

    return jax.tree_util.tree_map(_scale_leaf, per_example_grads), norms


def _tree_batch_dot(batch_tree: Any, tree: Any) -> jnp.ndarray:
    out = None
    _assert_same_tree_structure(
        batch_tree, jax.tree_util.tree_map(lambda x: x[None, ...], tree)
    )
    for g_leaf, r_leaf in zip(
        _tree_leaves_strict(batch_tree), _tree_leaves_strict(tree)
    ):
        g_arr = jnp.asarray(g_leaf).reshape((g_leaf.shape[0], -1))
        r_arr = jnp.asarray(r_leaf).reshape(-1)
        term = jnp.sum(g_arr * r_arr[None, :], axis=1)
        out = term if out is None else (out + term)
    return jnp.zeros((0,), dtype=jnp.float32) if out is None else out


def _tree_batch_self_dot(batch_tree: Any) -> jnp.ndarray:
    out = None
    for g_leaf in _tree_leaves_strict(batch_tree):
        g_arr = jnp.asarray(g_leaf).reshape((g_leaf.shape[0], -1))
        term = jnp.sum(g_arr * g_arr, axis=1)
        out = term if out is None else (out + term)
    return jnp.zeros((0,), dtype=jnp.float32) if out is None else out


def _tree_batch_sq_distance(batch_tree: Any, tree: Any) -> jnp.ndarray:
    self_dot = _tree_batch_self_dot(batch_tree)
    cross = _tree_batch_dot(batch_tree, tree)
    tree_dot = _tree_l2_sq(tree)
    return jnp.maximum(
        self_dot - 2.0 * cross + tree_dot, jnp.asarray(0.0, dtype=self_dot.dtype)
    )


def _single_example_grad(
    model: Any,
    state: Any,
    loss_fn: ExampleLossFn,
    x: jnp.ndarray,
    y: jnp.ndarray,
    key: jax.Array,
) -> Any:
    params, static = partition_model(model)

    def _loss_of_params(params: Any) -> jnp.ndarray:
        mdl = combine_model(params, static)
        return loss_fn(mdl, state, x, y, key)

    return jax.grad(_loss_of_params)(params)


def _per_candidate_grads(
    model: Any,
    state: Any,
    loss_fn: ExampleLossFn,
    xb: jnp.ndarray,
    yb: jnp.ndarray,
    key: jax.Array,
) -> Any:
    params, static = partition_model(model)

    def _loss_of_params(
        params: Any, x: jnp.ndarray, y: jnp.ndarray, key: jax.Array
    ) -> jnp.ndarray:
        mdl = combine_model(params, static)
        return loss_fn(mdl, state, x, y, key)

    grad_one = jax.grad(_loss_of_params)
    keys = jr.split(key, xb.shape[0])
    return jax.vmap(lambda x, y, k: grad_one(params, x, y, k), in_axes=(0, 0, 0))(
        xb, yb, keys
    )


def _as_feature_shape(
    *,
    feature_shape: Optional[Sequence[int]],
    true_record: Optional[Record],
    metadata: Optional[Dict[str, Any]],
) -> Tuple[int, ...]:
    if feature_shape is not None:
        return tuple(int(v) for v in feature_shape)
    if true_record is not None:
        return tuple(np.asarray(true_record.features).shape)
    if metadata is not None and "feature_shape" in metadata:
        return tuple(int(v) for v in metadata["feature_shape"])
    raise ValueError(
        "feature_shape must be provided unless true_record or observation.metadata['feature_shape'] is available."
    )


def _project_guess(
    x: jnp.ndarray,
    *,
    box_bounds: Optional[Tuple[float, float]],
    ball_center: Optional[np.ndarray],
    ball_radius: Optional[float],
) -> jnp.ndarray:
    out = jnp.asarray(x)
    if box_bounds is not None:
        lo, hi = float(box_bounds[0]), float(box_bounds[1])
        out = jnp.clip(out, lo, hi)
    if ball_center is not None and ball_radius is not None:
        center = jnp.asarray(ball_center, dtype=out.dtype).reshape(out.shape)
        diff = out - center
        norm = jnp.linalg.norm(jnp.ravel(diff))
        radius = jnp.asarray(float(ball_radius), dtype=out.dtype)
        scale = jnp.minimum(
            jnp.asarray(1.0, dtype=out.dtype),
            radius / jnp.maximum(norm, jnp.asarray(1e-12, dtype=out.dtype)),
        )
        out = center + diff * scale
    return out


def _total_variation(x: jnp.ndarray) -> jnp.ndarray:
    arr = jnp.asarray(x)
    if arr.ndim == 1:
        return (
            jnp.mean(jnp.abs(arr[1:] - arr[:-1]))
            if arr.size > 1
            else jnp.asarray(0.0, dtype=arr.dtype)
        )
    if arr.ndim == 2:
        dh = jnp.abs(arr[1:, :] - arr[:-1, :]).mean()
        dw = jnp.abs(arr[:, 1:] - arr[:, :-1]).mean()
        return dh + dw
    if arr.ndim == 3:
        if arr.shape[0] in {1, 3}:
            arr = jnp.transpose(arr, (1, 2, 0))
        dh = jnp.abs(arr[1:, :, :] - arr[:-1, :, :]).mean()
        dw = jnp.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean()
        return dh + dw
    return jnp.asarray(0.0, dtype=arr.dtype)


def _gradient_match_loss(
    candidate_grad: Any, observed_grad: Any, mode: str
) -> jnp.ndarray:
    key = str(mode).lower()
    if key == "dlg":
        return _tree_sq_distance(candidate_grad, observed_grad)
    if key == "geiping":
        dot = _tree_dot(candidate_grad, observed_grad)
        denom = jnp.sqrt(_tree_l2_sq(candidate_grad)) * jnp.sqrt(
            _tree_l2_sq(observed_grad)
        )
        return jnp.asarray(1.0, dtype=jnp.float32) - dot / jnp.maximum(
            denom, jnp.asarray(1e-12, dtype=jnp.float32)
        )
    if key == "gradinversion":
        return _tree_sq_distance(candidate_grad, observed_grad)
    raise ValueError(f"Unsupported gradient attack mode: {mode!r}")


def _default_init(
    shape: Sequence[int],
    *,
    rng: np.random.Generator,
    init: str,
    init_scale: float,
    box_bounds: Optional[Tuple[float, float]],
    ball_center: Optional[np.ndarray],
) -> np.ndarray:
    if str(init) == "center" and ball_center is not None:
        return np.asarray(ball_center, dtype=np.float32).reshape(shape)
    if str(init) == "uniform":
        if box_bounds is not None:
            lo, hi = float(box_bounds[0]), float(box_bounds[1])
            return rng.uniform(lo, hi, size=shape).astype(np.float32)
        return rng.uniform(-float(init_scale), float(init_scale), size=shape).astype(
            np.float32
        )
    return rng.normal(scale=float(init_scale), size=shape).astype(np.float32)


def run_gradient_attack(
    observation: GradientObservation,
    *,
    loss_name: str = "softmax_cross_entropy",
    loss_fn: Optional[ExampleLossFn] = None,
    mode: Literal["dlg", "geiping", "gradinversion"] = "dlg",
    feature_shape: Optional[Sequence[int]] = None,
    label_space: Optional[Sequence[int]] = None,
    known_label: Optional[int] = None,
    clip_norm: Optional[float] = None,
    num_steps: int = 2000,
    learning_rate: float = 1e-2,
    num_restarts: int = 5,
    tv_weight: float = 0.0,
    l2_weight: float = 0.0,
    batch_stat_weight: float = 0.0,
    batch_stat_regularizer: Optional[BatchStatRegularizer] = None,
    box_bounds: Optional[Tuple[float, float]] = None,
    ball_center: Optional[np.ndarray] = None,
    ball_radius: Optional[float] = None,
    init: Literal["uniform", "normal", "center"] = "uniform",
    init_scale: float = 1.0,
    true_record: Optional[Record] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
    seed: int = 0,
) -> AttackResult:
    """Run a single-record gradient inversion baseline.

    This is a baseline interface for DLG / Geiping / GradInversion-style attacks. It is appropriate
    when the observed gradient corresponds to a single target record, either directly or after offline
    subtraction of all known batch contributions.

    It is **not** a faithful implementation of exact large-batch inversion methods such as SPEAR, nor
    a full reproduction of the DP-SGD multi-step attack from Guo et al. (2023).
    """
    resolved_loss_fn = resolve_loss_fn(loss_name) if loss_fn is None else loss_fn
    x_shape = _as_feature_shape(
        feature_shape=feature_shape,
        true_record=true_record,
        metadata=observation.metadata,
    )
    labels = [int(known_label)] if known_label is not None else None
    if labels is None:
        if label_space is None:
            raise ValueError(
                "known_label or label_space must be provided for gradient attacks."
            )
        labels = [int(v) for v in label_space]

    rng = np.random.default_rng(int(seed))
    best_obj = float("inf")
    best_x = None
    best_label = None
    per_label_best: Dict[int, float] = {}

    observation_grad = observation.gradient
    batch_size = max(int(observation.batch_size), 1)
    reduction = str(observation.reduction)

    for label in labels:
        label_best = float("inf")
        for restart in range(max(1, int(num_restarts))):
            x0 = _default_init(
                x_shape,
                rng=rng,
                init=str(init),
                init_scale=float(init_scale),
                box_bounds=box_bounds,
                ball_center=ball_center,
            )
            x = jnp.asarray(x0, dtype=jnp.float32)
            opt = optax.adam(float(learning_rate))
            opt_state = opt.init(x)

            def _objective(x_var: jnp.ndarray) -> jnp.ndarray:
                attack_key = jr.PRNGKey(int(seed + 17 * restart + 131 * label))
                grad = _single_example_grad(
                    observation.model,
                    observation.state,
                    resolved_loss_fn,
                    x_var,
                    jnp.asarray(label),
                    attack_key,
                )
                if clip_norm is not None:
                    grad = _clip_single_grad(grad, float(clip_norm))
                if reduction == "mean":
                    grad = _tree_scalar_mul(grad, 1.0 / float(batch_size))
                loss = _gradient_match_loss(grad, observation_grad, mode)
                if float(tv_weight) != 0.0:
                    loss = loss + float(tv_weight) * _total_variation(x_var)
                if float(l2_weight) != 0.0:
                    center = (
                        jnp.zeros_like(x_var)
                        if ball_center is None
                        else jnp.asarray(ball_center, dtype=x_var.dtype).reshape(
                            x_var.shape
                        )
                    )
                    loss = loss + float(l2_weight) * jnp.mean(
                        jnp.square(x_var - center)
                    )
                if (
                    batch_stat_regularizer is not None
                    and float(batch_stat_weight) != 0.0
                ):
                    reg_key = jr.PRNGKey(int(seed + 23 * restart + 211 * label))
                    loss = loss + float(batch_stat_weight) * jnp.asarray(
                        batch_stat_regularizer(
                            observation.model, observation.state, x_var, reg_key
                        ),
                        dtype=jnp.float32,
                    )
                return loss

            value_and_grad = jax.value_and_grad(_objective)
            best_restart_obj = float("inf")
            best_restart_x = x
            for _ in range(max(1, int(num_steps))):
                loss_value, grad_x = value_and_grad(x)
                updates, opt_state = opt.update(grad_x, opt_state, x)
                x = optax.apply_updates(x, updates)
                x = _project_guess(
                    x,
                    box_bounds=box_bounds,
                    ball_center=ball_center,
                    ball_radius=ball_radius,
                )
                fv = float(loss_value)
                if fv < best_restart_obj:
                    best_restart_obj = fv
                    best_restart_x = x
            final_obj = float(_objective(best_restart_x))
            if final_obj < label_best:
                label_best = final_obj
            if final_obj < best_obj:
                best_obj = final_obj
                best_x = np.asarray(best_restart_x, dtype=np.float32)
                best_label = int(label)
        per_label_best[int(label)] = float(label_best)

    if best_x is None or best_label is None:
        raise RuntimeError(
            "Gradient attack failed to produce a candidate reconstruction."
        )

    pred_record = Record(
        features=np.asarray(best_x).reshape(x_shape), label=int(best_label)
    )
    metrics = (
        {}
        if true_record is None
        else reconstruction_metrics(true_record, pred_record, eta_grid=eta_grid)
    )

    return AttackResult(
        attack_family=f"gradient_{mode}",
        z_hat=np.asarray(pred_record.features),
        y_hat=int(pred_record.label),
        status="ok_known_label" if known_label is not None else "ok",
        diagnostics={
            "objective": float(best_obj),
            "per_label_best_objective": dict(per_label_best),
            "mode": str(mode),
            "num_steps": int(num_steps),
            "num_restarts": int(num_restarts),
            "clip_norm": None if clip_norm is None else float(clip_norm),
        },
        metrics=metrics,
    )


def run_dlg_attack(*args, **kwargs) -> AttackResult:
    kwargs = dict(kwargs)
    kwargs["mode"] = "dlg"
    return run_gradient_attack(*args, **kwargs)


def run_geiping_attack(*args, **kwargs) -> AttackResult:
    kwargs = dict(kwargs)
    kwargs["mode"] = "geiping"
    return run_gradient_attack(*args, **kwargs)


def run_gradinversion_attack(*args, **kwargs) -> AttackResult:
    kwargs = dict(kwargs)
    kwargs["mode"] = "gradinversion"
    return run_gradient_attack(*args, **kwargs)


def subtract_known_batch_gradients(
    trace: DPSGDTrace,
    dataset: ArrayDataset,
    *,
    target_index: int,
    loss_name: Optional[str] = None,
    loss_fn: Optional[ExampleLossFn] = None,
    seed: int = 0,
) -> DPSGDTrace:
    """Subtract all known non-target contributions from each privatized trace step.

    The returned trace still contains the Gaussian noise. When ``target_index`` is present in a batch,
    the residual is approximately the (clipped, optionally averaged) target contribution plus noise. When
    it is absent, the residual is pure noise.
    """
    if not trace.steps:
        return trace
    resolved_loss_fn = (
        resolve_loss_fn(trace.loss_name if loss_name is None else loss_name)
        if loss_fn is None
        else loss_fn
    )

    new_steps: list[DPSGDTraceStep] = []
    for step in trace.steps:
        if step.model_before is None:
            raise ValueError(
                "Trace step is missing model_before snapshots; cannot subtract known gradients offline."
            )
        batch_idx = np.asarray(step.batch_indices, dtype=np.int64)
        if batch_idx.size == 0:
            raise ValueError(
                "Trace step is missing batch_indices; cannot subtract known gradients offline."
            )
        known_idx = batch_idx[batch_idx != int(target_index)]
        if known_idx.size == 0:
            known_agg = _tree_zeros_like(step.observed_private_gradient)
        else:
            xb = jnp.asarray(np.asarray(dataset.X)[known_idx], dtype=jnp.float32)
            yb = jnp.asarray(np.asarray(dataset.y)[known_idx])
            grad_key = jr.PRNGKey(int(seed + 1009 * int(step.step)))
            per_example = _per_candidate_grads(
                step.model_before,
                trace.state,
                resolved_loss_fn,
                xb,
                yb,
                grad_key,
            )
            clipped, _ = _clip_per_example_grad_tree(per_example, float(step.clip_norm))
            known_agg = jax.tree_util.tree_map(lambda g: jnp.sum(g, axis=0), clipped)
            if str(trace.reduction) == "mean":
                known_agg = _tree_scalar_mul(known_agg, 1.0 / float(len(batch_idx)))
        residual = _tree_sub(step.observed_private_gradient, known_agg)
        new_steps.append(
            DPSGDTraceStep(
                step=int(step.step),
                model_before=step.model_before,
                observed_private_gradient=residual,
                batch_indices=batch_idx,
                clip_norm=float(step.clip_norm),
                noise_multiplier=float(step.noise_multiplier),
                effective_noise_std=float(step.effective_noise_std),
            )
        )
    new_metadata = dict(trace.metadata)
    new_metadata["residualized_against_known_batch"] = True
    new_metadata["target_index"] = int(target_index)
    return DPSGDTrace(
        steps=new_steps,
        state=trace.state,
        loss_name=trace.loss_name if loss_name is None else str(loss_name),
        reduction=str(trace.reduction),
        metadata=new_metadata,
    )


def run_prior_aware_trace_attack(
    trace: DPSGDTrace,
    prior_records: Sequence[Record],
    *,
    loss_name: Optional[str] = None,
    loss_fn: Optional[ExampleLossFn] = None,
    score_mode: Literal["sum", "top_q", "present_steps"] = "present_steps",
    sampling_probability: Optional[float] = None,
    normalize_by_self_norm: bool = False,
    known_label: Optional[int] = None,
    target_index: Optional[int] = None,
    true_record: Optional[Record] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
    seed: int = 0,
) -> AttackResult:
    """Rank a finite prior of candidate target points using a DP-SGD trace.

    This is a stronger and more principled version of the prior-aware trace ranker than a pure inner-product
    heuristic: the score is a per-step Gaussian log-likelihood surrogate based on the recorded effective noise
    standard deviation. When ``target_index`` is known, the default ``score_mode='present_steps'`` uses only
    the steps in which the target was actually sampled.

    It is still a *trace-ranking attack*, not a complete reproduction of every optimization detail from the
    Guo et al. (2023) attack. Treat it as a strong, production-friendly baseline rather than a claim of exact
    paper reproduction.
    """
    if not trace.steps:
        raise ValueError("Trace is empty.")
    filtered_prior = [
        r
        for r in prior_records
        if known_label is None or int(r.label) == int(known_label)
    ]
    if not filtered_prior:
        raise ValueError("Prior became empty after applying known_label filtering.")

    resolved_loss_fn = (
        resolve_loss_fn(trace.loss_name if loss_name is None else loss_name)
        if loss_fn is None
        else loss_fn
    )
    first_shape = np.asarray(filtered_prior[0].features).shape
    x_prior = jnp.asarray(
        np.stack(
            [
                np.asarray(r.features, dtype=np.float32).reshape(first_shape)
                for r in filtered_prior
            ],
            axis=0,
        ),
        dtype=jnp.float32,
    )
    y_prior = jnp.asarray(
        np.asarray([int(r.label) for r in filtered_prior]), dtype=jnp.int32
    )

    target_index = (
        int(target_index)
        if target_index is not None
        else trace.metadata.get("target_index", None)
    )

    per_step_scores: list[np.ndarray] = []
    used_step_mask: list[bool] = []
    for step in trace.steps:
        if step.model_before is None:
            raise ValueError(
                "Prior-aware attack requires model snapshots at every retained trace step."
            )

        target_present = True
        if target_index is not None and np.asarray(step.batch_indices).size > 0:
            target_present = bool(
                np.any(
                    np.asarray(step.batch_indices, dtype=np.int64) == int(target_index)
                )
            )
        if str(score_mode) == "present_steps" and not target_present:
            used_step_mask.append(False)
            continue

        grad_key = jr.PRNGKey(int(seed + 8191 * int(step.step)))
        per_example = _per_candidate_grads(
            step.model_before,
            trace.state,
            resolved_loss_fn,
            x_prior,
            y_prior,
            grad_key,
        )
        clipped, _ = _clip_per_example_grad_tree(per_example, float(step.clip_norm))
        if str(trace.reduction) == "mean":
            batch_size = max(1, int(len(step.batch_indices)))
            clipped = _tree_scalar_mul(clipped, 1.0 / float(batch_size))
            obs_noise_std = float(step.effective_noise_std) / float(batch_size)
        else:
            obs_noise_std = float(step.effective_noise_std)

        sq_dist = _tree_batch_sq_distance(clipped, step.observed_private_gradient)
        if bool(normalize_by_self_norm):
            denom = _tree_batch_self_dot(clipped)
            sq_dist = sq_dist / jnp.maximum(
                denom, jnp.asarray(1e-12, dtype=sq_dist.dtype)
            )
        sigma2 = float(obs_noise_std) ** 2
        if sigma2 > 0.0:
            step_scores = -0.5 * np.asarray(sq_dist, dtype=np.float32) / float(sigma2)
        else:
            step_scores = -np.asarray(sq_dist, dtype=np.float32)
        per_step_scores.append(step_scores)
        used_step_mask.append(True)

    if not per_step_scores:
        raise ValueError("No usable trace steps remained for scoring.")

    score_matrix = np.stack(per_step_scores, axis=1)
    if str(score_mode) in {"sum", "present_steps"}:
        final_scores = score_matrix.sum(axis=1)
        k = None
    else:
        q = sampling_probability
        if q is None:
            q = trace.metadata.get("sample_rate", None)
        if q is None:
            n_total = trace.metadata.get("dataset_size", None)
            if n_total is not None:
                avg_bs = float(
                    np.mean([max(1, len(s.batch_indices)) for s in trace.steps])
                )
                q = avg_bs / float(n_total)
        if q is None:
            raise ValueError(
                "score_mode='top_q' requires sampling_probability=... or trace.metadata['sample_rate'] / ['dataset_size']."
            )
        k = max(1, int(np.ceil(float(q) * score_matrix.shape[1])))
        part = np.partition(score_matrix, kth=max(score_matrix.shape[1] - k, 0), axis=1)
        final_scores = part[:, -k:].sum(axis=1)

    best_idx = int(np.argmax(final_scores))
    best_record = filtered_prior[best_idx]
    metrics = (
        {}
        if true_record is None
        else reconstruction_metrics(true_record, best_record, eta_grid=eta_grid)
    )

    top_order = np.argsort(-final_scores)[: min(10, len(final_scores))]
    candidates = [
        (
            int(filtered_prior[int(i)].label),
            np.asarray(filtered_prior[int(i)].features, dtype=np.float32),
        )
        for i in top_order.tolist()
    ]

    return AttackResult(
        attack_family="dpsgd_prior_aware_trace",
        z_hat=np.asarray(best_record.features, dtype=np.float32),
        y_hat=int(best_record.label),
        status="ok_known_label" if known_label is not None else "ok",
        diagnostics={
            "final_scores": np.asarray(final_scores, dtype=np.float32),
            "score_matrix": score_matrix,
            "score_mode": str(score_mode),
            "top_k_steps": k,
            "normalize_by_self_norm": bool(normalize_by_self_norm),
            "prior_size": int(len(filtered_prior)),
            "used_step_count": int(score_matrix.shape[1]),
            "target_index": None if target_index is None else int(target_index),
        },
        candidates=candidates,
        metrics=metrics,
    )
