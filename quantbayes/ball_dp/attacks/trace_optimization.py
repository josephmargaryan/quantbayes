# quantbayes/ball_dp/attacks/trace_optimization.py

from __future__ import annotations

import dataclasses as dc
from typing import Any, Dict, Optional, Sequence, Tuple, Literal

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
from .gradient_based import DPSGDTrace, subtract_known_batch_gradients


@dc.dataclass
class TraceOptimizationAttackConfig:
    """Configuration for continuous reconstruction from a DP-SGD trace.

    This attack targets the intermediate-gradient observation model:
    the adversary observes a trace of privatized gradients and model snapshots,
    knows D_- and optionally the target label.

    Parameters
    ----------
    loss_name:
        Built-in per-example loss name if loss_fn is not supplied.
    objective:
        'guo2023' implements the multi-step gradient-based objective inspired by
        Hayes/Mahloujifar/Balle (NeurIPS 2023):
            sum_t [ - <g_t(x), gbar_t> + ||g_t(x) - gbar_t||_1 ].
        'l2_match' is a simpler fallback baseline:
            sum_t ||g_t(x) - gbar_t||_2^2.
    step_mode:
        'all' uses every retained trace step.
        'present_steps' uses only steps where the target index is present in the batch.
        This is the strongest setting when the adversary knows membership of the target
        in each batch.
    num_steps:
        Number of optimization iterations per restart.
    learning_rate:
        Adam learning rate for optimizing the candidate input.
    num_restarts:
        Number of random restarts.
    tv_weight:
        Total-variation regularizer weight for image-like inputs.
    l2_weight:
        L2 regularizer weight around ball_center if supplied, else around 0.
    box_bounds:
        Optional coordinate-wise projection box.
    ball_center, ball_radius:
        Optional L2-ball projection.
    init:
        Initialization scheme for candidate input.
    init_scale:
        Scale used by 'normal' or box-free 'uniform' initialization.
    seed:
        PRNG seed.
    """

    loss_name: str = "softmax_cross_entropy"
    objective: Literal["guo2023", "l2_match"] = "guo2023"
    step_mode: Literal["all", "present_steps"] = "present_steps"
    num_steps: int = 4000
    learning_rate: float = 1e-2
    num_restarts: int = 5
    tv_weight: float = 0.0
    l2_weight: float = 0.0
    box_bounds: Optional[Tuple[float, float]] = None
    ball_center: Optional[np.ndarray] = None
    ball_radius: Optional[float] = None
    init: Literal["uniform", "normal", "center"] = "uniform"
    init_scale: float = 1.0
    seed: int = 0


def _tree_leaves(tree: Any) -> list[Any]:
    return list(jax.tree_util.tree_leaves(tree))


def _assert_same_tree(a: Any, b: Any) -> None:
    if jax.tree_util.tree_structure(a) != jax.tree_util.tree_structure(b):
        raise ValueError("Tree structures do not match.")


def _tree_sub(a: Any, b: Any) -> Any:
    _assert_same_tree(a, b)
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def _tree_scalar_mul(tree: Any, scalar: float | jnp.ndarray) -> Any:
    scalar = jnp.asarray(scalar)
    return jax.tree_util.tree_map(
        lambda x: x * scalar.astype(jnp.asarray(x).dtype), tree
    )


def _tree_dot(a: Any, b: Any) -> jnp.ndarray:
    _assert_same_tree(a, b)
    out = jnp.asarray(0.0, dtype=jnp.float32)
    for x, y in zip(_tree_leaves(a), _tree_leaves(b)):
        x = jnp.asarray(x).reshape(-1)
        y = jnp.asarray(y).reshape(-1)
        out = out + jnp.sum(x * y)
    return out


def _tree_l2_sq(tree: Any) -> jnp.ndarray:
    out = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in _tree_leaves(tree):
        arr = jnp.asarray(leaf)
        out = out + jnp.sum(arr * arr)
    return out


def _tree_l1_distance(a: Any, b: Any) -> jnp.ndarray:
    _assert_same_tree(a, b)
    out = jnp.asarray(0.0, dtype=jnp.float32)
    for x, y in zip(_tree_leaves(a), _tree_leaves(b)):
        out = out + jnp.sum(jnp.abs(jnp.asarray(x) - jnp.asarray(y)))
    return out


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


def _as_feature_shape(
    *,
    feature_shape: Optional[Sequence[int]],
    true_record: Optional[Record],
    trace: DPSGDTrace,
) -> Tuple[int, ...]:
    if feature_shape is not None:
        return tuple(int(v) for v in feature_shape)
    if true_record is not None:
        return tuple(np.asarray(true_record.features).shape)
    if "feature_shape" in trace.metadata:
        return tuple(int(v) for v in trace.metadata["feature_shape"])
    raise ValueError(
        "feature_shape must be provided unless true_record or trace.metadata['feature_shape'] is available."
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


def _select_steps(
    trace: DPSGDTrace,
    *,
    step_mode: str,
    target_index: Optional[int],
) -> list[Any]:
    steps = list(trace.steps)
    if not steps:
        raise ValueError("Trace is empty.")

    mode = str(step_mode).lower()
    if mode == "all":
        return steps

    if mode == "present_steps":
        if target_index is None:
            raise ValueError(
                "step_mode='present_steps' requires target_index to be known."
            )
        selected = []
        for step in steps:
            batch_idx = np.asarray(step.batch_indices, dtype=np.int64)
            if batch_idx.size == 0:
                raise ValueError(
                    "Trace step is missing batch indices; cannot determine target presence."
                )
            if np.any(batch_idx == int(target_index)):
                selected.append(step)
        if not selected:
            raise ValueError("No retained trace step contains the target index.")
        return selected

    raise ValueError("step_mode must be one of {'all', 'present_steps'}.")


def run_trace_optimization_attack(
    trace: DPSGDTrace,
    *,
    cfg: Optional[TraceOptimizationAttackConfig] = None,
    dataset: Optional[ArrayDataset] = None,
    target_index: Optional[int] = None,
    loss_fn: Optional[ExampleLossFn] = None,
    feature_shape: Optional[Sequence[int]] = None,
    known_label: Optional[int] = None,
    label_space: Optional[Sequence[int]] = None,
    true_record: Optional[Record] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
) -> AttackResult:
    """Optimize a candidate record directly against a privatized DP-SGD trace.

    This attack targets the intermediate-gradient threat model.

    If `dataset` and `target_index` are supplied, the function first subtracts all
    known non-target contributions from each privatized step. The remaining trace
    then approximates the clipped target contribution plus Gaussian noise on steps
    where the target was sampled, and pure noise elsewhere.

    Notes
    -----
    - `objective='guo2023'` is the recommended default and mirrors the multi-step
      gradient-based attack style from Hayes/Mahloujifar/Balle (NeurIPS 2023).
    - `step_mode='present_steps'` is the strongest setting when batch membership of
      the target is known.
    - This attack is intended as the stable flagship attack for trace access, while
      DLG / Geiping / GradInversion remain gradient-baseline attacks.
    """
    cfg = TraceOptimizationAttackConfig() if cfg is None else cfg
    resolved_loss_fn = resolve_loss_fn(cfg.loss_name) if loss_fn is None else loss_fn

    work_trace = trace
    if dataset is not None and target_index is not None:
        work_trace = subtract_known_batch_gradients(
            trace,
            dataset,
            target_index=int(target_index),
            loss_name=cfg.loss_name,
            loss_fn=resolved_loss_fn,
            seed=int(cfg.seed),
        )

    if target_index is None:
        target_index = work_trace.metadata.get("target_index", None)
    target_index = None if target_index is None else int(target_index)

    x_shape = _as_feature_shape(
        feature_shape=feature_shape,
        true_record=true_record,
        trace=work_trace,
    )

    labels = [int(known_label)] if known_label is not None else None
    if labels is None:
        if label_space is None:
            raise ValueError(
                "known_label or label_space must be provided for trace optimization."
            )
        labels = [int(v) for v in label_space]

    selected_steps = _select_steps(
        work_trace,
        step_mode=str(cfg.step_mode),
        target_index=target_index,
    )

    rng = np.random.default_rng(int(cfg.seed))
    best_obj = float("inf")
    best_x = None
    best_label = None
    per_label_best: Dict[int, float] = {}

    for label in labels:
        label_best = float("inf")
        for restart in range(max(1, int(cfg.num_restarts))):
            x0 = _default_init(
                x_shape,
                rng=rng,
                init=str(cfg.init),
                init_scale=float(cfg.init_scale),
                box_bounds=cfg.box_bounds,
                ball_center=cfg.ball_center,
            )
            x = jnp.asarray(x0, dtype=jnp.float32)
            opt = optax.adam(float(cfg.learning_rate))
            opt_state = opt.init(x)

            def _objective(x_var: jnp.ndarray) -> jnp.ndarray:
                total = jnp.asarray(0.0, dtype=jnp.float32)
                for step in selected_steps:
                    step_key = jr.PRNGKey(
                        int(cfg.seed + 10007 * int(step.step) + 131 * int(label))
                    )
                    cand = _single_example_grad(
                        step.model_before,
                        work_trace.state,
                        resolved_loss_fn,
                        x_var,
                        jnp.asarray(label),
                        step_key,
                    )
                    cand = _clip_single_grad(cand, float(step.clip_norm))

                    if str(work_trace.reduction) == "mean":
                        batch_size = max(1, int(len(step.batch_indices)))
                        cand = _tree_scalar_mul(cand, 1.0 / float(batch_size))

                    obs = step.observed_private_gradient

                    if str(cfg.objective).lower() == "guo2023":
                        total = total + (
                            -_tree_dot(cand, obs) + _tree_l1_distance(cand, obs)
                        )
                    elif str(cfg.objective).lower() == "l2_match":
                        total = total + _tree_l2_sq(_tree_sub(cand, obs))
                    else:
                        raise ValueError(
                            "objective must be one of {'guo2023', 'l2_match'}."
                        )

                total = total / float(len(selected_steps))

                if float(cfg.tv_weight) != 0.0:
                    total = total + float(cfg.tv_weight) * _total_variation(x_var)

                if float(cfg.l2_weight) != 0.0:
                    center = (
                        jnp.zeros_like(x_var)
                        if cfg.ball_center is None
                        else jnp.asarray(cfg.ball_center, dtype=x_var.dtype).reshape(
                            x_var.shape
                        )
                    )
                    total = total + float(cfg.l2_weight) * jnp.mean(
                        jnp.square(x_var - center)
                    )

                return total

            value_and_grad = jax.value_and_grad(_objective)
            best_restart_obj = float("inf")
            best_restart_x = x

            for _ in range(max(1, int(cfg.num_steps))):
                loss_value, grad_x = value_and_grad(x)
                updates, opt_state = opt.update(grad_x, opt_state, x)
                x = optax.apply_updates(x, updates)
                x = _project_guess(
                    x,
                    box_bounds=cfg.box_bounds,
                    ball_center=cfg.ball_center,
                    ball_radius=cfg.ball_radius,
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
            "Trace optimization attack failed to produce a candidate reconstruction."
        )

    pred_record = Record(
        features=np.asarray(best_x, dtype=np.float32).reshape(x_shape),
        label=int(best_label),
    )
    metrics = (
        {}
        if true_record is None
        else reconstruction_metrics(true_record, pred_record, eta_grid=eta_grid)
    )

    return AttackResult(
        attack_family="dpsgd_trace_optimization",
        z_hat=np.asarray(pred_record.features, dtype=np.float32),
        y_hat=int(pred_record.label),
        status="ok_known_label" if known_label is not None else "ok",
        diagnostics={
            "objective": float(best_obj),
            "per_label_best_objective": dict(per_label_best),
            "objective_mode": str(cfg.objective),
            "step_mode": str(cfg.step_mode),
            "selected_step_count": int(len(selected_steps)),
            "selected_steps": [int(s.step) for s in selected_steps],
            "num_steps": int(cfg.num_steps),
            "num_restarts": int(cfg.num_restarts),
            "target_index": None if target_index is None else int(target_index),
        },
        metrics=metrics,
    )
