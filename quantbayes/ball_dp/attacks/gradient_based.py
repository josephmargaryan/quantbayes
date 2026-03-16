# quantbayes/ball_dp/attacks/gradient_based.py

from __future__ import annotations

import dataclasses as dc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from ..metrics import reconstruction_metrics
from .ball_priors import _project_box_ball_intersection_jax
from ..nonconvex.per_example import (
    ExampleLossFn,
    combine_model,
    partition_model,
    resolve_loss_fn,
)
from ..types import ArrayDataset, AttackResult, Record


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
    """Optional recorder for the sanitized gradient transcript of DP-SGD."""

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


@dc.dataclass
class TraceOptimizationAttackConfig:
    """Configuration for the DP-SGD trace optimization attack.

    Defaults match the paper's optimization recipe:
      - optimize Equation (1),
      - uniform-noise initialization,
      - learning rate 0.01,
      - five random restarts,
      - known label assumed unless exhaustive label search is requested.
    """

    step_mode: Literal["all", "present_steps"] = "all"
    num_steps: int = 1_000_000
    learning_rate: float = 1e-2
    num_restarts: int = 5
    box_bounds: Optional[Tuple[float, float]] = None
    ball_center: Optional[np.ndarray] = None
    ball_radius: Optional[float] = None
    seed: int = 0


def _tree_leaves(tree: Any) -> list[Any]:
    return list(jax.tree_util.tree_leaves(tree))


def _assert_same_tree(a: Any, b: Any) -> None:
    if jax.tree_util.tree_structure(a) != jax.tree_util.tree_structure(b):
        raise ValueError("Tree structures do not match.")


def _tree_add(a: Any, b: Any) -> Any:
    _assert_same_tree(a, b)
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_sub(a: Any, b: Any) -> Any:
    _assert_same_tree(a, b)
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def _tree_scalar_mul(tree: Any, scalar: float | jnp.ndarray) -> Any:
    scalar = jnp.asarray(scalar)
    return jax.tree_util.tree_map(
        lambda x: x * scalar.astype(jnp.asarray(x).dtype), tree
    )


def _tree_zeros_like(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_dot(a: Any, b: Any) -> jnp.ndarray:
    _assert_same_tree(a, b)
    out = jnp.asarray(0.0, dtype=jnp.float32)
    for x, y in zip(_tree_leaves(a), _tree_leaves(b)):
        out = out + jnp.sum(jnp.asarray(x).reshape(-1) * jnp.asarray(y).reshape(-1))
    return out


def _tree_l1_distance(a: Any, b: Any) -> jnp.ndarray:
    _assert_same_tree(a, b)
    out = jnp.asarray(0.0, dtype=jnp.float32)
    for x, y in zip(_tree_leaves(a), _tree_leaves(b)):
        out = out + jnp.sum(jnp.abs(jnp.asarray(x) - jnp.asarray(y)))
    return out


def _tree_l2_sq(tree: Any) -> jnp.ndarray:
    out = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in _tree_leaves(tree):
        arr = jnp.asarray(leaf)
        out = out + jnp.sum(arr * arr)
    return out


def _tree_batch_dot(batch_tree: Any, tree: Any) -> jnp.ndarray:
    out = None
    for g_leaf, r_leaf in zip(_tree_leaves(batch_tree), _tree_leaves(tree)):
        g_arr = jnp.asarray(g_leaf).reshape((g_leaf.shape[0], -1))
        r_arr = jnp.asarray(r_leaf).reshape(-1)
        term = jnp.sum(g_arr * r_arr[None, :], axis=1)
        out = term if out is None else (out + term)
    return jnp.zeros((0,), dtype=jnp.float32) if out is None else out


def _tree_batch_l2_norms(tree: Any) -> jnp.ndarray:
    leaves = _tree_leaves(tree)
    if not leaves:
        return jnp.zeros((0,), dtype=jnp.float32)

    sq = None
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        flat = arr.reshape((arr.shape[0], -1))
        term = jnp.sum(flat * flat, axis=1)
        sq = term if sq is None else (sq + term)
    return jnp.sqrt(jnp.maximum(sq, jnp.asarray(0.0, dtype=sq.dtype)))


def _records_close(a: Record, b: Record, *, atol: float = 1e-7) -> bool:
    xa = np.asarray(a.features, dtype=np.float32).reshape(-1)
    xb = np.asarray(b.features, dtype=np.float32).reshape(-1)
    return (
        int(a.label) == int(b.label)
        and xa.shape == xb.shape
        and np.allclose(xa, xb, atol=atol, rtol=0.0)
    )


def _find_record_index(
    records: Sequence[Record],
    target: Record,
    *,
    atol: float = 1e-7,
) -> Optional[int]:
    matches = [i for i, r in enumerate(records) if _records_close(r, target, atol=atol)]
    if not matches:
        return None
    return int(matches[0])


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
    out = jnp.asarray(x, dtype=jnp.float32)
    has_box = box_bounds is not None
    has_ball = ball_center is not None and ball_radius is not None

    if has_box and has_ball:
        return _project_box_ball_intersection_jax(
            out,
            np.asarray(ball_center, dtype=np.float32).reshape(out.shape),
            float(ball_radius),
            box_bounds,
        )

    if has_box:
        lo, hi = float(box_bounds[0]), float(box_bounds[1])
        out = jnp.clip(out, lo, hi)

    if has_ball:
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


def _default_init(
    shape: Sequence[int],
    *,
    rng: np.random.Generator,
    box_bounds: Optional[Tuple[float, float]],
) -> np.ndarray:
    if box_bounds is not None:
        lo, hi = float(box_bounds[0]), float(box_bounds[1])
        return rng.uniform(lo, hi, size=shape).astype(np.float32)
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def _select_steps(
    trace: DPSGDTrace,
    *,
    step_mode: str,
    target_index: Optional[int],
) -> list[DPSGDTraceStep]:
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


def subtract_known_batch_gradients(
    trace: DPSGDTrace,
    dataset: ArrayDataset,
    *,
    target_index: int,
    loss_name: Optional[str] = None,
    loss_fn: Optional[ExampleLossFn] = None,
    seed: int = 0,
) -> DPSGDTrace:
    """Subtract all known non-target contributions from each sanitized trace step."""
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
                "Trace step is missing model_before snapshots; cannot subtract known gradients."
            )
        batch_idx = np.asarray(step.batch_indices, dtype=np.int64)
        if batch_idx.size == 0:
            raise ValueError(
                "Trace step is missing batch_indices; cannot subtract known gradients."
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
    algorithm: Literal["auto", "algorithm2", "algorithm3"] = "auto",
    sampling_probability: Optional[float] = None,
    known_label: Optional[int] = None,
    true_record: Optional[Record] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
    seed: int = 0,
) -> AttackResult:
    """Run the prior-aware DP-SGD trace attack from the NeurIPS 2023 paper.

    Algorithm 2 (full-batch):
        score_i = sum_t <clip_C(∇ℓ(θ_t, z_i)), ḡ_t>

    Algorithm 3 (mini-batch):
        score_i = sum of the top qT values of
                  <clip_C(∇ℓ(θ_t, z_i)), ḡ_t>
        across training steps t.
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

    alg = str(algorithm).lower()
    if alg == "auto":
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
        alg = "algorithm2" if (q is None or float(q) >= 1.0 - 1e-12) else "algorithm3"

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
        np.asarray([int(r.label) for r in filtered_prior]),
        dtype=jnp.int32,
    )

    per_step_scores: list[np.ndarray] = []
    for step in trace.steps:
        if step.model_before is None:
            raise ValueError(
                "Prior-aware trace attack requires model snapshots at every retained step."
            )

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

        step_scores = np.asarray(
            _tree_batch_dot(clipped, step.observed_private_gradient),
            dtype=np.float32,
        )
        per_step_scores.append(step_scores)

    score_matrix = np.stack(per_step_scores, axis=1)

    if alg == "algorithm2":
        final_scores = score_matrix.sum(axis=1)
        top_k_steps = None
    elif alg == "algorithm3":
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
                "Algorithm 3 requires sampling_probability=... or trace metadata containing sample_rate / dataset_size."
            )
        top_k_steps = max(1, int(np.ceil(float(q) * score_matrix.shape[1])))
        part = np.partition(
            score_matrix,
            kth=max(score_matrix.shape[1] - top_k_steps, 0),
            axis=1,
        )
        final_scores = part[:, -top_k_steps:].sum(axis=1)
    else:
        raise ValueError(
            "algorithm must be one of {'auto', 'algorithm2', 'algorithm3'}."
        )

    best_idx = int(np.argmax(final_scores))
    best_record = filtered_prior[best_idx]

    metrics = (
        {}
        if true_record is None
        else reconstruction_metrics(true_record, best_record, eta_grid=eta_grid)
    )
    if true_record is not None:
        true_prior_index = _find_record_index(filtered_prior, true_record)
        if true_prior_index is not None:
            order = np.argsort(-final_scores)
            rank = int(np.where(order == true_prior_index)[0][0]) + 1
            metrics["prior_exact_hit"] = float(best_idx == true_prior_index)
            metrics["prior_rank"] = float(rank)
            for kk in (1, 5, 10):
                metrics[f"prior_hit@{kk}"] = float(rank <= kk)
    else:
        true_prior_index = None

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
            "algorithm": str(alg),
            "top_k_steps": top_k_steps,
            "prior_size": int(len(filtered_prior)),
            "true_prior_index": true_prior_index,
            "predicted_prior_index": int(best_idx),
        },
        candidates=candidates,
        metrics=metrics,
    )


def run_trace_optimization_attack(
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
    """Run the gradient-based DP-SGD trace attack using Equation (1).

    The optimized objective is
        L(z) = sum_t [ -<clip_C(∇ℓ(θ_t, z)), ḡ_t>
                       + ||clip_C(∇ℓ(θ_t, z)) - ḡ_t||_1 ].

    If `dataset` and `target_index` are provided, the trace is first residualized
    by subtracting all known non-target contributions from each retained sanitized
    gradient step.
    """
    cfg = TraceOptimizationAttackConfig() if cfg is None else cfg
    resolved_loss_fn = (
        resolve_loss_fn(trace.loss_name if loss_name is None else loss_name)
        if loss_fn is None
        else loss_fn
    )

    work_trace = trace
    if dataset is not None and target_index is not None:
        work_trace = subtract_known_batch_gradients(
            trace,
            dataset,
            target_index=int(target_index),
            loss_name=trace.loss_name if loss_name is None else str(loss_name),
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
                box_bounds=cfg.box_bounds,
            )
            x = jnp.asarray(x0, dtype=jnp.float32)
            optimizer = optax.sgd(float(cfg.learning_rate))
            opt_state = optimizer.init(x)

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
                    total = total + (
                        -_tree_dot(cand, obs) + _tree_l1_distance(cand, obs)
                    )
                return total

            value_and_grad = jax.value_and_grad(_objective)
            best_restart_obj = float("inf")
            best_restart_x = x

            for _ in range(max(1, int(cfg.num_steps))):
                loss_value, grad_x = value_and_grad(x)
                updates, opt_state = optimizer.update(grad_x, opt_state, x)
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
        raise RuntimeError("Trace optimization attack failed to produce a candidate.")

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
            "equation": "paper_equation_1",
            "step_mode": str(cfg.step_mode),
            "selected_step_count": int(len(selected_steps)),
            "selected_steps": [int(s.step) for s in selected_steps],
            "num_steps": int(cfg.num_steps),
            "num_restarts": int(cfg.num_restarts),
            "target_index": None if target_index is None else int(target_index),
        },
        metrics=metrics,
    )
