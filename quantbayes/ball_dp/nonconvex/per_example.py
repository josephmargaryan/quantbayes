# quantbayes/ball_dp/nonconvex/per_example.py
from __future__ import annotations

from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax


ExampleLossFn = Callable[[Any, Any, jnp.ndarray, jnp.ndarray, jax.Array], jnp.ndarray]
PredictFn = Callable[[Any, Any, jnp.ndarray, jax.Array], jnp.ndarray]


def call_model(model: Any, state: Any, x: jnp.ndarray, key: jax.Array):
    out = model(x, key=key, state=state)
    if not (isinstance(out, tuple) and len(out) == 2):
        raise TypeError(
            "Custom models must implement "
            "__call__(x, *, key=None, state=None) and return (output, state)."
        )
    return out


def model_logits(model: Any, state: Any, x: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    logits, _ = call_model(model, state, x, key)
    logits = jnp.asarray(logits)
    if logits.ndim == 0:
        return logits[None]
    return logits


def _binary_target01(y: jnp.ndarray, dtype) -> jnp.ndarray:
    y = jnp.asarray(y)
    return jnp.where(
        y > 0,
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
    )


def binary_logistic_example_loss(
    model: Any,
    state: Any,
    x: jnp.ndarray,
    y: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    logits = model_logits(model, state, x, key)
    if logits.size != 1:
        raise ValueError(
            "binary_logistic requires a scalar-output model (one logit per example). "
            f"Got logits.shape={tuple(logits.shape)}."
        )
    logit = jnp.reshape(logits, ())
    target = _binary_target01(y, logit.dtype)
    return optax.sigmoid_binary_cross_entropy(logit, target).mean()


def multiclass_cross_entropy_example_loss(
    model: Any,
    state: Any,
    x: jnp.ndarray,
    y: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    logits = model_logits(model, state, x, key)
    if logits.ndim != 1 or logits.size < 2:
        raise ValueError(
            "softmax_cross_entropy requires a vector-output model with at least 2 logits. "
            f"Got logits.shape={tuple(logits.shape)}."
        )
    return optax.softmax_cross_entropy_with_integer_labels(
        logits[None, :], jnp.asarray(y, dtype=jnp.int32)[None]
    ).mean()


def resolve_loss_fn(loss_name: str) -> ExampleLossFn:
    name = str(loss_name).lower()
    if name in {"binary_logistic", "binary_cross_entropy_with_logits", "bce_logits"}:
        return binary_logistic_example_loss
    if name in {"softmax_cross_entropy", "cross_entropy", "multiclass_logistic"}:
        return multiclass_cross_entropy_example_loss
    raise ValueError(
        f"Unsupported loss_name={loss_name!r}. "
        "Supported: softmax_cross_entropy, binary_logistic."
    )


def default_predict_fn(
    model: Any, state: Any, x: jnp.ndarray, key: jax.Array
) -> jnp.ndarray:
    return model_logits(model, state, x, key)


def partition_model(model: Any):
    return eqx.partition(model, eqx.is_inexact_array)


def combine_model(params: Any, static: Any):
    return eqx.combine(params, static)


def tree_add(a: Any, b: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_scalar_mul(tree: Any, scalar: jnp.ndarray | float) -> Any:
    scalar = jnp.asarray(scalar)
    return jax.tree_util.tree_map(lambda x: x * scalar.astype(x.dtype), tree)


def tree_zeros_like(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_batch_l2_norms(tree: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.zeros((0,), dtype=jnp.float32)

    sq = None
    for leaf in leaves:
        flat = leaf.reshape((leaf.shape[0], -1))
        term = jnp.sum(flat * flat, axis=1)
        sq = term if sq is None else (sq + term)
    return jnp.sqrt(jnp.maximum(sq, jnp.asarray(0.0, dtype=sq.dtype)))


def clip_and_aggregate_per_example_grads(
    per_example_grads: Any,
    clip_norm: float | jnp.ndarray,
) -> tuple[Any, jnp.ndarray, jnp.ndarray]:
    norms = _tree_batch_l2_norms(per_example_grads)

    clip = jnp.asarray(clip_norm, dtype=norms.dtype)
    finite_clip = jnp.isfinite(clip)
    safe_clip = jnp.where(finite_clip, clip, jnp.asarray(1.0, dtype=norms.dtype))

    scales = jnp.minimum(
        jnp.asarray(1.0, dtype=norms.dtype),
        safe_clip / jnp.maximum(norms, jnp.asarray(1e-12, dtype=norms.dtype)),
    )
    scales = jnp.where(finite_clip, scales, jnp.ones_like(scales))

    def _scale_leaf(g):
        shape = (g.shape[0],) + (1,) * (g.ndim - 1)
        return g * scales.reshape(shape).astype(g.dtype)

    clipped = jax.tree_util.tree_map(_scale_leaf, per_example_grads)
    summed = jax.tree_util.tree_map(lambda g: jnp.sum(g, axis=0), clipped)
    clip_frac = jnp.where(
        finite_clip, jnp.mean(norms > clip), jnp.asarray(0.0, dtype=norms.dtype)
    )
    return summed, norms, clip_frac


def add_gaussian_noise(
    grad_tree: Any,
    noise_std: float | jnp.ndarray,
    key: jax.Array,
) -> Any:
    leaves, treedef = jax.tree_util.tree_flatten(grad_tree)
    if not leaves:
        return grad_tree

    keys = jr.split(key, len(leaves))
    out_leaves = []
    for leaf, k in zip(leaves, keys):
        std = jnp.asarray(noise_std, dtype=leaf.dtype)
        noise = std * jr.normal(k, shape=leaf.shape, dtype=leaf.dtype)
        out_leaves.append(leaf + noise)
    return treedef.unflatten(out_leaves)


def make_per_example_grad_fn(
    *,
    static: Any,
    state: Any,
    loss_fn: ExampleLossFn,
):
    def loss_of_params(params: Any, x: jnp.ndarray, y: jnp.ndarray, key: jax.Array):
        model = combine_model(params, static)
        return loss_fn(model, state, x, y, key)

    grad_one = jax.grad(loss_of_params)

    @eqx.filter_jit
    def per_example(params: Any, xb: jnp.ndarray, yb: jnp.ndarray, key: jax.Array):
        keys = jr.split(key, xb.shape[0])
        return jax.vmap(grad_one, in_axes=(None, 0, 0, 0))(params, xb, yb, keys)

    return per_example


def make_parameter_regularizer_grad_fn(
    *,
    static: Any,
    state: Any,
    parameter_regularizer: Optional[Callable[[Any, Any], jnp.ndarray]],
):
    if parameter_regularizer is None:
        return None

    def regularizer_of_params(params: Any):
        model = combine_model(params, static)
        return jnp.asarray(parameter_regularizer(model, state))

    grad_fn = jax.grad(regularizer_of_params)

    @eqx.filter_jit
    def apply(params: Any):
        return grad_fn(params)

    return apply


def make_batched_predict_fn(
    *,
    static: Any,
    state: Any,
    predict_fn: PredictFn,
):
    @eqx.filter_jit
    def batched_predict(params: Any, xb: jnp.ndarray, key: jax.Array):
        model = combine_model(params, static)
        keys = jr.split(key, xb.shape[0])
        return jax.vmap(lambda x, k: predict_fn(model, state, x, k), in_axes=(0, 0))(
            xb, keys
        )

    return batched_predict
