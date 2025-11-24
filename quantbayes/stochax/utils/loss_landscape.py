# quantbayes/stochax/utils/loss_landscape.py
from __future__ import annotations
from typing import Any, Callable, List, Sequence, Tuple, Optional
import math
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
import optax
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 0) Generic eval-mode copy (works for any Equinox model)
# ---------------------------------------------------------------------
def make_eval_copy(model):
    """
    Return a copy suitable for evaluation: Dropout off, BatchNorm in inference
    mode etc. Uses eqx.nn.inference_mode if available; otherwise, falls back to
    a safe no-op copy (most Equinox layers expose `.inference` anyway).
    """
    inf = getattr(eqx.nn, "inference_mode", None)
    if inf is not None:
        try:
            return inf(model)
        except Exception:
            pass
    # If inference_mode isn't available, just return the model as-is. Most
    # users will be on recent Equinox where inference_mode exists.
    return model


# ---------------------------------------------------------------------
# 0.1) Pool replacements by full-module substitution (no attribute writes)
# ---------------------------------------------------------------------
def _to_pair(x) -> Tuple[int, int]:
    if isinstance(x, (tuple, list)):
        assert len(x) == 2
        return int(x[0]), int(x[1])
    return int(x), int(x)


class AvgPoolViaConv(eqx.Module):
    """Average pooling implemented as depthwise convolution (fast under JAX/XLA).

    Assumes inputs are CHW or NCHW. Padding is 'VALID' (same as typical pool).
    """

    kernel: Tuple[int, int]
    stride: Tuple[int, int]

    def __call__(self, x):
        # Normalize to NCHW
        added_batch = False
        if x.ndim == 3:  # CHW
            x = x[None, ...]
            added_batch = True
        assert x.ndim == 4 and x.shape[1] is not None, "Expect NCHW or CHW."

        C = x.shape[1]
        kH, kW = self.kernel
        sH, sW = self.stride
        w = jnp.ones((C, 1, kH, kW), dtype=x.dtype) / float(kH * kW)
        y = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=w,
            window_strides=(sH, sW),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=C,  # depthwise
        )
        if added_batch:
            y = y[0]
        return y


def _is_pool_module(x) -> bool:
    nn = getattr(eqx, "nn", None)
    if nn is None:
        return False
    pool_names = {"MaxPool2d", "AvgPool2d"}
    return type(x).__name__ in pool_names


def _make_pool_replacement(old_pool, mode: str):
    """Return a new module to replace a pooling layer."""
    name = type(old_pool).__name__
    # Best-effort: read kernel_size/stride by attribute if present; default 2
    k = getattr(old_pool, "kernel_size", 2)
    s = getattr(old_pool, "stride", 2)
    k = _to_pair(k)
    s = _to_pair(s)

    if mode == "avgpool":
        # Replace MaxPool2d with AvgPool2d; keep AvgPool2d as AvgPool2d.
        if name == "AvgPool2d":
            return old_pool
        # Else build a fresh AvgPool2d
        AP = getattr(eqx.nn, "AvgPool2d")
        return AP(kernel_size=k, stride=s)

    elif mode == "avgconv":
        # Replace both with fast depthwise-conv avg pooling
        return AvgPoolViaConv(kernel=k, stride=s)

    else:  # "none"
        return old_pool


# Represent a path to a submodule as a sequence of steps:
#   ("attr", name)  .name
#   ("idx", i)      [i]
#   ("key", k)      [k] for dict
Path = List[Tuple[str, Any]]


def _where_from_path(path: Path) -> Callable[[Any], Any]:
    def where(root):
        cur = root
        for kind, val in path:
            if kind == "attr":
                cur = getattr(cur, val)
            elif kind == "idx":
                cur = cur[val]
            elif kind == "key":
                cur = cur[val]
            else:
                raise ValueError(f"Unknown path kind {kind}")
        return cur

    return where


def _collect_pool_paths(node, path: Path, out: List[Tuple[Path, Any]], mode: str):
    # Pool? collect + stop recursing into it
    if _is_pool_module(node):
        out.append((path, _make_pool_replacement(node, mode)))
        return

    # Recurse structure
    if isinstance(node, eqx.Module):
        for name, child in vars(node).items():
            _collect_pool_paths(child, path + [("attr", name)], out, mode)
    elif isinstance(node, (list, tuple)):
        for i, child in enumerate(node):
            _collect_pool_paths(child, path + [("idx", i)], out, mode)
    elif isinstance(node, dict):
        # Use deterministic order
        for k in sorted(list(node.keys()), key=lambda z: str(z)):
            _collect_pool_paths(node[k], path + [("key", k)], out, mode)
    else:
        return


def _rand_like(key, x):
    """Sample a random tangent with same shape and dtype as x.

    - For real x: N(0, 1) with dtype of x.
    - For complex x: (N(0,1) + i N(0,1)) / sqrt(2) with dtype of x.
    """
    import jax.random as jr
    import jax.numpy as jnp

    if jnp.iscomplexobj(x):
        # Use two keys for real and imag
        k1, k2 = jr.split(key, 2)
        real = jr.normal(k1, x.shape, dtype=x.real.dtype)
        imag = jr.normal(k2, x.shape, dtype=x.real.dtype)
        z = (real + 1j * imag) / jnp.sqrt(2.0)
        return z.astype(x.dtype)
    else:
        return jr.normal(key, x.shape, dtype=x.dtype)


def make_analysis_copy(model, analysis_mode: str = "avgconv", verbose: bool = False):
    """
    Start from eval-mode model, then (optionally) replace pooling layers
    by whole-module substitution.

    analysis_mode:
      - "none":     keep pools as-is.
      - "avgpool":  MaxPool2d -> AvgPool2d (still uses reduce-window).
      - "avgconv":  {Max,Avg}Pool2d -> AvgPoolViaConv (fast; recommended).
    """
    base = make_eval_copy(model)
    if analysis_mode not in ("none", "avgpool", "avgconv"):
        raise ValueError("analysis_mode must be one of {'none','avgpool','avgconv'}")

    if analysis_mode == "none":
        return base

    repls: List[Tuple[Path, Any]] = []
    _collect_pool_paths(base, [], repls, analysis_mode)

    if verbose:
        n_max = sum(1 for p, _ in repls if type(_).__name__ != "AvgPool2d")
        n_all = len(repls)
        print(f"[analysis] pool replacements: {n_all} total (mode={analysis_mode})")

    out = base
    for path, new_mod in repls:
        out = eqx.tree_at(_where_from_path(path), out, new_mod)
    return out


# ---------------------------------------------------------------------
# 1) Loss builder (BN-state aware, vmapped over batch)
# ---------------------------------------------------------------------
def _softmax_ce_mean(logits, y_int):
    return optax.softmax_cross_entropy_with_integer_labels(logits, y_int).mean()


def make_scalar_ce_loss(best_state, x_batch, y_batch, rng_key):
    """
    Returns a function loss_fn(params, static) -> scalar CE loss on x_batch.
    - vmaps the model over batch so Conv2d sees CHW rather than BCHW
    - passes the (frozen) BN/other state through the call
    - handles multihead by averaging logits if list/tuple is returned
    """
    x_batch = jnp.asarray(x_batch)
    y_batch = jnp.asarray(y_batch)
    B = x_batch.shape[0]

    def loss_fn(params, static):
        mdl = eqx.combine(params, static)
        keys = jr.split(rng_key, B)
        logits_or_list, _ = jax.vmap(
            mdl, in_axes=(0, 0, None), out_axes=(0, None), axis_name="batch"
        )(x_batch, keys, best_state)
        logits = (
            sum(logits_or_list) / float(len(logits_or_list))
            if isinstance(logits_or_list, (list, tuple))
            else logits_or_list
        )
        return _softmax_ce_mean(logits, y_batch)

    return loss_fn


# ---------------------------------------------------------------------
# 2) Filter-normalized random directions (Li et al., NeurIPS'18)
# ---------------------------------------------------------------------
def _filter_normalized_dirs(params, seed=0):
    leaves, treedef = jtu.tree_flatten(params)
    key = jr.PRNGKey(seed)
    ks1 = jr.split(key, len(leaves))
    ks2 = jr.split(jr.fold_in(key, 1), len(leaves))

    def norm2(x):
        v = jnp.vdot(x, x)
        v = v.real if jnp.iscomplexobj(v) else v
        return jnp.sqrt(jnp.maximum(v, 0.0)) + 1e-12

    d1 = [jr.normal(k, x.shape) for k, x in zip(ks1, leaves)]
    d2 = [jr.normal(k, x.shape) for k, x in zip(ks2, leaves)]
    d1 = [u * (norm2(x) / norm2(u)) for u, x in zip(d1, leaves)]
    d2 = [v * (norm2(x) / norm2(v)) for v, x in zip(d2, leaves)]
    return jtu.tree_unflatten(treedef, d1), jtu.tree_unflatten(treedef, d2)


def _apply_2dir(params0, d1, d2, a, b):
    return jax.tree.map(lambda p, u, v: p + a * u + b * v, params0, d1, d2)


def _restrict_dirs_like(params0, d, getters: Optional[Sequence[Callable]]):
    """Zero directions outside selected submodules (e.g., only classifier head).

    `getters` are callables like `lambda m: m.head` or something that returns a
    subtree (module, tuple of modules, etc.) of `params0`.
    We select *all array leaves* under those subtrees and zero everything else.
    """
    if not getters:
        return d

    # Flatten params0 to get a stable leaf order.
    leaves, treedef = jtu.tree_flatten(params0)

    # Collect the ids of all leaves that should be kept.
    keep_ids = set()
    for g in getters:
        sub = g(params0)  # subtree(s) we want to keep
        for leaf in jtu.tree_leaves(sub):
            # Only arrays actually appear in params0; ints etc. live in `static`.
            keep_ids.add(id(leaf))

    # Build a flat boolean mask aligned with `leaves`.
    flat_mask = [id(leaf) in keep_ids for leaf in leaves]

    # Unflatten to a tree mask with same structure as params0.
    mask = jtu.tree_unflatten(treedef, flat_mask)

    # Finally, zero-out directions where mask is False.
    return jax.tree.map(
        lambda di, keep: di if keep else jnp.zeros_like(di),
        d,
        mask,
    )


# ---------------------------------------------------------------------
# 3) Helpers to prepare an analysis model + loss
# ---------------------------------------------------------------------
def _prepare_analysis_loss(
    model,
    state,
    Xv,
    yv,
    analysis_mode: str,
    batch_for_loss: int,
    seed: int,
    verbose: bool,
):
    m_eval = make_analysis_copy(model, analysis_mode=analysis_mode, verbose=verbose)
    params0, static = eqx.partition(m_eval, eqx.is_array)
    x_small = jnp.asarray(Xv[:batch_for_loss])
    y_small = jnp.asarray(yv[:batch_for_loss])
    base_loss = make_scalar_ce_loss(state, x_small, y_small, jr.PRNGKey(seed))
    return params0, static, base_loss


# ---------------------------------------------------------------------
# 4) Fast 2D loss slice (vmapped and jitted). Works for any Equinox model.
# ---------------------------------------------------------------------
def plot_2d_landscape_fast(
    model,
    state,
    Xv,
    yv,
    *,
    analysis_mode: str = "avgconv",
    grid_halfwidth=0.4,
    grid_n=41,
    seed=0,
    log_scale=True,
    restrict_getters: Optional[Sequence[Callable]] = None,
    batch_for_loss=128,  # subsample batch used for the loss evaluation
    verbose=False,
):
    params0, static, base_loss = _prepare_analysis_loss(
        model, state, Xv, yv, analysis_mode, batch_for_loss, seed=123, verbose=verbose
    )

    d1, d2 = _filter_normalized_dirs(params0, seed=seed)
    d1 = _restrict_dirs_like(params0, d1, restrict_getters)
    d2 = _restrict_dirs_like(params0, d2, restrict_getters)

    alphas = jnp.linspace(-grid_halfwidth, grid_halfwidth, grid_n)
    betas = jnp.linspace(-grid_halfwidth, grid_halfwidth, grid_n)

    def loss_at_point(a, b, static_):
        p = _apply_2dir(params0, d1, d2, a, b)
        return base_loss(p, static_)

    # Warm-up compile
    _ = jax.jit(lambda a, b: loss_at_point(a, b, static))(0.0, 0.0).block_until_ready()

    loss_grid = jax.jit(
        jax.vmap(
            jax.vmap(lambda a, b: loss_at_point(a, b, static), in_axes=(None, 0)),
            in_axes=(0, None),
        )
    )
    Z = loss_grid(alphas, betas)  # (len(alphas), len(betas))

    Z_np = np.array(Z)
    if log_scale:
        Z_min = float(Z_np.min())
        Z_np = np.log10(np.maximum(Z_np - Z_min + 1e-8, 1e-8))

    plt.figure(figsize=(6, 5))
    extent = [float(betas[0]), float(betas[-1]), float(alphas[0]), float(alphas[-1])]
    plt.imshow(Z_np, origin="lower", extent=extent, aspect="auto")
    CS = plt.contour(np.array(betas), np.array(alphas), Z_np, levels=20, linewidths=0.8)
    plt.clabel(CS, inline=True, fontsize=8)
    plt.xlabel("direction β")
    plt.ylabel("direction α")
    ttl = "2D loss landscape (filter-normalized{})".format(
        ", log10" if log_scale else ""
    )
    plt.title(ttl)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 5) 3D surface version (same grid)
# ---------------------------------------------------------------------
def plot_3d_landscape_surface(
    model,
    state,
    Xv,
    yv,
    *,
    analysis_mode: str = "avgconv",
    grid_halfwidth=0.4,
    grid_n=41,
    seed=0,
    log_scale=True,
    restrict_getters: Optional[Sequence[Callable]] = None,  # kept for symmetry
    batch_for_loss=128,
    verbose=False,
):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    params0, static, base_loss = _prepare_analysis_loss(
        model, state, Xv, yv, analysis_mode, batch_for_loss, seed=124, verbose=verbose
    )

    d1, d2 = _filter_normalized_dirs(params0, seed=seed)
    d1 = _restrict_dirs_like(params0, d1, restrict_getters)
    d2 = _restrict_dirs_like(params0, d2, restrict_getters)

    alphas = jnp.linspace(-grid_halfwidth, grid_halfwidth, grid_n)
    betas = jnp.linspace(-grid_halfwidth, grid_halfwidth, grid_n)

    def loss_at_point(a, b, static_):
        p = _apply_2dir(params0, d1, d2, a, b)
        return base_loss(p, static_)

    # Warm-up compile
    _ = jax.jit(lambda a, b: loss_at_point(a, b, static))(0.0, 0.0).block_until_ready()

    loss_grid = jax.jit(
        jax.vmap(
            jax.vmap(lambda a, b: loss_at_point(a, b, static), in_axes=(None, 0)),
            in_axes=(0, None),
        )
    )
    Z = loss_grid(alphas, betas)

    A, B = np.meshgrid(np.array(betas), np.array(alphas))
    Z_np = np.array(Z)
    if log_scale:
        Z_min = float(Z_np.min())
        Z_np = np.log10(np.maximum(Z_np - Z_min + 1e-8, 1e-8))

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        A, B, Z_np, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.95
    )
    ax.set_xlabel("β")
    ax.set_ylabel("α")
    ax.set_zlabel("loss" + (" (log10)" if log_scale else ""))
    ax.set_title("3D loss landscape (filter-normalized)")
    fig.colorbar(surf, shrink=0.6, aspect=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 6) Top eigenvalues (Hessian and Gauss–Newton) with optional tqdm
# ---------------------------------------------------------------------
def _maybe_tqdm(it, progress: bool):
    if not progress:
        return it
    try:
        from tqdm.auto import tqdm

        return tqdm(it, desc="power-iter", leave=True)
    except Exception:
        return it


def hessian_top_eig(
    loss_fn, params0, static, iters=20, key=jr.PRNGKey(0), progress: bool = False
):
    """Power iteration on the Hessian of loss_fn(params, static)."""
    leaves, treedef = jtu.tree_flatten(params0)

    # Need 2 keys per leaf in worst case (for complex), so oversample.
    ks = jr.split(key, len(leaves) * 2)
    v_leaves = []
    for i, x in enumerate(leaves):
        k = ks[2 * i]  # _rand_like will split again if needed
        v_leaves.append(_rand_like(k, x))
    v = jtu.tree_unflatten(treedef, v_leaves)

    def dot(a, b):
        return sum(
            jnp.vdot(x, y).real for x, y in zip(jax.tree_leaves(a), jax.tree_leaves(b))
        )

    def norm(a):
        return jnp.sqrt(jnp.maximum(dot(a, a), 1e-30))

    def grad_wrt_params(p):
        return jax.grad(lambda pp: loss_fn(pp, static))(p)

    lam = jnp.array(0.0)
    for _ in _maybe_tqdm(range(iters), progress):
        v = jax.tree.map(lambda x: x / norm(v), v)
        Hv = jax.jvp(grad_wrt_params, (params0,), (v,))[1]
        lam = dot(v, Hv) / jnp.maximum(dot(v, v), 1e-30)
        v = Hv
    return float(lam)


def _ce_hessian_logits_vec(logits, v):
    """(diag(p)-pp^T) v for CE/softmax; logits/v shape (..., K)."""
    p = jax.nn.softmax(logits, axis=-1)
    pv = (p * v).sum(axis=-1, keepdims=True)
    return p * (v - pv)


def gauss_newton_top_eig_for_ce(
    f_logits, params0, static, iters=20, key=jr.PRNGKey(0), progress: bool = False
):
    """
    Power iteration on J^T H_z J where f_logits(params, static)->logits.
    CE Hessian wrt logits is (diag(p)-pp^T).
    """
    leaves, treedef = jtu.tree_flatten(params0)

    ks = jr.split(key, len(leaves) * 2)
    v_leaves = []
    for i, x in enumerate(leaves):
        k = ks[2 * i]
        v_leaves.append(_rand_like(k, x))
    v = jtu.tree_unflatten(treedef, v_leaves)

    def dot(a, b):
        return sum(
            jnp.vdot(x, y).real for x, y in zip(jax.tree_leaves(a), jax.tree_leaves(b))
        )

    def norm(a):
        return jnp.sqrt(jnp.maximum(dot(a, a), 1e-30))

    lam = jnp.array(0.0)
    for _ in _maybe_tqdm(range(iters), progress):
        v = jax.tree.map(lambda x: x / norm(v), v)

        # JVP: δz = J v
        z0, pullback = jax.vjp(lambda p: f_logits(p, static), params0)
        dz = jax.jvp(lambda p: f_logits(p, static), (params0,), (v,))[1]

        # Apply CE Hessian wrt logits
        ybar = _ce_hessian_logits_vec(z0, dz)

        # VJP: J^T ybar
        Hv = pullback(ybar)[0]

        lam = dot(v, Hv) / jnp.maximum(dot(v, v), 1e-30)
        v = Hv
    return float(lam)


def estimate_top_hessian_eigenvalue(
    model,
    state,
    Xv,
    yv,
    *,
    iters=10,
    batch_for_loss=128,
    analysis_mode="avgconv",
    progress: bool = False,
):
    """Convenience wrapper: CE loss on a small batch, then power iteration (true Hessian)."""
    params0, static, base_loss = _prepare_analysis_loss(
        model, state, Xv, yv, analysis_mode, batch_for_loss, seed=321, verbose=False
    )
    lam = hessian_top_eig(
        base_loss, params0, static, iters=iters, key=jr.PRNGKey(9), progress=progress
    )
    print(f"Estimated top Hessian eigenvalue (final model): {lam:.6g}")
    return lam


def estimate_top_gn_eigenvalue(
    model,
    state,
    Xv,
    yv,
    *,
    iters=10,
    batch_for_loss=128,
    analysis_mode="avgconv",
    progress: bool = False,
):
    """
    Convenience wrapper: top eigenvalue of Gauss–Newton (CE) using f_logits JVP/VJP.
    Typically faster/more stable around pooling layers.
    """
    # Build logits function on a small batch
    m_eval = make_analysis_copy(model, analysis_mode=analysis_mode, verbose=False)
    params0, static = eqx.partition(m_eval, eqx.is_array)

    x_small = jnp.asarray(Xv[:batch_for_loss])
    y_small = jnp.asarray(yv[:batch_for_loss])  # not used; CE Hessian uses logits only
    B = x_small.shape[0]

    def f_logits(p, s):
        mdl = eqx.combine(p, s)
        keys = jr.split(jr.PRNGKey(777), B)
        logits_or_list, _ = jax.vmap(
            mdl, in_axes=(0, 0, None), out_axes=(0, None), axis_name="batch"
        )(x_small, keys, state)
        logits = (
            sum(logits_or_list) / float(len(logits_or_list))
            if isinstance(logits_or_list, (list, tuple))
            else logits_or_list
        )
        return logits  # (B, K)

    lam = gauss_newton_top_eig_for_ce(
        f_logits, params0, static, iters=iters, key=jr.PRNGKey(8), progress=progress
    )
    print(f"Estimated top Gauss–Newton eigenvalue (final model): {lam:.6g}")
    return lam


def compute_2d_landscape_grid(
    model,
    state,
    Xv,
    yv,
    *,
    analysis_mode: str = "avgconv",
    grid_halfwidth=0.4,
    grid_n=41,
    seed=0,
    log_scale=True,
    restrict_getters: Optional[Sequence[Callable]] = None,
    batch_for_loss=128,
    verbose=False,
):
    params0, static, base_loss = _prepare_analysis_loss(
        model, state, Xv, yv, analysis_mode, batch_for_loss, seed=123, verbose=verbose
    )

    d1, d2 = _filter_normalized_dirs(params0, seed=seed)
    d1 = _restrict_dirs_like(params0, d1, restrict_getters)
    d2 = _restrict_dirs_like(params0, d2, restrict_getters)

    alphas = jnp.linspace(-grid_halfwidth, grid_halfwidth, grid_n)
    betas = jnp.linspace(-grid_halfwidth, grid_halfwidth, grid_n)

    def loss_at_point(a, b, static_):
        p = _apply_2dir(params0, d1, d2, a, b)
        return base_loss(p, static_)

    _ = jax.jit(lambda a, b: loss_at_point(a, b, static))(0.0, 0.0).block_until_ready()

    loss_grid = jax.jit(
        jax.vmap(
            jax.vmap(lambda a, b: loss_at_point(a, b, static), in_axes=(None, 0)),
            in_axes=(0, None),
        )
    )
    Z = loss_grid(alphas, betas)
    Z_np = np.array(Z)

    if log_scale:
        Z_min = float(Z_np.min())
        Z_np = np.log10(np.maximum(Z_np - Z_min + 1e-8, 1e-8))

    return np.array(alphas), np.array(betas), Z_np


if __name__ == "__main__":
    # Example with ViT on random data

    import equinox as eqx
    import matplotlib.pyplot as plt
    from quantbayes.stochax.utils.loss_landscape import (
        plot_3d_landscape_surface,
        estimate_top_hessian_eigenvalue,
        estimate_top_gn_eigenvalue,
    )
    from quantbayes.stochax.vision_classification.models.rfft_vit import (
        VisionTransformer,
    )
    from quantbayes.stochax import train, multiclass_loss

    # Generate random data
    num_classes = 10
    batch_size = 10
    image_size = 32
    num_channels = 3
    key = jr.PRNGKey(0)
    X = jr.normal(key, (batch_size, num_channels, image_size, image_size))
    y = jr.randint(key, (batch_size,), 0, num_classes)
    print(X.shape, y.shape)

    IMAGE_SIZE = 32
    PATCH_SIZE = 4
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 64

    EMBED_DIM = 128
    HIDDEN_DIM = 512
    NUM_HEADS = 4
    NUM_LAYERS = 6
    DROPOUT = 0.1
    NUM_CLASSES = 10

    key = jr.PRNGKey(0)
    k_dense, k_rfft = jr.split(key, 2)

    # Spectral ViT (RFFTCirculant1D on all D→D linears)
    model_rfft, state_rfft = eqx.nn.make_with_state(VisionTransformer)(
        embedding_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        num_classes=NUM_CLASSES,
        key=k_rfft,
        channels=3,
        use_spectral_proj=True,
    )

    # Baseline dense ViT
    model, state = eqx.nn.make_with_state(VisionTransformer)(
        embedding_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        num_classes=NUM_CLASSES,
        key=k_dense,
        channels=3,
        use_spectral_proj=False,
    )

    def train_model(
        model, state, X_train, y_train, X_test, y_test, num_epochs=200, batch_size=128
    ):

        optimizer = optax.adan(
            learning_rate=0.01,
            b1=0.95,
            b2=0.99,
            eps=1e-8,
            weight_decay=1e-4,
        )

        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        best_model, best_state, tr_loss, va_loss = train(
            model,
            state,
            opt_state,
            optimizer,
            multiclass_loss,
            jnp.array(X_train),
            jnp.array(y_train),
            jnp.array(X_test),
            jnp.array(y_test),
            batch_size=batch_size,
            num_epochs=num_epochs,
            patience=200,
            key=jr.key(0),
            # augment_fn=augment_fn,
            lambda_spec=0.0,
            log_global_bound_every=None,
            bound_conv_mode="tn",
            bound_tn_iters=4,
            bound_input_shape=(32, 32),
            bound_recorder=None,
            ckpt_path=None,
            checkpoint_interval=1,  # save every epoch
        )
        return best_model, best_state, tr_loss, va_loss

    best_model, best_state, tr_loss, va_loss = train_model(
        model, state, X, y, X, y, num_epochs=2, batch_size=10
    )

    best_model_rfft, best_state_rfft, tr_loss_rfft, va_loss_rfft = train_model(
        model_rfft, state_rfft, X, y, X, y, num_epochs=2, batch_size=10
    )

    # --- Basic components ---

    def head_getter(m):
        """Classifier head (Linear 128 -> num_classes)."""
        return m.head

    def patch_embed_linear_getter(m):
        """Patch embedding linear (48 -> 128)."""
        return m.patch_embedding.linear

    def all_attn_projections_getter(m):
        """
        All q/k/v/out projection modules across all transformer blocks.
        - Standard ViT: Linear layers.
        - RFFT ViT: RFFTCirculant1D layers.
        """
        return tuple(
            (
                block.attention.q_proj,
                block.attention.k_proj,
                block.attention.v_proj,
                block.attention.out_proj,
            )
            for block in m.attention_blocks
        )

    def all_ffn_layers_getter(m):
        """
        All FFN layers (linear1, linear2) across blocks.
        These are Linear in both models.
        """
        return tuple(
            (
                block.linear1,
                block.linear2,
            )
            for block in m.attention_blocks
        )

    # --- Per-block versions (first / middle / last) ---

    def make_block_attn_getter(block_idx: int):
        """Self-attention (all projections) for a specific block."""

        def getter(m):
            block = m.attention_blocks[block_idx]
            return block.attention

        return getter

    def make_block_ffn_getter(block_idx: int):
        """FFN (linear1 + linear2) for a specific block."""

        def getter(m):
            block = m.attention_blocks[block_idx]
            return (block.linear1, block.linear2)

        return getter

    restrict_attn = [all_attn_projections_getter]

    plot_3d_landscape_surface(
        model=best_model,
        state=best_state,
        Xv=X,
        yv=y,
        analysis_mode="none",
        grid_halfwidth=0.3,
        grid_n=17,  # smaller than 2D; 3D is expensive
        seed=0,
        log_scale=True,
        restrict_getters=restrict_attn,
        batch_for_loss=64,
        verbose=False,
    )

    plot_3d_landscape_surface(
        model=best_model_rfft,
        state=best_state_rfft,
        Xv=X,
        yv=y,
        analysis_mode="none",
        grid_halfwidth=0.3,
        grid_n=17,
        seed=0,
        log_scale=True,
        restrict_getters=restrict_attn,
        batch_for_loss=64,
        verbose=False,
    )

    cfg = dict(
        Xv=X,
        yv=y,
        batch_for_loss=128,
        analysis_mode="none",
        iters=10,
        progress=True,
    )

    lam_H_dense = estimate_top_hessian_eigenvalue(best_model, best_state, **cfg)
    lam_H_rfft = estimate_top_hessian_eigenvalue(
        best_model_rfft, best_state_rfft, **cfg
    )

    lam_GN_dense = estimate_top_gn_eigenvalue(best_model, best_state, **cfg)
    lam_GN_rfft = estimate_top_gn_eigenvalue(best_model_rfft, best_state_rfft, **cfg)
