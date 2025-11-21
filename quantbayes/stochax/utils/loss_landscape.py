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
    return jax.tree_map(lambda p, u, v: p + a * u + b * v, params0, d1, d2)


def _restrict_dirs_like(params0, d, getters: Optional[Sequence[Callable]]):
    """Zero directions outside selected submodules (e.g., only classifier head)."""
    if not getters:
        return d
    mask = jtu.tree_map(lambda _: False, params0)
    for g in getters:
        mask = eqx.tree_at(g, mask, True)
    return jax.tree_map(lambda di, keep: di if keep else jnp.zeros_like(di), d, mask)


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
    ks = jr.split(key, len(leaves))
    v = jtu.tree_unflatten(treedef, [jr.normal(k, x.shape) for k, x in zip(ks, leaves)])

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
        v = jax.tree_map(lambda x: x / norm(v), v)
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
    ks = jr.split(key, len(leaves))
    v = jtu.tree_unflatten(treedef, [jr.normal(k, x.shape) for k, x in zip(ks, leaves)])

    def dot(a, b):
        return sum(
            jnp.vdot(x, y).real for x, y in zip(jax.tree_leaves(a), jax.tree_leaves(b))
        )

    def norm(a):
        return jnp.sqrt(jnp.maximum(dot(a, a), 1e-30))

    lam = jnp.array(0.0)
    for _ in _maybe_tqdm(range(iters), progress):
        v = jax.tree_map(lambda x: x / norm(v), v)

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


if __name__ == "__main__":
    plot_2d_landscape_fast(
        model=model,
        state=state,
        Xv=X_val,
        yv=y_val,
        analysis_mode="avgconv",  # "avgconv" recommended for conv nets
        grid_halfwidth=0.4,  # how far you move in each direction
        grid_n=41,  # resolution of the grid
        seed=0,  # for the random directions
        log_scale=True,  # log10 on the loss for nicer plots
        batch_for_loss=256,  # subset of validation batch used for loss eval
        verbose=True,  # prints pooling replacements etc.
    )
    plot_3d_landscape_surface(
        model=model,
        state=state,
        Xv=X_val,
        yv=y_val,
        analysis_mode="avgconv",
        grid_halfwidth=0.4,
        grid_n=41,
        seed=0,
        log_scale=True,
        batch_for_loss=256,
        verbose=True,
    )
    lam_h = estimate_top_hessian_eigenvalue(
        model=model,
        state=state,
        Xv=X_val,
        yv=y_val,
        iters=20,  # number of power iterations
        batch_for_loss=256,
        analysis_mode="avgconv",
        progress=True,  # uses tqdm if available
    )

    print("Top Hessian eigenvalue:", lam_h)
    lam_gn = estimate_top_gn_eigenvalue(
        model=model,
        state=state,
        Xv=X_val,
        yv=y_val,
        iters=20,
        batch_for_loss=256,
        analysis_mode="avgconv",
        progress=True,
    )

    print("Top Gauss–Newton eigenvalue:", lam_gn)
    # Example: assume your model has an attribute `head`
    head_getter = lambda m: m.head

    plot_2d_landscape_fast(
        model=model,
        state=state,
        Xv=X_val,
        yv=y_val,
        restrict_getters=[head_getter],  # directions only in head params
        analysis_mode="avgconv",
        grid_halfwidth=0.4,
        grid_n=41,
        seed=0,
        log_scale=True,
        batch_for_loss=256,
        verbose=True,
    )
