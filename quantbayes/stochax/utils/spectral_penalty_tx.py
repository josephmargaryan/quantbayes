# quantbayes/stochax/utils/spectral_penalty_tx.py
from __future__ import annotations
from typing import Callable, Optional, NamedTuple, Dict, Any, List

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

"""
Example Usage: 

from quantbayes.stochax.utils.optim_util import (
    OptimizerConfig, DecayMaskConfig, LabelConfig, build_optimizer
)
from quantbayes.stochax.utils.spectral_penalty_tx import (
    add_spectral_penalty_transform, make_lambda_spec_schedule, make_soft_barrier
)
from quantbayes.stochax.utils.regularizers import global_spectral_norm_penalty

# 1) Model
class Net(eqx.Module):
    l1: eqx.nn.Linear
    def __init__(self, key):
        self.l1 = eqx.nn.Linear(51, 1, key=key)
    def __call__(self, x, key, state):
        return self.l1(x), state

key = jr.PRNGKey(0)
model, state = eqx.nn.make_with_state(Net)(key=key)

# 2) LR schedule
total_steps = 50_000
warmup_steps = 1_000
lr_sched = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3, warmup_steps=warmup_steps,
    decay_steps=total_steps - warmup_steps, end_value=1e-5
)

# 3) Config (AGC before global clip; decoupled masked WD)
cfg = OptimizerConfig(
    algorithm="adamw",
    lr=lr_sched,
    weight_decay=1e-4,
    agc_clip=0.01,            # AGC first
    clip_global_norm=1.0,     # then global clip
    decay_mask=DecayMaskConfig(),  # bias/norm/spectral off by default
)

# 4) Penalty transform: sum of per-layer operator norms (TN for convs by default)
lam_sched = make_lambda_spec_schedule(
    peak=1e-3, warmup_steps=5_000, hold_steps=10_000, decay_steps=30_000, floor=1e-5
)
barrier = make_soft_barrier(target=8.0, sharpness=4.0, max_mult=8.0)

specnorm_tx = add_spectral_penalty_transform(
    like_model=model,
    penalty_fn=partial(global_spectral_norm_penalty, conv_mode="tn", conv_tn_iters=8),
    schedule=lam_sched,
    barrier=barrier,
    barrier_every=200,     # compute proxy every 200 steps
)

# 5) Build optimizer with the transform PREPENDED
opt, opt_state, aux = build_optimizer(model, cfg, prepend=[specnorm_tx])

"""

# ---------------- Schedules & recorder ---------------- #


class SpecPenRecorder:
    def __init__(self):
        self.hist: List[Dict[str, Any]] = []

    def __call__(self, m: Dict[str, Any]):
        self.hist.append(m)

    @property
    def step(self):
        return [x["step"] for x in self.hist]

    @property
    def L(self):
        return [x["last_lip"] for x in self.hist]

    @property
    def lam(self):
        return [x["last_lambda"] for x in self.hist]

    @property
    def lam_base(self):
        return [x["last_lambda_base"] for x in self.hist]

    @property
    def bar(self):
        return [x["last_bar"] for x in self.hist]

    def to_df(self):
        try:
            import pandas as pd
        except Exception:
            raise RuntimeError(
                "pandas not available; install it or use the list properties."
            )
        return pd.DataFrame(self.hist)


def make_lambda_spec_schedule(
    peak: float,
    warmup_steps: int,
    hold_steps: int = 0,
    decay_steps: int = 0,
    floor: float = 0.0,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """0→peak linear warmup, optional hold, cosine decay to floor."""
    warmup_steps = int(warmup_steps)
    hold_steps = int(hold_steps)
    decay_steps = int(decay_steps)

    def sched(step: jnp.ndarray) -> jnp.ndarray:
        step = jnp.asarray(step, dtype=jnp.int32)
        t = step.astype(jnp.float32)

        def warm():
            denom = jnp.maximum(1.0, float(warmup_steps))
            frac = jnp.clip(t / denom, 0.0, 1.0)
            return jnp.float32(peak) * frac

        def hold():
            return jnp.float32(peak)

        def decay():
            if decay_steps <= 0:
                return jnp.float32(peak)
            dt = t - float(warmup_steps + hold_steps)
            denom = jnp.maximum(1.0, float(decay_steps))
            frac = jnp.clip(dt / denom, 0.0, 1.0)
            return jnp.float32(floor) + 0.5 * (
                jnp.float32(peak) - jnp.float32(floor)
            ) * (1.0 + jnp.cos(jnp.pi * frac))

        return jnp.where(
            t < warmup_steps,
            warm(),
            jnp.where(t < warmup_steps + hold_steps, hold(), decay()),
        )

    return sched


def make_soft_barrier(
    target: float,
    sharpness: float = 5.0,
    max_mult: float = 10.0,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Smoothly increase strength when a Lipschitz proxy exceeds target. Returns factor in [1, max_mult]."""
    t = jnp.float32(target)
    sh = jnp.float32(sharpness)
    mm = jnp.float32(max_mult)

    def barrier(lip: jnp.ndarray) -> jnp.ndarray:
        z = jax.nn.softplus(sh * (lip / t - 1.0))
        return jnp.minimum(1.0 + z, mm)

    return barrier


# ---------------- Fast Lipschitz proxy (self-contained) ---------------- #


def _sv_max_power_flat(W: jnp.ndarray, iters: int = 1) -> jnp.ndarray:
    """Approximate top singular value of a weight tensor by flattening to (out, -1)."""
    W2 = jnp.reshape(W, (W.shape[0], -1))
    v = jnp.ones((W2.shape[1],), dtype=W2.dtype)
    v = v / (jnp.linalg.norm(v) + 1e-12)

    def body(i, v):
        u = W2 @ v
        u = u / (jnp.linalg.norm(u) + 1e-12)
        v = W2.T @ u
        v = v / (jnp.linalg.norm(v) + 1e-12)
        return v

    v = jax.lax.fori_loop(0, max(iters, 1), body, v)
    return jnp.linalg.norm(W2 @ v)


def _collect_sigmas(obj: Any, out: List[jnp.ndarray], n_iter: int) -> bool:
    """
    Append a σ estimate for `obj` to `out` if `obj` is a linear-like module.
    Returns True if we handled this object and should not recurse into it.
    """
    # Prefer explicit hint if provided (your spectral wrappers expose this).
    if hasattr(obj, "__operator_norm_hint__"):
        try:
            s = obj.__operator_norm_hint__()
            if s is not None:
                out.append(jnp.asarray(s, jnp.float32))
                return True
        except Exception:
            pass  # fall through and try structure-based rules

    name = type(obj).__name__

    # eqx spectral-normalisation wrapper → ≈1 by design
    if name == "SpectralNorm":
        out.append(jnp.array(1.0, dtype=jnp.float32))
        return True

    # Common linear ops
    try:
        import equinox.nn as nn  # local import to avoid hard dependency if renamed
    except Exception:
        nn = None

    if nn is not None:
        if isinstance(obj, getattr(nn, "Linear", ())) and hasattr(obj, "weight"):
            out.append(_sv_max_power_flat(obj.weight, iters=n_iter).astype(jnp.float32))
            return True

        # Conv / ConvTranspose families; flatten kernel as in standard SN implementations
        conv_like = tuple(
            getattr(nn, k)
            for k in ("Conv", "Conv1d", "Conv2d", "Conv3d")
            if hasattr(nn, k)
        ) + tuple(
            getattr(nn, k)
            for k in (
                "ConvTranspose",
                "ConvTranspose1d",
                "ConvTranspose2d",
                "ConvTranspose3d",
            )
            if hasattr(nn, k)
        )
        if isinstance(obj, conv_like) and hasattr(obj, "weight"):
            out.append(_sv_max_power_flat(obj.weight, iters=n_iter).astype(jnp.float32))
            return True

    # Otherwise, recurse
    if isinstance(obj, eqx.Module):
        for v in vars(obj).values():
            _collect_sigmas(v, out, n_iter)
        return False
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_sigmas(v, out, n_iter)
        return False
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_sigmas(v, out, n_iter)
        return False
    return False


def prod_sigma_proxy(model: Any, *, n_iter: int = 1) -> jnp.ndarray:
    """
    Cheap upper-bound proxy for ‖f‖_Lip ≈ ∏ σ_i.
    - Uses module-provided `__operator_norm_hint__` when available (exact/tight for your spectral layers).
    - Else treats eqx.nn.SpectralNorm ≈ 1.0.
    - Else approximates σ of Linear/Conv/ConvTranspose by flatten+power-iter.
    Stable: computes exp(sum log σ_i)).
    """
    sigs: List[jnp.ndarray] = []
    _collect_sigmas(model, sigs, n_iter=n_iter)
    if not sigs:
        return jnp.array(1.0, dtype=jnp.float32)
    sigs = [jnp.clip(jnp.asarray(s, jnp.float32), 1e-12, 1e12) for s in sigs]
    logs = jnp.stack([jnp.log(s) for s in sigs])
    return jnp.exp(jnp.sum(logs)).astype(jnp.float32)


# -------------- Optax transform -------------- #


class _SpecPenaltyState(NamedTuple):
    count: jnp.ndarray  # step counter (int32)
    last_lip: jnp.ndarray  # cached Lipschitz proxy (float32)
    last_lambda: jnp.ndarray  # effective λ used (after barrier)
    last_lambda_base: jnp.ndarray  # raw schedule λ before barrier
    last_bar: jnp.ndarray  # barrier multiplier used
    lip_updated: jnp.ndarray  # bool flag (True when we recomputed L this step)


def add_spectral_penalty_transform(
    like_model,  # pytree with same structure as your model
    penalty_fn: Optional[Callable[[Any], jnp.ndarray]] = None,
    schedule: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    *,
    lipschitz_fn: Optional[Callable[[Any], jnp.ndarray]] = None,
    barrier: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    barrier_every: int = 0,  # 0 = never compute feedback; >0 = every K steps
) -> optax.GradientTransformation:
    """
    Returns a GradientTransformation that adds λ(t) * ∇_θ R(θ) to the grads.

    - like_model: Equinox model used only for static structure.
    - penalty_fn(model) -> scalar: e.g., global_spectral_penalty(model).
    - schedule(step) -> λ_t.

    Optional feedback:
    - lipschitz_fn(model) -> scalar proxy for Lipschitz constant (default: prod_sigma_proxy).
    - barrier(L) -> multiplicative factor >= 1 that grows when L > target.
    - barrier_every: compute L every K steps (cache otherwise).
    """
    # Default penalty: repo-global spectral penalty (back-compat)
    if penalty_fn is None:
        from quantbayes.stochax.utils.regularizers import (
            global_spectral_penalty as _gsp,
        )

        penalty_fn = _gsp

    if schedule is None:
        schedule = lambda _: jnp.array(0.0, dtype=jnp.float32)

    if lipschitz_fn is None:
        lipschitz_fn = lambda m: prod_sigma_proxy(m, n_iter=1)

    params_like, static_like = eqx.partition(like_model, eqx.is_inexact_array)

    # penalty over params only (rebuild full model under the hood)
    def _penalty_over_params(params):
        model = eqx.combine(params, static_like)
        return jnp.asarray(penalty_fn(model), dtype=jnp.float32)

    penalty_grad = jax.grad(_penalty_over_params)

    def _compute_lip(params):
        model = eqx.combine(params, static_like)
        return jnp.asarray(lipschitz_fn(model), dtype=jnp.float32)

    def init_fn(params):
        del params  # optax API
        return _SpecPenaltyState(
            count=jnp.array(0, dtype=jnp.int32),
            last_lip=jnp.array(1.0, dtype=jnp.float32),
            last_lambda=jnp.array(0.0, dtype=jnp.float32),
            last_lambda_base=jnp.array(0.0, dtype=jnp.float32),
            last_bar=jnp.array(1.0, dtype=jnp.float32),
            lip_updated=jnp.array(False),
        )

    def update_fn(updates, state, params=None):
        count = state.count
        lam_base = jnp.asarray(schedule(count), dtype=jnp.float32)

        # Defaults (no feedback)
        lip = state.last_lip
        lip_updated_flag = jnp.array(False)
        bar_mult = jnp.array(1.0, dtype=jnp.float32)

        # Only decide in Python on NON-traced things
        do_feedback = (params is not None) and (barrier is not None)

        if do_feedback:
            # Avoid Python boolean on traced `lam_base`; gate with lax.cond instead.
            def _with_lambda(_):
                # Optionally recompute L every `barrier_every` steps.
                if barrier_every and barrier_every > 0:
                    do_now = (count % jnp.int32(barrier_every)) == 0

                    def _recompute(_):
                        L = _compute_lip(params)
                        return L, jnp.array(True)

                    def _reuse(_):
                        return state.last_lip, jnp.array(False)

                    return jax.lax.cond(do_now, _recompute, _reuse, operand=None)
                else:
                    return state.last_lip, jnp.array(False)

            def _no_lambda(_):
                return state.last_lip, jnp.array(False)

            lip, lip_updated_flag = jax.lax.cond(
                lam_base > 0.0, _with_lambda, _no_lambda, operand=None
            )
            bar_mult = barrier(lip)

        lam_eff = lam_base * bar_mult

        def add_penalty(_):
            if params is None:
                return updates
            pgrad = penalty_grad(params)
            # Cast to grads' dtype per leaf
            return jax.tree_map(
                lambda g, pg: g + (lam_eff * pg).astype(g.dtype), updates, pgrad
            )

        new_updates = jax.lax.cond(
            lam_eff > 0.0, add_penalty, lambda _: updates, operand=None
        )

        new_state = _SpecPenaltyState(
            count=count + 1,
            last_lip=lip,
            last_lambda=lam_eff,
            last_lambda_base=lam_base,
            last_bar=bar_mult,
            lip_updated=lip_updated_flag,
        )
        return new_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# ----------------------- Metrics helpers ----------------------- #


def _find_spec_state(tree) -> Optional[_SpecPenaltyState]:
    """Recursively search an Optax state tree for our _SpecPenaltyState."""
    if isinstance(tree, _SpecPenaltyState):
        return tree
    if isinstance(tree, (tuple, list)):
        for t in tree:
            out = _find_spec_state(t)
            if out is not None:
                return out
    if isinstance(tree, dict):
        for t in tree.values():
            out = _find_spec_state(t)
            if out is not None:
                return out
    for name in ("inner_state", "state", "base_state", "mu", "nu"):
        if hasattr(tree, name):
            out = _find_spec_state(getattr(tree, name))
            if out is not None:
                return out
    return None


def specpen_metrics_from_opt_state(opt_state) -> Optional[Dict[str, Any]]:
    """
    Extract metrics if the spectral-penalty transform is present in the Optax chain.
    Returns:
      {"step": int, "last_lip": float, "last_lambda": float,
       "last_lambda_base": float, "last_bar": float, "lip_updated": bool}
    or None if not found.
    """
    st = _find_spec_state(opt_state)
    if st is None:
        return None
    return {
        "step": int(st.count),
        "last_lip": float(st.last_lip),
        "last_lambda": float(st.last_lambda),
        "last_lambda_base": float(st.last_lambda_base),
        "last_bar": float(st.last_bar),
        "lip_updated": bool(st.lip_updated),
    }
