# spectral_penalty_tx.py
from __future__ import annotations
from typing import Callable, Optional, NamedTuple, Dict, Any

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from quantbayes.stochax.layers.theory_tools import (
    spectral_frobenius_lists,
    lipschitz_serial,
)

"""
Example usage:

```python
import optax, equinox as eqx, jax.random as jr
from spectral_penalty_tx import (
    make_lambda_spec_schedule, make_soft_barrier,
    add_spectral_penalty_transform, prod_sigma_fast,
    specpen_metrics_from_opt_state
)

model, state = eqx.nn.make_with_state(YourModel)(key=jr.PRNGKey(0))

lam_sched = make_lambda_spec_schedule(
    peak=1e-3, warmup_steps=5_000, hold_steps=10_000, decay_steps=30_000, floor=1e-5
)
bar = make_soft_barrier(target=8.0, sharpness=4.0, max_mult=8.0)

tx = optax.chain(
    add_spectral_penalty_transform(
        like_model=model,
        schedule=lam_sched,
        lipschitz_fn=prod_sigma_fast,
        barrier=bar,
        barrier_every=200,  # compute L every 200 steps
    ),
    optax.adamw(learning_rate=1e-3, weight_decay=1e-4),
)

params = eqx.filter(model, eqx.is_inexact_array)
opt_state = tx.init(params)
```

# IMPORTANT: set lambda_spec=0.0 in your loss call; the transform adds the spectral penalty gradient.
# ...pass `tx` as the optimizer and `opt_state` into your train loop...


Want logging without affecting λ? You can pass a “neutral” barrier:

```python
tx = optax.chain(
    add_spectral_penalty_transform(
        like_model=model,
        schedule=lam_sched,
        lipschitz_fn=prod_sigma_fast,
        barrier=lambda L: jnp.ones_like(L),  # logs L but leaves λ unchanged
        barrier_every=200,
    ),
    optax.adamw(...),
)
```


## TODO 
# in trainer.train 
# at top of file
```python
from quantbayes.stochax.spectral_penalty_tx import specpen_metrics_from_opt_state

def train(...,
          log_lipschitz_from_tx: bool = False,
          return_lipschitz_logs: bool = False,
          ...):
    lip_hist, lam_hist = [], []

    ...
    for xb, yb in data_loader(...):
        rng, step_rng = jr.split(rng)
        # IMPORTANT: pass the *tx* and its opt_state:
        model, state, opt_state, batch_loss = train_step(
            model, state, opt_state, xb, yb, step_rng,
            loss_fn, optimizer,   # here `optimizer` should be your `tx`
            lambda_spec=0.0,      # <-- prevent double-counting; transform adds spec penalty
            lambda_frob=lambda_frob,
        )

        if log_lipschitz_from_tx:
            m = specpen_metrics_from_opt_state(opt_state)
            if m and m["lip_updated"]:
                lip_hist.append((m["step"], m["last_lip"]))
                lam_hist.append((m["step"], m["last_lambda"]))

    ...
    if return_lipschitz_logs:
        return best_model, best_state, train_losses, val_losses, penalty_history, (lip_hist, lam_hist)
    else:
        return best_model, best_state, train_losses, val_losses, penalty_history
```
"""


# ---------------- Schedules ---------------- #


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
            # cosine from peak → floor
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
    """
    Smoothly increase strength when a Lipschitz proxy L exceeds 'target'.
    Returns a multiplicative factor in [1, max_mult].
    """
    t = jnp.float32(target)
    sh = jnp.float32(sharpness)
    mm = jnp.float32(max_mult)

    def barrier(lip: jnp.ndarray) -> jnp.ndarray:
        # 1 + softplus(sh * (L/target - 1)), clipped at max_mult
        z = jax.nn.softplus(sh * (lip / t - 1.0))
        return jnp.minimum(1.0 + z, mm)

    return barrier


# -------------- Optax transform -------------- #


class _SpecPenaltyState(NamedTuple):
    count: jnp.ndarray  # step counter (int32)
    last_lip: jnp.ndarray  # cached Lipschitz proxy (float32)
    last_lambda: jnp.ndarray  # effective λ used (after barrier) (float32)
    last_lambda_base: jnp.ndarray  # raw schedule λ before barrier (float32)
    last_bar: jnp.ndarray  # barrier multiplier used (float32)
    lip_updated: jnp.ndarray  # bool flag (True when we recomputed L this step)


def add_spectral_penalty_transform(
    like_model,  # a template pytree with same structure as your model
    penalty_fn: Optional[Callable[[Any], jnp.ndarray]] = None,
    schedule: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    *,
    # Optional feedback:
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
    - lipschitz_fn(model) -> scalar proxy for Lipschitz constant.
    - barrier(L) -> multiplicative factor >= 1 that grows when L > target.
    - barrier_every: compute L every K steps (cache otherwise).
    """
    # Default penalty: your repo's global_spectral_penalty; graceful fallback otherwise.
    if penalty_fn is None:
        try:
            from quantbayes.stochax.trainer.train import global_spectral_penalty as _gsp

            penalty_fn = _gsp  # type: ignore
        except Exception:

            def penalty_fn(m):
                params = eqx.filter(m, eqx.is_inexact_array)
                leaves = jax.tree_util.tree_leaves(params)
                acc = jnp.array(0.0, dtype=jnp.float32)
                for p in leaves:
                    if isinstance(p, jnp.ndarray) and p.dtype.kind in ("f", "c"):
                        acc = acc + jnp.sum(p**2)
                return acc

    if schedule is None:
        schedule = lambda _: jnp.array(0.0, dtype=jnp.float32)

    params_like, static_like = eqx.partition(like_model, eqx.is_inexact_array)

    # penalty over params only (rebuild full model under the hood)
    def _penalty_over_params(params):
        model = eqx.combine(params, static_like)
        return jnp.asarray(penalty_fn(model), dtype=jnp.float32)

    penalty_grad = jax.grad(_penalty_over_params)

    # Lipschitz proxy
    def _compute_lip(params):
        if lipschitz_fn is None:
            return jnp.array(1.0, dtype=jnp.float32)
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
        # 'updates' are the incoming grads; add λ ∇R on top.
        count = state.count
        lam_base = jnp.asarray(schedule(count), dtype=jnp.float32)

        # Optional barrier modulation
        lip = state.last_lip
        lip_updated_flag = state.lip_updated
        bar_mult = jnp.array(1.0, dtype=jnp.float32)

        if (
            (barrier is not None)
            and (lipschitz_fn is not None)
            and (params is not None)
        ):
            if barrier_every and barrier_every > 0:
                do_now = (count % jnp.int32(barrier_every)) == 0

                def _recompute(_):
                    L = _compute_lip(params)
                    return (L, jnp.array(True))

                def _reuse(_):
                    return (state.last_lip, jnp.array(False))

                lip, lip_updated_flag = jax.lax.cond(
                    do_now, _recompute, _reuse, operand=None
                )
            else:
                lip = _compute_lip(params)
                lip_updated_flag = jnp.array(True)

            bar_mult = barrier(lip)

        lam_eff = lam_base * bar_mult

        def add_penalty(_):
            if params is None:
                return updates
            pgrad = penalty_grad(params)
            return jax.tree_map(lambda g, pg: g + lam_eff * pg, updates, pgrad)

        new_updates = jax.lax.cond(
            lam_eff > 0.0,
            add_penalty,
            lambda _: updates,
            operand=None,
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
    # NamedTuples in Optax are subclasses of tuple → iterate positionally
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
    # Some wrappers store nested state in attributes; best-effort
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
      {
        "step": int,
        "last_lip": float,
        "last_lambda": float,        # effective λ (after barrier)
        "last_lambda_base": float,   # raw schedule λ (before barrier)
        "last_bar": float,           # barrier multiplier
        "lip_updated": bool,         # True if L was recomputed on this step
      }
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


# ----------------------- Fast Lipschitz proxy ----------------------- #


def prod_sigma_fast(model) -> jnp.ndarray:
    """
    Very light proxy for ‖f‖_Lip ≈ ∏ σ_i(model).
    Uses n_iter=1 power iteration for 2D blocks to stay cheap.
    """
    sigmas, _ = spectral_frobenius_lists(model, method2d="power", n_iter=1)
    return jnp.asarray(lipschitz_serial(sigmas), dtype=jnp.float32)
