# quantbayes/pkdiffusion/samplers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


Array = jax.Array


@dataclass(frozen=True)
class VPSDESamplerConfig:
    """
    Reverse-time SDE Euler-Maruyama sampler for VP SDE, with optional guidance.

    We assume you trained a score model sθ(t,x) ≈ ∇_x log p_t(x).

    Reverse SDE drift (VP):
      dx = [-0.5β(t)x - β(t)sθ(t,x)] dt + sqrt(β(t)) dW_t,  with t decreasing
    """

    t1: float = 10.0
    num_steps: int = 500
    num_samples: int = 2000
    seed: int = 0


def _beta_from_int_beta(int_beta_fn, t: Array) -> Array:
    # beta(t) = d/dt int_beta(t) via JVP
    _, beta = jax.jvp(int_beta_fn, (t,), (jnp.ones_like(t),))
    return beta


def sample_reverse_vp_sde_euler(
    score_model,  # eqx module: (t, x, key=..., train=...) -> score
    int_beta_fn,  # callable
    *,
    sample_shape: tuple[int, ...],
    key: Array,
    t1: float,
    num_steps: int,
    guidance_fn: Optional[
        Callable
    ] = None,  # (t, x, score, int_beta_fn=...) -> extra_score
) -> Array:
    """
    Sample one trajectory end point x0 from t=t1 to t=0.

    guidance_fn returns an extra score term, added to the base score.
    """
    t1 = float(t1)
    num_steps = int(num_steps)

    # Time grid: t0=t1 -> 0
    t_grid = jnp.linspace(t1, 0.0, num_steps + 1)

    # Start from standard normal at time t1
    key_init, key_scan = jr.split(key)
    x = jr.normal(key_init, shape=sample_shape)

    def step(carry, t_pair):
        x, key = carry
        t, t_next = t_pair
        dt = t_next - t  # negative

        beta = _beta_from_int_beta(int_beta_fn, t)
        beta = jnp.maximum(beta, 1e-8)

        score = score_model(t, x, key=None, train=False)
        if guidance_fn is not None:
            score = score + guidance_fn(t, x, score, int_beta_fn=int_beta_fn)

        # Reverse drift
        drift = -0.5 * beta * x - beta * score

        key, sub = jr.split(key)
        noise = jr.normal(sub, shape=x.shape)

        x = x + drift * dt + jnp.sqrt(beta) * jnp.sqrt(-dt) * noise
        return (x, key), x

    t_pairs = jnp.stack([t_grid[:-1], t_grid[1:]], axis=1)  # (num_steps,2)
    (x_final, _), _ = jax.lax.scan(step, (x, key_scan), t_pairs)
    return x_final


def sample_many_reverse_vp_sde_euler(
    score_model,
    int_beta_fn,
    *,
    sample_shape: tuple[int, ...],
    key: Array,
    num_samples: int,
    t1: float,
    num_steps: int,
    guidance_fn: Optional[Callable] = None,
) -> Array:
    keys = jr.split(key, int(num_samples))

    # We jit the per-sample function for speed; eqx.filter_jit is safe here.
    @eqx.filter_jit
    def one(k):
        return sample_reverse_vp_sde_euler(
            score_model,
            int_beta_fn,
            sample_shape=sample_shape,
            key=k,
            t1=t1,
            num_steps=num_steps,
            guidance_fn=guidance_fn,
        )

    return jax.vmap(one)(keys)
