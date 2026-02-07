# quantbayes/stochax/energy_based/train.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from .base import BaseEBM


@dataclass(frozen=True)
class PCDTrainConfig:
    steps: int = 50_000
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0

    # persistent negatives
    reinit_prob: float = 0.05
    init_scale: float = 1.0

    # SGLD for negatives
    sgld_steps: int = 60
    sgld_step_size: float = 1e-2

    clamp_min: Optional[float] = 0.0
    clamp_max: Optional[float] = 1.0

    # stabilizers
    l2_energy: float = 0.0  # e.g. 1e-4 can help
    seed: int = 0
    print_every: int = 500


def _dataloader_x(
    x: jnp.ndarray, batch_size: int, *, key: jr.PRNGKey, drop_last: bool = True
):
    n = x.shape[0]
    idx = jnp.arange(n)
    while True:
        key, sub = jr.split(key)
        perm = jr.permutation(sub, idx)
        end = n - (n % batch_size) if drop_last else n
        for i in range(0, end, batch_size):
            bidx = perm[i : i + batch_size]
            yield x[bidx]


def _init_replay_buffer(
    key: jr.PRNGKey,
    batch_shape: tuple[int, ...],
    *,
    init_scale: float,
    clamp_min: Optional[float],
    clamp_max: Optional[float],
) -> jnp.ndarray:
    x = jr.normal(key, batch_shape) * float(init_scale)
    if clamp_min is not None and clamp_max is not None:
        x = jnp.clip(x, float(clamp_min), float(clamp_max))
    return x


def _sgld_chain(
    ebm: BaseEBM,
    x0: jnp.ndarray,
    *,
    key: jr.PRNGKey,
    n_steps: int,
    step_size: float,
    clamp_min: Optional[float],
    clamp_max: Optional[float],
) -> jnp.ndarray:
    """
    Runs SGLD on x to obtain negatives.
    NOTE: sampling is treated as *non-differentiable* in training by stop_gradient at the end.
    """
    step_size = float(step_size)

    def sum_energy(z):
        return jnp.sum(ebm.energy_batch(z))

    def body(i, x_t):
        k = jr.fold_in(key, i)
        noise = jr.normal(k, x_t.shape)
        gradE = jax.grad(sum_energy)(x_t)  # ∇x E
        # SGLD: x <- x - 0.5*η*∇E + sqrt(η)*N
        x_t = x_t - 0.5 * step_size * gradE + jnp.sqrt(step_size) * noise
        if clamp_min is not None and clamp_max is not None:
            x_t = jnp.clip(x_t, float(clamp_min), float(clamp_max))
        return x_t

    return jax.lax.fori_loop(0, int(n_steps), body, x0)


@eqx.filter_jit
def pcd_train_step(
    ebm: BaseEBM,
    opt_state,
    replay: jnp.ndarray,
    x_real: jnp.ndarray,
    key: jr.PRNGKey,
    *,
    optimizer: optax.GradientTransformation,
    cfg: PCDTrainConfig,
):
    """
    One PCD step:
      - choose init negatives from replay (with reinit_prob random restart)
      - run SGLD to get x_neg
      - stop-gradient through x_neg
      - update parameters with grad of E(real)-E(neg) (+ optional L2 energy penalty)
      - update replay = x_neg
    """
    b = x_real.shape[0]
    k_reinit, k_init, k_sgld = jr.split(key, 3)

    # broadcast mask over sample dims
    mask_shape = (b,) + (1,) * (x_real.ndim - 1)
    reinit = jr.uniform(k_reinit, mask_shape) < jnp.asarray(
        cfg.reinit_prob, jnp.float32
    )

    x_rand = jr.normal(k_init, x_real.shape) * float(cfg.init_scale)
    if cfg.clamp_min is not None and cfg.clamp_max is not None:
        x_rand = jnp.clip(x_rand, float(cfg.clamp_min), float(cfg.clamp_max))

    x0 = jnp.where(reinit, x_rand, replay)

    def loss_fn(m: BaseEBM):
        x_neg = _sgld_chain(
            m,
            x0,
            key=k_sgld,
            n_steps=cfg.sgld_steps,
            step_size=cfg.sgld_step_size,
            clamp_min=cfg.clamp_min,
            clamp_max=cfg.clamp_max,
        )
        x_neg = jax.lax.stop_gradient(x_neg)

        e_real = jnp.mean(m.energy_batch(x_real))
        e_neg = jnp.mean(m.energy_batch(x_neg))

        loss = e_real - e_neg
        if cfg.l2_energy > 0.0:
            loss = loss + float(cfg.l2_energy) * (e_real * e_real + e_neg * e_neg)

        return loss, (e_real, e_neg, x_neg)

    (loss, (e_real, e_neg, x_neg)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(ebm)

    params = eqx.filter(ebm, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    ebm = eqx.apply_updates(ebm, updates)

    replay = x_neg  # already stop_grad
    return ebm, opt_state, replay, loss, e_real, e_neg


def train_ebm_pcd(
    ebm: BaseEBM,
    x_train: jnp.ndarray,
    *,
    cfg: PCDTrainConfig,
    replay_init_key: Optional[jr.PRNGKey] = None,
) -> Tuple[BaseEBM, jnp.ndarray]:
    """
    Simple training loop for array datasets (JAX arrays).
    Returns (trained_ebm, replay_buffer).
    """
    opt = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(cfg.lr, weight_decay=cfg.weight_decay),
    )
    opt_state = opt.init(eqx.filter(ebm, eqx.is_inexact_array))

    key = jr.PRNGKey(cfg.seed)
    loader = _dataloader_x(
        x_train, cfg.batch_size, key=jr.PRNGKey(cfg.seed + 123), drop_last=True
    )

    # init replay buffer to batch shape
    if replay_init_key is None:
        replay_init_key = jr.PRNGKey(cfg.seed + 999)
    replay = _init_replay_buffer(
        replay_init_key,
        (cfg.batch_size, *x_train.shape[1:]),
        init_scale=cfg.init_scale,
        clamp_min=cfg.clamp_min,
        clamp_max=cfg.clamp_max,
    )

    run = 0.0
    ct = 0
    for step in range(1, cfg.steps + 1):
        x_real = next(loader)
        key, sub = jr.split(key)

        ebm, opt_state, replay, loss, e_real, e_neg = pcd_train_step(
            ebm,
            opt_state,
            replay,
            x_real,
            sub,
            optimizer=opt,
            cfg=cfg,
        )

        run += float(loss)
        ct += 1
        if cfg.print_every > 0 and step % cfg.print_every == 0:
            print(
                f"[EBM] step {step:6d} | loss={run/max(ct,1):.6f} | "
                f"E(real)={float(e_real):.4f} E(neg)={float(e_neg):.4f}"
            )
            run, ct = 0.0, 0

    return ebm, replay
