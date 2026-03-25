# quantbayes/stochax/vae/train_vae.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from . import losses
from .schedules import make_beta_schedule as make_beta

Likelihood = Literal["bernoulli", "gaussian"]


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    beta_schedule: str = "linear"  # "linear" | "cosine" | "sigmoid" | "constant"
    beta_warmup_steps: int = 5_000
    free_bits: float = 0.0  # e.g. 0.02 * latent_dim

    likelihood: Likelihood = "gaussian"
    gaussian_learn_logvar: bool = False  # uses model.gauss_logvar_param if True
    logvar_clamp: Tuple[float, float] = (-10.0, 10.0)

    seed: int = 42
    verbose: bool = True
    drop_last: bool = True  # keeps batch shapes static (fewer recompiles)


def _iterate_minibatches(rng, X, batch_size: int, *, drop_last: bool):
    n = X.shape[0]
    perm = jax.random.permutation(rng, n)
    end = n - (n % batch_size) if drop_last else n
    for start in range(0, end, batch_size):
        yield X[perm[start : start + batch_size]]


def train_vae(model, data: jnp.ndarray, cfg: TrainConfig):
    """
    Trainer expects model(x, rng, train=True) -> (decoder_out, mu, logvar).

    IMPORTANT JAX NOTE:
      - step is passed into jitted fns as a *JAX scalar* to avoid recompiling every batch.
      - step counts optimizer updates (not number of examples).
    """
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay),
    )
    params0 = eqx.filter(model, eqx.is_inexact_array)
    opt_state = tx.init(params0)

    beta_fn = make_beta(cfg.beta_schedule, cfg.beta_warmup_steps)

    @eqx.filter_jit
    def loss_fn(m, x, step_scalar: jnp.ndarray, rng):
        dec_out, mu, logvar = m(x, rng, train=True)

        lo, hi = cfg.logvar_clamp
        logvar = jnp.clip(logvar, lo, hi)

        if cfg.likelihood == "bernoulli":
            recon = losses.recon_bernoulli_logits(x, dec_out)
        else:
            if cfg.gaussian_learn_logvar:
                glv = jnp.asarray(m.gauss_logvar_param, dtype=x.dtype)
                glv = jnp.broadcast_to(glv, x.shape)
                recon = losses.recon_gaussian(x, dec_out, glv)
            else:
                recon = losses.recon_gaussian(x, dec_out, None)

        kl = losses.kl_std_normal(mu, logvar)

        beta = beta_fn(step_scalar)
        beta = jnp.asarray(beta, dtype=x.dtype)

        obj = losses.elbo(recon, kl, beta=beta, free_bits=cfg.free_bits)
        return jnp.mean(obj)

    @eqx.filter_jit
    def step_fn(m, opt_state, x, step_scalar: jnp.ndarray, rng):
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(m, x, step_scalar, rng)
        params = eqx.filter(m, eqx.is_inexact_array)
        updates, opt_state = tx.update(grads, opt_state, params)
        m = eqx.apply_updates(m, updates)
        return m, opt_state, loss_val

    rng = jax.random.PRNGKey(cfg.seed)
    step_i = 0  # optimizer step counter
    model_out = model

    for epoch in range(cfg.epochs):
        rng, perm_key = jax.random.split(rng)
        running = 0.0
        batches = 0

        for x_batch in _iterate_minibatches(
            perm_key, data, cfg.batch_size, drop_last=cfg.drop_last
        ):
            rng, bkey = jax.random.split(rng)

            # CRITICAL: pass step as a JAX scalar to avoid recompiling each batch
            step_scalar = jnp.asarray(step_i, dtype=jnp.int32)

            model_out, opt_state, loss_val = step_fn(
                model_out, opt_state, x_batch, step_scalar, bkey
            )

            running += loss_val
            batches += 1
            step_i += 1

        if cfg.verbose:
            beta_val = float(beta_fn(jnp.asarray(step_i, dtype=jnp.int32)))
            avg = float(running / jnp.maximum(batches, 1))
            print(
                f"[VAE] epoch {epoch+1}/{cfg.epochs} | loss={avg:.4f} | beta={beta_val:.3f}"
            )

    return model_out
