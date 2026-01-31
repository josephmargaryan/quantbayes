# train_vae.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

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
    beta_schedule: str = "linear"  # "linear" | "cosine" | "constant"
    beta_warmup_steps: int = 5_000
    free_bits: float = 0.0  # e.g. 0.02 * latent_dim
    likelihood: Likelihood = "gaussian"
    gaussian_learn_logvar: bool = (
        False  # if True, model must expose `gauss_logvar_param` (scalar or broadcastable)
    )
    logvar_clamp: Tuple[float, float] = (-10.0, 10.0)
    seed: int = 42
    verbose: bool = True


def _iterate_minibatches(rng, X, batch_size):
    n = X.shape[0]
    perm = jax.random.permutation(rng, n)
    for start in range(0, n, batch_size):
        yield X[perm[start : start + batch_size]]


def train_vae(model, data: jnp.ndarray, cfg: TrainConfig):
    """
    Generic VAE trainer. Expects model(x, rng, train=True) -> (decoder_out, mu, logvar).
    - If cfg.likelihood == "bernoulli": decoder_out are logits.
    - If cfg.likelihood == "gaussian": decoder_out are means; if gaussian_learn_logvar=True,
      model must expose `gauss_logvar_param` (scalar or broadcastable).
    Returns the updated (trained) model.
    """
    # Optax pipeline
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay),
    )
    opt_state = tx.init(eqx.filter(model, eqx.is_array))
    beta_fn = make_beta(cfg.beta_schedule, cfg.beta_warmup_steps)

    @eqx.filter_jit
    def loss_fn(m, x, step, rng):
        rng_e, rng_s = jax.random.split(rng)
        # Forward
        dec_out, mu, logvar = m(x, rng_e, train=True)
        # Clamp encoder logvar for numeric stability
        lo, hi = cfg.logvar_clamp
        logvar = jnp.clip(logvar, lo, hi)
        # Recon term
        if cfg.likelihood == "bernoulli":
            recon = losses.recon_bernoulli_logits(x, dec_out)
        else:
            if cfg.gaussian_learn_logvar:
                # expect m.gauss_logvar_param (scalar or broadcastable to x shape)
                glv = m.gauss_logvar_param  # pytrees supported by Eqx
                glv = jnp.broadcast_to(glv, x.shape)
                recon = losses.recon_gaussian(x, dec_out, glv)
            else:
                recon = losses.recon_gaussian(x, dec_out, None)
        # KL
        kl = losses.kl_std_normal(mu, logvar)
        beta = beta_fn(step)
        beta = jnp.asarray(beta, dtype=x.dtype)  # ensure same dtype as inputs
        elbo = losses.elbo(recon, kl, beta=beta, free_bits=cfg.free_bits)
        return jnp.mean(elbo)

    @eqx.filter_jit
    def step_fn(m, opt_state, x, step, rng):
        (loss_val), grads = eqx.filter_value_and_grad(loss_fn)(m, x, step, rng)
        updates, opt_state = tx.update(grads, opt_state, m)
        m = eqx.apply_updates(m, updates)
        return m, opt_state, loss_val

    rng = jax.random.PRNGKey(cfg.seed)
    step = 0
    model_out = model
    for epoch in range(cfg.epochs):
        rng, perm_key = jax.random.split(rng)
        running = 0.0
        batches = 0
        for x_batch in _iterate_minibatches(perm_key, data, cfg.batch_size):
            rng, bkey = jax.random.split(rng)
            model_out, opt_state, loss_val = step_fn(
                model_out, opt_state, x_batch, step, bkey
            )
            running += loss_val
            batches += 1
            step += x_batch.shape[0]
        if cfg.verbose:
            print(
                f"[VAE] epoch {epoch+1}/{cfg.epochs} | loss={float(running/max(1,batches)):.4f} | beta={beta_fn(step):.3f}"
            )
    return model_out
