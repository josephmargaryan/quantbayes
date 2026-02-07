# quantbayes/stochax/vae/pk/train_score.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from .utils import dataloader, inference_copy, clamp_logvar
from .features import FeatureMap
from .score_model import LatentScoreNet


LossWeight = Literal["none", "sigma2"]


@dataclass(frozen=True)
class ScoreDSMTrainConfig:
    steps: int = 20_000
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    print_every: int = 500
    seed: int = 0

    # noise schedule for DSM
    sigma_min: float = 0.01
    sigma_max: float = 1.0
    sample: Literal["log_uniform", "uniform"] = "log_uniform"
    loss_weight: LossWeight = "sigma2"

    # optional checkpoint
    save_path: Optional[Path] = None
    save_every: int = 0


def _sample_log_sigma(
    key: jr.PRNGKey, b: int, *, sigma_min: float, sigma_max: float, sample: str
):
    if sample == "log_uniform":
        lo = jnp.log(jnp.asarray(sigma_min, jnp.float32))
        hi = jnp.log(jnp.asarray(sigma_max, jnp.float32))
        rho = jr.uniform(key, (b,), minval=lo, maxval=hi)
        return rho
    if sample == "uniform":
        s = jr.uniform(key, (b,), minval=sigma_min, maxval=sigma_max)
        return jnp.log(jnp.maximum(s, 1e-12))
    raise ValueError(f"Unknown sample={sample!r}")


def train_score_on_array(
    model: LatentScoreNet,
    u_data: jnp.ndarray,  # (N,M)
    cfg: ScoreDSMTrainConfig,
) -> LatentScoreNet:
    """
    DSM training on a precomputed array of u samples.
    """
    if u_data.ndim != 2:
        raise ValueError(f"u_data must be (N,M). Got {u_data.shape}")

    # load if exists
    if cfg.save_path is not None:
        p = Path(cfg.save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            with open(p, "rb") as f:
                return eqx.tree_deserialise_leaves(f, model)

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(cfg.lr, weight_decay=cfg.weight_decay),
    )
    params0 = eqx.filter(model, eqx.is_inexact_array)
    opt_state = tx.init(params0)

    key = jr.PRNGKey(cfg.seed)
    loader = dataloader(u_data, cfg.batch_size, key=jr.PRNGKey(cfg.seed + 123))

    @eqx.filter_jit
    def step(m: LatentScoreNet, opt_state, u_clean: jnp.ndarray, key: jr.PRNGKey):
        b = u_clean.shape[0]
        k_sig, k_noise = jr.split(key, 2)
        log_sigma = _sample_log_sigma(
            k_sig,
            b,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            sample=cfg.sample,
        )
        sigma = jnp.exp(log_sigma).reshape((b, 1))

        noise = jr.normal(k_noise, u_clean.shape) * sigma
        u_tilde = u_clean + noise
        target = -(u_tilde - u_clean) / (sigma**2 + 1e-12)

        def loss_fn(mm: LatentScoreNet):
            pred = mm(log_sigma, u_tilde)
            err = pred - target
            per = jnp.mean(err * err, axis=-1)  # (B,)
            if cfg.loss_weight == "sigma2":
                per = per * (sigma[:, 0] ** 2)
            return jnp.mean(per)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        params = eqx.filter(m, eqx.is_inexact_array)
        updates, opt_state = tx.update(grads, opt_state, params)
        m = eqx.apply_updates(m, updates)
        return m, opt_state, loss

    running = 0.0
    for i in range(1, cfg.steps + 1):
        u_batch = next(loader)
        key, sub = jr.split(key)
        model, opt_state, loss = step(model, opt_state, u_batch, sub)
        running += float(loss)

        if cfg.print_every > 0 and i % cfg.print_every == 0:
            print(f"[score] step {i:6d} | loss {running / cfg.print_every:.6f}")
            running = 0.0

        if cfg.save_path is not None and cfg.save_every > 0 and i % cfg.save_every == 0:
            with open(Path(cfg.save_path), "wb") as f:
                eqx.tree_serialise_leaves(f, model)

    if cfg.save_path is not None:
        with open(Path(cfg.save_path), "wb") as f:
            eqx.tree_serialise_leaves(f, model)

    return model


def train_score_from_vae_aggregate(
    model: LatentScoreNet,
    vae,
    x_data: jnp.ndarray,
    feature_map: FeatureMap,
    cfg: ScoreDSMTrainConfig,
    *,
    key: jr.PRNGKey,
    use_mu: bool = False,
    logvar_clamp_range: Tuple[float, float] = (-10.0, 10.0),
    save_path: Optional[Path] = None,
) -> LatentScoreNet:
    """
    DSM training where u=F(z), z~q(z|x) aggregated over the dataset.
    Generates z on-the-fly from the VAE encoder (no giant latent cache).
    """
    save_path = save_path if save_path is not None else cfg.save_path

    # load if exists
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            with open(p, "rb") as f:
                return eqx.tree_deserialise_leaves(f, model)

    vae_eval = inference_copy(vae)

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(cfg.lr, weight_decay=cfg.weight_decay),
    )
    params0 = eqx.filter(model, eqx.is_inexact_array)
    opt_state = tx.init(params0)

    loader = dataloader(x_data, cfg.batch_size, key=jr.PRNGKey(cfg.seed + 77))
    lo, hi = logvar_clamp_range

    @eqx.filter_jit
    def step(m: LatentScoreNet, opt_state, x_batch: jnp.ndarray, key: jr.PRNGKey):
        b = x_batch.shape[0]
        k_enc, k_lat, k_sig, k_noise = jr.split(key, 4)

        # encode -> sample z
        mu, logvar = vae_eval.encoder(x_batch, rng=k_enc, train=False)
        logvar = clamp_logvar(logvar, lo, hi)

        if use_mu:
            z = mu
        else:
            eps = jr.normal(k_lat, mu.shape)
            z = mu + jnp.exp(0.5 * logvar) * eps

        # u = F(z)
        u_clean = feature_map(z)  # (B,M)

        # DSM noise
        log_sigma = _sample_log_sigma(
            k_sig,
            b,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            sample=cfg.sample,
        )
        sigma = jnp.exp(log_sigma).reshape((b, 1))

        noise = jr.normal(k_noise, u_clean.shape) * sigma
        u_tilde = u_clean + noise
        target = -(u_tilde - u_clean) / (sigma**2 + 1e-12)

        def loss_fn(mm: LatentScoreNet):
            pred = mm(log_sigma, u_tilde)
            err = pred - target
            per = jnp.mean(err * err, axis=-1)
            if cfg.loss_weight == "sigma2":
                per = per * (sigma[:, 0] ** 2)
            return jnp.mean(per)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        params = eqx.filter(m, eqx.is_inexact_array)
        updates, opt_state = tx.update(grads, opt_state, params)
        m = eqx.apply_updates(m, updates)
        return m, opt_state, loss

    running = 0.0
    for i in range(1, cfg.steps + 1):
        xb = next(loader)
        key, sub = jr.split(key)
        model, opt_state, loss = step(model, opt_state, xb, sub)
        running += float(loss)

        if cfg.print_every > 0 and i % cfg.print_every == 0:
            print(f"[score agg] step {i:6d} | loss {running / cfg.print_every:.6f}")
            running = 0.0

        if save_path is not None and cfg.save_every > 0 and i % cfg.save_every == 0:
            with open(Path(save_path), "wb") as f:
                eqx.tree_serialise_leaves(f, model)

    if save_path is not None:
        with open(Path(save_path), "wb") as f:
            eqx.tree_serialise_leaves(f, model)

    return model
