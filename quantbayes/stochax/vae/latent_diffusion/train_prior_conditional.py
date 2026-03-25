# quantbayes/stochax/vae/latent_diffusion/train_prior_conditional.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax

from quantbayes.stochax.diffusion.checkpoint import save_checkpoint, load_checkpoint
from quantbayes.stochax.diffusion.edm import edm_precond_scalars, edm_lambda_weight


def dataloader_xy(z: jnp.ndarray, y: jnp.ndarray, batch_size: int, *, key: jr.PRNGKey):
    n = z.shape[0]
    idx = jnp.arange(n)
    while True:
        key, sub = jr.split(key)
        perm = jr.permutation(sub, idx)
        for i in range(0, n, batch_size):
            bidx = perm[i : i + batch_size]
            yield z[bidx], y[bidx]


def _ema_update(old, new, decay: float):
    old_params, old_static = eqx.partition(old, eqx.is_inexact_array)
    new_params, _ = eqx.partition(new, eqx.is_inexact_array)
    ema_params = jtu.tree_map(
        lambda e, n: decay * e + (1.0 - decay) * n, old_params, new_params
    )
    return eqx.combine(ema_params, old_static)


@dataclass(frozen=True)
class LatentEDMCondTrainConfig:
    batch_size: int = 512
    num_steps: int = 50_000
    lr: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    print_every: int = 500
    checkpoint_every: int = 5000
    keep_last: int = 3
    seed: int = 0

    sigma_data: float = 0.5
    sigma_min_train: float = 0.002
    sigma_max_train: float = 80.0

    # CFG training: drop labels with prob p_uncond to learn unconditional branch
    p_uncond: float = 0.10


def _sample_sigmas_uniform_logspace(
    key: jr.PRNGKey, b: int, *, sigma_min: float, sigma_max: float
):
    lo = jnp.log(jnp.asarray(sigma_min, jnp.float32))
    hi = jnp.log(jnp.asarray(sigma_max, jnp.float32))
    rho = jr.uniform(key, (b,), minval=lo, maxval=hi)
    return jnp.exp(rho)  # (B,)


def _edm_single_loss_cond(
    model,
    x: jnp.ndarray,  # (D,)
    sigma: jnp.ndarray,  # scalar
    label: jnp.ndarray,  # scalar int32
    *,
    sigma_data: float,
    key: jr.PRNGKey,
) -> jnp.ndarray:
    k_noise, k_model = jr.split(key, 2)
    noise = jr.normal(k_noise, x.shape)
    y = x + sigma * noise

    c_in, c_skip, c_out = edm_precond_scalars(sigma, sigma_data)
    log_sigma = jnp.log(jnp.maximum(sigma, 1e-12))

    y_in = c_in * y
    out = model(log_sigma, y_in, key=k_model, train=True, label=label)
    denoised = c_skip * y + c_out * out

    w = edm_lambda_weight(sigma, sigma_data)
    return w * jnp.mean((denoised - x) ** 2)


def edm_batch_loss_conditional(
    model,
    z_batch: jnp.ndarray,  # (B,D)
    y_batch: jnp.ndarray,  # (B,)
    key: jr.PRNGKey,
    *,
    sigma_data: float,
    sigma_min: float,
    sigma_max: float,
    p_uncond: float,
    null_label: int,
) -> jnp.ndarray:
    b = z_batch.shape[0]
    k_sig, k_noise, k_drop = jr.split(key, 3)

    sigmas = _sample_sigmas_uniform_logspace(
        k_sig, b, sigma_min=sigma_min, sigma_max=sigma_max
    )

    drop = jr.uniform(k_drop, (b,)) < jnp.asarray(p_uncond, jnp.float32)
    y_in = jnp.where(
        drop, jnp.asarray(null_label, jnp.int32), y_batch.astype(jnp.int32)
    )

    keys = jr.split(k_noise, b)

    per = jax.vmap(
        lambda x_i, s_i, y_i, k_i: _edm_single_loss_cond(
            model, x_i, s_i, y_i, sigma_data=sigma_data, key=k_i
        ),
        in_axes=(0, 0, 0, 0),
        out_axes=0,
    )(z_batch, sigmas, y_in, keys)

    return jnp.mean(per)


@eqx.filter_jit
def _train_step(
    model,
    ema_model,
    opt_state,
    z_batch,
    y_batch,
    key,
    *,
    optimizer,
    cfg,
    null_label: int,
):
    def loss_callable(m, zb, yb, k):
        return edm_batch_loss_conditional(
            m,
            zb,
            yb,
            k,
            sigma_data=cfg.sigma_data,
            sigma_min=cfg.sigma_min_train,
            sigma_max=cfg.sigma_max_train,
            p_uncond=cfg.p_uncond,
            null_label=null_label,
        )

    loss_fn = eqx.filter_value_and_grad(loss_callable)
    loss, grads = loss_fn(model, z_batch, y_batch, key)

    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)

    ema_model = _ema_update(ema_model, model, cfg.ema_decay)
    return loss, model, ema_model, opt_state


def train_or_load_latent_edm_prior_conditional(
    *,
    ckpt_dir: Path,
    model_template,
    z_dataset: jnp.ndarray,  # (N,D)
    y_dataset: jnp.ndarray,  # (N,)
    null_label: int,
    cfg: LatentEDMCondTrainConfig,
) -> Tuple[Any, Any]:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = model_template
    ema_model = jtu.tree_map(lambda x: x, model_template)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(cfg.lr, weight_decay=cfg.weight_decay),
    )
    opt_state_template = optimizer.init(
        eqx.filter(model_template, eqx.is_inexact_array)
    )
    opt_state = opt_state_template

    start_step = 0
    if (ckpt_dir / "latest.txt").exists():
        model, ema_model, opt_state, start_step = load_checkpoint(
            ckpt_dir, model_template, model_template, opt_state_template, step=None
        )
        print(
            f"[latent EDM cond] Loaded checkpoint at step {start_step} from {ckpt_dir}"
        )

    if start_step >= cfg.num_steps:
        print(
            f"[latent EDM cond] Already trained to {start_step} >= {cfg.num_steps}. Skipping."
        )
        return model, ema_model

    loader = dataloader_xy(
        z_dataset, y_dataset, cfg.batch_size, key=jr.PRNGKey(cfg.seed + 101)
    )
    key = jr.PRNGKey(cfg.seed)

    run, ct = 0.0, 0
    for step in range(start_step + 1, cfg.num_steps + 1):
        zb, yb = next(loader)
        key, sub = jr.split(key)

        loss, model, ema_model, opt_state = _train_step(
            model,
            ema_model,
            opt_state,
            zb,
            yb,
            sub,
            optimizer=optimizer,
            cfg=cfg,
            null_label=null_label,
        )

        run += float(loss)
        ct += 1

        if cfg.print_every > 0 and step % cfg.print_every == 0:
            print(f"[latent EDM cond] step {step:6d} | loss {run / max(ct,1):.6f}")
            run, ct = 0.0, 0

        if cfg.checkpoint_every > 0 and step % cfg.checkpoint_every == 0:
            save_checkpoint(
                ckpt_dir,
                model=model,
                ema_model=ema_model,
                opt_state=opt_state,
                step=step,
                extras={"loss": float(loss)},
                keep_last=cfg.keep_last,
            )

    save_checkpoint(
        ckpt_dir,
        model=model,
        ema_model=ema_model,
        opt_state=opt_state,
        step=cfg.num_steps,
        extras={"final": True},
        keep_last=cfg.keep_last,
    )
    print(f"[latent EDM cond] Training complete. Saved {cfg.num_steps} to {ckpt_dir}")

    return model, ema_model
