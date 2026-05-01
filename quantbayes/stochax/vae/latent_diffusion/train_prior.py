# quantbayes/stochax/vae/latent_diffusion/train_prior.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Optional, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax

from quantbayes.stochax.diffusion.edm import edm_batch_loss
from quantbayes.stochax.diffusion.checkpoint import save_checkpoint, load_checkpoint
from quantbayes.stochax.diffusion.dataloaders import dataloader as jax_dataloader


@dataclass(frozen=True)
class LatentEDMTrainConfig:
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


def _ema_update(old, new, decay: float):
    old_params, old_static = eqx.partition(old, eqx.is_inexact_array)
    new_params, _ = eqx.partition(new, eqx.is_inexact_array)
    ema_params = jtu.tree_map(
        lambda e, n: decay * e + (1.0 - decay) * n, old_params, new_params
    )
    return eqx.combine(ema_params, old_static)


@eqx.filter_jit
def _step(
    model,
    ema_model,
    opt_state,
    batch,
    key,
    *,
    optimizer,
    loss_callable,
    ema_decay: float,
):
    loss_fn = eqx.filter_value_and_grad(loss_callable)
    loss, grads = loss_fn(model, batch, key)
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    ema_model = _ema_update(ema_model, model, ema_decay)
    return loss, model, ema_model, opt_state


def train_or_load_latent_edm_prior(
    *,
    ckpt_dir: Path,
    model_template,
    z_dataset: jnp.ndarray,  # (N,D)
    cfg: LatentEDMTrainConfig,
) -> Tuple[Any, Any]:
    """
    Train EDM prior on latent dataset z. Returns (model, ema_model).
    """
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
            ckpt_dir,
            model_template,
            model_template,
            opt_state_template,
            step=None,
        )
        print(f"[latent EDM] Loaded checkpoint at step {start_step} from {ckpt_dir}")

    if start_step >= cfg.num_steps:
        print(
            f"[latent EDM] Already trained to {start_step} >= {cfg.num_steps}. Skipping."
        )
        return model, ema_model

    import math

    rho_min = float(math.log(cfg.sigma_min_train))
    rho_max = float(math.log(cfg.sigma_max_train))

    def loss_callable(m, batch, key):
        return edm_batch_loss(
            m,
            batch,
            key,
            sigma_data=cfg.sigma_data,
            rho_min=rho_min,
            rho_max=rho_max,
            sample="uniform",
        )

    loader = jax_dataloader(z_dataset, cfg.batch_size, key=jr.PRNGKey(cfg.seed + 101))
    key = jr.PRNGKey(cfg.seed)

    run, ct = 0.0, 0
    for step in range(start_step + 1, cfg.num_steps + 1):
        batch = next(loader)
        key, sub = jr.split(key)

        loss, model, ema_model, opt_state = _step(
            model,
            ema_model,
            opt_state,
            batch,
            sub,
            optimizer=optimizer,
            loss_callable=loss_callable,
            ema_decay=cfg.ema_decay,
        )

        run += float(loss)
        ct += 1

        if cfg.print_every > 0 and step % cfg.print_every == 0:
            print(f"[latent EDM] step {step:6d} | loss {run / max(ct,1):.6f}")
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
    print(f"[latent EDM] Training complete. Saved {cfg.num_steps} to {ckpt_dir}")

    return model, ema_model
