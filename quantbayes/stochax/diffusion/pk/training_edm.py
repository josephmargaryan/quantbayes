from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import equinox as eqx
import jax.random as jr
import jax.tree_util as jtu
import optax

from quantbayes.stochax.diffusion.checkpoint import save_checkpoint, load_checkpoint
from quantbayes.stochax.diffusion.dataloaders import dataloader as jax_dataloader
from quantbayes.stochax.diffusion.edm import edm_batch_loss
from quantbayes.stochax.diffusion.trainer import _make_step


@dataclass(frozen=True)
class EDMTrainConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    num_steps: int = 40000
    ema_decay: float = 0.999
    grad_clip_norm: float = 1.0
    print_every: int = 500
    checkpoint_every: int = 5000
    keep_last: int = 3

    sigma_data: float = 0.5
    sigma_min_train: float = 0.002
    sigma_max_train: float = 80.0

    seed: int = 0


def train_or_load_edm_unconditional(
    *,
    ckpt_dir: Path,
    build_model_fn: Callable[[], eqx.Module],
    dataset,
    cfg: EDMTrainConfig,
) -> Tuple[eqx.Module, eqx.Module]:
    """
    Train (or resume) unconditional EDM model with EMA, using your checkpoint format.
    Returns (model, ema_model).
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_template = build_model_fn()
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
        print(f"[EDM] Loaded checkpoint at step {start_step} from {ckpt_dir}")

    if start_step >= cfg.num_steps:
        print(f"[EDM] Already trained to >= {cfg.num_steps}. Skipping training.")
        return model, ema_model

    # EDM trains with rho in log-sigma space
    import math

    log_sigma_min = float(math.log(cfg.sigma_min_train))
    log_sigma_max = float(math.log(cfg.sigma_max_train))

    def loss_callable(m, batch, key):
        return edm_batch_loss(
            m,
            batch,
            key,
            sigma_data=cfg.sigma_data,
            rho_min=log_sigma_min,
            rho_max=log_sigma_max,
            sample="uniform",
        )

    loader = jax_dataloader(dataset, cfg.batch_size, key=jr.PRNGKey(cfg.seed + 303))
    train_key = jr.PRNGKey(cfg.seed + 404)

    running, count = 0.0, 0
    for step in range(start_step + 1, cfg.num_steps + 1):
        batch = next(loader)
        train_key, sub = jr.split(train_key)

        loss, model, ema_model, opt_state = _make_step(
            model,
            ema_model,
            opt_state,
            batch,
            sub,
            loss_callable=loss_callable,
            optimizer=optimizer,
            ema_decay=cfg.ema_decay,
        )

        running += float(loss)
        count += 1

        if cfg.print_every > 0 and (step % cfg.print_every == 0):
            print(f"[EDM] step {step:6d} | loss {running / max(count,1):.6f}")
            running, count = 0.0, 0

        if cfg.checkpoint_every > 0 and (step % cfg.checkpoint_every == 0):
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
    print(f"[EDM] Training complete. Saved step {cfg.num_steps} to {ckpt_dir}")

    return model, ema_model
