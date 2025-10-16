# quantbayes/stochax/diffusion/trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import equinox as eqx
import jax.random as jr
import jax.tree_util as jtu
import optax

from quantbayes.stochax.diffusion.checkpoint import save_checkpoint
from quantbayes.stochax.diffusion.sde import batch_loss_fn


def _ema_update(old, new, decay: float):
    # EMA only on inexact array leaves; keep non-arrays as-is.
    old_params, old_static = eqx.partition(old, eqx.is_inexact_array)
    new_params, _ = eqx.partition(new, eqx.is_inexact_array)
    ema_params = jtu.tree_map(
        lambda e, n: decay * e + (1.0 - decay) * n, old_params, new_params
    )
    return eqx.combine(ema_params, old_static)


@eqx.filter_jit
def _make_step(
    model,
    ema_model,
    opt_state,
    batch,
    key,
    *,
    loss_callable: Callable,  # static: (model, batch, key) -> scalar
    optimizer: optax.GradientTransformation,  # static
    ema_decay: float,  # static
):
    loss_fn = eqx.filter_value_and_grad(loss_callable)
    loss, grads = loss_fn(model, batch, key)
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    ema_model = _ema_update(ema_model, model, ema_decay)
    return loss, model, ema_model, opt_state


def train_model(
    model,
    dataset,
    t1,
    lr,
    num_steps,
    batch_size,
    weight_fn,
    int_beta_fn,
    print_every,
    seed,
    *,
    data_loader_func,
    grad_clip_norm: float = 1.0,
    ema_decay: float = 0.999,
    weight_decay: float = 0.0,
    optimizer: optax.GradientTransformation | None = None,
    # --- objective selection ---
    loss_impl: str | None = None,  # "score" (default) | "edm"
    custom_loss: Optional[Callable] = None,  # if given, overrides loss_impl
    # --- checkpointing ---
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 0,
    keep_last: int = 3,
    # --- best-checkpoint support ---
    eval_every: int = 0,  # if > 0, run eval every N steps
    eval_fn: Optional[Callable] = None,  # Callable(ema_model_inference, key) -> float
    best_mode: str = "min",  # "min" or "max"
):
    """
    Returns the EMA model. If checkpoint_dir is provided, saves periodic checkpoints.
    If eval_* is set, also tracks a best metric on the EMA model and writes best.txt.

    Notes
    -----
    - `eval_fn` should accept the **EMA** model already set to inference mode and a PRNGKey,
      and return a Python float metric. Smaller is better if best_mode="min", else larger.
    - For EDM, you can pass eval_fn that computes a small held-out edm_batch_loss.
    """
    key = jr.PRNGKey(seed)
    model_key, loader_key = jr.split(key)

    if optimizer is None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip_norm),
            optax.adamw(lr, weight_decay=weight_decay),
        )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Start EMA as a pure copy (no serialisation side effects).
    ema_model = jtu.tree_map(lambda x: x, model)

    # Build the loss callable
    if custom_loss is not None:
        loss_callable = custom_loss
    else:
        which = (loss_impl or "score").lower()
        if which == "score":

            def loss_callable(m, batch, k):
                return batch_loss_fn(m, weight_fn, int_beta_fn, batch, t1, k)

        elif which == "edm":
            raise ValueError(
                "EDM selected but no custom EDM loss passed. "
                "Import edm_batch_loss from edm.py and pass via custom_loss=..."
            )
        else:
            raise ValueError(f"Unknown loss_impl: {which!r}")

    # dataloader
    loader = data_loader_func(dataset, batch_size, key=loader_key)

    # running stats for pretty printing
    total_value = 0.0
    total_size = 0

    # best checkpoint tracking
    mode = best_mode.lower()
    if mode not in ("min", "max"):
        raise ValueError("best_mode must be 'min' or 'max'")
    best_metric: Optional[float] = None
    best_step: Optional[int] = None
    ckpt_path = Path(checkpoint_dir) if checkpoint_dir else None

    for step in range(1, num_steps + 1):
        batch = next(loader)
        subkey, model_key = jr.split(model_key)

        loss, model, ema_model, opt_state = _make_step(
            model,
            ema_model,
            opt_state,
            batch,
            subkey,
            loss_callable=loss_callable,
            optimizer=optimizer,
            ema_decay=ema_decay,
        )

        total_value += float(loss)
        total_size += 1

        if (step % print_every) == 0 or step == num_steps:
            print(f"Step={step}  Loss={total_value / max(total_size,1):.6f}")
            total_value = 0.0
            total_size = 0

        # periodic checkpoint
        if ckpt_path and checkpoint_every > 0 and (step % checkpoint_every) == 0:
            save_checkpoint(
                ckpt_path,
                model=model,
                ema_model=ema_model,
                opt_state=opt_state,
                step=step,
                extras={"loss_avg": float(loss)},
                keep_last=keep_last,
            )

        # best checkpoint via user eval
        if (
            ckpt_path
            and eval_every > 0
            and eval_fn is not None
            and (step % eval_every) == 0
        ):
            eval_key, model_key = jr.split(model_key)
            # Turn on inference mode for eval
            ema_eval = eqx.tree_inference(ema_model, value=True)
            metric = float(eval_fn(ema_eval, eval_key))

            def _is_better(cur, best):
                if best is None:
                    return True
                return (cur < best) if mode == "min" else (cur > best)

            if _is_better(metric, best_metric):
                best_metric = metric
                best_step = step

                save_checkpoint(
                    ckpt_path,
                    model=model,
                    ema_model=ema_model,
                    opt_state=opt_state,
                    step=step,
                    extras={"best": True, "metric": metric},
                    keep_last=keep_last,
                )
                # write/update best.txt pointer
                (ckpt_path / "best.txt").write_text(str(step))
                print(f"[best] step={step} metric={metric:.6f}")

    # final save
    if ckpt_path:
        save_checkpoint(
            ckpt_path,
            model=model,
            ema_model=ema_model,
            opt_state=opt_state,
            step=num_steps,
            extras={"final": True},
            keep_last=keep_last,
        )
        # If no eval ran, optionally set best to final
        if best_step is None and (ckpt_path / "best.txt").exists() is False:
            (ckpt_path / "best.txt").write_text(str(num_steps))

    return ema_model
