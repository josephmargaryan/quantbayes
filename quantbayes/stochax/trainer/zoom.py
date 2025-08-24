# quantbayes/stochax/trainer/train_zoom.py
from __future__ import annotations

import pathlib
from typing import Any, Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx

from quantbayes.stochax.trainer.train import (
    data_loader,
    multiclass_loss,  # default; pass binary/regression when needed
    global_spectral_penalty,
    global_frobenius_penalty,
    AugmentFn,
)

Array = jnp.ndarray
Key = jr.PRNGKey


def _make_value_fn(
    static_model_part: eqx.Module,
    state: Any,
    xb: Array,
    yb: Array,
    key: Key,
    loss_fn: Callable,
    lambda_spec: float,
    lambda_frob: float,
    deterministic_objective: bool,
) -> Callable[[eqx.Module], Array]:
    """
    Returns f(params) -> scalar objective on a *fixed* mini-batch.

    If deterministic_objective=True, we run the model in inference_mode
    (freezes BN/Dropout), which makes the line search well-posed (smooth,
    reproducible objective along the search direction).
    """

    def f(params: eqx.Module) -> Array:
        mdl = eqx.combine(params, static_model_part)
        if deterministic_objective:
            mdl = eqx.nn.inference_mode(mdl)
        base, _ = loss_fn(mdl, state, xb, yb, key)
        val = base
        if lambda_spec:
            val = val + lambda_spec * global_spectral_penalty(mdl)
        if lambda_frob:
            val = val + lambda_frob * global_frobenius_penalty(mdl)
        return val

    return f


def train_zoom(
    model: eqx.Module,
    state: Any,
    X_train: Array,
    y_train: Array,
    X_val: Array,
    y_val: Array,
    *,
    batch_size: int = 512,
    num_epochs: int = 20,
    patience: int = 5,
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    key: Key,
    augment_fn: Optional[AugmentFn] = None,
    loss_fn: Callable = multiclass_loss,
    # Zoom (Strong-Wolfe) knobs
    max_linesearch_steps: int = 10,
    zoom_c1: float = 1e-4,  # Armijo
    zoom_c2: float = 0.9,  # strong Wolfe
    deterministic_objective: bool = True,
    # checkpoints
    ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    checkpoint_interval: Optional[int] = None,
    return_penalty_history: bool = False,
) -> Union[
    Tuple[eqx.Module, Any, List[float], List[float]],
    Tuple[eqx.Module, Any, List[float], List[float], List[float]],
]:
    """
    Mini-batch Zoom (Strong-Wolfe) line-search with (negative) SGD direction.

    Best practice:
      * Keep f deterministic during the search (freeze BN/Dropout) and keep
        the sampled/augmented batch fixed while searching.
      * Penalties are folded into the objective, so line search reasons about them too.
    """
    if checkpoint_interval is not None:
        assert checkpoint_interval > 0, "`checkpoint_interval` must be > 0"

    # params/static split
    params, static = eqx.partition(model, eqx.is_inexact_array)

    # Direction = -grad (SGD with lr=1.0); Zoom scales it via line search.
    solver = optax.chain(
        optax.sgd(learning_rate=1.0),
        optax.scale_by_zoom_linesearch(
            max_linesearch_steps=max_linesearch_steps, c1=zoom_c1, c2=zoom_c2
        ),
    )
    opt_state = solver.init(params)

    rng, eval_rng = jr.split(key)
    best_val = jnp.inf
    best_params, best_state = params, state

    train_hist: List[float] = []
    val_hist: List[float] = []
    pen_hist: List[float] = []
    patience_ctr = 0

    for epoch in range(1, num_epochs + 1):
        rng, perm = jr.split(rng)
        epoch_loss, n_seen = 0.0, 0

        for xb, yb in data_loader(
            X_train,
            y_train,
            batch_size=batch_size,
            shuffle=True,
            key=perm,
            augment_fn=augment_fn,
        ):
            # Freeze mini-batch for the entire line search
            rng, sk = jr.split(rng)

            f = _make_value_fn(
                static_model_part=static,
                state=state,
                xb=xb,
                yb=yb,
                key=sk,
                loss_fn=loss_fn,
                lambda_spec=lambda_spec,
                lambda_frob=lambda_frob,
                deterministic_objective=deterministic_objective,
            )

            # Reuse (value, grad) across LS via optax helper
            value_and_grad = optax.value_and_grad_from_state(f)
            value, grad = value_and_grad(params, state=opt_state)

            updates, opt_state = solver.update(
                grad,
                opt_state,
                params,
                value=value,
                grad=grad,
                value_fn=f,
            )
            params = optax.apply_updates(params, updates)

            epoch_loss += float(value) * xb.shape[0]
            n_seen += xb.shape[0]

        train_hist.append(epoch_loss / max(1, n_seen))

        # ----- Validation (always inference mode) -----
        mdl_val = eqx.nn.inference_mode(eqx.combine(params, static))
        v_loss, n_val = 0.0, 0
        for xb, yb in data_loader(
            X_val,
            y_val,
            batch_size=batch_size,
            shuffle=False,
            key=eval_rng,
            augment_fn=None,
        ):
            eval_rng, vk = jr.split(eval_rng)
            l, _ = loss_fn(mdl_val, state, xb, yb, vk)
            v_loss += float(l) * xb.shape[0]
            n_val += xb.shape[0]
        val_hist.append(v_loss / max(1, n_val))

        # Track penalty magnitude on current params (for monitoring)
        pen_hist.append(float(global_spectral_penalty(eqx.combine(params, static))))

        print(
            f"[ZOOM | Epoch {epoch:3d}/{num_epochs}] "
            f"Train={train_hist[-1]:.4f} | Val={val_hist[-1]:.4f}"
        )

        # Early stopping
        if val_hist[-1] < best_val - 1e-6:
            best_val = val_hist[-1]
            best_params, best_state = params, state
            patience_ctr = 0
            if ckpt_path is not None:
                best_file = pathlib.Path(str(ckpt_path).format(epoch=epoch))
                best_file.parent.mkdir(parents=True, exist_ok=True)
                eqx.tree_serialise_leaves(
                    best_file,
                    {"model": eqx.combine(best_params, static), "state": best_state},
                )
        else:
            patience_ctr += 1
            if patience_ctr > patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Periodic checkpoint of the *current* weights
        if (
            ckpt_path is not None
            and checkpoint_interval is not None
            and epoch % checkpoint_interval == 0
        ):
            ckpt_file = pathlib.Path(str(ckpt_path).format(epoch=epoch))
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(
                ckpt_file,
                {"model": eqx.combine(params, static), "state": state},
            )

    final_model = eqx.combine(best_params, static)
    if return_penalty_history:
        return final_model, best_state, train_hist, val_hist, pen_hist
    else:
        return final_model, best_state, train_hist, val_hist


def train_zoom_full_data(
    model: eqx.Module,
    state: Any,
    X_train: Array,
    y_train: Array,
    X_val: Array,
    y_val: Array,
    *,
    batch_size: int = 512,
    num_epochs: int = 20,
    key: Key,
    augment_fn: Optional[AugmentFn] = None,
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    loss_fn: Callable = multiclass_loss,
    max_linesearch_steps: int = 10,
    zoom_c1: float = 1e-4,
    zoom_c2: float = 0.9,
    deterministic_objective: bool = True,
    ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    checkpoint_interval: Optional[int] = None,
    return_penalty_history: bool = False,
):
    """
    Combine train+val; disable early stop (patience=num_epochs).
    """
    X_full = jnp.concatenate([X_train, X_val], axis=0)
    y_full = jnp.concatenate([y_train, y_val], axis=0)
    X_dummy, y_dummy = X_full[:1], y_full[:1]
    return train_zoom(
        model,
        state,
        X_full,
        y_full,
        X_dummy,
        y_dummy,
        batch_size=batch_size,
        num_epochs=num_epochs,
        patience=num_epochs,
        lambda_spec=lambda_spec,
        lambda_frob=lambda_frob,
        key=key,
        augment_fn=augment_fn,
        loss_fn=loss_fn,
        max_linesearch_steps=max_linesearch_steps,
        zoom_c1=zoom_c1,
        zoom_c2=zoom_c2,
        deterministic_objective=deterministic_objective,
        ckpt_path=ckpt_path,
        checkpoint_interval=checkpoint_interval,
        return_penalty_history=return_penalty_history,
    )
