# quantbayes/stochax/privacy/dp_train.py
from __future__ import annotations
from typing import Any, Callable, List, Tuple
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
import optax
from .dp import DPPrivacyEngine, DPSGDConfig

Array = jnp.ndarray


def _iter_batches(X: Array, y: Array, B: int):
    N = int(X.shape[0])
    for s in range(0, N, B):
        e = min(s + B, N)
        yield X[s:e], y[s:e]


def dp_eqx_train(
    model: eqx.Module,
    state: Any,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable[[eqx.Module, Any, Array, Array, jax.Array], Tuple[Array, Any]],
    X_train: Array,
    y_train: Array,
    X_val: Array,
    y_val: Array,
    *,
    dp_config: DPSGDConfig,
    batch_size: int,
    num_epochs: int,
    patience: int,
    key: jax.Array,
    shuffle: bool = True,
):
    engine = DPPrivacyEngine(dp_config)
    train_hist: List[float] = []
    val_hist: List[float] = []

    best_loss = jnp.inf
    best_model, best_state = model, state
    no_improve = 0
    rng = key

    for _ in range(num_epochs):
        if shuffle and X_train.shape[0] > batch_size:
            rng, sid = jr.split(rng)
            idx = jr.permutation(sid, X_train.shape[0])
            X_train = X_train[idx]
            y_train = y_train[idx]

        epoch_loss = 0.0
        nb = 0
        for xb, yb in _iter_batches(X_train, y_train, batch_size):
            rng, sub = jr.split(rng)
            noisy_grad, new_state = engine.noisy_grad(
                loss_fn, model, state, xb, yb, key=sub
            )
            updates, opt_state = optimizer.update(
                noisy_grad, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
            )
            model = eqx.apply_updates(model, updates)
            state = new_state

            rng, esub = jr.split(rng)
            loss_val, _ = loss_fn(model, state, xb, yb, esub)
            epoch_loss += float(loss_val)
            nb += 1

        epoch_loss = epoch_loss / max(1, nb)
        train_hist.append(epoch_loss)

        rng, vsub = jr.split(rng)
        val_loss, _ = loss_fn(model, state, X_val, y_val, vsub)
        val_loss = float(val_loss)
        val_hist.append(val_loss)

        if val_loss + 1e-8 < best_loss:
            best_loss = val_loss
            best_model, best_state = model, state
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_model, best_state, train_hist, val_hist
