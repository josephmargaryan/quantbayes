# quantbayes/ball_dp/privacy/dpsgd_train.py
from __future__ import annotations

from typing import Any, Callable, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .dpsgd import DPPrivacyEngine, DPSGDConfig

Array = jnp.ndarray


def _iter_batches_shuffled(X: Array, y: Array, B: int, key: jax.Array):
    N = int(X.shape[0])
    idx = jr.permutation(key, N)
    for s in range(0, N, B):
        e = min(s + B, N)
        sl = idx[s:e]
        yield X[sl], y[sl]


def dp_eqx_train(
    model: eqx.Module,
    state: Any,
    opt_state: Any,
    optimizer: Any,
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
    """
    Standard DP-SGD training loop with WOR-RDP accountant.

    - Uses fixed-size minibatches (shuffle) with sample_rate q = batch_size / N.
    - Early stopping uses val loss; set patience big if you want fixed epochs.
    """
    engine = DPPrivacyEngine(dp_config)

    train_hist: List[float] = []
    val_hist: List[float] = []

    best_val = jnp.inf
    best_model, best_state = model, state
    no_improve = 0
    rng = key

    N = int(X_train.shape[0])
    q = float(batch_size) / max(1, N)
    q = min(q, 1.0)

    for epoch in range(1, int(num_epochs) + 1):
        epoch_loss = 0.0
        nb = 0
        rng, ek = jr.split(rng)

        batch_iter = _iter_batches_shuffled(
            X_train, y_train, int(batch_size), key=ek if shuffle else jr.PRNGKey(0)
        )

        for xb, yb in batch_iter:
            if xb.shape[0] == 0:
                continue
            rng, sub = jr.split(rng)

            noisy_mean_grad, new_state = engine.noisy_grad(
                loss_fn, model, state, xb, yb, key=sub, sample_rate=q
            )

            params = eqx.filter(model, eqx.is_inexact_array)
            updates, opt_state = optimizer.update(
                noisy_mean_grad, opt_state, params=params
            )
            model = eqx.apply_updates(model, updates)
            state = new_state

            rng, em = jr.split(rng)
            loss_val, _ = loss_fn(
                eqx.nn.inference_mode(model, value=True), state, xb, yb, em
            )
            epoch_loss += float(loss_val)
            nb += 1

        train_hist.append(epoch_loss / max(1, nb))

        rng, vk = jr.split(rng)
        m_eval = eqx.nn.inference_mode(model, True)
        val_loss, _ = loss_fn(m_eval, state, X_val, y_val, vk)
        v = float(val_loss)
        val_hist.append(v)

        if v + 1e-8 < float(best_val):
            best_val = v
            best_model, best_state = model, state
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(patience):
                break

        try:
            eps = engine.epsilon()
            print(
                f"[DP-SGD] epoch {epoch:03d} | train={train_hist[-1]:.4f} | val={v:.4f} | eps≈{eps:.3f} (δ={dp_config.delta:g})"
            )
        except Exception:
            print(
                f"[DP-SGD] epoch {epoch:03d} | train={train_hist[-1]:.4f} | val={v:.4f}"
            )

    return best_model, best_state, train_hist, val_hist
