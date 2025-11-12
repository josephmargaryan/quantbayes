# quantbayes/stochax/privacy/dp_train.py
from __future__ import annotations
from typing import Any, Callable, List, Optional, Tuple

import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
import optax

from .dp import DPPrivacyEngine, DPSGDConfig

Array = jnp.ndarray


def _iter_batches_shuffled(X: Array, y: Array, B: int, key: jax.Array):
    """Classic shuffled minibatches."""
    N = int(X.shape[0])
    idx = jr.permutation(key, N)
    for s in range(0, N, B):
        e = min(s + B, N)
        sl = idx[s:e]
        yield X[sl], y[sl]


def _iter_batches_poisson(X: Array, y: Array, *, q: float, steps: int, key: jax.Array):
    """
    Poisson subsampling (Bernoulli(q) per example) for a fixed number of steps per epoch.
    Each step, draw a fresh subset; if empty, resample once (rare when q*N is decent).
    """
    N = int(X.shape[0])
    for t in range(steps):
        key, kk = jr.split(key)
        mask = jr.bernoulli(kk, p=jnp.asarray(q, jnp.float32), shape=(N,))
        sel = jnp.where(mask)[0]
        # rare safeguard: resample once if empty
        sel = jax.lax.cond(
            sel.size == 0,
            lambda _: jnp.where(
                jr.bernoulli(
                    jr.fold_in(kk, 1), p=jnp.asarray(q, jnp.float32), shape=(N,)
                )
            )[0],
            lambda _: sel,
            operand=None,
        )
        yield X[sel], y[sel]


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
    """
    SOTA DP-SGD training loop:
      - Poisson subsampling (optional, cfg.poisson_sampling) with rate q = batch_size / N if not provided.
      - Per-example clipping (norm C), Gaussian noise on the sum (std = sigma*C), then divide by batch size.
      - RDP accountant for the Sampled Gaussian Mechanism.
      - Per-example grad pass uses inference mode (state-safe); state is updated once per batch.

    Notes:
      - If you need extra regularization, wrap `loss_fn` before calling this (unchanged).
      - For small batches or DP, consider switching BN → GroupNorm/LayerNorm in your models.
    """
    engine = DPPrivacyEngine(dp_config)

    train_hist: List[float] = []
    val_hist: List[float] = []

    best_val = jnp.inf
    best_model, best_state = model, state
    no_improve = 0
    rng = key

    N = int(X_train.shape[0])
    q = (
        float(dp_config.sampling_rate)
        if (dp_config.sampling_rate is not None)
        else float(batch_size) / max(1, N)
    )
    q = min(q, 1.0)  # clamp to a valid sampling rate
    steps_per_epoch = max(1, int(jnp.ceil(N / max(1, batch_size))))

    for epoch in range(1, num_epochs + 1):
        # -------- epoch loop --------
        epoch_loss = 0.0
        nb = 0
        rng, ek = jr.split(rng)

        if dp_config.poisson_sampling:
            batch_iter = _iter_batches_poisson(
                X_train, y_train, q=q, steps=steps_per_epoch, key=ek
            )
        else:
            # classic shuffled minibatches (approximate accounting still uses q = B/N)
            batch_iter = _iter_batches_shuffled(
                X_train, y_train, batch_size, key=ek if shuffle else jr.PRNGKey(0)
            )

        for xb, yb in batch_iter:
            if xb.shape[0] == 0:
                continue
            rng, sub = jr.split(rng)

            noisy_mean_grad, new_state = engine.noisy_grad(
                loss_fn, model, state, xb, yb, key=sub, sample_rate=q
            )

            # Optimizer step
            params = eqx.filter(model, eqx.is_inexact_array)
            updates, opt_state = optimizer.update(
                noisy_mean_grad, opt_state, params=params
            )
            model = eqx.apply_updates(model, updates)
            state = new_state

            # Track training loss (no DP accounting is needed here)
            rng, em = jr.split(rng)
            loss_val, _ = loss_fn(
                eqx.nn.inference_mode(model, value=True), state, xb, yb, em
            )
            epoch_loss += float(loss_val)
            nb += 1

        train_hist.append(epoch_loss / max(1, nb))

        # -------- validation --------
        rng, vk = jr.split(rng)
        m_eval = eqx.nn.inference_mode(model, True)
        val_loss, _ = loss_fn(m_eval, state, X_val, y_val, vk)
        v = float(val_loss)
        val_hist.append(v)

        # early stopping
        if v + 1e-8 < float(best_val):
            best_val = v
            best_model, best_state = model, state
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        # optional: print ε progress (cheap)
        try:
            eps = engine.epsilon()
            print(
                f"[DP] epoch {epoch:03d} | train={train_hist[-1]:.4f} | val={v:.4f} | eps≈{eps:.3f} (δ={dp_config.delta:g})"
            )
        except Exception:
            print(f"[DP] epoch {epoch:03d} | train={train_hist[-1]:.4f} | val={v:.4f}")

    return best_model, best_state, train_hist, val_hist
