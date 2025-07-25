#!/usr/bin/env python3
# ---------------------------------------------------------------------
#   Strong‑Wolfe back‑tracking line‑search + SGD direction
# ---------------------------------------------------------------------
import jax, jax.numpy as jnp, jax.random as jr
import optax, equinox as eqx
from typing import Any, Callable, List, Optional, Tuple

from quantbayes.stochax.trainer.train import (
    data_loader,
    multiclass_loss,
    global_spectral_penalty,
    AugmentFn,
)


# ---------------------------------------------------------------------
def make_value_and_grad(
    static_model_part: Any,
    loss_fn: Callable,
    lambda_spec: float,
) -> Callable:
    def _oracle(params, state, xb, yb, key):
        mdl = eqx.combine(params, static_model_part)
        (base, new_state), grads = eqx.filter_value_and_grad(
            lambda m, s: loss_fn(m, s, xb, yb, key),
            has_aux=True,
        )(mdl, state)
        value = base
        if lambda_spec:
            value += lambda_spec * global_spectral_penalty(mdl)
        return value, grads, new_state

    return _oracle


# ---------------------------------------------------------------------
def train_backtrack(
    model: eqx.Module,
    state: Any,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_val: jnp.ndarray,
    y_val: jnp.ndarray,
    *,
    batch_size: int = 512,
    num_epochs: int = 20,
    patience: int = 5,
    lambda_spec: float = 0.0,
    key: jr.key,
    augment_fn: Optional[AugmentFn] = None,
    loss_fn: Callable = multiclass_loss,
    return_penalty_history: bool = False,
) -> (
    Tuple[eqx.Module, Any, List[float], List[float]]
    | Tuple[eqx.Module, Any, List[float], List[float], List[float]]
):
    params, static = eqx.partition(model, eqx.is_inexact_array)
    oracle = make_value_and_grad(static, loss_fn, lambda_spec)

    solver = optax.chain(
        optax.sgd(learning_rate=1.0),
        optax.scale_by_backtracking_linesearch(max_backtracking_steps=10),
    )
    opt_state = solver.init(params)

    rng, eval_rng = jr.split(key)
    best_val = jnp.inf
    best_params = params

    train_hist: List[float] = []
    val_hist: List[float] = []
    pen_hist: List[float] = []
    patience_ctr = 0

    for ep in range(1, num_epochs + 1):
        rng, perm = jr.split(rng)
        epoch_loss = 0.0
        n_seen = 0

        for xb, yb in data_loader(
            X_train,
            y_train,
            batch_size,
            shuffle=True,
            key=perm,
            augment_fn=augment_fn,
        ):
            rng, sk = jr.split(rng)
            value, grads, state = oracle(params, state, xb, yb, sk)

            def _value_fn(p):
                v, _, _ = oracle(p, state, xb, yb, sk)
                return v

            updates, opt_state = solver.update(
                grads,
                opt_state,
                params,
                value=value,
                grad=grads,
                value_fn=_value_fn,
            )
            params = optax.apply_updates(params, updates)

            epoch_loss += float(value) * xb.shape[0]
            n_seen += xb.shape[0]

        train_hist.append(epoch_loss / n_seen)

        # ---------- validation ----------
        mdl = eqx.combine(params, static)
        v_loss = 0.0
        n_val = 0
        for xb, yb in data_loader(
            X_val, y_val, batch_size, shuffle=False, key=eval_rng
        ):
            eval_rng, vk = jr.split(eval_rng)
            l, _ = loss_fn(mdl, state, xb, yb, vk)
            v_loss += float(l) * xb.shape[0]
            n_val += xb.shape[0]
        v_loss /= n_val
        val_hist.append(v_loss)

        pen_hist.append(float(global_spectral_penalty(mdl)))

        print(f"[{ep:3d}] train={train_hist[-1]:.4f} | val={v_loss:.4f}")

        if v_loss < best_val - 1e-6:
            best_val, best_params = v_loss, params
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr > patience:
                break

    final_model = eqx.combine(best_params, static)

    if return_penalty_history:
        return final_model, state, train_hist, val_hist, pen_hist
    else:
        return final_model, state, train_hist, val_hist


# ---------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    from quantbayes.stochax.trainer.test import (
        SimpleCNN,
        X_train,
        X_val,
        y_train,
        y_val,
        augment_fn,
    )

    model, state = eqx.nn.make_with_state(SimpleCNN)(jr.key(10))

    train_backtrack(
        model,
        state,
        jnp.array(X_train),
        jnp.array(y_train),
        jnp.array(X_val),
        jnp.array(y_val),
        batch_size=256,
        num_epochs=20,
        patience=4,
        key=jr.key(314),
        augment_fn=augment_fn,
    )
