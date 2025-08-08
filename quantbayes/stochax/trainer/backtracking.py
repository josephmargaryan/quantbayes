#   Strong-Wolfe back-tracking line-search + SGD direction
import pathlib
import jax, jax.numpy as jnp, jax.random as jr
import optax, equinox as eqx
from equinox import combine
from typing import Any, Callable, List, Optional, Tuple, Union

from quantbayes.stochax.trainer.train import (
    data_loader,
    multiclass_loss,
    global_spectral_penalty,
    global_frobenius_penalty,
    AugmentFn,
)


def make_value_and_grad(
    static_model_part: Any,
    loss_fn: Callable,
    lambda_spec: float,
    lambda_frob: float = 0.0,
) -> Callable:
    def _oracle(params, state, xb, yb, key):
        mdl = combine(params, static_model_part)
        (base, new_state), grads = eqx.filter_value_and_grad(
            lambda m, s: loss_fn(m, s, xb, yb, key),
            has_aux=True,
        )(mdl, state)

        value = base
        if lambda_spec:
            value += lambda_spec * global_spectral_penalty(mdl)
        if lambda_frob:
            value += lambda_frob * global_frobenius_penalty(mdl)
        return value, grads, new_state

    return _oracle


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
    lambda_frob: float = 0.0,
    key: jr.PRNGKey,
    augment_fn: Optional[Callable[[jr.PRNGKey, jnp.ndarray], jnp.ndarray]] = None,
    loss_fn: Callable = multiclass_loss,
    ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    checkpoint_interval: Optional[int] = None,
    return_penalty_history: bool = False,
) -> Union[
    Tuple[eqx.Module, Any, List[float], List[float]],
    Tuple[eqx.Module, Any, List[float], List[float], List[float]],
]:
    if checkpoint_interval is not None:
        assert checkpoint_interval > 0, "checkpoint_interval must be > 0"

    params, static = eqx.partition(model, eqx.is_inexact_array)
    oracle = make_value_and_grad(static, loss_fn, lambda_spec, lambda_frob)
    solver = optax.chain(
        optax.sgd(learning_rate=1.0),
        optax.scale_by_backtracking_linesearch(max_backtracking_steps=10),
    )
    opt_state = solver.init(params)

    rng, eval_rng = jr.split(key)
    best_val = jnp.inf
    best_params, best_state = params, state

    train_hist: List[float] = []
    val_hist: List[float] = []
    pen_hist: List[float] = []
    patience_ctr = 0

    for ep in range(1, num_epochs + 1):
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

        mdl = eqx.combine(params, static)
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
            l, _ = loss_fn(mdl, state, xb, yb, vk)
            v_loss += float(l) * xb.shape[0]
            n_val += xb.shape[0]
        val_hist.append(v_loss / n_val)

        pen_hist.append(float(global_spectral_penalty(mdl)))
        print(f"[{ep:3d}] train={train_hist[-1]:.4f} | val={val_hist[-1]:.4f}")

        if val_hist[-1] < best_val - 1e-6:
            best_val = val_hist[-1]
            best_params, best_state = params, state
            patience_ctr = 0
            if ckpt_path is not None:
                tpl = str(ckpt_path)
                best_file = pathlib.Path(tpl.format(epoch=ep))
                best_file.parent.mkdir(parents=True, exist_ok=True)
                eqx.tree_serialise_leaves(
                    best_file,
                    {"model": eqx.combine(best_params, static), "state": best_state},
                )
        else:
            patience_ctr += 1
            if patience_ctr > patience:
                break

        if (
            ckpt_path is not None
            and checkpoint_interval is not None
            and ep % checkpoint_interval == 0
        ):
            tpl = str(ckpt_path)
            ckpt_file = pathlib.Path(tpl.format(epoch=ep))
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


def train_backtrack_full_data(
    model: eqx.Module,
    state: Any,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_val: jnp.ndarray,
    y_val: jnp.ndarray,
    *,
    batch_size: int = 512,
    num_epochs: int = 20,
    key: jr.PRNGKey,
    augment_fn: Optional[Callable[[jr.PRNGKey, jnp.ndarray], jnp.ndarray]] = None,
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    loss_fn: Callable = multiclass_loss,
    ckpt_path: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    return_penalty_history: bool = False,
):
    """
    Final run on 100% of data using a 1-sample dummy validation and
    patience=num_epochs to disable early stopping.
    """
    X_full = jnp.concatenate([X_train, X_val], axis=0)
    y_full = jnp.concatenate([y_train, y_val], axis=0)
    X_dummy, y_dummy = X_full[:1], y_full[:1]
    return train_backtrack(
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
        ckpt_path=ckpt_path,
        checkpoint_interval=checkpoint_interval,
        return_penalty_history=return_penalty_history,
    )


if __name__ == "__main__":
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
