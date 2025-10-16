# quantbayes/stochax/trainer/train_backtrack.py
from __future__ import annotations

import pathlib
from typing import Any, Callable, List, Optional, Tuple, Union, Literal, Dict

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx

from quantbayes.stochax.trainer.train import (
    data_loader,
    multiclass_loss,  # default; pass binary/regression when needed
    AugmentFn,
)

from quantbayes.stochax.utils.regularizers import (
    global_spectral_penalty,
    global_frobenius_penalty,
    global_spectral_norm_penalty,
    sobolev_kernel_smoothness,
    sobolev_jacobian_penalty,
    lip_product_penalty,
    network_lipschitz_upper,  # certified global Lipschitz UB
)

Array = jnp.ndarray


# ------------------------------ value fn -------------------------------------


def _make_value_fn(
    static_model_part: eqx.Module,
    state: Any,
    xb: Array,
    yb: Array,
    key: jr.key,
    loss_fn: Callable,
    # --- regularizer knobs (parity with SGD/Zoom trainers) ---
    lambda_spec: float,
    lambda_frob: float,
    lambda_specnorm: float,
    lambda_sob_jac: float,
    lambda_sob_kernel: float,
    lambda_liplog: float,
    sob_jac_apply: Optional[Callable[[Any, Any, jnp.ndarray, jr.PRNGKey], jnp.ndarray]],
    sob_jac_samples: int,
    # legacy fallbacks (preserved)
    conv_tn_iters: int,
    conv_fft_shape: Optional[Tuple[int, int]],
    # ---- Σσ penalty config ----
    specnorm_conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ],
    specnorm_conv_tn_iters: Optional[int],
    specnorm_conv_gram_iters: Optional[int],
    specnorm_conv_fft_shape: Optional[Tuple[int, int]],
    specnorm_conv_input_shape: Optional[Tuple[int, int]],
    # ---- Lip product penalty config ----
    lip_conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ],
    lip_conv_tn_iters: Optional[int],
    lip_conv_gram_iters: Optional[int],
    lip_conv_fft_shape: Optional[Tuple[int, int]],
    lip_conv_input_shape: Optional[Tuple[int, int]],
    # ----------------------------------------------- #
    deterministic_objective: bool,
) -> Callable[[eqx.Module], Array]:
    """
    Build f(params) -> scalar objective on a fixed mini-batch (xb, yb, key).

    If deterministic_objective=True, the model is evaluated in eqx.nn.inference_mode
    (freezes BatchNorm/Dropout), making the line search well-posed.
    All regularizers mirror the main SGD/Zoom trainer semantics.
    """

    def _pick(v, default):
        return default if v is None else v

    # Fallbacks to preserve older signatures
    sn_tn_iters = _pick(specnorm_conv_tn_iters, conv_tn_iters)
    sn_gram_iters = _pick(specnorm_conv_gram_iters, 5)
    sn_fft_shape = _pick(specnorm_conv_fft_shape, conv_fft_shape)

    lp_tn_iters = _pick(lip_conv_tn_iters, conv_tn_iters)
    lp_gram_iters = _pick(lip_conv_gram_iters, 5)
    lp_fft_shape = _pick(lip_conv_fft_shape, conv_fft_shape)

    def f(params: eqx.Module) -> Array:
        mdl = eqx.combine(params, static_model_part)
        if deterministic_objective:
            mdl = eqx.nn.inference_mode(mdl)

        base, _ = loss_fn(mdl, state, xb, yb, key)

        # --- penalties (identical semantics to SGD/Zoom trainers) ---
        pen_spec = (
            lambda_spec * global_spectral_penalty(mdl) if lambda_spec > 0.0 else 0.0
        )
        pen_frob = (
            lambda_frob * global_frobenius_penalty(mdl) if lambda_frob > 0.0 else 0.0
        )

        pen_specnorm = (
            lambda_specnorm
            * global_spectral_norm_penalty(
                mdl,
                conv_mode=specnorm_conv_mode,
                conv_tn_iters=sn_tn_iters,
                conv_gram_iters=sn_gram_iters,
                conv_fft_shape=sn_fft_shape,
                conv_input_shape=specnorm_conv_input_shape,
            )
            if lambda_specnorm > 0.0
            else 0.0
        )

        pen_sob_k = (
            lambda_sob_kernel * sobolev_kernel_smoothness(mdl)
            if lambda_sob_kernel > 0.0
            else 0.0
        )

        pen_sob_j = (
            lambda_sob_jac
            * sobolev_jacobian_penalty(
                mdl, state, xb, key, sob_jac_apply, num_samples=sob_jac_samples
            )
            if (lambda_sob_jac > 0.0 and sob_jac_apply is not None)
            else 0.0
        )

        # BN-aware Lipschitz product penalty (pass state so BN uses running stats)
        pen_lip = (
            lip_product_penalty(
                mdl,
                state=state,
                tau=lambda_liplog,
                conv_mode=lip_conv_mode,
                conv_tn_iters=lp_tn_iters,
                conv_gram_iters=lp_gram_iters,
                conv_fft_shape=lp_fft_shape,
                conv_input_shape=lip_conv_input_shape,
            )
            if lambda_liplog > 0.0
            else 0.0
        )

        return base + (
            pen_spec + pen_frob + pen_specnorm + pen_sob_k + pen_sob_j + pen_lip
        )

    return f


# ------------------------------ trainer --------------------------------------


def train_backtrack(
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
    # base penalties
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    # regularizer parity with SGD/Zoom trainers
    lambda_specnorm: float = 0.0,
    lambda_sob_jac: float = 0.0,
    lambda_sob_kernel: float = 0.0,
    lambda_liplog: float = 0.0,
    sob_jac_apply: Optional[
        Callable[[Any, Any, jnp.ndarray, jr.PRNGKey], jnp.ndarray]
    ] = None,
    sob_jac_samples: int = 1,
    # legacy conv knobs (fall back for both Σσ and Lip penalties)
    conv_tn_iters: int = 8,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    # ---- Σσ regulariser config ----
    specnorm_conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = "tn",
    specnorm_conv_tn_iters: Optional[int] = None,
    specnorm_conv_gram_iters: Optional[int] = None,
    specnorm_conv_fft_shape: Optional[Tuple[int, int]] = None,
    specnorm_conv_input_shape: Optional[Tuple[int, int]] = None,
    # ---- Lip product penalty config ----
    lip_conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = "tn",
    lip_conv_tn_iters: Optional[int] = None,
    lip_conv_gram_iters: Optional[int] = None,
    lip_conv_fft_shape: Optional[Tuple[int, int]] = None,
    lip_conv_input_shape: Optional[Tuple[int, int]] = None,
    # RNG + obj
    key: jr.key = jr.PRNGKey(0),
    augment_fn: Optional[AugmentFn] = None,
    loss_fn: Callable = multiclass_loss,
    # Backtracking knobs
    max_backtracking_steps: int = 10,
    deterministic_objective: bool = True,
    # checkpoints
    ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    checkpoint_interval: Optional[int] = None,
    return_penalty_history: bool = False,
    # -------- Lipschitz logging (certified global UB) --------
    log_global_bound_every: Optional[int] = None,
    bound_conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = "tn",
    bound_tn_iters: int = 8,
    bound_gram_iters: int = 5,
    bound_fft_shape: Optional[Tuple[int, int]] = None,
    bound_input_shape: Optional[Tuple[int, int]] = None,
    bound_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Union[
    Tuple[eqx.Module, Any, List[float], List[float]],
    Tuple[eqx.Module, Any, List[float], List[float], List[float]],
]:
    """
    Mini-batch backtracking line search (Armijo backtracking) with SGD direction.

    • Deterministic objective (recommended): set `deterministic_objective=True` to
      freeze BN/Dropout during the line search, and keep (xb, yb, key) fixed.
    • Regularizers: same as SGD/Zoom (spectral, Frobenius, Σ per-layer σ with configurable
      conv bounds, Sobolev kernel/Jacobian, and log-Lipschitz product).
    • Certified Lipschitz logging: set `log_global_bound_every` to compute a global UB
      every k epochs; BN running stats are used (via `state`).
    """
    if checkpoint_interval is not None:
        assert checkpoint_interval > 0, "`checkpoint_interval` must be > 0"

    # params/static split
    params, static = eqx.partition(model, eqx.is_inexact_array)

    # Direction = -grad (lr=1.0); line search scales it.
    solver = optax.chain(
        optax.sgd(learning_rate=1.0),
        optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=max_backtracking_steps
        ),
    )
    opt_state = solver.init(params)

    # --- optional Lipschitz UB fn (certified, BN-aware) ---
    if log_global_bound_every is not None:

        def lip_fn(m):
            return network_lipschitz_upper(
                m,
                state=state,  # <-- use running BN stats
                conv_mode=bound_conv_mode,
                conv_tn_iters=bound_tn_iters,
                conv_gram_iters=bound_gram_iters,
                conv_fft_shape=bound_fft_shape,
                conv_input_shape=bound_input_shape,
            )

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
            rng, sk = jr.split(rng)

            f = _make_value_fn(
                static_model_part=static,
                state=state,
                xb=xb,
                yb=yb,
                key=sk,  # fixed subkey across the line search
                loss_fn=loss_fn,
                lambda_spec=lambda_spec,
                lambda_frob=lambda_frob,
                lambda_specnorm=lambda_specnorm,
                lambda_sob_jac=lambda_sob_jac,
                lambda_sob_kernel=lambda_sob_kernel,
                lambda_liplog=lambda_liplog,
                sob_jac_apply=sob_jac_apply,
                sob_jac_samples=sob_jac_samples,
                conv_tn_iters=conv_tn_iters,
                conv_fft_shape=conv_fft_shape,
                # Σσ config
                specnorm_conv_mode=specnorm_conv_mode,
                specnorm_conv_tn_iters=specnorm_conv_tn_iters,
                specnorm_conv_gram_iters=specnorm_conv_gram_iters,
                specnorm_conv_fft_shape=specnorm_conv_fft_shape,
                specnorm_conv_input_shape=specnorm_conv_input_shape,
                # Lip config
                lip_conv_mode=lip_conv_mode,
                lip_conv_tn_iters=lip_conv_tn_iters,
                lip_conv_gram_iters=lip_conv_gram_iters,
                lip_conv_fft_shape=lip_conv_fft_shape,
                lip_conv_input_shape=lip_conv_input_shape,
                deterministic_objective=deterministic_objective,
            )

            # reuse (value, grad) via optax cache
            value_and_grad = optax.value_and_grad_from_state(f)
            value, grad = value_and_grad(params, state=opt_state)

            updates, opt_state = solver.update(
                grad, opt_state, params, value=value, grad=grad, value_fn=f
            )
            params = optax.apply_updates(params, updates)

            epoch_loss += float(value) * xb.shape[0]
            n_seen += xb.shape[0]

        train_hist.append(epoch_loss / max(1, n_seen))

        # ----- Validation (inference mode) -----
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

        # Track penalty magnitude on current params (monitoring only)
        pen_hist.append(float(global_spectral_penalty(eqx.combine(params, static))))

        print(
            f"[BACKTRACK | Epoch {epoch:3d}/{num_epochs}] "
            f"Train={train_hist[-1]:.4f} | Val={val_hist[-1]:.4f}"
        )

        # --- certified Lipschitz UB (epoch-level) ---
        if (log_global_bound_every is not None) and (
            (epoch % max(1, log_global_bound_every) == 0) or (epoch == num_epochs)
        ):
            mdl_raw = eqx.combine(params, static)
            L_raw = float(lip_fn(mdl_raw))
            L_eval = float(lip_fn(mdl_val))  # eval = inference_mode
            rec = {
                "epoch": int(epoch),
                "L_raw": L_raw,
                "L_eval": L_eval,
                "mode": bound_conv_mode,
            }
            if bound_recorder is not None:
                bound_recorder(rec)
            print(
                f"    [Lipschitz UB] raw={L_raw:.6g}  eval={L_eval:.6g}  ({bound_conv_mode})"
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

        # Periodic checkpoint
        if (
            ckpt_path is not None
            and checkpoint_interval is not None
            and epoch % checkpoint_interval == 0
        ):
            ckpt_file = pathlib.Path(str(ckpt_path).format(epoch=epoch))
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(
                ckpt_file, {"model": eqx.combine(params, static), "state": state}
            )

    final_model = eqx.combine(best_params, static)
    if return_penalty_history:
        return final_model, best_state, train_hist, val_hist, pen_hist
    else:
        return final_model, best_state, train_hist, val_hist


def train_backtrack_full_data(
    model: eqx.Module,
    state: Any,
    X_train: Array,
    y_train: Array,
    X_val: Array,
    y_val: Array,
    *,
    batch_size: int = 512,
    num_epochs: int = 20,
    # base penalties (INCLUDED)
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    # extended regularizers
    lambda_specnorm: float = 0.0,
    lambda_sob_jac: float = 0.0,
    lambda_sob_kernel: float = 0.0,
    lambda_liplog: float = 0.0,
    sob_jac_apply: Optional[
        Callable[[Any, Any, jnp.ndarray, jr.PRNGKey], jnp.ndarray]
    ] = None,
    sob_jac_samples: int = 1,
    # legacy conv fallbacks
    conv_tn_iters: int = 8,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    # Σσ config
    specnorm_conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = "tn",
    specnorm_conv_tn_iters: Optional[int] = None,
    specnorm_conv_gram_iters: Optional[int] = None,
    specnorm_conv_fft_shape: Optional[Tuple[int, int]] = None,
    specnorm_conv_input_shape: Optional[Tuple[int, int]] = None,
    # Lip config
    lip_conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = "tn",
    lip_conv_tn_iters: Optional[int] = None,
    lip_conv_gram_iters: Optional[int] = None,
    lip_conv_fft_shape: Optional[Tuple[int, int]] = None,
    lip_conv_input_shape: Optional[Tuple[int, int]] = None,
    # obj + misc
    key: jr.key = jr.PRNGKey(0),
    augment_fn: Optional[AugmentFn] = None,
    loss_fn: Callable = multiclass_loss,
    max_backtracking_steps: int = 10,
    deterministic_objective: bool = True,
    ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    checkpoint_interval: Optional[int] = None,
    return_penalty_history: bool = False,
    # Lipschitz logging passthrough
    log_global_bound_every: Optional[int] = None,
    bound_conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = "tn",
    bound_tn_iters: int = 8,
    bound_gram_iters: int = 5,
    bound_fft_shape: Optional[Tuple[int, int]] = None,
    bound_input_shape: Optional[Tuple[int, int]] = None,
    bound_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """
    Combine train+val; disable early stop (patience=num_epochs).
    Keeps checkpoints + optional certified Lipschitz logging.
    Full pass-through of all regularizer and bound knobs.
    """
    X_full = jnp.concatenate([X_train, X_val], axis=0)
    y_full = jnp.concatenate([y_train, y_val], axis=0)
    # Minimal dummy val to keep loop shape; early stopping is disabled anyway.
    X_dummy, y_dummy = X_full[:1], y_full[:1]

    return train_backtrack(
        model=model,
        state=state,
        X_train=X_full,
        y_train=y_full,
        X_val=X_dummy,
        y_val=y_dummy,
        batch_size=batch_size,
        num_epochs=num_epochs,
        patience=num_epochs,  # effectively no early stop
        # regularizers (FULL passthrough)
        lambda_spec=lambda_spec,
        lambda_frob=lambda_frob,
        lambda_specnorm=lambda_specnorm,
        lambda_sob_jac=lambda_sob_jac,
        lambda_sob_kernel=lambda_sob_kernel,
        lambda_liplog=lambda_liplog,
        sob_jac_apply=sob_jac_apply,
        sob_jac_samples=sob_jac_samples,
        conv_tn_iters=conv_tn_iters,
        conv_fft_shape=conv_fft_shape,
        # Σσ config
        specnorm_conv_mode=specnorm_conv_mode,
        specnorm_conv_tn_iters=specnorm_conv_tn_iters,
        specnorm_conv_gram_iters=specnorm_conv_gram_iters,
        specnorm_conv_fft_shape=specnorm_conv_fft_shape,
        specnorm_conv_input_shape=specnorm_conv_input_shape,
        # Lip config
        lip_conv_mode=lip_conv_mode,
        lip_conv_tn_iters=lip_conv_tn_iters,
        lip_conv_gram_iters=lip_conv_gram_iters,
        lip_conv_fft_shape=lip_conv_fft_shape,
        lip_conv_input_shape=lip_conv_input_shape,
        # obj + misc
        key=key,
        augment_fn=augment_fn,
        loss_fn=loss_fn,
        max_backtracking_steps=max_backtracking_steps,
        deterministic_objective=deterministic_objective,
        ckpt_path=ckpt_path,
        checkpoint_interval=checkpoint_interval,
        return_penalty_history=return_penalty_history,
        # Lipschitz logging
        log_global_bound_every=log_global_bound_every,
        bound_conv_mode=bound_conv_mode,
        bound_tn_iters=bound_tn_iters,
        bound_gram_iters=bound_gram_iters,
        bound_fft_shape=bound_fft_shape,
        bound_input_shape=bound_input_shape,
        bound_recorder=bound_recorder,
    )
