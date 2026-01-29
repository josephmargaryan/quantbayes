from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Literal,
)

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx

Array = jnp.ndarray


# ----------------------------
# Small utilities
# ----------------------------
class BoundLogger:
    """Kept for API compatibility with your original code."""

    def __init__(self):
        self.data: List[Dict[str, Any]] = []

    def __call__(self, rec: Dict[str, Any]):
        out = {}
        for k, v in rec.items():
            try:
                out[k] = float(v)
            except Exception:
                out[k] = v
        self.data.append(out)


def data_loader(
    X: Array,
    y: Array,
    batch_size: int,
    *,
    shuffle: bool = True,
    key: Optional[jr.PRNGKey] = None,
    augment_fn: Optional[Callable[[jr.PRNGKey, Any], Any]] = None,
) -> Iterator[Tuple[Array, Array]]:
    """Mini-batch generator with optional on-device augmentation."""
    n = X.shape[0]
    idx = jnp.arange(n)

    if shuffle:
        if key is None:
            raise ValueError("shuffle=True requires a PRNG key.")
        key, sk = jr.split(key)
        idx = jr.permutation(sk, idx)

    for i in range(0, n, batch_size):
        batch_idx = idx[i : i + batch_size]
        xb, yb = X[batch_idx], y[batch_idx]

        if augment_fn is not None:
            if key is None:
                raise ValueError("augment_fn provided but key is None.")
            key, sk = jr.split(key)
            # match your existing calling convention: augment_fn(key, [xb, yb])
            xb, yb = augment_fn(sk, [xb, yb])

        yield xb, yb


# ----------------------------
# Losses
# ----------------------------
def make_loss_fn(
    per_example_loss: Callable[[Array, Array], Array],
    expect_num_classes: bool = False,
    *,
    head_weights: Optional[Sequence[float]] = None,
):
    """
    Builds loss_fn(model, state, x, y, key) -> (loss_scalar, new_state)

    Assumes your model signature is: model(x, key, state) -> (logits_or_list, new_state)
    Supports deep supervision if logits_or_list is a list/tuple.
    """

    def loss_fn(model: Any, state: Any, x: Array, y: Array, key: jr.PRNGKey):
        keys = jr.split(key, x.shape[0])

        logits_or_list, new_state = jax.vmap(
            model,
            in_axes=(0, 0, None),
            out_axes=(0, None),
            axis_name="batch",
        )(x, keys, state)

        preds = (
            logits_or_list
            if isinstance(logits_or_list, (list, tuple))
            else [logits_or_list]
        )

        if head_weights is None:
            w = jnp.ones((len(preds),), dtype=jnp.float32) / max(1, len(preds))
        else:
            w = jnp.asarray(head_weights, dtype=jnp.float32)
            w = w / jnp.maximum(1e-12, w.sum())

        total = 0.0
        for logit, weight in zip(preds, w):
            # If regression/binary head has trailing singleton channel, squeeze it.
            if (
                (not expect_num_classes)
                and (logit.ndim == y.ndim + 1)
                and (logit.shape[-1] == 1)
            ):
                logit = logit.squeeze(-1)

            per_el = per_example_loss(logit, y)  # (B, ...) or (B,)
            b = per_el.shape[0]
            per_el = per_el.reshape((b, -1)).mean(
                axis=1
            )  # mean over spatial dims if any
            total = total + weight * per_el.mean()

        return total, new_state

    return eqx.filter_jit(loss_fn)


multiclass_loss = make_loss_fn(
    optax.softmax_cross_entropy_with_integer_labels,
    expect_num_classes=True,
)


# ----------------------------
# Train step / eval step
# ----------------------------
@eqx.filter_jit
def train_step(
    model: Any,
    state: Any,
    opt_state: Any,
    x: Array,
    y: Array,
    key: jr.PRNGKey,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
) -> Tuple[Any, Any, Any, Array]:
    def _loss(m, s, xb, yb, k):
        return loss_fn(m, s, xb, yb, k)

    (loss, new_state), grads = eqx.filter_value_and_grad(_loss, has_aux=True)(
        model, state, x, y, key
    )

    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    model = eqx.apply_updates(model, updates)
    return model, new_state, opt_state, loss


@eqx.filter_jit
def eval_step(
    model: Any, state: Any, x: Array, y: Array, key: jr.PRNGKey, loss_fn: Callable
) -> Array:
    loss, _ = loss_fn(model, state, x, y, key)
    return loss


# ----------------------------
# Main training loop
# ----------------------------
def train(
    model: Any,
    state: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable[[Any, Any, Array, Array, jr.PRNGKey], Tuple[Array, Any]],
    X_train: Array,
    y_train: Array,
    X_val: Array,
    y_val: Array,
    *,
    batch_size: int,
    num_epochs: int,
    patience: int,
    key: jr.PRNGKey,
    augment_fn: Optional[Callable[[jr.PRNGKey, Any], Any]] = None,
    ckpt_path: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    # --- compatibility kwargs (accepted, but intentionally ignored in this minimal version) ---
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    lambda_specnorm: float = 0.0,
    lambda_sob_jac: float = 0.0,
    lambda_sob_kernel: float = 0.0,
    lambda_liplog: float = 0.0,
    log_global_bound_every: Optional[int] = None,
    bound_conv_mode: str = "tn",
    bound_tn_iters: int = 8,
    bound_input_shape: Optional[Tuple[int, int]] = None,
    bound_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
    **kwargs,
):
    """
    Minimal trainer:
      - early stopping on val loss
      - optional checkpointing
      - supports BN/Dropout freezing during validation via eqx.nn.inference_mode
      - accepts extra kwargs for API compatibility with your original trainer
    """
    if checkpoint_interval is not None and checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be > 0 if provided.")

    rng, eval_rng = jr.split(key)
    train_losses: List[float] = []
    val_losses: List[float] = []

    best_val = float("inf")
    best_model = model
    best_state = state
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        epoch_train_loss = 0.0
        n_train = 0

        rng, perm_rng = jr.split(rng)
        for xb, yb in data_loader(
            X_train,
            y_train,
            batch_size=batch_size,
            shuffle=True,
            key=perm_rng,
            augment_fn=augment_fn,
        ):
            rng, step_rng = jr.split(rng)
            model, state, opt_state, batch_loss = train_step(
                model, state, opt_state, xb, yb, step_rng, loss_fn, optimizer
            )
            epoch_train_loss += float(batch_loss) * xb.shape[0]
            n_train += xb.shape[0]

        train_losses.append(epoch_train_loss / max(1, n_train))

        # ---- Validate (frozen) ----
        epoch_val_loss = 0.0
        n_val = 0

        val_model = eqx.nn.inference_mode(model, value=True)
        for xb, yb in data_loader(
            X_val,
            y_val,
            batch_size=batch_size,
            shuffle=False,
            key=eval_rng,
            augment_fn=None,
        ):
            eval_rng, vk = jr.split(eval_rng)
            vloss = eval_step(val_model, state, xb, yb, vk, loss_fn)
            epoch_val_loss += float(vloss) * xb.shape[0]
            n_val += xb.shape[0]

        val_losses.append(epoch_val_loss / max(1, n_val))

        # ---- Progress ----
        if epoch == 1 or epoch == num_epochs or (epoch % max(1, num_epochs // 10) == 0):
            print(
                f"[Epoch {epoch:3d}/{num_epochs}] Train={train_losses[-1]:.6f} | Val={val_losses[-1]:.6f}"
            )

        # ---- Best model checkpoint (on improvement) ----
        improved = val_losses[-1] < best_val
        if improved:
            best_val = val_losses[-1]
            best_model, best_state = model, state
            patience_counter = 0

            if ckpt_path is not None:
                best_file = pathlib.Path(str(ckpt_path).format(epoch=epoch))
                best_file.parent.mkdir(parents=True, exist_ok=True)
                eqx.tree_serialise_leaves(
                    best_file, {"model": best_model, "state": best_state}
                )
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # ---- Periodic checkpoints (current weights) ----
        if (
            ckpt_path is not None
            and checkpoint_interval is not None
            and (epoch % checkpoint_interval == 0)
        ):
            ckpt_file = pathlib.Path(str(ckpt_path).format(epoch=epoch))
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(ckpt_file, {"model": model, "state": state})

    return best_model, best_state, train_losses, val_losses


@dataclass
class EMA:
    # arrays-only pytree whose structure matches `eqx.partition(model, eqx.is_inexact_array)[0]`
    params: Any
    decay: float


def init_ema(model: Any, decay: float = 0.999) -> EMA:
    # arrays-only params tree; no Nones mixed in (partition is preferable to filter here)
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    return EMA(params=params, decay=float(decay))


# JIT **arrays-only** function; do not pass the EMA object through JIT.
@eqx.filter_jit
def _ema_update_arrays(ema_params, model_params, decay: float):
    # Correct EMA update: new = decay*old + (1-decay)*current
    return jax.tree_map(
        lambda e, p: decay * e + (1.0 - decay) * p, ema_params, model_params
    )


def update_ema(ema: EMA, model: Any) -> EMA:
    # Partition model to arrays-only, call the jitted arrays-only updater.
    model_params, _ = eqx.partition(model, eqx.is_inexact_array)
    new_params = _ema_update_arrays(ema.params, model_params, float(ema.decay))
    return EMA(params=new_params, decay=ema.decay)


def swap_ema_params(model: Any, ema: EMA) -> Any:
    """Return a copy of `model` with EMA params swapped in."""
    params, static = eqx.partition(model, eqx.is_inexact_array)
    # Ensure trees align: ema.params must match `params` structure.
    return eqx.combine(ema.params, static)


def load_model(
    ckpt_path: Union[str, pathlib.Path],
    tmpl_model: eqx.Module,
    tmpl_state: Any,
    *,
    # How to handle EMA stored in the checkpoint (if any)
    ema_policy: Literal["prefer", "require", "ignore"] = "prefer",
    # If True, and EMA is used (per policy), we swap EMA params into model before returning
    finalize_ema: Optional[bool] = None,  # None -> defaults to for_inference
    # Set eval/inference switches (e.g., dropout off, BN eval) on the returned model
    for_inference: bool = True,
    # Try to load an optimizer state if present (for resume training)
    load_opt_state: bool = False,
    tmpl_opt_state: Any = None,  # required if load_opt_state=True and ckpt has opt_state
) -> Tuple[eqx.Module, Any, Optional[EMA], Optional[Any], Dict[str, Any]]:
    """
    Load a checkpoint saved by the provided training loops.

    Args:
        ckpt_path: Path to the saved checkpoint file.
        tmpl_model: A template model with the exact same architecture used at save time.
        tmpl_state: A template state object (e.g., BN stats) for 'like='.
        ema_policy:
            - "prefer"  : use EMA if present; else fall back to raw.
            - "require" : raise if EMA missing.
            - "ignore"  : ignore EMA even if present.
        finalize_ema: If True and EMA is selected by policy, swap EMA params into the model.
                      If None, defaults to `for_inference`.
        for_inference: If True, return model wrapped with `eqx.nn.inference_mode`.
        load_opt_state: If True, also attempt to load optimizer state if it was saved.
        tmpl_opt_state: Template opt state for 'like=' (only needed if load_opt_state=True).

    Returns:
        (model, state, ema_or_none, opt_state_or_none, meta_dict)

    Example Usage:
        ```python
            tmpl_model, tmpl_state = eqx.nn.make_with_state(NN)(key=jr.key(0))

            model, state, ema, _, meta = load_model(
                "checkpoints/unet_eqx-epoch-050.ckpt",
                tmpl_model, tmpl_state,
                ema_policy="prefer",     # use EMA if available
                finalize_ema=True,       # swap EMA params into model
                for_inference=True,      # eval mode
                load_opt_state=False
            )

            # Now just call your existing predict:
            logits = predict(model, state, X, key)
            probs  = jax.nn.sigmoid(logits).squeeze()
            # Or for finetuning
            model, state, ema, opt_state, meta = load_model(
            "checkpoints/unet_eqx-epoch-050.ckpt",
            tmpl_model, tmpl_state,
            ema_policy="prefer",     # keep the EMA object if present…
            finalize_ema=False,      # …but do NOT swap params; continue from raw weights
            for_inference=False,     # keep training mode semantics
            load_opt_state=False     # set True + provide tmpl_opt_state if you saved it
        )```
    """
    ckpt_path = pathlib.Path(ckpt_path)
    if finalize_ema is None:
        finalize_ema = bool(for_inference)

    # Build candidate 'like=' dictionaries. We don't know exactly what was saved,
    # so we try a few combinations in order of most-informative to least.
    common_like = {"model": tmpl_model, "state": tmpl_state}
    params_like, _ = eqx.partition(tmpl_model, eqx.is_inexact_array)

    likes_to_try = []

    if load_opt_state and tmpl_opt_state is not None:
        likes_to_try.append(
            {
                **common_like,
                "ema_params": params_like,
                "ema_decay": jnp.asarray(0.0),
                "opt_state": tmpl_opt_state,
            }
        )
        likes_to_try.append({**common_like, "opt_state": tmpl_opt_state})
        # If opt_state wasn’t saved, these two may fail; we’ll keep falling back.
    else:
        likes_to_try.append(
            {**common_like, "ema_params": params_like, "ema_decay": jnp.asarray(0.0)}
        )

    likes_to_try.append(common_like)

    last_exc = None
    ckpt = None
    used_like = None
    for like in likes_to_try:
        try:
            ckpt = eqx.tree_deserialise_leaves(ckpt_path, like=like)
            used_like = like
            break
        except Exception as e:
            last_exc = e

    if ckpt is None:
        raise RuntimeError(
            f"Failed to load checkpoint '{ckpt_path}'. Last error:\n{last_exc}"
        )

    # Extract always-present fields
    model = ckpt["model"]
    state = ckpt["state"]

    # Optional fields
    meta = ckpt.get("meta", {})
    opt_state = ckpt.get("opt_state", None)

    # EMA presence
    has_ema = ("ema_params" in ckpt) and ("ema_decay" in ckpt)
    ema_obj = None
    if has_ema:
        ema_obj = EMA(params=ckpt["ema_params"], decay=float(ckpt["ema_decay"]))

    # Decide EMA usage per policy
    use_ema_now = False
    if ema_policy == "ignore":
        use_ema_now = False
    elif ema_policy == "require":
        if not has_ema:
            raise ValueError("Checkpoint has no EMA but ema_policy='require'.")
        use_ema_now = True
    else:  # "prefer"
        use_ema_now = has_ema

    # Finalize to EMA params if requested
    if use_ema_now and finalize_ema:
        model = swap_ema_params(model, ema_obj)  # params replaced with EMA

    # Toggle inference switches if requested (safe even if no BN/Dropout)
    if for_inference:
        model = eqx.nn.inference_mode(model)

    return model, state, ema_obj, opt_state, meta
