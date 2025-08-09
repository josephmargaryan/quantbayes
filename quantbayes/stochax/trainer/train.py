import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import augmax
from typing import (
    Callable,
    Optional,
    Iterator,
    Tuple,
    Union,
    Any,
    List,
    Sequence,
    Literal,
)
from quantbayes.stochax.utils import EMA, init_ema, update_ema, swap_ema_params

Array = jnp.ndarray
Key = jr.key


AugmentFn = Callable[[jr.key, jnp.ndarray], jnp.ndarray]


def make_augmax_augment(transform):
    @jax.jit
    def _augment(key, batch_pair):
        imgs_chw, second = batch_pair
        subkeys = jr.split(key, imgs_chw.shape[0])

        imgs_hwc = jnp.transpose(imgs_chw, (0, 2, 3, 1))

        if second.ndim == 4:
            masks_hwc = jnp.transpose(second, (0, 2, 3, 1))
        else:
            masks_hwc = second

        (img_hwc_aug, second_hwc_aug) = jax.vmap(transform)(
            subkeys, [imgs_hwc, masks_hwc]
        )

        img_chw_aug = jnp.transpose(img_hwc_aug, (0, 3, 1, 2))

        if second.ndim == 4:
            second_chw_aug = jnp.transpose(second_hwc_aug, (0, 3, 1, 2))
        else:
            second_chw_aug = second_hwc_aug

        return img_chw_aug, second_chw_aug

    return _augment


def data_loader(
    X: jnp.ndarray,
    y: jnp.ndarray,
    batch_size: int,
    *,
    shuffle: bool = True,
    key: Optional[jr.key] = None,
    augment_fn: Optional[Callable[[jr.key, jnp.ndarray], jnp.ndarray]] = None,
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Mini‐batch generator with optional Augmax augmentation on‐device.
    """
    n = X.shape[0]
    idx = jnp.arange(n)

    if shuffle:
        if key is None:
            raise ValueError("`shuffle=True` but no key provided.")
        key, sk = jr.split(key)
        idx = jr.permutation(sk, idx)

    for i in range(0, n, batch_size):
        batch_idx = idx[i : i + batch_size]
        xb, yb = X[batch_idx], y[batch_idx]
        if augment_fn is not None:
            if key is None:
                raise ValueError("`augment_fn` given but no PRNG key supplied.")
            key, sk = jr.split(key)
            xb, yb = augment_fn(sk, [xb, yb])
        yield xb, yb


def make_loss_fn(
    per_example_loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    expect_num_classes: bool = False,
    *,
    head_weights: Sequence[float] | None = None,
):
    """
    per_example_loss:  any pixel-wise loss that returns shape (B,H,W) or (B,)
    head_weights:      None → average equally (1/k)
                       else a list/tuple of weights, same length as
    """

    def loss_fn(model, state, x: jnp.ndarray, y: jnp.ndarray, key):
        keys = jax.random.split(key, x.shape[0])

        logits_or_list, state = jax.vmap(
            model,
            in_axes=(0, 0, None),
            out_axes=(0, None),
            axis_name="batch",
        )(x, keys, state)

        if isinstance(logits_or_list, (list, tuple)):
            preds = logits_or_list  # deep supervision (UnetPP, etc.)
        else:
            preds = [logits_or_list]  # single head

        if head_weights is None:
            w = [1.0 / len(preds)] * len(preds)
        else:
            assert len(head_weights) == len(preds), "head_weights length mismatch"
            w = jnp.asarray(head_weights, dtype=jnp.float32)
            w = w / w.sum()  # normalise

        # compute weighted loss over heads
        total = 0.0
        for logit, weight in zip(preds, w):
            if (
                not expect_num_classes
                and logit.ndim == y.ndim + 1
                and logit.shape[-1] == 1
            ):
                logit = logit.squeeze(-1)

            # per-example, then spatial mean, then batch mean
            per_el = per_example_loss(logit, y)
            b = per_el.shape[0]
            per_el = per_el.reshape((b, -1)).mean(axis=1)
            total = total + weight * per_el.mean()

        return total, state

    return eqx.filter_jit(loss_fn)


binary_loss = make_loss_fn(optax.sigmoid_binary_cross_entropy, expect_num_classes=False)
multiclass_loss = make_loss_fn(
    optax.softmax_cross_entropy_with_integer_labels, expect_num_classes=True
)
regression_loss = make_loss_fn(lambda p, t: (p - t) ** 2, expect_num_classes=False)


def make_dice_bce_loss(
    pos_weight: float | None = None,
    dice_weight: float = 0.5,
    bce_weight: float = 0.5,
):
    """
    Returns a per-example loss fn compatible with make_loss_fn().
    dice_weight + bce_weight should sum to 1.
    """

    def per_example(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        # Dice
        probs = jax.nn.sigmoid(logits)
        inter = jnp.sum(probs * targets, axis=(-2, -1))
        union = jnp.sum(probs, axis=(-2, -1)) + jnp.sum(targets, axis=(-2, -1))
        dice = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)

        # Weighted BCE:  w * CE(pos) + CE(neg)
        if pos_weight is None:
            bce = optax.sigmoid_binary_cross_entropy(logits, targets)
        else:
            # CE =  - [ y·log σ + (1-y)·log(1-σ) ]
            ce = optax.sigmoid_binary_cross_entropy(logits, targets)
            w = 1 + (pos_weight - 1) * targets
            bce = w * ce
        bce = jnp.mean(bce, axis=(-2, -1))

        return dice_weight * dice + bce_weight * bce

    return make_loss_fn(per_example, expect_num_classes=False)


def make_focal_dice_loss(
    gamma: float = 2.0,
    dice_weight: float = 0.5,
    focal_weight: float = 0.5,
):
    """Returns a per-example loss fn for make_loss_fn()."""

    def per_example(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        # Dice term
        probs = jax.nn.sigmoid(logits)
        inter = jnp.sum(probs * targets, axis=(-2, -1))
        union = jnp.sum(probs, axis=(-2, -1)) + jnp.sum(targets, axis=(-2, -1))
        dice = 1.0 - (2 * inter + 1e-6) / (union + 1e-6)

        bce = optax.sigmoid_binary_cross_entropy(logits, targets)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        mod = (1 - p_t) ** gamma
        focal = mod * bce
        focal = jnp.mean(focal, axis=(-2, -1))

        return dice_weight * dice + focal_weight * focal

    return make_loss_fn(per_example, expect_num_classes=False)


def make_tversky_loss(alpha=0.3, beta=0.7, gamma=1.0):
    def per_example(logits, targets):
        p = jax.nn.sigmoid(logits)
        tp = jnp.sum(p * targets, axis=(-2, -1))
        fp = jnp.sum(p * (1 - targets), axis=(-2, -1))
        fn = jnp.sum((1 - p) * targets, axis=(-2, -1))
        tversky = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
        return (1 - tversky) ** gamma

    return make_loss_fn(per_example, expect_num_classes=False)


def global_spectral_penalty(model) -> jnp.ndarray:
    """
    Sum β·θ² over every spectral weight θ in the network,
    plus any delta_alpha terms, without per-module normalization.
    Scale the *total* by your lambda_spec in train_step.
    """
    total = jnp.array(0.0, dtype=jnp.float32)

    if hasattr(model, "__spectral_penalty__"):
        total += model.__spectral_penalty__()
        if hasattr(model, "delta_alpha"):
            total += jnp.sum(model.delta_alpha**2)

    if isinstance(model, eqx.Module):
        for v in vars(model).values():
            total += global_spectral_penalty(v)
    elif isinstance(model, (list, tuple)):
        for v in model:
            total += global_spectral_penalty(v)
    elif isinstance(model, dict):
        for v in model.values():
            total += global_spectral_penalty(v)

    return total


def global_frobenius_penalty(model) -> jnp.ndarray:
    """
    Sum ‖W‖^2 over every weight-matrix (ndim=2) and Conv2D kernel (ndim=4),
    but skip all 1D biases (and any other 0D/1D arrays).
    """
    params = eqx.filter(model, lambda x: eqx.is_inexact_array(x) and x.ndim >= 2)
    leaves = jax.tree_util.tree_leaves(params)
    return sum(jnp.sum(p**2) for p in leaves)


@eqx.filter_jit
def train_step(
    model,
    state,
    opt_state,
    x: jnp.ndarray,
    y: jnp.ndarray,
    key: jr.key,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
) -> Tuple[Any, Any, Any, jnp.ndarray]:
    def total_loss_fn(m, s, xb, yb, k):
        base, new_s = loss_fn(m, s, xb, yb, k)
        # spectral penalty
        pen_spec = (
            lambda_spec * global_spectral_penalty(m) if lambda_spec > 0.0 else 0.0
        )
        # Frobenius (L2) penalty
        pen_frob = (
            lambda_frob * global_frobenius_penalty(m) if lambda_frob > 0.0 else 0.0
        )
        pen = pen_spec + pen_frob
        return base + pen, (new_s, base, pen)

    (tot, (new_state, _, _)), grads = eqx.filter_value_and_grad(
        total_loss_fn, has_aux=True
    )(model, state, x, y, key)

    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    model = eqx.apply_updates(model, updates)

    return model, new_state, opt_state, tot


@eqx.filter_jit
def train_step_sam(
    model,
    state,
    opt_state,
    x: jnp.ndarray,
    y: jnp.ndarray,
    key: jax.random.PRNGKey,
    loss_fn,
    optimizer,
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    *,
    sam_rho: float = 0.05,
    sam_mode: Literal["sam", "asam"] = "sam",
    asam_eps: float = 1e-12,
    freeze_norm_on_perturbed: bool = True,  # <— new
) -> Tuple[Any, Any, Any, jnp.ndarray]:

    def total_loss_fn(m, s, xb, yb, k):
        base, new_s = loss_fn(m, s, xb, yb, k)
        pen_spec = lambda_spec * global_spectral_penalty(m) if lambda_spec > 0 else 0.0
        pen_frob = lambda_frob * global_frobenius_penalty(m) if lambda_frob > 0 else 0.0
        return base + pen_spec + pen_frob, (new_s, base, pen_spec + pen_frob)

    # 1) grads at current params
    (loss1, (state_after, _, _)), grads = eqx.filter_value_and_grad(
        total_loss_fn, has_aux=True
    )(model, state, x, y, key)

    # 2) epsilon direction
    params = eqx.filter(model, eqx.is_inexact_array)
    g_params = eqx.filter(grads, eqx.is_inexact_array)
    scaled = (
        jax.tree_map(lambda g, p: g * (jnp.abs(p) + asam_eps), g_params, params)
        if sam_mode == "asam"
        else g_params
    )
    gnorm = optax.global_norm(scaled)
    epsilon = jax.tree_map(lambda s: s * (sam_rho / (gnorm + 1e-12)), scaled)

    # 3) second grads at perturbed params (optionally freeze norm layers)
    model_pert = eqx.apply_updates(model, epsilon)
    model_pert = (
        eqx.nn.inference_mode(model_pert) if freeze_norm_on_perturbed else model_pert
    )

    (loss2, _), grads2 = eqx.filter_value_and_grad(total_loss_fn, has_aux=True)(
        model_pert, state, x, y, key
    )

    # 4) optimizer step using grads2; keep only state from the first pass
    updates, opt_state = optimizer.update(grads2, opt_state, params=params)
    model = eqx.apply_updates(model, updates)
    return model, state_after, opt_state, loss2


def _contains_batchnorm(tree) -> bool:
    # cheap recursive scan; extend if you have custom BN blocks
    if isinstance(tree, eqx.nn.BatchNorm):
        return True
    if isinstance(tree, eqx.Module):
        return any(_contains_batchnorm(v) for v in vars(tree).values())
    if isinstance(tree, (list, tuple)):
        return any(_contains_batchnorm(v) for v in tree)
    if isinstance(tree, dict):
        return any(_contains_batchnorm(v) for v in tree.values())
    return False


@eqx.filter_jit
def eval_step(model, state, x, y, key, loss_fn):
    loss, _ = loss_fn(model, state, x, y, key)
    return loss


def train(
    model: Any,
    state: Any,
    opt_state: Any,
    optimizer: Any,
    loss_fn: Callable[
        [Any, Any, jnp.ndarray, jnp.ndarray, jr.PRNGKey], Tuple[jnp.ndarray, Any]
    ],
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_val: jnp.ndarray,
    y_val: jnp.ndarray,
    batch_size: int,
    num_epochs: int,
    patience: int,
    key: jr.PRNGKey,
    *,
    augment_fn: Optional[Callable[[jr.PRNGKey, jnp.ndarray], jnp.ndarray]] = None,
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    ckpt_path: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    return_penalty_history: bool = False,
    # -------------------- EMA knobs (optional) -------------------- #
    use_ema: bool = False,
    ema_decay: float = 0.999,
    eval_with_ema: bool = True,
    return_ema: bool = False,
) -> Union[
    Tuple[Any, Any, List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], EMA],
    Tuple[Any, Any, List[float], List[float], List[float], EMA],
]:
    """
    Train a model with early stopping, spectral/Frobenius penalties, and optional periodic checkpointing.

    Args:
        model: Equinox model to train.
        state: Model state (e.g., for batch-norm).
        opt_state: Optimizer state.
        optimizer: Optax or similar optimizer.
        loss_fn: Function (model, state, xb, yb, key) -> (loss, new_state_info).
        X_train, y_train: Training data arrays.
        X_val, y_val: Validation data arrays.
        batch_size: Samples per batch.
        num_epochs: Max epochs to run.
        patience: Epochs to wait for improvement before stopping.
        key: JAX PRNG key.

    Keyword Args:
        augment_fn: Optional augmentation fn taking (key, xb) -> xb_aug.
        lambda_spec: Weight for spectral norm penalty.
        lambda_frob: Weight for Frobenius norm penalty.
        ckpt_path: If provided, path to save checkpoints.
        checkpoint_interval: If provided (>0), save a checkpoint every N epochs.
            Example:
                CKPT_PATH  = pathlib.Path("checkpoints/unet_eqx-epoch-{epoch:03d}.ckpt")
                train(..., ckpt_path=CKPT_PATH, checkpoint_interval=5)
                Then after train, load the weights with:
                    ckpt = eqx.tree_deserialise_leaves(CKPT, like={"model": tmpl_model,
                                                                "state": tmpl_state})
                    model, state = ckpt["model"], ckpt["state"]
        return_penalty_history: If True, also return the penalty history.

    EMA behavior:
      - If use_ema=True, we maintain a shadow EMA of params updated *after* each optimizer step.
      - If eval_with_ema=True, validation + early stopping use EMA weights (recommended).
      - Checkpoints store both raw params and ema params (if enabled).
      - If return_ema=True, returns the *best-epoch* EMA object at the end.

    Returns (by flags):
      base:        (best_model, best_state, train_losses, val_losses)
      + penalty:   (..., penalty_history)
      + ema:       (..., ema_best)
      + both:      (..., penalty_history, ema_best)
    """
    if checkpoint_interval is not None:
        assert checkpoint_interval > 0, "`checkpoint_interval` must be > 0"

    rng, eval_rng = jr.split(key)
    train_losses, val_losses, penalty_history = [], [], []

    # -------------------- EMA init -------------------- #
    ema: Optional[EMA] = init_ema(model, decay=ema_decay) if use_ema else None
    # Track the *best-epoch* EMA separately so we can return it
    best_ema_params = ema.params if (use_ema and ema is not None) else None
    best_ema_decay = ema.decay if (use_ema and ema is not None) else None

    best_val = float("inf")
    best_model = model
    best_state = state
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        # -------------------- Train -------------------- #
        epoch_train_loss, n_train = 0.0, 0
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
                model,
                state,
                opt_state,
                xb,
                yb,
                step_rng,
                loss_fn,
                optimizer,
                lambda_spec,
                lambda_frob,
            )
            # -------------------- EMA update (outside jit) -------------------- #
            if use_ema and ema is not None:
                ema = update_ema(ema, model)

            epoch_train_loss += float(batch_loss) * xb.shape[0]
            n_train += xb.shape[0]

        train_losses.append(epoch_train_loss / max(1, n_train))

        # -------------------- Validate (optionally with EMA) -------------------- #
        epoch_val_loss, n_val = 0.0, 0
        # choose which params to evaluate
        val_model = (
            swap_ema_params(model, ema)
            if (use_ema and ema is not None and eval_with_ema)
            else model
        )

        for xb, yb in data_loader(
            X_val,
            y_val,
            batch_size=batch_size,
            shuffle=False,
            key=eval_rng,
            augment_fn=None,
        ):
            eval_rng, vk = jr.split(eval_rng)
            val_loss, _ = loss_fn(val_model, state, xb, yb, vk)
            epoch_val_loss += float(val_loss) * xb.shape[0]
            n_val += xb.shape[0]
        val_losses.append(epoch_val_loss / max(1, n_val))

        # Track (raw) spectral penalty to monitor regularization magnitude
        penalty_history.append(float(global_spectral_penalty(model)))

        if epoch % max(1, num_epochs // 10) == 0 or epoch == num_epochs:
            print(
                f"[Epoch {epoch:3d}/{num_epochs}] "
                f"Train={train_losses[-1]:.4f} | Val={val_losses[-1]:.4f}"
            )

        improved = val_losses[-1] < best_val
        if improved:
            best_val = val_losses[-1]
            best_model, best_state = model, state
            patience_counter = 0

            # Freeze best-epoch EMA snapshot in-memory so we can return it later
            if use_ema and ema is not None:
                best_ema_params = eqx.tree_map(
                    lambda x: x, ema.params, is_leaf=eqx.is_inexact_array
                )
                best_ema_decay = float(ema.decay)

            # Save "best" checkpoint for this epoch
            if ckpt_path is not None:
                best_file = pathlib.Path(str(ckpt_path).format(epoch=epoch))
                best_file.parent.mkdir(parents=True, exist_ok=True)
                to_save = {"model": best_model, "state": best_state}
                if use_ema and ema is not None:
                    to_save["ema_params"] = best_ema_params
                    to_save["ema_decay"] = best_ema_decay
                eqx.tree_serialise_leaves(best_file, to_save)
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Periodic (non-best) checkpoints of the *current* weights
        if (
            ckpt_path is not None
            and checkpoint_interval is not None
            and epoch % checkpoint_interval == 0
        ):
            ckpt_file = pathlib.Path(str(ckpt_path).format(epoch=epoch))
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            to_save = {"model": model, "state": state}
            if use_ema and ema is not None:
                to_save["ema_params"] = ema.params
                to_save["ema_decay"] = ema.decay
            eqx.tree_serialise_leaves(ckpt_file, to_save)

    # -------------------- Assemble return tuple -------------------- #
    base = (best_model, best_state, train_losses, val_losses)
    both = (
        return_penalty_history
        and return_ema
        and (use_ema and best_ema_params is not None)
    )

    if both:
        return base + (
            penalty_history,
            EMA(params=best_ema_params, decay=best_ema_decay),
        )
    if return_penalty_history:
        return base + (penalty_history,)
    if return_ema and use_ema and best_ema_params is not None:
        return base + (EMA(params=best_ema_params, decay=best_ema_decay),)
    return base


def train_on_full_data(
    model,
    state,
    opt_state,
    optimizer,
    loss_fn,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size,
    num_epochs,
    key,
    *,
    augment_fn=None,
    lambda_spec=0.0,
    lambda_frob=0.0,
    ckpt_path=None,
    checkpoint_interval=None,
    return_penalty_history=False,
    # propagate EMA knobs
    use_ema: bool = False,
    ema_decay: float = 0.999,
    eval_with_ema: bool = True,
    return_ema: bool = False,
):
    """Combine train+val, disable early-stop, but keep periodic checkpoints and EMA support."""
    X_full = jnp.concatenate([X_train, X_val], axis=0)
    y_full = jnp.concatenate([y_train, y_val], axis=0)
    # Minimal dummy val to keep shapes consistent; early stopping is disabled anyway.
    X_dummy, y_dummy = X_full[:1], y_full[:1]

    return train(
        model,
        state,
        opt_state,
        optimizer,
        loss_fn,
        X_full,
        y_full,
        X_dummy,
        y_dummy,
        batch_size,
        num_epochs,
        patience=num_epochs,  # effectively no early stop
        key=key,
        augment_fn=augment_fn,
        lambda_spec=lambda_spec,
        lambda_frob=lambda_frob,
        ckpt_path=ckpt_path,
        checkpoint_interval=checkpoint_interval,
        return_penalty_history=return_penalty_history,
        use_ema=use_ema,
        ema_decay=ema_decay,
        eval_with_ema=eval_with_ema,
        return_ema=return_ema,
    )


def train_sam(
    model: Any,
    state: Any,
    opt_state: Any,
    optimizer: Any,
    loss_fn,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_val: jnp.ndarray,
    y_val: jnp.ndarray,
    batch_size: int,
    num_epochs: int,
    patience: int,
    key: jr.PRNGKey,
    *,
    augment_fn=None,
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    ckpt_path: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    return_penalty_history: bool = False,
    # EMA
    use_ema: bool = False,
    ema_decay: float = 0.999,
    eval_with_ema: bool = True,
    return_ema: bool = False,
    # SAM/ASAM
    sam_rho: float = 0.05,
    sam_mode: Literal["sam", "asam"] = "sam",
    asam_eps: float = 1e-12,
    require_no_batchnorm: bool = True,
) -> Union[
    Tuple[Any, Any, List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], "EMA"],
    Tuple[Any, Any, List[float], List[float], List[float], "EMA"],
]:
    """
    SAM/ASAM trainer.
    - Doubles forward/backward cost per step (two passes).
    - By default refuses to run if BatchNorm is present (SAM perturbed pass breaks BN stats).
      Set `require_no_batchnorm=False` only if you know how to freeze BN externally.
    """
    if checkpoint_interval is not None:
        assert checkpoint_interval > 0, "`checkpoint_interval` must be > 0"

    if require_no_batchnorm and _contains_batchnorm(model):
        raise ValueError(
            "BatchNorm detected. SAM/ASAM is unsafe with BN unless BN stats are frozen. "
            "Remove BN or set require_no_batchnorm=False and handle BN manually."
        )

    rng, eval_rng = jr.split(key)
    train_losses, val_losses, penalty_history = [], [], []

    # EMA init
    ema = init_ema(model, decay=ema_decay) if use_ema else None
    best_ema_params = ema.params if (use_ema and ema is not None) else None
    best_ema_decay = ema.decay if (use_ema and ema is not None) else None

    best_val = float("inf")
    best_model = model
    best_state = state
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        # ---------------- Train ----------------
        epoch_train_loss, n_train = 0.0, 0
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
            model, state, opt_state, batch_loss = train_step_sam(
                model,
                state,
                opt_state,
                xb,
                yb,
                step_rng,
                loss_fn,
                optimizer,
                lambda_spec,
                lambda_frob,
                sam_rho=sam_rho,
                sam_mode=sam_mode,
                asam_eps=asam_eps,
            )
            if use_ema and ema is not None:
                ema = update_ema(ema, model)

            epoch_train_loss += float(batch_loss) * xb.shape[0]
            n_train += xb.shape[0]

        train_losses.append(epoch_train_loss / max(1, n_train))

        # ---------------- Validate (EMA optional) ----------------
        epoch_val_loss, n_val = 0.0, 0
        val_model = (
            swap_ema_params(model, ema)
            if (use_ema and ema is not None and eval_with_ema)
            else model
        )

        for xb, yb in data_loader(
            X_val,
            y_val,
            batch_size=batch_size,
            shuffle=False,
            key=eval_rng,
            augment_fn=None,
        ):
            eval_rng, vk = jr.split(eval_rng)
            val_loss, _ = loss_fn(val_model, state, xb, yb, vk)
            epoch_val_loss += float(val_loss) * xb.shape[0]
            n_val += xb.shape[0]

        val_losses.append(epoch_val_loss / max(1, n_val))

        penalty_history.append(float(global_spectral_penalty(model)))

        if epoch % max(1, num_epochs // 10) == 0 or epoch == num_epochs:
            print(
                f"[SAM {sam_mode.upper()} | Epoch {epoch:3d}/{num_epochs}] "
                f"Train={train_losses[-1]:.4f} | Val={val_losses[-1]:.4f}"
            )

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            best_model, best_state = model, state
            patience_counter = 0

            if use_ema and ema is not None:
                # snapshot EMA aligned with best epoch
                best_ema_params = eqx.tree_map(
                    lambda x: x, ema.params, is_leaf=eqx.is_inexact_array
                )
                best_ema_decay = float(ema.decay)

            if ckpt_path is not None:
                best_file = pathlib.Path(str(ckpt_path).format(epoch=epoch))
                best_file.parent.mkdir(parents=True, exist_ok=True)
                to_save = {"model": best_model, "state": best_state}
                if use_ema and ema is not None:
                    to_save["ema_params"] = best_ema_params
                    to_save["ema_decay"] = best_ema_decay
                eqx.tree_serialise_leaves(best_file, to_save)
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if (
            ckpt_path is not None
            and checkpoint_interval is not None
            and epoch % checkpoint_interval == 0
        ):
            ckpt_file = pathlib.Path(str(ckpt_path).format(epoch=epoch))
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            to_save = {"model": model, "state": state}
            if use_ema and ema is not None:
                to_save["ema_params"] = ema.params
                to_save["ema_decay"] = ema.decay
            eqx.tree_serialise_leaves(ckpt_file, to_save)

    base = (best_model, best_state, train_losses, val_losses)
    both = (
        return_penalty_history
        and return_ema
        and (use_ema and best_ema_params is not None)
    )

    if both:
        return base + (
            penalty_history,
            EMA(params=best_ema_params, decay=best_ema_decay),
        )
    if return_penalty_history:
        return base + (penalty_history,)
    if return_ema and use_ema and best_ema_params is not None:
        return base + (EMA(params=best_ema_params, decay=best_ema_decay),)
    return base


def train_on_full_data_sam(
    model,
    state,
    opt_state,
    optimizer,
    loss_fn,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size,
    num_epochs,
    key,
    *,
    augment_fn=None,
    lambda_spec=0.0,
    lambda_frob=0.0,
    ckpt_path=None,
    checkpoint_interval=None,
    return_penalty_history=False,
    # EMA
    use_ema: bool = False,
    ema_decay: float = 0.999,
    eval_with_ema: bool = True,
    return_ema: bool = False,
    # SAM/ASAM
    sam_rho: float = 0.05,
    sam_mode: Literal["sam", "asam"] = "sam",
    asam_eps: float = 1e-12,
    require_no_batchnorm: bool = True,
    freeze_norm_on_perturbed: bool = True,
):
    """Combine train+val, disable early-stop, keep periodic checkpoints; SAM/ASAM backend."""
    X_full = jnp.concatenate([X_train, X_val], axis=0)
    y_full = jnp.concatenate([y_train, y_val], axis=0)
    X_dummy, y_dummy = X_full[:1], y_full[:1]

    return train_sam(
        model=model,
        state=state,
        opt_state=opt_state,
        optimizer=optimizer,
        loss_fn=loss_fn,
        X_train=X_full,
        y_train=y_full,
        X_val=X_dummy,
        y_val=y_dummy,
        batch_size=batch_size,
        num_epochs=num_epochs,
        patience=num_epochs,  # effectively no early stop
        key=key,
        augment_fn=augment_fn,
        lambda_spec=lambda_spec,
        lambda_frob=lambda_frob,
        ckpt_path=ckpt_path,
        checkpoint_interval=checkpoint_interval,
        return_penalty_history=return_penalty_history,
        use_ema=use_ema,
        ema_decay=ema_decay,
        eval_with_ema=eval_with_ema,
        return_ema=return_ema,
        sam_rho=sam_rho,
        sam_mode=sam_mode,
        asam_eps=asam_eps,
        require_no_batchnorm=require_no_batchnorm,
        # important for stability
        freeze_norm_on_perturbed=freeze_norm_on_perturbed,
    )


def _select_head(out):
    """Deep supervision support: if model returns a list of heads, use the last."""
    return out[-1] if isinstance(out, list) else out


def _channel_axis_to_last(x: Array, channel_axis: int) -> Array:
    """Move 'channel_axis' to last position for stable softmax/sigmoid ops."""
    if channel_axis == -1:
        return x
    return jnp.moveaxis(x, channel_axis, -1)


def _channel_axis_from_last(x: Array, channel_axis: int) -> Array:
    """Inverse of _channel_axis_to_last."""
    if channel_axis == -1:
        return x
    return jnp.moveaxis(x, -1, channel_axis)


def _infer_logits_slice_jitted(
    x_slice: Array, keys: jr.key, model: eqx.Module, state: Any
) -> Array:
    """JITted, slice-wise forward that returns logits (or raw outputs for regression)."""

    @eqx.filter_jit
    def _infer_slice(
        x_slice: Array, keys: jr.key, model: eqx.Module, state: Any
    ) -> Array:
        inference_model = eqx.nn.inference_mode(model)

        def single(x, k):
            out, _ = inference_model(x, k, state)
            logits = _select_head(out)
            return logits

        return jax.vmap(single, in_axes=(0, 0))(x_slice, keys)

    return _infer_slice(x_slice, keys, model, state)


def _apply_link(
    logits: Array,
    task: Literal["auto", "classification", "binary", "regression"],
    channel_axis: int,
) -> Array:
    """
    Convert logits to prediction space:
    - classification/segmentation: softmax along channel_axis (C ≥ 2)
    - binary: sigmoid (C == 1 or scalar)
    - regression: identity
    - auto: infer binary vs multiclass by channel size along channel_axis
    """
    if task == "regression":
        return logits

    # Move channels to last to apply softmax/sigmoid cleanly
    moved = _channel_axis_to_last(logits, channel_axis)

    if task == "binary":
        probs = jax.nn.sigmoid(moved)
        return _channel_axis_from_last(probs, channel_axis)

    if task == "classification":
        probs = jax.nn.softmax(moved, axis=-1)
        return _channel_axis_from_last(probs, channel_axis)

    # auto
    C = moved.shape[-1] if moved.ndim >= 1 else 1
    if C == 1:
        probs = jax.nn.sigmoid(moved)
    else:
        probs = jax.nn.softmax(moved, axis=-1)
    return _channel_axis_from_last(probs, channel_axis)


def _aggregate_models_probspace(probs_list: List[Array], weights: Array) -> Array:
    """Weighted sum of probabilities. Shapes must match."""
    acc = None
    for w, p in zip(weights, probs_list):
        acc = p * w if acc is None else acc + p * w
    return acc


def _aggregate_models_logitspace(logits_list: List[Array], weights: Array) -> Array:
    """Arithmetic mean of logits (not probability preserving, but often used)."""
    acc = None
    for w, z in zip(weights, logits_list):
        acc = z * w if acc is None else acc + z * w
    return acc


# ============================================================
# Single-model inference (drop-in replacements)
# ============================================================


@eqx.filter_jit
def predict(
    model: eqx.Module,
    state: Any,
    X: Array,
    key: jr.key,
) -> Array:
    """Vectorized single-model forward over batch. Returns raw outputs/logits."""
    inference_model = eqx.nn.inference_mode(model)

    def single(x, k):
        out, _ = inference_model(x, k, state)
        return _select_head(out)

    keys = jr.split(key, X.shape[0])
    return jax.vmap(single, in_axes=(0, 0))(X, keys)


@eqx.filter_jit
def predict_batched(
    model: eqx.Module,
    state: Any,
    X: Array,
    key: jr.key,
    batch_size: int = 256,
) -> Array:
    """Chunked single-model forward. Returns raw outputs/logits."""
    inference_model = eqx.nn.inference_mode(model)
    N = X.shape[0]
    num_batches = (N + batch_size - 1) // batch_size
    batch_keys = jr.split(key, num_batches)

    preds = []
    start = 0
    for bk in batch_keys:
        end = min(start + batch_size, N)
        xb = X[start:end]
        subkeys = jr.split(bk, xb.shape[0])

        def infer_one(x, sk):
            out, _ = inference_model(x, sk, state)
            return _select_head(out)

        batch_preds = jax.vmap(infer_one, in_axes=(0, 0))(xb, subkeys)
        preds.append(batch_preds)
        start = end

    return jnp.concatenate(preds, axis=0)


@eqx.filter_jit
def infer_slice(
    x_slice: Array,
    keys: jr.key,
    model: eqx.Module,
    state: Any,
) -> Array:
    """Slice-wise inference (very large batches / streaming). Returns logits/raw."""
    inference_model = eqx.nn.inference_mode(model)

    def single(x, k):
        out, _ = inference_model(x, k, state)
        logits = _select_head(out)
        return logits

    return jax.vmap(single, in_axes=(0, 0))(x_slice, keys)


def predict_batched_efficient(
    model: eqx.Module,
    state: Any,
    X: Array,
    key: jr.key,
    batch_size: int = 24,
) -> Array:
    """Memory-aware variant. Returns logits/raw. Uses device_get to free memory."""
    N = X.shape[0]
    all_logits = []
    for i in range(0, N, batch_size):
        x_slice = X[i : i + batch_size]
        key, slice_key = jr.split(key)
        subkeys = jr.split(slice_key, x_slice.shape[0])
        logits_slice = infer_slice(x_slice, subkeys, model, state)
        logits_slice = jax.device_get(logits_slice)
        jax.block_until_ready(logits_slice)
        all_logits.append(logits_slice)
    return jnp.concatenate(all_logits, axis=0)


# ============================================================
# Ensemble inference
# ============================================================


def predict_ensemble_batched(
    models: Sequence[eqx.Module],
    states: Sequence[Any],
    X: Array,
    key: jr.key,
    batch_size: int = 256,
    task: Literal["auto", "classification", "binary", "regression"] = "auto",
    channel_axis: int = -1,
    agg: Literal["probs", "logits"] = "probs",
    weights: Optional[Array] = None,
    return_variance: bool = False,  # only for regression
) -> Union[Array, Tuple[Array, Array]]:
    """
    Memory-safe ensemble prediction.

    - classification/segmentation:
        * agg="probs" (default): average probabilities (recommended) -> returns probs
        * agg="logits": average logits -> returns logits
    - binary: like classification but uses sigmoid
    - regression: returns mean; if return_variance=True, also epistemic variance

    Shapes:
      X: [N, ...]
      Output:
        classification: [N, ..., C] with channels at `channel_axis`
        binary: [N, ..., 1] (or scalar per spatial), channels at `channel_axis`
        regression: mean [N, ...] (+ var [N, ...] if requested)
    """
    assert len(models) == len(states), "models and states must be same length"
    M = len(models)
    if weights is None:
        weights = jnp.ones((M,), dtype=jnp.float32) / M
    else:
        weights = jnp.asarray(weights, dtype=jnp.float32)
        assert weights.shape == (M,), "weights must have shape [M]"
        weights = weights / jnp.sum(weights)

    N = X.shape[0]
    num_batches = (N + batch_size - 1) // batch_size

    # Allocate on first batch when we know output shape
    out_buffer = None
    var_buffer = None

    start = 0
    for _ in range(num_batches):
        end = min(start + batch_size, N)
        x_slice = X[start:end]
        B = x_slice.shape[0]

        # Accumulators for this batch
        if task == "regression":
            batch_mean = None
            batch_second_moment = None if return_variance else None
        else:
            if agg == "probs":
                batch_accum_probs: Optional[Array] = None
            else:
                batch_accum_logits: Optional[Array] = None

        # Loop over models (each forward is jitted)
        for m_idx, (model, state) in enumerate(zip(models, states)):
            key, slice_key = jr.split(key)
            subkeys = jr.split(slice_key, B)
            logits = _infer_logits_slice_jitted(
                x_slice, subkeys, model, state
            )  # [B, ...]

            if task == "regression":
                w = weights[m_idx]
                contrib = w * logits
                batch_mean = contrib if batch_mean is None else (batch_mean + contrib)
                if return_variance:
                    contrib2 = w * (logits**2)
                    batch_second_moment = (
                        contrib2
                        if batch_second_moment is None
                        else (batch_second_moment + contrib2)
                    )
            else:
                if agg == "probs":
                    probs = _apply_link(logits, task, channel_axis)
                    weighted = weights[m_idx] * probs
                    batch_accum_probs = (
                        weighted
                        if batch_accum_probs is None
                        else (batch_accum_probs + weighted)
                    )
                else:
                    weighted = weights[m_idx] * logits
                    batch_accum_logits = (
                        weighted
                        if batch_accum_logits is None
                        else (batch_accum_logits + weighted)
                    )

        # Allocate final buffers on first batch
        if out_buffer is None:
            if task == "regression":
                out_shape = (N,) + batch_mean.shape[1:]
                out_buffer = jnp.zeros(out_shape, dtype=batch_mean.dtype)
                if return_variance:
                    var_buffer = jnp.zeros(out_shape, dtype=batch_mean.dtype)
            else:
                if agg == "probs":
                    out_shape = (N,) + batch_accum_probs.shape[1:]
                    out_buffer = jnp.zeros(out_shape, dtype=batch_accum_probs.dtype)
                else:
                    out_shape = (N,) + batch_accum_logits.shape[1:]
                    out_buffer = jnp.zeros(out_shape, dtype=batch_accum_logits.dtype)

        # Write batch to buffers
        if task == "regression":
            out_buffer = out_buffer.at[start:end].set(batch_mean)
            if return_variance:
                # Var_w[y] = E_w[y^2] - (E_w[y])^2
                batch_var = jnp.maximum(0.0, batch_second_moment - batch_mean**2)
                var_buffer = var_buffer.at[start:end].set(batch_var)
        else:
            if agg == "probs":
                out_buffer = out_buffer.at[start:end].set(batch_accum_probs)
            else:
                out_buffer = out_buffer.at[start:end].set(batch_accum_logits)

        start = end

    if task == "regression":
        if return_variance:
            return out_buffer, var_buffer
        return out_buffer
    else:
        return out_buffer


def predict_ensemble(
    models: Sequence[eqx.Module],
    states: Sequence[Any],
    X: Array,
    key: jr.key,
    task: Literal["auto", "classification", "binary", "regression"] = "auto",
    channel_axis: int = -1,
    agg: Literal["probs", "logits"] = "probs",
    weights: Optional[Array] = None,
) -> Array:
    """Convenience: run ensemble in a single batch (uses the batched impl under the hood)."""
    return predict_ensemble_batched(
        models=models,
        states=states,
        X=X,
        key=key,
        batch_size=X.shape[0],
        task=task,
        channel_axis=channel_axis,
        agg=agg,
        weights=weights,
        return_variance=False,
    )


# ============================================================
# Optional: VMAP-based fast path (uniform model shapes)
# ============================================================


def predict_ensemble_fast_vmap(
    models: Sequence[eqx.Module],
    states: Sequence[Any],
    X: Array,
    key: jr.key,
    task: Literal["auto", "classification", "binary", "regression"] = "auto",
    channel_axis: int = -1,
    agg: Literal["probs", "logits"] = "probs",
    weights: Optional[Array] = None,
) -> Array:
    """
    Faster path when ALL models share identical tree structure and output shapes.
    Vectorizes across models (M) and batch (N). May use more memory than the batched loop.
    """
    M = len(models)
    assert M == len(states), "models and states must be same length"
    if weights is None:
        weights = jnp.ones((M,), dtype=jnp.float32) / M
    else:
        weights = jnp.asarray(weights, dtype=jnp.float32)
        assert weights.shape == (M,), "weights must have shape [M]"
        weights = weights / jnp.sum(weights)

    N = X.shape[0]
    keys = jr.split(key, N * M).reshape(N, M, 2)

    def forward_one_example(x, ks, models, states):
        def apply_model(m, s, k):
            inference_model = eqx.nn.inference_mode(m)
            out, _ = inference_model(x, k, s)
            return _select_head(out)

        # vmap across models axis -> [M, ...]
        outs_m = eqx.filter_vmap(apply_model, in_axes=(0, 0, 0))(models, states, ks)

        if task == "regression":
            mean = jnp.tensordot(weights, outs_m, axes=([0], [0]))  # [ ... ]
            return mean

        if agg == "probs":
            # Convert each model's output to probs, then weight-average
            probs_list = []
            for i in range(M):
                probs_list.append(_apply_link(outs_m[i], task, channel_axis))
            return _aggregate_models_probspace(probs_list, weights)
        else:
            # Weighted logits
            return _aggregate_models_logitspace(list(outs_m), weights)

    # vmap across batch
    return jax.vmap(forward_one_example, in_axes=(0, 0, None, None))(
        X, keys, models, states
    )


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import augmax
    from augmax import InputType

    num_epochs = 10
    batch_size = 32

    rng = np.random.RandomState(0)
    N, H, W, C, NUM_CLASSES = 2048, 1, 28, 28, 1
    X_np = rng.rand(N, H, W, C).astype("float32")  # [N, H, W, C]
    y_np = rng.randint(0, NUM_CLASSES, size=(N,)).astype("int32")

    # train / val split
    split = int(0.8 * N)
    X_train, X_val = X_np[:split], X_np[split:]
    y_train, y_val = y_np[:split], y_np[split:]

    transform = augmax.Chain(
        augmax.HorizontalFlip(),
        augmax.Rotate(angle_range=15),
        input_types=[InputType.IMAGE, InputType.METADATA],
    )
    augment_fn = make_augmax_augment(transform)

    class SimpleCNN(eqx.Module):
        conv1: eqx.nn.Conv2d
        bn1: eqx.nn.BatchNorm
        conv2: eqx.nn.Conv2d
        bn2: eqx.nn.BatchNorm
        pool1: eqx.nn.MaxPool2d
        pool2: eqx.nn.MaxPool2d
        fc1: eqx.nn.Linear
        bn3: eqx.nn.BatchNorm
        fc2: eqx.nn.Linear
        drop1: eqx.nn.Dropout
        drop2: eqx.nn.Dropout
        drop3: eqx.nn.Dropout

        def __init__(self, key):
            k1, k2, k3, k4 = jr.split(key, 4)
            self.conv1 = eqx.nn.Conv2d(1, 32, kernel_size=3, padding=1, key=k1)
            self.bn1 = eqx.nn.BatchNorm(input_size=32, axis_name="batch", mode="batch")
            self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=3, padding=1, key=k2)
            self.bn2 = eqx.nn.BatchNorm(input_size=64, axis_name="batch", mode="batch")
            self.pool1 = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = eqx.nn.Linear(64 * 7 * 7, 128, key=k3)
            self.bn3 = eqx.nn.BatchNorm(input_size=128, axis_name="batch", mode="batch")
            self.fc2 = eqx.nn.Linear(128, 10, key=k4)
            self.drop1 = eqx.nn.Dropout(0.25)
            self.drop2 = eqx.nn.Dropout(0.25)
            self.drop3 = eqx.nn.Dropout(0.50)

        def __call__(self, x, key, state):
            d1, d2, d3 = jr.split(key, 3)
            x = self.conv1(x)
            x, state = self.bn1(x, state)
            x = jax.nn.relu(x)
            x = self.pool1(x)
            x = self.drop1(x, key=d1)
            x = self.conv2(x)
            x, state = self.bn2(x, state)
            x = jax.nn.relu(x)
            x = self.pool2(x)
            x = self.drop2(x, key=d2)
            x = x.reshape(-1)
            x = self.fc1(x)
            x, state = self.bn3(x, state)
            x = jax.nn.relu(x)
            x = self.drop3(x, key=d3)
            x = self.fc2(x)
            return x, state

    master_key = jr.PRNGKey(42)
    model_key, train_key = jr.split(master_key)

    model, state = eqx.nn.make_with_state(SimpleCNN)(model_key)
    steps_per_epoch = int(jnp.ceil(X_train.shape[0] / batch_size))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(100, steps_per_epoch)  # ~1 epoch or 100 steps

    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,  # your peak LR
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=1e-5,
    )
    optimizer = optax.adan(
        learning_rate=lr_sched,
        b1=0.95,
        b2=0.99,
        eps=1e-8,
        weight_decay=1e-4,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    best_model, best_state, tr_loss, va_loss = train(
        model,
        state,
        opt_state,
        optimizer,
        multiclass_loss,
        jnp.array(X_train),
        jnp.array(y_train),
        jnp.array(X_val),
        jnp.array(y_val),
        batch_size=batch_size,
        num_epochs=num_epochs,
        patience=5,
        key=train_key,
        augment_fn=augment_fn,
        lambda_spec=0.0,
    )
    print("Training complete.")
    logits = predict(best_model, best_state, jnp.array(X_val), train_key)
    print("Predictions shape:", logits.shape)

    plt.plot(tr_loss, label="train")
    plt.plot(va_loss, label="val")
    plt.legend()
    plt.show()
