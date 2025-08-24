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
    Dict,
)
from quantbayes.stochax.utils import EMA, init_ema, update_ema, swap_ema_params
from quantbayes.stochax.utils.spectral_penalty_tx import specpen_metrics_from_opt_state
from quantbayes.stochax.utils.regularizers import (
    global_spectral_penalty,  # backward-compat (your custom spectral hooks)
    global_frobenius_penalty,  # L2 on weights (skips bias)
    global_spectral_norm_penalty,  # Σ per-layer σ (exact/TN/FFT)
    sobolev_jacobian_penalty,  # Jacobian Sobolev
    sobolev_kernel_smoothness,  # kernel ∇ smoothness
    lip_product_penalty,  # τ·log(Lip_upper)
)
from quantbayes.stochax.utils.regularizers import network_lipschitz_upper


Array = jnp.ndarray

AugmentFn = Callable[[jr.key, jnp.ndarray], jnp.ndarray]


def make_augmax_augment(transform):
    """
    Build an Augmax augmentor that supports 1- or 2-input chains.

    - 1 input (IMAGE): labels/metadata bypass unchanged.
    - 2 inputs (IMAGE + MASK/METADATA):
        * (B,C,H,W) -> transposed to (B,H,W,C)
        * (B,H,W)   -> expanded to (B,H,W,1)
        * otherwise -> passed through unchanged (e.g., vectors)
    """
    t_input_types = getattr(transform, "input_types", None)
    n_inputs = len(t_input_types) if t_input_types is not None else 1
    if n_inputs not in (1, 2):
        raise ValueError(f"Unsupported chain arity {n_inputs}; expected 1 or 2.")

    @jax.jit
    def _augment(key: jr.key, batch_pair: Tuple[jnp.ndarray, Any]):
        imgs_chw, second = batch_pair
        B = imgs_chw.shape[0]
        subkeys = jr.split(key, B)

        # CHW -> HWC
        imgs_hwc = jnp.transpose(imgs_chw, (0, 2, 3, 1))

        if n_inputs == 1:
            img_hwc_aug = jax.vmap(transform)(subkeys, imgs_hwc)
            img_chw_aug = jnp.transpose(img_hwc_aug, (0, 3, 1, 2))
            return img_chw_aug, second

        # n_inputs == 2
        if hasattr(second, "ndim") and second.ndim == 4:
            masks_hwc = jnp.transpose(second, (0, 2, 3, 1))
        elif hasattr(second, "ndim") and second.ndim == 3:
            masks_hwc = second[..., None]
        else:
            masks_hwc = second

        outputs = jax.vmap(transform)(subkeys, [imgs_hwc, masks_hwc])

        # normalize return structure
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            img_hwc_aug, second_hwc_aug = outputs
        else:
            img_hwc_aug, second_hwc_aug = outputs[0], outputs[1]

        img_chw_aug = jnp.transpose(img_hwc_aug, (0, 3, 1, 2))

        if hasattr(second_hwc_aug, "ndim") and second_hwc_aug.ndim == 4:
            second_chw_aug = jnp.transpose(second_hwc_aug, (0, 3, 1, 2))
        elif hasattr(second_hwc_aug, "ndim") and second_hwc_aug.ndim == 3:
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


def make_lipschitz_upper_fn(
    *,
    conv_mode: Literal[
        "tn", "circular_fft", "min_tn_circ_embed", "circ_plus_lr"
    ] = "tn",
    conv_tn_iters: int = 8,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    conv_input_shape: Optional[Tuple[int, int]] = None,
) -> Callable[[Any], jnp.ndarray]:
    """Returns a JIT'd function L(model) = certified global Lipschitz upper bound."""

    def _L(m):
        return network_lipschitz_upper(
            m,
            conv_mode=conv_mode,
            conv_tn_iters=conv_tn_iters,
            conv_fft_shape=conv_fft_shape,
            conv_input_shape=conv_input_shape,
        )

    return eqx.filter_jit(_L)


class BoundLogger:
    def __init__(self):
        self.data = []

    def __call__(self, rec):
        # keep it JSON-safe
        self.data.append(
            {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in rec.items()
            }
        )


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
    lambda_spec: float = 0.0,  # legacy: your __spectral_penalty__ hook
    lambda_frob: float = 0.0,  # Frobenius/L2 on weights
    # --- NEW knobs ---
    lambda_specnorm: float = 0.0,  # Σ per-layer σ (Linear exact; Conv TN/FFT)
    lambda_sob_jac: float = 0.0,  # Jacobian Sobolev (data-dependent)
    lambda_sob_kernel: float = 0.0,  # kernel smoothness
    lambda_liplog: float = 0.0,  # τ·log Lip_upper (very small τ)
    sob_jac_apply: Optional[
        Callable[[Any, Any, jnp.ndarray, jr.PRNGKey], jnp.ndarray]
    ] = None,
    sob_jac_samples: int = 1,
    conv_tn_iters: int = 8,
    conv_fft_shape: Optional[
        Tuple[int, int]
    ] = None,  # only if you use circular FFT exact
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
        # NEW: Σ per-layer operator norms (exact/TN/FFT)
        pen_specnorm = (
            lambda_specnorm
            * global_spectral_norm_penalty(
                m,
                conv_mode=("tn" if conv_fft_shape is None else "circular_fft"),
                conv_tn_iters=conv_tn_iters,
                conv_fft_shape=conv_fft_shape,
            )
            if lambda_specnorm > 0.0
            else 0.0
        )
        # NEW: kernel Sobolev
        pen_sob_k = (
            lambda_sob_kernel * sobolev_kernel_smoothness(m)
            if lambda_sob_kernel > 0.0
            else 0.0
        )
        # NEW: function-space Sobolev (Jacobian). Requires f_apply.
        pen_sob_j = (
            lambda_sob_jac
            * sobolev_jacobian_penalty(
                m, s, xb, k, sob_jac_apply, num_samples=sob_jac_samples
            )
            if (lambda_sob_jac > 0.0 and sob_jac_apply is not None)
            else 0.0
        )
        # NEW: gentle Lipschitz product penalty (small τ suggested)
        pen_lip = (
            lip_product_penalty(
                m,
                tau=lambda_liplog,
                conv_mode=("tn" if conv_fft_shape is None else "circular_fft"),
                conv_tn_iters=conv_tn_iters,
                conv_fft_shape=conv_fft_shape,
            )
            if lambda_liplog > 0.0
            else 0.0
        )
        pen = pen_spec + pen_frob + pen_specnorm + pen_sob_k + pen_sob_j + pen_lip
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
    x,
    y,
    key,
    loss_fn,
    optimizer,
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    # --- NEW knobs ---
    lambda_specnorm: float = 0.0,
    lambda_sob_jac: float = 0.0,
    lambda_sob_kernel: float = 0.0,
    lambda_liplog: float = 0.0,
    sob_jac_apply: Optional[
        Callable[[Any, Any, jnp.ndarray, jr.PRNGKey], jnp.ndarray]
    ] = None,
    sob_jac_samples: int = 1,
    conv_tn_iters: int = 8,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    *,
    sam_rho: float = 0.05,
    sam_mode: Literal["sam", "asam"] = "sam",
    asam_eps: float = 1e-12,
    freeze_norm_on_perturbed: bool = True,
):
    # --- 1) data-only loss for epsilon ---
    def base_loss_fn(m, s, xb, yb, k):
        base, new_s = loss_fn(m, s, xb, yb, k)  # no penalties here
        return base, new_s

    (base1, state_after), grads_base = eqx.filter_value_and_grad(
        base_loss_fn, has_aux=True
    )(model, state, x, y, key)

    params = eqx.filter(model, eqx.is_inexact_array)
    g_params = eqx.filter(grads_base, eqx.is_inexact_array)

    def _scale(g, p):
        return g * (jnp.abs(p) + asam_eps) if sam_mode == "asam" else g

    scaled = jax.tree_map(_scale, g_params, params)
    gnorm = optax.global_norm(scaled)
    epsilon = jax.tree_map(lambda s: s * (sam_rho / (gnorm + 1e-12)), scaled)
    epsilon = jax.tree_map(jax.lax.stop_gradient, epsilon)  # defensive

    # --- 2) total loss (data + penalties) at perturbed params for update ---
    def total_loss_fn(m, s, xb, yb, k):
        base, _ = loss_fn(m, s, xb, yb, k)
        pen_spec = lambda_spec * global_spectral_penalty(m) if lambda_spec > 0 else 0.0
        pen_frob = lambda_frob * global_frobenius_penalty(m) if lambda_frob > 0 else 0.0
        pen_specnorm = (
            lambda_specnorm
            * global_spectral_norm_penalty(
                m,
                conv_mode=("tn" if conv_fft_shape is None else "circular_fft"),
                conv_tn_iters=conv_tn_iters,
                conv_fft_shape=conv_fft_shape,
            )
            if lambda_specnorm > 0
            else 0.0
        )
        pen_sob_k = (
            lambda_sob_kernel * sobolev_kernel_smoothness(m)
            if lambda_sob_kernel > 0
            else 0.0
        )
        pen_sob_j = (
            lambda_sob_jac
            * sobolev_jacobian_penalty(
                m, s, xb, k, sob_jac_apply, num_samples=sob_jac_samples
            )
            if (lambda_sob_jac > 0 and sob_jac_apply is not None)
            else 0.0
        )
        pen_lip = (
            lip_product_penalty(
                m,
                tau=lambda_liplog,
                conv_mode=("tn" if conv_fft_shape is None else "circular_fft"),
                conv_tn_iters=conv_tn_iters,
                conv_fft_shape=conv_fft_shape,
            )
            if lambda_liplog > 0
            else 0.0
        )
        return (
            base + pen_spec + pen_frob + pen_specnorm + pen_sob_k + pen_sob_j + pen_lip
        )

    model_pert = eqx.apply_updates(model, epsilon)
    if freeze_norm_on_perturbed:
        model_pert = eqx.nn.inference_mode(model_pert)

    loss2, grads2 = eqx.filter_value_and_grad(total_loss_fn)(
        model_pert, state, x, y, key
    )
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
    # NEW penalty knobs
    lambda_specnorm: float = 0.0,
    lambda_sob_jac: float = 0.0,
    lambda_sob_kernel: float = 0.0,
    lambda_liplog: float = 0.0,
    sob_jac_apply: Optional[
        Callable[[Any, Any, jnp.ndarray, jr.PRNGKey], jnp.ndarray]
    ] = None,
    sob_jac_samples: int = 1,
    conv_tn_iters: int = 8,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    ckpt_path: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    return_penalty_history: bool = False,
    # -------------------- EMA knobs (optional) -------------------- #
    use_ema: bool = False,
    ema_decay: float = 0.999,
    eval_with_ema: bool = True,
    return_ema: bool = False,
    specpen_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_global_bound_every: Optional[int] = None,  # e.g., 5 → compute every 5 epochs
    bound_conv_mode: Literal[
        "tn", "circular_fft", "min_tn_circ_embed", "circ_plus_lr"
    ] = "tn",
    bound_tn_iters: int = 8,
    bound_fft_shape: Optional[Tuple[int, int]] = None,  # for "circular_fft"
    bound_input_shape: Optional[
        Tuple[int, int]
    ] = None,  # for "min_tn_circ_embed"/"circ_plus_lr"
    bound_recorder: Optional[
        Callable[[Dict[str, Any]], None]
    ] = None,  # callback to collect logs
) -> Union[
    Tuple[Any, Any, List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], EMA],
    Tuple[Any, Any, List[float], List[float], List[float], EMA],
]:
    """
    Train a JAX/Equinox model with early stopping, optional EMA evaluation,
    checkpointing, spectral/Frobenius/Sobolev penalties, and (optionally)
    **certified global Lipschitz upper-bound logging** per epoch.

    Parameters
    ----------
    model, state : Any
        Equinox model and its mutable state (e.g., BatchNorm statistics).
    opt_state : Any
        Optimizer state returned by `optimizer.init(...)`.
    optimizer : optax.GradientTransformation
        Optax optimizer (e.g., `optax.adamw`, `optax.adan`, ...).
    loss_fn : Callable
        Signature `(model, state, xb, yb, key) -> (loss, new_state)`.
        The loop passes minibatches from `X_*`/`y_*` and manages `key`.
    X_train, y_train : jnp.ndarray
        Training inputs and labels.
    X_val, y_val : jnp.ndarray
        Validation inputs and labels (used for early stopping and reporting).
    batch_size : int
        Minibatch size.
    num_epochs : int
        Maximum number of epochs.
    patience : int
        Early-stopping patience (epochs without validation improvement).
    key : jr.PRNGKey
        RNG key for data-dependent pieces (loss, dropout, Sobolev estimates, etc.).

    Keyword Args
    ------------
    augment_fn : Optional[Callable[[jr.PRNGKey, jnp.ndarray], jnp.ndarray]]
        If provided, called as `xb_aug = augment_fn(key, xb)` each step.
    lambda_spec : float, default 0.0
        Weight for your legacy `global_spectral_penalty(model)` hook aggregator.
    lambda_frob : float, default 0.0
        Weight for `global_frobenius_penalty(model)` (classic L2 on weights).
    lambda_specnorm : float, default 0.0
        Weight for `global_spectral_norm_penalty(model)` = sum of per-layer
        operator norms (dense exact; conv via TN or exact circular depending
        on `conv_fft_shape`).
    lambda_sob_jac : float, default 0.0
        Weight for function-space Sobolev/Jacobian penalty
        `sobolev_jacobian_penalty(...)` (data-dependent).
    lambda_sob_kernel : float, default 0.0
        Weight for kernel smoothness `sobolev_kernel_smoothness(model)`.
    lambda_liplog : float, default 0.0
        τ for `lip_product_penalty(model, tau=τ)`; use very small τ.
    sob_jac_apply : Optional[Callable], sob_jac_samples : int
        Function used by the Jacobian penalty and number of Hutchinson probes.
    conv_tn_iters : int, default 8
        Iterations for tensor-norm (TN) bound on conv layers.
    conv_fft_shape : Optional[Tuple[int, int]]
        If set, `lambda_specnorm` uses **exact** circular conv norms via FFT
        for stride=1; otherwise convs use the TN bound.

    ckpt_path : Optional[str]
        Template path for checkpoints; `"{epoch}"` is formatted each save.
        Example: `"checkpoints/unet-e{epoch:03d}.ckpt"`.
    checkpoint_interval : Optional[int]
        Save a (non-best) checkpoint every N epochs. Must be > 0 if provided.
    return_penalty_history : bool, default False
        If True, also return a list with the epoch-wise value of
        `global_spectral_penalty(model)` (useful for monitoring regularization).

    # EMA (Exponential Moving Average)
    use_ema : bool, default False
        Maintain EMA shadow params updated *after* each optimizer step.
    ema_decay : float, default 0.999
        EMA decay factor.
    eval_with_ema : bool, default True
        If True and `use_ema`, validation + early-stopping use EMA params.
    return_ema : bool, default False
        If True and `use_ema`, return an EMA snapshot (params + decay) captured
        at the **best validation epoch**.

    # Logging for certified global Lipschitz upper bounds (optional)
    log_global_bound_every : Optional[int]
        If set (e.g., 1 or 5), compute and log a certified **global** Lipschitz
        upper bound every `k` epochs (and at the final epoch).
    bound_conv_mode : {"tn","circular_fft","min_tn_circ_embed","circ_plus_lr"}, default "tn"
        Which certified conv bound to use inside the global bound:
          • "tn" — tensor-norm bound (stride/dilation aware).
          • "circular_fft" — exact circular stride=1 (requires `bound_fft_shape`).
          • "min_tn_circ_embed" — min(TN, circulant-embedding UB) (requires `bound_input_shape`).
          • "circ_plus_lr" — our circulant+low-rank certificate (requires `bound_input_shape`).
    bound_tn_iters : int
        TN iterations for the bound (if mode uses TN).
    bound_fft_shape : Optional[Tuple[int, int]]
        Grid size for "circular_fft".
    bound_input_shape : Optional[Tuple[int, int]]
        Input H×W for "min_tn_circ_embed" and "circ_plus_lr".
    bound_recorder : Optional[Callable[[Dict[str, Any]], None]]
        Callback invoked with `{"epoch": int, "L_raw": float, "L_eval": float, "mode": str}`.
        `L_raw` uses the *current* params; `L_eval` uses the *validation model*
        (EMA params if `eval_with_ema=True`). Note: the Lipschitz bound is
        **data-independent**; “eval” here refers only to which parameters are used.

    Returns
    -------
    Tuple
        Always returns **four** values first:
          1. `best_model` : Any
             Model at the **best validation loss** epoch (raw params).
          2. `best_state` : Any
             State at that epoch (e.g., BatchNorm buffers).
          3. `train_losses` : List[float]
             Per-epoch average training loss.
          4. `val_losses` : List[float]
             Per-epoch average validation loss.

        If `return_penalty_history=True` (and `return_ema=False`):
          5. `penalty_history` : List[float]

        If `return_ema=True` (and `use_ema=True`) but `return_penalty_history=False`:
          5. `ema_best` : EMA
             Snapshot of EMA params/decay captured at the best-val epoch.

        If **both** `return_penalty_history=True` and `return_ema=True` (and `use_ema=True`):
          5. `penalty_history` : List[float]
          6. `ema_best` : EMA

        (If `use_ema=False`, `return_ema` is ignored and no EMA object is returned.)

    Notes
    -----
    • Checkpointing:
        - When validation improves, a "best" checkpoint is saved (if `ckpt_path` is set).
        - If `checkpoint_interval` is provided, periodic checkpoints of the *current*
          (not necessarily best) weights are also saved.
    • Certified Lipschitz logging:
        - Bounds are **certified** given the chosen conv mode and sizes.
        - They are *global* operator-norm bounds; no data is used in their computation.
        - Use a small cadence (e.g., `log_global_bound_every=5`) if runtime matters.
    • EMA semantics:
        - EMA is updated after each optimizer step.
        - If `eval_with_ema=True`, validation/early-stopping use EMA parameters;
          nevertheless, `best_model` returned is the raw-weights snapshot at the best epoch.
          The EMA snapshot at that epoch is returned separately as `ema_best` (if requested).

    Examples
    --------
    >>> CKPT = "checkpoints/model-e{epoch:03d}.ckpt"
    >>> best_model, best_state, tr, va, pen_hist, ema = train(
    ...     model, state, opt_state, optimizer, loss_fn,
    ...     X_tr, y_tr, X_va, y_va,
    ...     batch_size=64, num_epochs=50, patience=10, key=jr.PRNGKey(0),
    ...     lambda_specnorm=1e-4, conv_tn_iters=8,
    ...     use_ema=True, eval_with_ema=True, return_ema=True,
    ...     return_penalty_history=True, ckpt_path=CKPT, checkpoint_interval=5,
    ...     log_global_bound_every=5, bound_conv_mode="min_tn_circ_embed",
    ...     bound_input_shape=(224, 224),
    ... )

    See Also
    --------
    make_lipschitz_upper_fn : Builds the certified global Lipschitz callable.
    global_spectral_norm_penalty : Sum of per-layer operator norms (used in training penalty).
    """
    if checkpoint_interval is not None:
        assert checkpoint_interval > 0, "`checkpoint_interval` must be > 0"

    rng, eval_rng = jr.split(key)
    train_losses, val_losses, penalty_history = [], [], []

    # --- Build (optional) epoch-level Lipschitz bound fn BEFORE the loop ---
    if log_global_bound_every is not None:
        lip_fn = make_lipschitz_upper_fn(
            conv_mode=bound_conv_mode,
            conv_tn_iters=bound_tn_iters,
            conv_fft_shape=bound_fft_shape,
            conv_input_shape=bound_input_shape,
        )
    # optional local buffer (kept private; use bound_recorder to export)
    _bound_history: List[Dict[str, Any]] = []

    # -------------------- EMA init -------------------- #
    ema: Optional[EMA] = init_ema(model, decay=ema_decay) if use_ema else None
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
                lambda_specnorm=lambda_specnorm,
                lambda_sob_jac=lambda_sob_jac,
                lambda_sob_kernel=lambda_sob_kernel,
                lambda_liplog=lambda_liplog,
                sob_jac_apply=sob_jac_apply,
                sob_jac_samples=sob_jac_samples,
                conv_tn_iters=conv_tn_iters,
                conv_fft_shape=conv_fft_shape,
            )
            if specpen_recorder is not None:
                m = specpen_metrics_from_opt_state(opt_state)
                if m and m.get("lip_updated", False):
                    specpen_recorder(m)

            # -------------------- EMA update (outside jit) -------------------- #
            if use_ema and ema is not None:
                ema = update_ema(ema, model)

            epoch_train_loss += float(batch_loss) * xb.shape[0]
            n_train += xb.shape[0]

        train_losses.append(epoch_train_loss / max(1, n_train))

        # -------------------- Validate (optionally with EMA) -------------------- #
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

        # Track (raw) spectral penalty to monitor regularization magnitude
        penalty_history.append(float(global_spectral_penalty(model)))

        # --- Progress print ---
        if epoch % max(1, num_epochs // 10) == 0 or epoch == num_epochs:
            print(
                f"[Epoch {epoch:3d}/{num_epochs}] "
                f"Train={train_losses[-1]:.4f} | Val={val_losses[-1]:.4f}"
            )

        # --- OPTIONAL: certified global Lipschitz UB every k epochs (after val) ---
        if (log_global_bound_every is not None) and (
            (epoch % max(1, log_global_bound_every) == 0) or (epoch == num_epochs)
        ):
            L_raw = float(lip_fn(model))
            L_eval = float(lip_fn(val_model))  # EMA if eval_with_ema=True
            rec = {
                "epoch": int(epoch),
                "L_raw": L_raw,
                "L_eval": L_eval,
                "mode": bound_conv_mode,
            }
            _bound_history.append(rec)
            if bound_recorder is not None:
                bound_recorder(rec)
            print(
                f"    [Lipschitz UB] raw={L_raw:.6g}  eval={L_eval:.6g}  ({bound_conv_mode})"
            )

        # --- Early stopping bookkeeping ---
        improved = val_losses[-1] < best_val
        if improved:
            best_val = val_losses[-1]
            best_model, best_state = model, state
            patience_counter = 0

            if use_ema and ema is not None:
                best_ema_params = ema.params
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
    model: Any,
    state: Any,
    opt_state: Any,
    optimizer: Any,
    loss_fn: Callable,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_val: jnp.ndarray,
    y_val: jnp.ndarray,
    batch_size: int,
    num_epochs: int,
    key: jr.PRNGKey,
    *,
    augment_fn: Optional[Callable[[jr.PRNGKey, jnp.ndarray], jnp.ndarray]] = None,
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    lambda_specnorm: float = 0.0,
    lambda_sob_jac: float = 0.0,
    lambda_sob_kernel: float = 0.0,
    lambda_liplog: float = 0.0,
    sob_jac_apply: Optional[
        Callable[[Any, Any, jnp.ndarray, jr.PRNGKey], jnp.ndarray]
    ] = None,
    sob_jac_samples: int = 1,
    conv_tn_iters: int = 8,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    ckpt_path: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    return_penalty_history: bool = False,
    # EMA
    use_ema: bool = False,
    ema_decay: float = 0.999,
    eval_with_ema: bool = True,
    return_ema: bool = False,
    # spectral-penalty recorder passthrough
    specpen_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
    # ---- NEW: global bound logging passthrough ----
    log_global_bound_every: Optional[int] = None,
    bound_conv_mode: Literal[
        "tn", "circular_fft", "min_tn_circ_embed", "circ_plus_lr"
    ] = "tn",
    bound_tn_iters: int = 8,
    bound_fft_shape: Optional[Tuple[int, int]] = None,
    bound_input_shape: Optional[Tuple[int, int]] = None,
    bound_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Union[
    Tuple[Any, Any, List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], "EMA"],
    Tuple[Any, Any, List[float], List[float], List[float], "EMA"],
]:
    """Combine train+val, disable early stop, but keep checkpoints, EMA, and bound logging."""
    X_full = jnp.concatenate([X_train, X_val], axis=0)
    y_full = jnp.concatenate([y_train, y_val], axis=0)
    # minimal dummy val to reuse the same training loop
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
        patience=num_epochs,  # effectively disables early stop
        key=key,
        augment_fn=augment_fn,
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
        ckpt_path=ckpt_path,
        checkpoint_interval=checkpoint_interval,
        return_penalty_history=return_penalty_history,
        use_ema=use_ema,
        ema_decay=ema_decay,
        eval_with_ema=eval_with_ema,
        return_ema=return_ema,
        specpen_recorder=specpen_recorder,
        # ---- pass-through of bound logging knobs ----
        log_global_bound_every=log_global_bound_every,
        bound_conv_mode=bound_conv_mode,
        bound_tn_iters=bound_tn_iters,
        bound_fft_shape=bound_fft_shape,
        bound_input_shape=bound_input_shape,
        bound_recorder=bound_recorder,
    )


def train_sam(
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
    lambda_specnorm: float = 0.0,
    lambda_sob_jac: float = 0.0,
    lambda_sob_kernel: float = 0.0,
    lambda_liplog: float = 0.0,
    sob_jac_apply: Optional[
        Callable[[Any, Any, jnp.ndarray, jr.PRNGKey], jnp.ndarray]
    ] = None,
    sob_jac_samples: int = 1,
    conv_tn_iters: int = 8,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
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
    freeze_norm_on_perturbed: bool = True,
    # logging hooks
    specpen_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
    # ---- NEW: global bound logging knobs ----
    log_global_bound_every: Optional[int] = None,  # e.g. 5 → log every 5 epochs
    bound_conv_mode: Literal[
        "tn", "circular_fft", "min_tn_circ_embed", "circ_plus_lr"
    ] = "tn",
    bound_tn_iters: int = 8,
    bound_fft_shape: Optional[Tuple[int, int]] = None,
    bound_input_shape: Optional[Tuple[int, int]] = None,
    bound_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Union[
    Tuple[Any, Any, List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], "EMA"],
    Tuple[Any, Any, List[float], List[float], List[float], "EMA"],
]:
    """
    SAM/ASAM trainer with optional certified global Lipschitz logging.
    - Doubles forward/backward cost per step (two passes).
    - By default refuses to run if BatchNorm is present (unless you freeze BN).
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

    # --- (optional) Lipschitz bound fn ---
    lip_fn: Optional[Callable[[Any], float]] = None
    if log_global_bound_every is not None:
        lip_fn = make_lipschitz_upper_fn(
            conv_mode=bound_conv_mode,
            conv_tn_iters=bound_tn_iters,
            conv_fft_shape=bound_fft_shape,
            conv_input_shape=bound_input_shape,
        )

    # --- EMA init ---
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
                lambda_specnorm=lambda_specnorm,
                lambda_sob_jac=lambda_sob_jac,
                lambda_sob_kernel=lambda_sob_kernel,
                lambda_liplog=lambda_liplog,
                sob_jac_apply=sob_jac_apply,
                sob_jac_samples=sob_jac_samples,
                conv_tn_iters=conv_tn_iters,
                conv_fft_shape=conv_fft_shape,
                sam_rho=sam_rho,
                sam_mode=sam_mode,
                asam_eps=asam_eps,
                freeze_norm_on_perturbed=freeze_norm_on_perturbed,
            )
            if specpen_recorder is not None:
                m = specpen_metrics_from_opt_state(opt_state)
                if m and m.get("lip_updated", False):
                    specpen_recorder(m)

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

        # track raw spectral penalty magnitude (optional)
        penalty_history.append(float(global_spectral_penalty(model)))

        # --- optional: log global Lipschitz UB every k epochs ---
        if (lip_fn is not None) and (
            (epoch % max(1, log_global_bound_every) == 0) or (epoch == num_epochs)
        ):
            L_raw = float(lip_fn(model))
            L_eval = float(lip_fn(val_model))
            if bound_recorder is not None:
                bound_recorder(
                    {
                        "epoch": float(epoch),
                        "L_raw": L_raw,
                        "L_eval": L_eval,
                        "mode": bound_conv_mode,
                    }
                )
            print(
                f"    [Lipschitz UB] raw={L_raw:.6g}  eval={L_eval:.6g}  ({bound_conv_mode})"
            )

        # --- early stopping & checkpointing ---
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
                best_ema_params = ema.params
                best_ema_decay = float(ema.decay)

            if ckpt_path is not None:
                best_file = pathlib.Path(str(ckpt_path)).with_name(
                    pathlib.Path(str(ckpt_path)).name.format(epoch=epoch)
                )
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
            ckpt_file = pathlib.Path(str(ckpt_path)).with_name(
                pathlib.Path(str(ckpt_path)).name.format(epoch=epoch)
            )
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
    model: Any,
    state: Any,
    opt_state: Any,
    optimizer: Any,
    loss_fn: Callable,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_val: jnp.ndarray,
    y_val: jnp.ndarray,
    batch_size: int,
    num_epochs: int,
    key: jr.PRNGKey,
    *,
    augment_fn: Optional[Callable[[jr.PRNGKey, jnp.ndarray], jnp.ndarray]] = None,
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    lambda_specnorm: float = 0.0,
    lambda_sob_jac: float = 0.0,
    lambda_sob_kernel: float = 0.0,
    lambda_liplog: float = 0.0,
    sob_jac_apply: Optional[
        Callable[[Any, Any, jnp.ndarray, jr.PRNGKey], jnp.ndarray]
    ] = None,
    sob_jac_samples: int = 1,
    conv_tn_iters: int = 8,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
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
    freeze_norm_on_perturbed: bool = True,
    specpen_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
    # ---- NEW: global bound logging passthrough ----
    log_global_bound_every: Optional[int] = None,
    bound_conv_mode: Literal[
        "tn", "circular_fft", "min_tn_circ_embed", "circ_plus_lr"
    ] = "tn",
    bound_tn_iters: int = 8,
    bound_fft_shape: Optional[Tuple[int, int]] = None,
    bound_input_shape: Optional[Tuple[int, int]] = None,
    bound_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Union[
    Tuple[Any, Any, List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], "EMA"],
    Tuple[Any, Any, List[float], List[float], List[float], "EMA"],
]:
    """Combine train+val, disable early stop; SAM/ASAM backend with bound logging."""
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
        lambda_specnorm=lambda_specnorm,
        lambda_sob_jac=lambda_sob_jac,
        lambda_sob_kernel=lambda_sob_kernel,
        lambda_liplog=lambda_liplog,
        sob_jac_apply=sob_jac_apply,
        sob_jac_samples=sob_jac_samples,
        conv_tn_iters=conv_tn_iters,
        conv_fft_shape=conv_fft_shape,
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
        freeze_norm_on_perturbed=freeze_norm_on_perturbed,
        specpen_recorder=specpen_recorder,
        # bound logging
        log_global_bound_every=log_global_bound_every,
        bound_conv_mode=bound_conv_mode,
        bound_tn_iters=bound_tn_iters,
        bound_fft_shape=bound_fft_shape,
        bound_input_shape=bound_input_shape,
        bound_recorder=bound_recorder,
    )


def finalize_ema_for_deploy(model, ema):
    """Return a model whose params are permanently set to EMA (raw copy is unchanged)."""
    return swap_ema_params(model, ema)


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
    brec = BoundLogger()
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
        log_global_bound_every=1,  # log every epoch (raise if slow)
        bound_conv_mode="min_tn_circ_embed",  # tight, certified
        bound_tn_iters=8,
        bound_input_shape=(28, 28),  # REQUIRED for circ-embed flavors
        bound_recorder=brec,
    )
    print("Training complete.")
    print("First few bound logs:", brec.data[:3])

    # ---------------- final tightest certified bound (paper number) -----------
    L_min_tn_circ = float(
        make_lipschitz_upper_fn(
            conv_mode="min_tn_circ_embed", conv_tn_iters=8, conv_input_shape=(28, 28)
        )(best_model)
    )
    L_circ_plus_lr = float(
        make_lipschitz_upper_fn(
            conv_mode="circ_plus_lr", conv_tn_iters=8, conv_input_shape=(28, 28)
        )(best_model)
    )
    L_final = min(L_min_tn_circ, L_circ_plus_lr)
    print(
        f"Final certified L: min(min_tn_circ_embed={L_min_tn_circ:.6g}, "
        f"circ_plus_lr={L_circ_plus_lr:.6g}) = {L_final:.6g}"
    )

    # ---------------- sanity: predictions shape ----------------
    logits = predict(best_model, best_state, jnp.array(X_val), jr.PRNGKey(0))
    print("Predictions shape:", logits.shape)

    # ---------------- plot losses + Lipschitz curve on twin axis -------------
    epochs = np.arange(1, len(tr_loss) + 1)
    # brec logs only at chosen cadence; we plot eval (EMA if enabled)
    bepochs = np.array([r["epoch"] for r in brec.data], int)
    L_eval = np.array([r["L_eval"] for r in brec.data], float)
    L_raw = np.array([r["L_raw"] for r in brec.data], float)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs, tr_loss, label="train loss")
    ax1.plot(epochs, va_loss, label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(bepochs, L_eval, linestyle="--", marker="o", label="L_eval (certified)")
    ax2.plot(bepochs, L_raw, linestyle=":", marker="x", label="L_raw (certified)")
    ax2.set_ylabel("global Lipschitz upper bound")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
