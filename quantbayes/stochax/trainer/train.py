import pathlib
import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import augmax
import dataclasses as dc
from jaxtyping import PRNGKeyArray
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
from quantbayes.stochax.utils.lip_upper import make_lipschitz_upper_fn


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


def make_lmt_multiclass_loss(
    *,
    eps: float,
    alpha: float = 1.0,
    head_weights: Sequence[float] | None = None,
    conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = "tn",
    conv_tn_iters: int = 8,
    conv_gram_iters: int = 5,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    conv_input_shape: Optional[Tuple[int, int]] = None,
    stop_grad_L: bool = True,
):
    """
    LMT for softmax CE:
      z'_y = z_y - κ,   z'_j = z_j + κ (j≠y),  κ = α·ε·L_hat(model,state)

    Notes:
      • L_hat is BN/residual/concat/pooling-aware via certified global bound.
      • For circular stride=1, use conv_mode="circular_fft" with conv_fft_shape=(H,W).
      • For zero/reflect padding on a known grid, use "circ_plus_lr" or "min_tn_circ_embed"
        with conv_input_shape=(H,W).
    """

    @eqx.filter_jit
    def loss_fn(model, state, x, y, key):
        L_hat = network_lipschitz_upper(
            model,
            state=state,
            conv_mode=conv_mode,
            conv_tn_iters=conv_tn_iters,
            conv_gram_iters=conv_gram_iters,
            conv_fft_shape=conv_fft_shape,
            conv_input_shape=conv_input_shape,
        )
        if stop_grad_L:
            L_hat = jax.lax.stop_gradient(L_hat)

        kappa = jnp.asarray(alpha * eps, jnp.float32) * jnp.clip(L_hat, 1e-12, 1e12)

        keys = jax.random.split(key, x.shape[0])
        logits_or_list, state = jax.vmap(
            model, in_axes=(0, 0, None), out_axes=(0, None), axis_name="batch"
        )(x, keys, state)
        preds = (
            logits_or_list
            if isinstance(logits_or_list, (list, tuple))
            else [logits_or_list]
        )

        if head_weights is None:
            w = [1.0 / len(preds)] * len(preds)
        else:
            w = jnp.asarray(head_weights, dtype=jnp.float32)
            w = w / w.sum()

        total = 0.0
        for logit, weight in zip(preds, w):
            K = logit.shape[-1]
            assert K > 1, "LMT multiclass expects class dimension."
            onehot = jax.nn.one_hot(y, num_classes=K, dtype=logit.dtype)
            # z' = z + κ*(1 - 2·onehot)  ≡  z_y - κ, z_j + κ (j≠y)
            logit_adj = logit + kappa[..., None] * (1.0 - 2.0 * onehot)
            per_ex = optax.softmax_cross_entropy_with_integer_labels(logit_adj, y)
            B = per_ex.shape[0]
            total = total + weight * per_ex.reshape((B, -1)).mean(axis=1).mean()

        return total, state

    return loss_fn


def make_lmt_binary_loss(
    *,
    eps: float,
    alpha: float = 1.0,
    head_weights: Sequence[float] | None = None,
    conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = "tn",
    conv_tn_iters: int = 8,
    conv_gram_iters: int = 5,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    conv_input_shape: Optional[Tuple[int, int]] = None,
    stop_grad_L: bool = True,
):
    """
    LMT for sigmoid BCE with one logit per location:
      s = 2y-1 ∈ {-1,+1},   z' = z - (2κ)·s,   κ = α·ε·L_hat(model,state)
    """

    @eqx.filter_jit
    def loss_fn(model, state, x, y, key):
        L_hat = network_lipschitz_upper(
            model,
            state=state,
            conv_mode=conv_mode,
            conv_tn_iters=conv_tn_iters,
            conv_gram_iters=conv_gram_iters,
            conv_fft_shape=conv_fft_shape,
            conv_input_shape=conv_input_shape,
        )
        if stop_grad_L:
            L_hat = jax.lax.stop_gradient(L_hat)

        kappa = jnp.asarray(alpha * eps, jnp.float32) * jnp.clip(L_hat, 1e-12, 1e12)

        keys = jax.random.split(key, x.shape[0])
        logits_or_list, state = jax.vmap(
            model, in_axes=(0, 0, None), out_axes=(0, None), axis_name="batch"
        )(x, keys, state)
        preds = (
            logits_or_list
            if isinstance(logits_or_list, (list, tuple))
            else [logits_or_list]
        )

        if head_weights is None:
            w = [1.0 / len(preds)] * len(preds)
        else:
            w = jnp.asarray(head_weights, dtype=jnp.float32)
            w = w / w.sum()

        total = 0.0
        for logit, weight in zip(preds, w):
            # allow trailing singleton channel
            if logit.ndim == y.ndim + 1 and logit.shape[-1] == 1:
                logit = logit.squeeze(-1)

            sgn = 2.0 * y.astype(logit.dtype) - 1.0  # {-1,+1}
            logit_adj = logit - (2.0 * kappa) * sgn
            per_ex = optax.sigmoid_binary_cross_entropy(logit_adj, y)
            B = per_ex.shape[0]
            total = total + weight * per_ex.reshape((B, -1)).mean(axis=1).mean()

        return total, state

    return loss_fn


def _project_linf_ball(
    x_adv: Array,
    x_orig: Array,
    eps: float,
    clip_min: float,
    clip_max: float,
) -> Array:
    """
    Project x_adv back to the intersection of:
      • L∞ ball B_eps(x_orig)
      • box [clip_min, clip_max]
    """
    x_dtype = x_orig.dtype
    eps_arr = jnp.asarray(eps, dtype=x_dtype)
    clip_min_arr = jnp.asarray(clip_min, dtype=x_dtype)
    clip_max_arr = jnp.asarray(clip_max, dtype=x_dtype)

    # First clamp to valid data range
    x_adv = jnp.clip(x_adv, clip_min_arr, clip_max_arr)

    # Then clamp the perturbation in L∞
    lower = jnp.maximum(x_orig - eps_arr, clip_min_arr)
    upper = jnp.minimum(x_orig + eps_arr, clip_max_arr)
    x_adv = jnp.clip(x_adv, lower, upper)
    return x_adv


def make_adversarial_loss(
    base_loss_fn: Callable[
        [Any, Any, Array, Array, PRNGKeyArray],
        Tuple[jnp.ndarray, Any],
    ],
    *,
    attack: Literal["fgsm", "pgd"] = "pgd",
    eps: float = 8.0 / 255.0,
    step_size: Optional[float] = None,
    num_steps: int = 7,
    random_start: bool = True,
    clean_weight: float = 0.0,
    adv_weight: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    detach_adv: bool = True,
    freeze_bn_for_attack: bool = True,
) -> Callable[
    [Any, Any, Array, Array, PRNGKeyArray],
    Tuple[jnp.ndarray, Any],
]:
    """
    Wrap an existing loss_fn(model, state, x, y, key) with FGSM/PGD adversarial training.

    Parameters
    ----------
    base_loss_fn
        Your existing loss function, e.g. `multiclass_loss` or `binary_loss`.
        Signature: (model, state, x, y, key) -> (loss_scalar, new_state)

    attack
        "fgsm" : single-step L∞ attack (Fast Gradient Sign Method).
        "pgd"  : multi-step L∞ PGD with projection and optional random start.

    eps
        Radius of the L∞ ball (in data units, e.g. ~8/255 for 0–1 images).

    step_size
        Step-size for PGD. If None, defaults to eps / max(num_steps, 1).

    num_steps
        Number of PGD steps. If attack=="fgsm", this is ignored.

    random_start
        If True (PGD only), start from x + U(-eps, eps) before iterating.

    clean_weight, adv_weight
        Weights for natural vs adversarial loss.
        Internally normalized, so (0.0, 1.0) = adv-only, (1.0, 1.0) = 50/50.

    clip_min, clip_max
        Data range used to clamp adversarial examples.

    detach_adv
        If True, stop_gradient on x_adv before computing the outer loss.
        This avoids double backprop (no grad through the attack itself).

    freeze_bn_for_attack
        If True, attack is computed with `eqx.nn.inference_mode(model)`, so BN
        and dropout are “frozen” during PGD/FGSM. BN stats are updated only
        from the clean forward.

    Returns
    -------
    adv_loss_fn
        A new loss_fn with the same signature, suitable for passing into `train`.
    """
    if step_size is None:
        step_size = eps / max(num_steps, 1)

    if num_steps < 1 and attack == "pgd":
        raise ValueError("PGD requires num_steps >= 1")

    cw = float(clean_weight)
    aw = float(adv_weight)
    if cw < 0.0 or aw < 0.0:
        raise ValueError("clean_weight and adv_weight must be non-negative")
    if cw + aw == 0.0:
        raise ValueError("clean_weight + adv_weight must be > 0")
    norm = cw + aw
    cw /= norm
    aw /= norm

    step_size_f = float(step_size)
    eps_f = float(eps)

    def adv_loss_fn(
        model: Any,
        state: Any,
        x: Array,
        y: Array,
        key: PRNGKeyArray,
    ) -> Tuple[jnp.ndarray, Any]:
        """
        # PGD-7 with 8/255 radius, 2/255 step, 50% clean / 50% adv loss
        adv_loss = make_adversarial_loss(
            multiclass_loss,
            attack="pgd",
            eps=8.0 / 255.0,
            step_size=2.0 / 255.0,
            num_steps=7,
            random_start=True,
            clean_weight=1.0,
            adv_weight=1.0,   # 50/50 once normalized
            clip_min=0.0,
            clip_max=1.0,
            detach_adv=True,
            freeze_bn_for_attack=True,
        )

        best_model, best_state, tr_loss, va_loss = train(
            model,
            state,
            opt_state,
            optimizer,
            adv_loss,              # <<< just swap this in
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=batch_size,
            num_epochs=num_epochs,
            patience=patience,
            key=train_key,
            augment_fn=augment_fn,
            # all your usual penalties and Lipschitz logging still work
            lambda_spec=lambda_spec,
            lambda_frob=lambda_frob,
            lambda_specnorm=lambda_specnorm,
            lambda_sob_jac=lambda_sob_jac,
            lambda_sob_kernel=lambda_sob_kernel,
            lambda_liplog=lambda_liplog,
            sob_jac_apply=sob_jac_apply,
            sob_jac_samples=sob_jac_samples,
            log_global_bound_every=log_global_bound_every,
            bound_conv_mode="min_tn_circ_embed",
            bound_tn_iters=8,
            bound_input_shape=(H, W),
            bound_recorder=brec,
            ckpt_path=CKPT,
            checkpoint_interval=1,
        )

        #FGSM training (fast adversarial)
        adv_loss_fgsm = make_adversarial_loss(
            multiclass_loss,
            attack="fgsm",
            eps=8.0 / 255.0,
            clean_weight=1.0,
            adv_weight=1.0,
        )
        # then call train(...) with loss_fn=adv_loss_fgsm
        """
        # 1) Model to use inside the attack (optionally in inference mode)
        attack_model = (
            eqx.nn.inference_mode(model, value=True) if freeze_bn_for_attack else model
        )

        # Split RNG: attack, clean, adversarial
        key_attack, key_clean, key_adv = jr.split(key, 3)

        # Helper: loss as a function of x, for gradient wrt input
        def loss_wrt_x(x_in: Array, k: PRNGKeyArray) -> jnp.ndarray:
            loss, _ = base_loss_fn(attack_model, state, x_in, y, k)
            return loss

        # 2) Build adversarial examples (L∞-FGSM or L∞-PGD)
        if eps_f <= 0.0:
            # No perturbation requested → pure clean loss
            x_adv = x
        else:
            if attack == "fgsm":
                # Single-step FGSM
                grad_x = jax.grad(lambda z: loss_wrt_x(z, key_attack))(x)
                step = eps_f * jnp.sign(grad_x)
                x_adv = _project_linf_ball(
                    x + step,
                    x_orig=x,
                    eps=eps_f,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
            elif attack == "pgd":
                # Random start in L∞ ball (optional)
                if random_start:
                    noise = jr.uniform(
                        key_attack,
                        shape=x.shape,
                        minval=-eps_f,
                        maxval=eps_f,
                        dtype=x.dtype,
                    )
                    x_adv0 = _project_linf_ball(
                        x + noise,
                        x_orig=x,
                        eps=eps_f,
                        clip_min=clip_min,
                        clip_max=clip_max,
                    )
                else:
                    x_adv0 = x

                step_size_arr = jnp.asarray(step_size_f, dtype=x.dtype)

                def pgd_body(i, x_adv_curr):
                    # Fold in i so each step gets a different RNG without
                    # threading the key through the loop state.
                    step_key = jr.fold_in(key_attack, int(i))

                    grad_x = jax.grad(lambda z: loss_wrt_x(z, step_key))(x_adv_curr)
                    x_adv_next = x_adv_curr + step_size_arr * jnp.sign(grad_x)
                    x_adv_next = _project_linf_ball(
                        x_adv_next,
                        x_orig=x,
                        eps=eps_f,
                        clip_min=clip_min,
                        clip_max=clip_max,
                    )
                    return x_adv_next

                x_adv = jax.lax.fori_loop(0, num_steps, pgd_body, x_adv0)
            else:
                raise ValueError(
                    f"Unknown attack kind '{attack}' (expected 'fgsm' or 'pgd')."
                )

            if detach_adv:
                # Critical: prevent gradients w.r.t. params from flowing
                # through the PGD/FGSM construction graph.
                x_adv = jax.lax.stop_gradient(x_adv)

        # 3) Compute clean + adversarial loss using the *training* model
        #    (BN stats updated on clean data only; adv_loss uses same state).
        clean_loss, new_state = base_loss_fn(model, state, x, y, key_clean)

        if eps_f <= 0.0 or aw == 0.0:
            total_loss = clean_loss
        else:
            adv_loss, _ = base_loss_fn(model, state, x_adv, y, key_adv)
            total_loss = cw * clean_loss + aw * adv_loss

        return total_loss, new_state

    return adv_loss_fn


def _generate_adversarial_batch(
    model: Any,
    state: Any,
    x: Array,
    y: Array,
    key: PRNGKeyArray,
    base_loss_fn: Callable[
        [Any, Any, Array, Array, PRNGKeyArray], Tuple[jnp.ndarray, Any]
    ],
    *,
    attack: Literal["fgsm", "pgd"] = "pgd",
    eps: float = 8.0 / 255.0,
    step_size: Optional[float] = None,
    num_steps: int = 7,
    random_start: bool = True,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Array:
    """
    Build adversarial examples for evaluation only.

    - model should already be in inference_mode if you want BN/Dropout frozen.
    - state is passed into base_loss_fn but we *ignore* any returned new_state.
    """
    eps_f = float(eps)
    if eps_f <= 0.0:
        return x

    if attack == "pgd" and num_steps < 1:
        raise ValueError("PGD requires num_steps >= 1.")

    if step_size is None:
        step_size = eps_f / max(num_steps, 1)
    step_size_f = float(step_size)

    def loss_wrt_x(x_in: Array, k: PRNGKeyArray) -> jnp.ndarray:
        # BN/Dropout behaviour is controlled by model's mode.
        loss, _ = base_loss_fn(model, state, x_in, y, k)
        return loss

    if attack == "fgsm":
        # Single-step FGSM: x_adv = x + eps * sign(∂_x L)
        grad_x = jax.grad(lambda z: loss_wrt_x(z, key))(x)
        step = eps_f * jnp.sign(grad_x)
        x_adv = _project_linf_ball(
            x + step,
            x_orig=x,
            eps=eps_f,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        return x_adv

    elif attack == "pgd":
        # Random start in L∞ ball (optional)
        if random_start:
            noise = jr.uniform(
                key,
                shape=x.shape,
                minval=-eps_f,
                maxval=eps_f,
                dtype=x.dtype,
            )
            x_adv0 = _project_linf_ball(
                x + noise,
                x_orig=x,
                eps=eps_f,
                clip_min=clip_min,
                clip_max=clip_max,
            )
        else:
            x_adv0 = x

        step_size_arr = jnp.asarray(step_size_f, dtype=x.dtype)

        def pgd_body(i, x_adv_curr):
            step_key = jr.fold_in(key, int(i))
            grad_x = jax.grad(lambda z: loss_wrt_x(z, step_key))(x_adv_curr)
            x_adv_next = x_adv_curr + step_size_arr * jnp.sign(grad_x)
            x_adv_next = _project_linf_ball(
                x_adv_next,
                x_orig=x,
                eps=eps_f,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            return x_adv_next

        x_adv = jax.lax.fori_loop(0, num_steps, pgd_body, x_adv0)
        return x_adv

    else:
        raise ValueError(f"Unknown attack kind '{attack}' (expected 'fgsm' or 'pgd').")


def evaluate_clean_loss(
    model: Any,
    state: Any,
    X: Array,
    y: Array,
    batch_size: int,
    key: PRNGKeyArray,
    loss_fn: Callable[[Any, Any, Array, Array, PRNGKeyArray], Tuple[jnp.ndarray, Any]],
) -> float:
    """
    Compute mean loss on clean data (no adversary), with BN/Dropout frozen.
    """
    model_eval = eqx.nn.inference_mode(model, value=True)
    rng = key
    total_loss = 0.0
    n_total = 0

    for xb, yb in data_loader(
        X, y, batch_size=batch_size, shuffle=False, key=rng, augment_fn=None
    ):
        rng, subkey = jr.split(rng)
        loss, _ = loss_fn(model_eval, state, xb, yb, subkey)
        total_loss += float(loss) * xb.shape[0]
        n_total += xb.shape[0]

    return total_loss / max(1, n_total)


def evaluate_adversarial_loss(
    model: Any,
    state: Any,
    X: Array,
    y: Array,
    batch_size: int,
    key: PRNGKeyArray,
    base_loss_fn: Callable[
        [Any, Any, Array, Array, PRNGKeyArray], Tuple[jnp.ndarray, Any]
    ],
    *,
    attack: Literal["fgsm", "pgd"] = "pgd",
    eps: float = 8.0 / 255.0,
    step_size: Optional[float] = None,
    num_steps: int = 7,
    random_start: bool = True,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    freeze_bn_for_attack: bool = True,
) -> float:
    """
    Robust evaluation:
      - craft adversarial examples with FGSM/PGD
      - compute mean loss on these adversarial inputs.

    No BN state updates; model is kept in inference_mode if freeze_bn_for_attack=True.
    """
    rng = key
    total_loss = 0.0
    n_total = 0

    for xb, yb in data_loader(
        X, y, batch_size=batch_size, shuffle=False, key=rng, augment_fn=None
    ):
        rng, attack_key, eval_key = jr.split(rng, 3)

        attack_model = (
            eqx.nn.inference_mode(model, value=True) if freeze_bn_for_attack else model
        )

        x_adv = _generate_adversarial_batch(
            model=attack_model,
            state=state,
            x=xb,
            y=yb,
            key=attack_key,
            base_loss_fn=base_loss_fn,
            attack=attack,
            eps=eps,
            step_size=step_size,
            num_steps=num_steps,
            random_start=random_start,
            clip_min=clip_min,
            clip_max=clip_max,
        )

        # Evaluate loss on adversarial inputs (again with frozen semantics).
        loss_adv, _ = base_loss_fn(attack_model, state, x_adv, yb, eval_key)
        total_loss += float(loss_adv) * xb.shape[0]
        n_total += xb.shape[0]

    return total_loss / max(1, n_total)


def evaluate_adversarial_binary_acc(
    model: Any,
    state: Any,
    X: Array,
    y: Array,
    batch_size: int,
    key: PRNGKeyArray,
    base_loss_fn: Callable[
        [Any, Any, Array, Array, PRNGKeyArray], Tuple[jnp.ndarray, Any]
    ],
    *,
    attack: Literal["fgsm", "pgd"] = "pgd",
    eps: float = 8.0 / 255.0,
    step_size: Optional[float] = None,
    num_steps: int = 7,
    random_start: bool = True,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Tuple[float, float]:
    """
    Robust evaluation for binary classification (1-logit head):

    Returns:
        (robust_loss, robust_accuracy)
    """
    rng = key
    total_loss = 0.0
    total_correct = 0
    n_total = 0

    for xb, yb in data_loader(
        X, y, batch_size=batch_size, shuffle=False, key=rng, augment_fn=None
    ):
        rng, attack_key, eval_key, pred_key = jr.split(rng, 4)

        # Freeze semantics for attack and loss/accuracy
        model_eval = eqx.nn.inference_mode(model, value=True)

        x_adv = _generate_adversarial_batch(
            model=model_eval,
            state=state,
            x=xb,
            y=yb,
            key=attack_key,
            base_loss_fn=base_loss_fn,
            attack=attack,
            eps=eps,
            step_size=step_size,
            num_steps=num_steps,
            random_start=random_start,
            clip_min=clip_min,
            clip_max=clip_max,
        )

        # Robust loss
        loss_adv, _ = base_loss_fn(model_eval, state, x_adv, yb, eval_key)
        total_loss += float(loss_adv) * xb.shape[0]

        # Robust predictions via your existing predict()
        logits_adv = predict(model, state, x_adv, pred_key)  # raw logits
        # Assume shape [B] or [B,1] for binary
        if logits_adv.ndim == 2 and logits_adv.shape[-1] == 1:
            logits_flat = logits_adv.squeeze(-1)
        else:
            logits_flat = logits_adv

        preds = (logits_flat > 0.0).astype(yb.dtype)
        correct = jnp.sum(preds == yb)
        total_correct += int(correct)
        n_total += xb.shape[0]

    robust_loss = total_loss / max(1, n_total)
    robust_acc = total_correct / max(1, n_total)
    return robust_loss, robust_acc


_NORM_TYPES: Tuple[type, ...] = tuple(
    t
    for t in (
        getattr(eqx.nn, "BatchNorm", None),
        getattr(eqx.nn, "LayerNorm", None),
        getattr(eqx.nn, "GroupNorm", None),
    )
    if t is not None
)


def make_asam_mask_from_model(model: Any) -> Any:
    """
    Returns a PyTree of scalars (0./1.) matching eqx.filter(model, eqx.is_inexact_array).
    Rule: exclude any parameter whose field name contains 'bias' (scalar or vector),
    and exclude all params inside norm layers; include everything else.
    """
    # 1) Get params tree + treedef
    params = eqx.filter(model, eqx.is_inexact_array)
    leaves, treedef = jax.tree_util.tree_flatten(params)

    # 2) Collect metadata (field_name, in_norm_context) in *the same flatten order*
    meta: List[Tuple[str, bool]] = []

    def walk(node: Any, in_norm: bool = False):
        # Recurse in the same order JAX flattens dataclasses/lists/tuples/dicts.
        if isinstance(node, eqx.Module):
            is_norm = in_norm or isinstance(node, _NORM_TYPES)
            # dataclass fields keep a deterministic order
            for f in dc.fields(node):
                v = getattr(node, f.name)
                if eqx.is_inexact_array(v):
                    meta.append((f.name, is_norm))
                else:
                    # containers or submodules
                    if isinstance(v, (eqx.Module, list, tuple, dict)):
                        walk(v, is_norm)
                    # else: static leaf → no params here
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v, in_norm)
        elif isinstance(node, dict):
            # JAX flattens dicts in sorted key order
            for k in sorted(node.keys()):
                walk(node[k], in_norm)

    walk(model, False)
    assert len(meta) == len(leaves), (
        f"ASAM mask metadata ({len(meta)}) != param leaves ({len(leaves)}). "
        "If you have custom containers, ensure traversal order matches flatten."
    )

    # 3) Build mask leaves
    mask_leaves = []
    for name, in_norm in meta:
        exclude = in_norm or ("bias" in name.lower())  # catches scalar and vector_bias
        mask_leaves.append(jnp.array(0.0 if exclude else 1.0))

    # 4) Unflatten back to param-shaped PyTree
    return treedef.unflatten(mask_leaves)


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
    key: PRNGKeyArray,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    lambda_spec: float = 0.0,  # legacy: your __spectral_penalty__ hook
    lambda_frob: float = 0.0,  # Frobenius/L2 on weights
    # --- NEW knobs ---
    lambda_specnorm: float = 0.0,  # Σ per-layer σ (configurable conv bound)
    lambda_sob_jac: float = 0.0,  # Jacobian Sobolev (data-dependent)
    lambda_sob_kernel: float = 0.0,  # kernel smoothness
    lambda_liplog: float = 0.0,  # τ·log Lip_upper (very small τ)
    sob_jac_apply: Optional[
        Callable[[Any, Any, jnp.ndarray, PRNGKeyArray], jnp.ndarray]
    ] = None,
    sob_jac_samples: int = 1,
    # ---- Σσ penalty config ----
    specnorm_conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = "tn",
    specnorm_conv_tn_iters: int = 8,
    specnorm_conv_gram_iters: int = 5,
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
    lip_conv_tn_iters: int = 8,
    lip_conv_gram_iters: int = 5,
    lip_conv_fft_shape: Optional[Tuple[int, int]] = None,
    lip_conv_input_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Any, Any, Any, jnp.ndarray]:

    def total_loss_fn(m, s, xb, yb, k):
        base, new_s = loss_fn(m, s, xb, yb, k)

        # legacy spectral hook
        pen_spec = (
            lambda_spec * global_spectral_penalty(m) if lambda_spec > 0.0 else 0.0
        )

        # Frobenius (L2) on weights (skips biases)
        pen_frob = (
            lambda_frob * global_frobenius_penalty(m) if lambda_frob > 0.0 else 0.0
        )

        # Σ per-layer operator norms (dense exact; conv per selected bound)
        pen_specnorm = (
            lambda_specnorm
            * global_spectral_norm_penalty(
                m,
                conv_mode=specnorm_conv_mode,
                conv_tn_iters=specnorm_conv_tn_iters,
                conv_gram_iters=specnorm_conv_gram_iters,
                conv_fft_shape=specnorm_conv_fft_shape,
                conv_input_shape=specnorm_conv_input_shape,
            )
            if lambda_specnorm > 0.0
            else 0.0
        )

        # kernel smoothness (∑ ||∇K||²)
        pen_sob_k = (
            lambda_sob_kernel * sobolev_kernel_smoothness(m)
            if lambda_sob_kernel > 0.0
            else 0.0
        )

        # function-space Sobolev (Jacobian) via Hutchinson
        pen_sob_j = (
            lambda_sob_jac
            * sobolev_jacobian_penalty(
                m, s, xb, k, sob_jac_apply, num_samples=sob_jac_samples
            )
            if (lambda_sob_jac > 0.0 and sob_jac_apply is not None)
            else 0.0
        )

        # gentle Lip product penalty (uses BN running stats via `state=s`)
        pen_lip = (
            lip_product_penalty(
                m,
                state=s,
                tau=lambda_liplog,
                conv_mode=lip_conv_mode,
                conv_tn_iters=lip_conv_tn_iters,
                conv_gram_iters=lip_conv_gram_iters,
                conv_fft_shape=lip_conv_fft_shape,
                conv_input_shape=lip_conv_input_shape,
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
    model: Any,
    state: Any,
    opt_state: Any,
    x: Array,
    y: Array,
    key: jr.key,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    # penalties ...
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
    # Σσ config ...
    specnorm_conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = "tn",
    specnorm_conv_tn_iters: int = 8,
    specnorm_conv_gram_iters: int = 5,
    specnorm_conv_fft_shape: Optional[Tuple[int, int]] = None,
    specnorm_conv_input_shape: Optional[Tuple[int, int]] = None,
    # Lip config ...
    lip_conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = "tn",
    lip_conv_tn_iters: int = 8,
    lip_conv_gram_iters: int = 5,
    lip_conv_fft_shape: Optional[Tuple[int, int]] = None,
    lip_conv_input_shape: Optional[Tuple[int, int]] = None,
    # SAM/ASAM knobs
    sam_rho: float = 0.05,
    sam_mode: Literal["sam", "asam"] = "sam",
    asam_eps: float = 1e-12,
    freeze_norm_on_perturbed: bool = True,
    # NEW: epsilon mask (1.0 = perturb, 0.0 = skip)
    asam_mask: Optional[Any] = None,
) -> Tuple[Any, Any, Any, jnp.ndarray]:

    # 1) Base (data-only) loss → grads for epsilon
    def base_loss_fn(m, s, xb, yb, k):
        base, new_s = loss_fn(m, s, xb, yb, k)
        return base, new_s

    (base1, state_after), grads_base = eqx.filter_value_and_grad(
        base_loss_fn, has_aux=True
    )(model, state, x, y, key)

    params = eqx.filter(model, eqx.is_inexact_array)
    g_params = eqx.filter(grads_base, eqx.is_inexact_array)

    if asam_mask is None:
        # Default: perturb everything (you can pass a precomputed mask instead)
        asam_mask = jax.tree_map(lambda p: jnp.array(1.0), params)

    # ASAM/SAM scaling
    def _scale_no_mask(g, p):
        g_asam = g * (jnp.abs(p) + asam_eps) if sam_mode == "asam" else g
        return g_asam

    scaled = jax.tree_map(_scale_no_mask, g_params, params)
    # Apply mask AFTER scaling; guard non-array leaves
    scaled = jax.tree_map(
        lambda s, m: s * m if eqx.is_inexact_array(s) else s, scaled, asam_mask
    )

    gnorm = optax.global_norm(scaled)
    epsilon = jax.tree_map(lambda s: s * (sam_rho / (gnorm + 1e-12)), scaled)
    epsilon = jax.tree_map(jax.lax.stop_gradient, epsilon)

    # 2) Total loss at perturbed params (penalties included)
    def total_loss_fn(m, s, xb, yb, k):
        base, _ = loss_fn(m, s, xb, yb, k)
        pen_spec = (
            lambda_spec * global_spectral_penalty(m) if lambda_spec > 0.0 else 0.0
        )
        pen_frob = (
            lambda_frob * global_frobenius_penalty(m) if lambda_frob > 0.0 else 0.0
        )
        pen_specnorm = (
            lambda_specnorm
            * global_spectral_norm_penalty(
                m,
                conv_mode=specnorm_conv_mode,
                conv_tn_iters=specnorm_conv_tn_iters,
                conv_gram_iters=specnorm_conv_gram_iters,
                conv_fft_shape=specnorm_conv_fft_shape,
                conv_input_shape=specnorm_conv_input_shape,
            )
            if lambda_specnorm > 0.0
            else 0.0
        )
        pen_sob_k = (
            lambda_sob_kernel * sobolev_kernel_smoothness(m)
            if lambda_sob_kernel > 0.0
            else 0.0
        )
        pen_sob_j = (
            lambda_sob_jac
            * sobolev_jacobian_penalty(
                m, s, xb, k, sob_jac_apply, num_samples=sob_jac_samples
            )
            if (lambda_sob_jac > 0.0 and sob_jac_apply is not None)
            else 0.0
        )
        pen_lip = (
            lip_product_penalty(
                m,
                state=s,
                tau=lambda_liplog,
                conv_mode=lip_conv_mode,
                conv_tn_iters=lip_conv_tn_iters,
                conv_gram_iters=lip_conv_gram_iters,
                conv_fft_shape=lip_conv_fft_shape,
                conv_input_shape=lip_conv_input_shape,
            )
            if lambda_liplog > 0.0
            else 0.0
        )
        return (
            base + pen_spec + pen_frob + pen_specnorm + pen_sob_k + pen_sob_j + pen_lip
        )

    model_pert = eqx.apply_updates(model, epsilon)
    if freeze_norm_on_perturbed:
        model_pert = eqx.nn.inference_mode(model_pert, value=True)

    state_for_pert = state_after if freeze_norm_on_perturbed else state

    loss2, grads2 = eqx.filter_value_and_grad(total_loss_fn)(
        model_pert, state_for_pert, x, y, key
    )
    updates, opt_state = optimizer.update(grads2, opt_state, params=params)
    model = eqx.apply_updates(model, updates)
    return model, state_after, opt_state, loss2


def _contains_batchnorm(model: Any) -> bool:
    """Detect eqx.nn.BatchNorm anywhere in the tree."""
    found = False

    def visit(x):
        nonlocal found
        if found:
            return
        if hasattr(eqx.nn, "BatchNorm") and isinstance(x, eqx.nn.BatchNorm):
            found = True
            return
        if isinstance(x, eqx.Module):
            for v in vars(x).values():
                visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
        elif isinstance(x, dict):
            for v in x.values():
                visit(v)

    visit(model)
    return found


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
    # -------------------- NEW penalty knobs -------------------- #
    lambda_specnorm: float = 0.0,
    lambda_sob_jac: float = 0.0,
    lambda_sob_kernel: float = 0.0,
    lambda_liplog: float = 0.0,
    sob_jac_apply: Optional[
        Callable[[Any, Any, jnp.ndarray, jr.PRNGKey], jnp.ndarray]
    ] = None,
    sob_jac_samples: int = 1,
    # (back-compat) legacy conv knobs used as fallback if specific knobs are None
    conv_tn_iters: int = 8,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    # ---- Σσ regulariser config (specific) ----
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
    # ---- Lip product penalty config (specific) ----
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
    ckpt_path: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    return_penalty_history: bool = False,
    # -------------------- EMA knobs (optional) -------------------- #
    use_ema: bool = False,
    ema_decay: float = 0.999,
    eval_with_ema: bool = True,
    return_ema: bool = False,
    specpen_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
    # -------------------- Global Lipschitz logging -------------------- #
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
    Tuple[Any, Any, List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], "EMA"],
    Tuple[Any, Any, List[float], List[float], List[float], "EMA"],
]:
    """
    Train a JAX/Equinox model with early stopping, optional EMA evaluation,
    checkpointing, spectral/Frobenius/Sobolev penalties, and (optionally)
    **certified global Lipschitz upper-bound logging** per epoch.

    Key points
    ----------
    • Bounds are **certified** given the chosen convolution certificate and sizes.
    • BatchNorm is handled **BN-aware**: if `state` is provided and layers are in
    `mode="ema"`, running variances are used in the certificate.
    • Logging computes **one** global bound per epoch (the one in `bound_conv_mode`).
    For multi-method comparisons, reload checkpoints and compute bounds post-hoc.

    Parameters
    ----------
    model, state : Any
        Equinox model and its mutable state (e.g., BatchNorm statistics).
    opt_state : Any
        Optimizer state returned by `optimizer.init(...)`.
    optimizer : optax.GradientTransformation
        Optax optimizer (e.g., `optax.adamw`, `optax.adan`, ...).
    loss_fn : Callable
        `(model, state, xb, yb, key) -> (loss, new_state)`.
    X_train, y_train : jnp.ndarray
    X_val, y_val : jnp.ndarray
    batch_size : int
    num_epochs : int
    patience : int
    key : jr.PRNGKey

    Keyword Args
    ------------
    augment_fn : Optional[Callable[[jr.PRNGKey, jnp.ndarray], jnp.ndarray]]
    lambda_spec, lambda_frob, lambda_specnorm, lambda_sob_jac, lambda_sob_kernel, lambda_liplog : float
        Weights for classical penalties (see library docs).
    sob_jac_apply : Optional[Callable]
    sob_jac_samples : int

    # Convolution certificate defaults used by penalties & logging
    conv_tn_iters : int, default 8
    conv_fft_shape : Optional[Tuple[int, int]]
        If set and `bound_conv_mode="circular_fft"`, uses exact circular norms for stride=1.
    bound_conv_mode : {"tn","circular_fft","circular_gram","min_tn_circ_embed","circ_plus_lr","circ_embed_opt"}, default "tn"
        Convolution certificate used for per-epoch logging.
        • "tn": resolution-free tensor bound.
        • "circular_fft": exact circular stride=1 (requires `bound_fft_shape`).
        • "circular_gram": circular Gram UB (requires `bound_fft_shape`).
        • "min_tn_circ_embed": min(TN, circulant-embed UB) (requires `bound_input_shape`).
        • "circ_plus_lr": circular + border remainder UB (requires `bound_input_shape`).
        • "circ_embed_opt": ACE bound (optional candidate grids; see `make_lipschitz_upper_fn`).

    bound_tn_iters : int
    bound_gram_iters : int
    bound_fft_shape : Optional[Tuple[int, int]]
    bound_input_shape : Optional[Tuple[int, int]]
        Required for grid-aware modes (min_tn_circ_embed, circ_plus_lr, circ_embed_opt).
    log_global_bound_every : Optional[int]
        If set (e.g., 1 or 5), compute and log a certified global bound every `k` epochs.

    ckpt_path : Optional[str]
        Template path for checkpoints; `"{epoch}"` is formatted each save.
    checkpoint_interval : Optional[int]
        Save a (non-best) checkpoint every N epochs. Must be > 0 if provided.

    EMA (Exponential Moving Average)
    --------------------------------
    use_ema : bool, default False
    ema_decay : float, default 0.999
    eval_with_ema : bool, default True
    return_ema : bool, default False

    Returns
    -------
    Tuple
        Always returns: (best_model, best_state, train_losses, val_losses).
        Optionally returns penalty history and/or an EMA snapshot depending on flags.

    Notes
    -----
    • Certified Lipschitz logging uses the **selected** conv mode only. To compare
    multiple modes (e.g., min(TN,CE) vs C+R vs ACE), reload checkpoints and
    evaluate post-hoc with `make_lipschitz_upper_fn`.
    • Always pass the BN `state` to certification calls to avoid pessimistic
    fallbacks (`1/sqrt(eps)`).
    """
    # --- back-compat defaults for the specific knobs ---
    if specnorm_conv_tn_iters is None:
        specnorm_conv_tn_iters = conv_tn_iters
    if specnorm_conv_gram_iters is None:
        specnorm_conv_gram_iters = 5
    if specnorm_conv_fft_shape is None:
        specnorm_conv_fft_shape = conv_fft_shape

    if lip_conv_tn_iters is None:
        lip_conv_tn_iters = conv_tn_iters
    if lip_conv_gram_iters is None:
        lip_conv_gram_iters = 5
    if lip_conv_fft_shape is None:
        lip_conv_fft_shape = conv_fft_shape

    if checkpoint_interval is not None:
        assert checkpoint_interval > 0, "`checkpoint_interval` must be > 0"

    rng, eval_rng = jr.split(key)
    train_losses, val_losses, penalty_history = [], [], []

    # --- (optional) epoch-level Lipschitz bound callable ---
    if log_global_bound_every is not None:

        def lip_fn(m):
            return network_lipschitz_upper(
                m,
                state=state,  # BN-aware scaling
                conv_mode=bound_conv_mode,
                conv_tn_iters=bound_tn_iters,
                conv_gram_iters=bound_gram_iters,
                conv_fft_shape=bound_fft_shape,
                conv_input_shape=bound_input_shape,
            )

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
                # Σσ regulariser config
                specnorm_conv_mode=specnorm_conv_mode,
                specnorm_conv_tn_iters=specnorm_conv_tn_iters,
                specnorm_conv_gram_iters=specnorm_conv_gram_iters,
                specnorm_conv_fft_shape=specnorm_conv_fft_shape,
                specnorm_conv_input_shape=specnorm_conv_input_shape,
                # Lip product penalty config
                lip_conv_mode=lip_conv_mode,
                lip_conv_tn_iters=lip_conv_tn_iters,
                lip_conv_gram_iters=lip_conv_gram_iters,
                lip_conv_fft_shape=lip_conv_fft_shape,
                lip_conv_input_shape=lip_conv_input_shape,
            )

            if specpen_recorder is not None:
                try:
                    m = specpen_metrics_from_opt_state(opt_state)  # optional helper
                    if m and m.get("lip_updated", False):
                        specpen_recorder(m)
                except NameError:
                    pass  # helper not available; silently skip

            # EMA update (outside jit)
            if use_ema and ema is not None:
                ema = update_ema(ema, model)

            epoch_train_loss += float(batch_loss) * xb.shape[0]
            n_train += xb.shape[0]

        train_losses.append(epoch_train_loss / max(1, n_train))

        # -------------------- Validate (in inference mode) -------------------- #
        epoch_val_loss, n_val = 0.0, 0
        val_model = (
            swap_ema_params(model, ema)
            if (use_ema and ema is not None and eval_with_ema)
            else model
        )
        val_model_eval = eqx.nn.inference_mode(val_model, value=True)
        for xb, yb in data_loader(
            X_val,
            y_val,
            batch_size=batch_size,
            shuffle=False,
            key=eval_rng,  # unused when shuffle=False; fine
            augment_fn=None,
        ):
            eval_rng, vk = jr.split(eval_rng)
            # No context manager; just use the eval copy
            val_loss, _ = loss_fn(val_model_eval, state, xb, yb, vk)
            epoch_val_loss += float(val_loss) * xb.shape[0]
            n_val += xb.shape[0]
        val_losses.append(epoch_val_loss / max(1, n_val))

        # Track (raw) spectral hook to monitor regularization magnitude
        penalty_history.append(float(global_spectral_penalty(model)))

        # --- Progress print ---
        if epoch % max(1, num_epochs // 10) == 0 or epoch == num_epochs:
            print(
                f"[Epoch {epoch:3d}/{num_epochs}] "
                f"Train={train_losses[-1]:.4f} | Val={val_losses[-1]:.4f}"
            )

        # --- OPTIONAL: certified global Lipschitz UB every k epochs ---
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

        # Periodic (non-best) checkpoints of the current weights
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
    # legacy conv knobs (fallbacks if specific ones are None)
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
    # ---- global bound logging passthrough ----
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
        # penalties
        lambda_specnorm=lambda_specnorm,
        lambda_sob_jac=lambda_sob_jac,
        lambda_sob_kernel=lambda_sob_kernel,
        lambda_liplog=lambda_liplog,
        sob_jac_apply=sob_jac_apply,
        sob_jac_samples=sob_jac_samples,
        # legacy conv knobs (fallbacks)
        conv_tn_iters=conv_tn_iters,
        conv_fft_shape=conv_fft_shape,
        # Σσ regulariser config
        specnorm_conv_mode=specnorm_conv_mode,
        specnorm_conv_tn_iters=specnorm_conv_tn_iters,
        specnorm_conv_gram_iters=specnorm_conv_gram_iters,
        specnorm_conv_fft_shape=specnorm_conv_fft_shape,
        specnorm_conv_input_shape=specnorm_conv_input_shape,
        # Lip product penalty config
        lip_conv_mode=lip_conv_mode,
        lip_conv_tn_iters=lip_conv_tn_iters,
        lip_conv_gram_iters=lip_conv_gram_iters,
        lip_conv_fft_shape=lip_conv_fft_shape,
        lip_conv_input_shape=lip_conv_input_shape,
        ckpt_path=ckpt_path,
        checkpoint_interval=checkpoint_interval,
        return_penalty_history=return_penalty_history,
        # EMA passthrough
        use_ema=use_ema,
        ema_decay=ema_decay,
        eval_with_ema=eval_with_ema,
        return_ema=return_ema,
        specpen_recorder=specpen_recorder,
        # global Lipschitz logging passthrough
        log_global_bound_every=log_global_bound_every,
        bound_conv_mode=bound_conv_mode,
        bound_tn_iters=bound_tn_iters,
        bound_gram_iters=bound_gram_iters,
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
    # base penalties
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    # extended penalties
    lambda_specnorm: float = 0.0,
    lambda_sob_jac: float = 0.0,
    lambda_sob_kernel: float = 0.0,
    lambda_liplog: float = 0.0,
    sob_jac_apply: Optional[
        Callable[[Any, Any, jnp.ndarray, jr.PRNGKey], jnp.ndarray]
    ] = None,
    sob_jac_samples: int = 1,
    # ---- Σσ penalty config ----
    specnorm_conv_mode: Literal[
        "tn", "circular_fft", "circular_gram", "min_tn_circ_embed", "circ_plus_lr"
    ] = "tn",
    specnorm_conv_tn_iters: int = 8,
    specnorm_conv_gram_iters: int = 5,
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
    lip_conv_tn_iters: int = 8,
    lip_conv_gram_iters: int = 5,
    lip_conv_fft_shape: Optional[Tuple[int, int]] = None,
    lip_conv_input_shape: Optional[Tuple[int, int]] = None,
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
    # ---- certified global Lipschitz logging ----
    log_global_bound_every: Optional[int] = None,  # e.g. 5 → compute every 5 epochs
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
    # checkpoints
    ckpt_path: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    return_penalty_history: bool = False,
) -> Union[
    Tuple[Any, Any, List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], "EMA"],
    Tuple[Any, Any, List[float], List[float], List[float], "EMA"],
]:
    """
    SAM/ASAM trainer with penalties + optional certified global Lipschitz logging.
    """
    if checkpoint_interval is not None:
        assert checkpoint_interval > 0, "`checkpoint_interval` must be > 0"

    if require_no_batchnorm and _contains_batchnorm(model):
        raise ValueError(
            "BatchNorm detected. SAM/ASAM can be unstable with BN unless stats are frozen.\n"
            "Either remove BN, set `require_no_batchnorm=False` and ensure stability, or "
            "keep `freeze_norm_on_perturbed=True` to avoid drifting stats during the SAM pass."
        )

    rng, eval_rng = jr.split(key)
    train_losses, val_losses, penalty_history = [], [], []

    # --- optional BN-aware Lipschitz bound fn ---
    if log_global_bound_every is not None:

        def lip_fn(m):
            return network_lipschitz_upper(
                m,
                state=state,  # use BN running stats if available
                conv_mode=bound_conv_mode,
                conv_tn_iters=bound_tn_iters,
                conv_gram_iters=bound_gram_iters,
                conv_fft_shape=bound_fft_shape,
                conv_input_shape=bound_input_shape,
            )

    else:
        lip_fn = None

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
            asam_mask = make_asam_mask_from_model(model)
            model, state, opt_state, batch_loss = train_step_sam(
                model=model,
                state=state,
                opt_state=opt_state,
                x=xb,
                y=yb,
                key=step_rng,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lambda_spec=lambda_spec,
                lambda_frob=lambda_frob,
                lambda_specnorm=lambda_specnorm,
                lambda_sob_jac=lambda_sob_jac,
                lambda_sob_kernel=lambda_sob_kernel,
                lambda_liplog=lambda_liplog,
                sob_jac_apply=sob_jac_apply,
                sob_jac_samples=sob_jac_samples,
                # Σσ config
                specnorm_conv_mode=specnorm_conv_mode,
                specnorm_conv_tn_iters=specnorm_conv_tn_iters,
                specnorm_conv_gram_iters=specnorm_conv_gram_iters,
                specnorm_conv_fft_shape=specnorm_conv_fft_shape,
                specnorm_conv_input_shape=specnorm_conv_input_shape,
                # Lip product config
                lip_conv_mode=lip_conv_mode,
                lip_conv_tn_iters=lip_conv_tn_iters,
                lip_conv_gram_iters=lip_conv_gram_iters,
                lip_conv_fft_shape=lip_conv_fft_shape,
                lip_conv_input_shape=lip_conv_input_shape,
                # SAM/ASAM
                sam_rho=sam_rho,
                sam_mode=sam_mode,
                asam_eps=asam_eps,
                freeze_norm_on_perturbed=freeze_norm_on_perturbed,
                asam_mask=asam_mask,
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

        # ---------------- Validate (EMA optional, in inference mode) ----------------
        epoch_val_loss, n_val = 0.0, 0
        val_model = (
            swap_ema_params(model, ema)
            if (use_ema and ema is not None and eval_with_ema)
            else model
        )
        val_model_eval = eqx.nn.inference_mode(val_model, value=True)

        for xb, yb in data_loader(
            X_val,
            y_val,
            batch_size=batch_size,
            shuffle=False,
            key=eval_rng,
            augment_fn=None,
        ):
            eval_rng, vk = jr.split(eval_rng)
            val_loss, _ = loss_fn(val_model_eval, state, xb, yb, vk)
            epoch_val_loss += float(val_loss) * xb.shape[0]
            n_val += xb.shape[0]

        val_losses.append(epoch_val_loss / max(1, n_val))

        # monitor legacy spectral penalty magnitude
        penalty_history.append(float(global_spectral_penalty(model)))

        # --- optional: certified global Lipschitz UB every k epochs ---
        if (lip_fn is not None) and (
            (epoch % max(1, log_global_bound_every) == 0) or (epoch == num_epochs)
        ):
            L_raw = float(lip_fn(model))
            L_eval = float(lip_fn(val_model_eval))
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

        # --- progress + early stopping & checkpointing ---
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
    # penalties
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
    # Σσ config
    specnorm_conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = "tn",
    specnorm_conv_tn_iters: int = 8,
    specnorm_conv_gram_iters: int = 5,
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
    lip_conv_tn_iters: int = 8,
    lip_conv_gram_iters: int = 5,
    lip_conv_fft_shape: Optional[Tuple[int, int]] = None,
    lip_conv_input_shape: Optional[Tuple[int, int]] = None,
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
    # checkpoints
    ckpt_path: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    return_penalty_history: bool = False,
) -> Union[
    Tuple[Any, Any, List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], "EMA"],
    Tuple[Any, Any, List[float], List[float], List[float], "EMA"],
]:
    """Combine train+val; disable early stop; SAM/ASAM backend with bound logging."""
    X_full = jnp.concatenate([X_train, X_val], axis=0)
    y_full = jnp.concatenate([y_train, y_val], axis=0)
    # Minimal dummy val to reuse loop shape; early stopping disabled anyway.
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
        # penalties
        lambda_spec=lambda_spec,
        lambda_frob=lambda_frob,
        lambda_specnorm=lambda_specnorm,
        lambda_sob_jac=lambda_sob_jac,
        lambda_sob_kernel=lambda_sob_kernel,
        lambda_liplog=lambda_liplog,
        sob_jac_apply=sob_jac_apply,
        sob_jac_samples=sob_jac_samples,
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
        # EMA
        use_ema=use_ema,
        ema_decay=ema_decay,
        eval_with_ema=eval_with_ema,
        return_ema=return_ema,
        # SAM/ASAM
        sam_rho=sam_rho,
        sam_mode=sam_mode,
        asam_eps=asam_eps,
        require_no_batchnorm=require_no_batchnorm,
        freeze_norm_on_perturbed=freeze_norm_on_perturbed,
        specpen_recorder=specpen_recorder,
        # Lipschitz logging
        log_global_bound_every=log_global_bound_every,
        bound_conv_mode=bound_conv_mode,
        bound_tn_iters=bound_tn_iters,
        bound_gram_iters=bound_gram_iters,
        bound_fft_shape=bound_fft_shape,
        bound_input_shape=bound_input_shape,
        bound_recorder=bound_recorder,
        # checkpoints
        ckpt_path=ckpt_path,
        checkpoint_interval=checkpoint_interval,
        return_penalty_history=return_penalty_history,
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
    key: jr.key = None,
) -> Array:
    """Vectorized single-model forward over batch. Returns raw outputs/logits."""
    k = jr.PRNGKey(0) if key is None else key
    inference_model = eqx.nn.inference_mode(model)

    def single(x, k):
        out, _ = inference_model(x, k, state)
        return _select_head(out)

    keys = jr.split(k, X.shape[0])
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
    import re
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx
    import optax
    import numpy as np
    import matplotlib.pyplot as plt

    # --- your public APIs (adjust imports if your paths differ) ---
    from quantbayes.stochax import train, predict, BoundLogger

    # If multiclass_loss isn't exported at package root, use:
    #   from quantbayes.stochax.trainer.train import multiclass_loss
    from quantbayes.stochax.trainer.train import multiclass_loss, make_augmax_augment
    from quantbayes.stochax.utils.lip_upper import make_lipschitz_upper_fn
    from quantbayes.stochax.utils.regularizers import conv_norm_bracket

    import augmax
    from augmax import InputType

    # ------------------------ data ---------------------------------
    num_epochs = 10
    batch_size = 32

    rng = np.random.RandomState(0)
    N, C, H, W, NUM_CLASSES = 2048, 1, 28, 28, 10
    X_np = rng.rand(N, C, H, W).astype("float32")  # [N, C, H, W]
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

    # ------------------------ model --------------------------------
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
            self.bn1 = eqx.nn.BatchNorm(input_size=32, axis_name="batch", mode="ema")
            self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=3, padding=1, key=k2)
            self.bn2 = eqx.nn.BatchNorm(input_size=64, axis_name="batch", mode="ema")
            self.pool1 = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = eqx.nn.Linear(64 * 7 * 7, 128, key=k3)
            self.bn3 = eqx.nn.BatchNorm(input_size=128, axis_name="batch", mode="ema")
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

    # ------------------------ training -----------------------------
    master_key = jr.PRNGKey(42)
    model_key, train_key = jr.split(master_key)

    model, state = eqx.nn.make_with_state(SimpleCNN)(model_key)
    steps_per_epoch = int(jnp.ceil(X_train.shape[0] / batch_size))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(100, steps_per_epoch)

    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=1e-5,
    )
    optimizer = optax.adan(
        learning_rate=lr_sched, b1=0.95, b2=0.99, eps=1e-8, weight_decay=1e-4
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    brec = BoundLogger()
    CKPT = "checkpoints/cnn-e{epoch:03d}.eqx"

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
        log_global_bound_every=1,  # log one method each epoch
        bound_conv_mode="min_tn_circ_embed",  # min(TN,CE) during training
        bound_tn_iters=8,
        bound_input_shape=(H, W),
        bound_recorder=brec,
        ckpt_path=CKPT,
        checkpoint_interval=1,  # save every epoch
    )
    print("Training complete.")
    print("First few bound logs:", brec.data[:3])

    # ------------------------ final (single-method) numbers -------------------
    L_min_tn_circ = float(
        make_lipschitz_upper_fn(
            conv_mode="min_tn_circ_embed",
            conv_tn_iters=8,
            conv_input_shape=(H, W),
        )(best_model, best_state)
    )
    L_circ_plus_lr = float(
        make_lipschitz_upper_fn(
            conv_mode="circ_plus_lr",
            conv_tn_iters=8,
            conv_input_shape=(H, W),
        )(best_model, best_state)
    )

    # Optional ACE candidates (treat as a tiny certified hyperparam)
    ACE_CANDS = (
        (H + 2, W + 2),
        (H + 4, W + 4),
        (1 << (H - 1).bit_length(), 1 << (W - 1).bit_length()),
    )
    L_ace = float(
        make_lipschitz_upper_fn(
            conv_mode="circ_embed_opt",
            conv_input_shape=(H, W),
            conv_circ_embed_candidates=ACE_CANDS,  # comment out to use defaults
        )(best_model, best_state)
    )
    print(
        f"Final certified UBs:\n"
        f"  min(TN,CE)  : {L_min_tn_circ:.6g}\n"
        f"  C+R         : {L_circ_plus_lr:.6g}\n"
        f"  ACE         : {L_ace:.6g}\n"
    )
    finals = {
        "min_tn_circ_embed": L_min_tn_circ,
        "circ_plus_lr": L_circ_plus_lr,
        "circ_embed_opt": L_ace,
    }
    winner = min(finals, key=finals.get)
    print(f"Tightest final UB: {winner} = {finals[winner]:.6g}")

    # ------------------------ bracket demo (layer-level) ----------------------
    K = best_model.conv1.weight
    lb, ub, iters = conv_norm_bracket(
        K,
        Cin=int(K.shape[1]),
        Hin=H,
        Win=W,
        padding=((1, 1), (1, 1)),  # match conv1 padding=1
        strides=(1, 1),
        rhs_dilation=(1, 1),
        in_hw=(H, W),
        tol=1e-2,
    )
    print(f"[conv1] bracket: LB={lb:.6g}, UB={ub:.6g}, iters={iters}")

    # ------------------------ plot loss + training-time bound -----------------
    epochs = np.arange(1, len(tr_loss) + 1)
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
    ax2.semilogy(bepochs, L_eval, "--o", label="L_eval (cert.)")
    ax2.semilogy(bepochs, L_raw, ":x", label="L_raw  (cert.)")
    ax2.set_ylabel("global Lipschitz upper bound (log)")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # ------------------------ post-hoc per-epoch comparison -------------------
    # No trainer edits: load snapshots and evaluate all modes per epoch
    ckpt_dir = Path("checkpoints")
    pat = re.compile(r"cnn-e(\d{3})\.eqx$")
    ckpts = sorted(
        [
            (int(pat.match(p.name).group(1)), p)
            for p in ckpt_dir.glob("cnn-e*.eqx")
            if pat.match(p.name)
        ],
        key=lambda z: z[0],
    )
    epochs_ckpt = np.array([e for e, _ in ckpts], dtype=int)

    # Build BN-aware callables for each mode (jitted)
    MODES = ("min_tn_circ_embed", "circ_plus_lr", "circ_embed_opt")
    L_fns = {
        m: make_lipschitz_upper_fn(
            conv_mode=m,
            conv_tn_iters=8,
            conv_gram_iters=5,
            conv_input_shape=(H, W),
            conv_circ_embed_candidates=(ACE_CANDS if m == "circ_embed_opt" else None),
        )
        for m in MODES
    }

    def load_snapshot(path, like_model, like_state):
        like = {"model": like_model, "state": like_state}
        try:
            return eqx.tree_deserialise_leaves(path, like)
        except Exception:
            ema_like = {
                "model": like_model,
                "state": like_state,
                "ema_params": eqx.filter(like_model, eqx.is_inexact_array),
                "ema_decay": 0.0,
            }
            return eqx.tree_deserialise_leaves(path, ema_like)

    vals = {m: [] for m in MODES}
    for e, p in ckpts:
        snap = load_snapshot(p, best_model, best_state)
        m_e, s_e = snap["model"], snap["state"]  # BN-aware state
        for m in MODES:
            vals[m].append(float(L_fns[m](m_e, s_e)))

    for m in MODES:
        vals[m] = np.array(vals[m], dtype=float)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(np.arange(1, len(tr_loss) + 1), tr_loss, label="train loss")
    ax1.plot(np.arange(1, len(va_loss) + 1), va_loss, label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    styles = {
        "min_tn_circ_embed": ("--o", "min(TN, CE)"),
        "circ_plus_lr": (":s", "C+R"),
        "circ_embed_opt": ("-.^", "ACE"),
    }
    for m in MODES:
        ls, lab = styles[m]
        ax2.semilogy(epochs_ckpt, vals[m], ls, label=lab)
    ax2.set_ylabel("global Lipschitz upper bound (log)")
    ax2.legend(loc="upper right")
    ax2.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()
