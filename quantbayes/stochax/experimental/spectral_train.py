# quantbayes/stochax/experiments/spectral_train.py
from __future__ import annotations
from typing import Any, Callable, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx

# Reuse your utilities
from quantbayes.stochax.trainer.train import (
    data_loader,
    train_step,
    global_spectral_penalty,
)

# Spectral modules (used only to build Stage-A mask)
from quantbayes.stochax.layers.spectral_layers import (
    SpectralConv2d,
    AdaptiveSpectralConv2d,
    SpectralDense,
    AdaptiveSpectralDense,
    SpectralCirculantLayer2d,
    AdaptiveSpectralCirculantLayer2d,
    SVDDense,
)


"""
example usage:
from quantbayes.stochax.vision_classification.models import ResNetClassifier
from spectral_surgery import spectralize_resnet_classifier
from quantbayes.stochax.experiments.spectral_train import train_two_stage
from spectral_penalty_tx import make_lambda_spec_schedule, make_soft_barrier, prod_sigma_fast
import jax.random as jr

key = jr.PRNGKey(0)
model, state = eqx.nn.make_with_state(ResNetClassifier)(backbone="resnet34", num_classes=1000, key=key)

# Replace 3x3 convs + FC with spectral variants
spec_model = spectralize_resnet_classifier(model, which_convs="3x3+fc", alpha_init=1.0, key=key)

# Schedules / barrier (optional)
lam_sched = make_lambda_spec_schedule(peak=1e-3, warmup_steps=5_000, hold_steps=10_000, decay_steps=30_000, floor=1e-5)
bar = make_soft_barrier(target=8.0, sharpness=4.0, max_mult=8.0)

best_model, best_state, tr, va, pen, lip_log, lam_log = train_two_stage(
    model=spec_model,
    state=state,
    loss_fn=your_loss_fn,                # e.g., multiclass_loss
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    batch_size=64,
    key=key,
    # penalties if NOT using transform:
    lambda_spec_A=0.0, lambda_spec_B=0.0,  # disabled here; handled by transform
    # enable transform and logging:
    use_specpen_tx=True,
    lam_schedule=lam_sched,
    lipschitz_fn=prod_sigma_fast,        # fast proxy
    barrier=bar,
    barrier_every=200,
    log_lipschitz=True,
)

"""
from quantbayes.stochax.spectral_penalty_tx import (
    add_spectral_penalty_transform,
    specpen_metrics_from_opt_state,
)


# ------------------------ masking helpers (Stage A) ------------------------ #


def _collect_train_ids_stageA(model) -> set[int]:
    """Return Python ids of arrays to train in Stage A (s & bias only)."""
    wanted: set[int] = set()

    def visit(m):
        if isinstance(m, (SVDDense, SpectralDense, AdaptiveSpectralDense)):
            if hasattr(m, "s"):
                wanted.add(id(m.s))
            if hasattr(m, "bias") and m.bias is not None:
                wanted.add(id(m.bias))
        if isinstance(m, (SpectralConv2d, AdaptiveSpectralConv2d)):
            if hasattr(m, "s"):
                wanted.add(id(m.s))
            if hasattr(m, "bias") and m.bias is not None:
                wanted.add(id(m.bias))
        if isinstance(m, (SpectralCirculantLayer2d, AdaptiveSpectralCirculantLayer2d)):
            if hasattr(m, "bias") and m.bias is not None:
                wanted.add(id(m.bias))
        # Recurse
        if isinstance(m, eqx.Module):
            for v in vars(m).values():
                visit(v)
        elif isinstance(m, (list, tuple)):
            for v in m:
                visit(v)
        elif isinstance(m, dict):
            for v in m.values():
                visit(v)

    visit(model)
    return wanted


def _make_mask_tree_for_params(model) -> Tuple[optax.Params, optax.Params]:
    """Build (params_tree, mask_tree) aligned to eqx.filter(model, is_inexact_array)."""
    params = eqx.filter(model, eqx.is_inexact_array)
    flat, treedef = jax.tree_util.tree_flatten(params)
    id2idx = {id(leaf): i for i, leaf in enumerate(flat)}
    wanted_ids = _collect_train_ids_stageA(model)
    mask_flat = [False] * len(flat)
    for wid in wanted_ids:
        i = id2idx.get(wid, None)
        if i is not None:
            mask_flat[i] = True
    mask_tree = jax.tree_util.tree_unflatten(treedef, mask_flat)
    return params, mask_tree


def _maybe_masked(optim: optax.GradientTransformation, model, stage: Literal["A", "B"]):
    if stage == "B":
        return optim, optim.init(eqx.filter(model, eqx.is_inexact_array))
    # Stage A: mask to s & bias
    params, mask = _make_mask_tree_for_params(model)
    masked = optax.masked(optim, mask)
    return masked, masked.init(params)


# ------------------------------- epoch loop -------------------------------- #


def _epoch_loop(
    *,
    model,
    state,
    opt_state,
    optimizer,
    loss_fn: Callable,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_val: jnp.ndarray,
    y_val: jnp.ndarray,
    batch_size: int,
    num_epochs: int,
    key: jr.PRNGKey,
    lambda_spec_in_loss: float,
    lambda_frob: float,
    print_prefix: str = "",
    # logging from specpen transform:
    log_lipschitz: bool = False,
) -> Tuple[
    Any,
    Any,
    Any,
    List[float],
    List[float],
    List[float],
    List[Tuple[int, float]],
    List[Tuple[int, float]],
]:
    """
    Returns:
      best_model, best_state, opt_state, train_losses, val_losses, penalty_hist,
      lip_hist (list of (step, L)), lam_hist (list of (step, λ_eff))
    """
    train_losses: List[float] = []
    val_losses: List[float] = []
    pen_hist: List[float] = []
    lip_hist: List[Tuple[int, float]] = []
    lam_hist: List[Tuple[int, float]] = []

    best_val = float("inf")
    best_model, best_state = model, state

    rng, eval_rng = jr.split(key)
    for ep in range(1, num_epochs + 1):
        # --- Train ---
        running, seen = 0.0, 0
        rng, perm = jr.split(rng)
        for xb, yb in data_loader(
            X_train, y_train, batch_size=batch_size, shuffle=True, key=perm
        ):
            rng, sk = jr.split(rng)
            model, state, opt_state, loss = train_step(
                model,
                state,
                opt_state,
                xb,
                yb,
                sk,
                loss_fn,
                optimizer,
                lambda_spec=lambda_spec_in_loss,
                lambda_frob=lambda_frob,
            )
            running += float(loss) * xb.shape[0]
            seen += xb.shape[0]

            # --- NEW: read Lipschitz/λ from opt_state if transform is attached ---
            if log_lipschitz:
                m = specpen_metrics_from_opt_state(opt_state)
                if m and m["lip_updated"]:
                    lip_hist.append((int(m["step"]), float(m["last_lip"])))
                    lam_hist.append((int(m["step"]), float(m["last_lambda"])))

        train_losses.append(running / max(1, seen))

        # --- Val ---
        v_running, v_seen = 0.0, 0
        for xb, yb in data_loader(
            X_val, y_val, batch_size=batch_size, shuffle=False, key=eval_rng
        ):
            eval_rng, vk = jr.split(eval_rng)
            val_loss, _ = loss_fn(model, state, xb, yb, vk)
            v_running += float(val_loss) * xb.shape[0]
            v_seen += xb.shape[0]
        v = v_running / max(1, v_seen)
        val_losses.append(v)

        # --- Monitor penalty magnitude (model-side) ---
        pen_hist.append(float(global_spectral_penalty(model)))

        if (ep % max(1, num_epochs // 5) == 0) or (ep == num_epochs):
            msg = f"{print_prefix}[epoch {ep:03d}/{num_epochs}] train={train_losses[-1]:.4f} val={v:.4f} pen={pen_hist[-1]:.3e}"
            if log_lipschitz and len(lip_hist) > 0:
                last_L = lip_hist[-1][1]
                last_lam = lam_hist[-1][1]
                msg += f" | L≈{last_L:.3g} λ_eff={last_lam:.2e}"
            print(msg)

        if v < best_val:
            best_val = v
            best_model, best_state = model, state

    return (
        best_model,
        best_state,
        opt_state,
        train_losses,
        val_losses,
        pen_hist,
        lip_hist,
        lam_hist,
    )


# ------------------------------- public API -------------------------------- #


def train_two_stage(
    *,
    model,
    state,
    loss_fn: Callable[
        [Any, Any, jnp.ndarray, jnp.ndarray, jr.PRNGKey], Tuple[jnp.ndarray, Any]
    ],
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_val: jnp.ndarray,
    y_val: jnp.ndarray,
    batch_size: int,
    key: jr.PRNGKey,
    # schedules / penalties
    lr_stageA: float = 5e-4,
    lr_stageB: float = 2e-4,
    wd: float = 1e-4,
    epochs_stageA: int = 5,
    epochs_stageB: int = 20,
    lambda_spec_A: float = 0.0,
    lambda_spec_B: float = 1e-4,
    lambda_frob: float = 0.0,
    optimizer_family: Literal["adamw", "adan", "lion"] = "adamw",
    # ---- NEW: optional spectral-penalty transform + logging ----
    use_specpen_tx: bool = False,
    lam_schedule: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    lipschitz_fn: Optional[Callable[[Any], jnp.ndarray]] = None,
    barrier: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    barrier_every: int = 0,
    log_lipschitz: bool = False,
) -> Tuple[
    Any,
    Any,
    List[float],
    List[float],
    List[float],
    List[Tuple[int, float]],
    List[Tuple[int, float]],
]:
    """
    Two-stage spectral fine-tuning:
      Stage A: update only (s, bias) with lr_stageA and lambda_spec_A (if not using tx).
      Stage B: unfreeze all, smaller lr_stageB and lambda_spec_B (if not using tx).

    If `use_specpen_tx=True`, the spectral penalty is injected by the Optax transform.
    In that case set lambda_spec_A/B=0 and the effective λ(t) is controlled by `lam_schedule`
    (optionally modulated by `barrier(lipschitz_fn(model))` every `barrier_every` steps).

    Returns:
      best_model, best_state, train_losses, val_losses, penalties, lip_hist, lam_hist
      (lip_hist/lam_hist are empty lists if not logging or no transform present).
    """

    # --- pick base optimizer family
    def make_base(lr):
        if optimizer_family == "adamw":
            return optax.adamw(lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=wd)
        if optimizer_family == "adan":
            return optax.adan(lr, b1=0.98, b2=0.99, eps=1e-8, weight_decay=wd)
        if optimizer_family == "lion":
            return optax.chain(optax.add_decayed_weights(wd), optax.lion(lr))
        raise ValueError("unknown optimizer family")

    def maybe_chain_with_specpen(base_opt, like_model):
        if not use_specpen_tx:
            return base_opt
        tx_pen = add_spectral_penalty_transform(
            like_model=like_model,
            schedule=lam_schedule,
            lipschitz_fn=lipschitz_fn,
            barrier=barrier,
            barrier_every=barrier_every,
        )
        return optax.chain(tx_pen, base_opt)

    # --- Stage A (mask to s & bias)
    optA_base = make_base(lr_stageA)
    optA = maybe_chain_with_specpen(optA_base, model)
    optA, opt_stateA = _maybe_masked(optA, model, stage="A")

    lamA_in_loss = 0.0 if use_specpen_tx else float(lambda_spec_A)
    bmA, bsA, opt_stateA, trA, vaA, penA, lipA, lamA = _epoch_loop(
        model=model,
        state=state,
        opt_state=opt_stateA,
        optimizer=optA,
        loss_fn=loss_fn,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=batch_size,
        num_epochs=epochs_stageA,
        key=key,
        lambda_spec_in_loss=lamA_in_loss,
        lambda_frob=lambda_frob,
        print_prefix="[Stage A] ",
        log_lipschitz=log_lipschitz and use_specpen_tx,
    )

    # --- Stage B (unfreeze all)
    optB_base = make_base(lr_stageB)
    optB = maybe_chain_with_specpen(optB_base, bmA)
    optB, opt_stateB = _maybe_masked(optB, bmA, stage="B")
    keyB = jr.fold_in(key, 1)

    lamB_in_loss = 0.0 if use_specpen_tx else float(lambda_spec_B)
    bmB, bsB, opt_stateB, trB, vaB, penB, lipB, lamB = _epoch_loop(
        model=bmA,
        state=bsA,
        opt_state=opt_stateB,
        optimizer=optB,
        loss_fn=loss_fn,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=batch_size,
        num_epochs=epochs_stageB,
        key=keyB,
        lambda_spec_in_loss=lamB_in_loss,
        lambda_frob=lambda_frob,
        print_prefix="[Stage B] ",
        log_lipschitz=log_lipschitz and use_specpen_tx,
    )

    # concat histories
    train_losses = trA + trB
    val_losses = vaA + vaB
    penalties = penA + penB
    lip_hist = lipA + lipB
    lam_hist = lamA + lamB
    return bmB, bsB, train_losses, val_losses, penalties, lip_hist, lam_hist
