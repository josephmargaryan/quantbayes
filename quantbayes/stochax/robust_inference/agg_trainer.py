# quantbayes/stochax/robust_inference/agg_trainer.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Callable, Literal

import os
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from quantbayes.stochax.trainer.train import (
    train,  # generic trainer with bound_recorder + checkpoints
    multiclass_loss,
    make_lmt_multiclass_loss,
)
from quantbayes.stochax.robust_inference.simplex import project_rows_to_simplex
from quantbayes.stochax.robust_inference.masks import choose_m_probs  # p(m) ∝ C(n,m)

Array = jnp.ndarray
PRNG = jax.Array


# ------------------------ adversarial training loss: paper policy (m ~ C(n,m)) ------------------------ #


def make_adv_probit_loss(
    f: int,
    *,
    pgd_steps: int = 20,
    pgd_step_size: float = 5e-2,
    tries_per_batch: int = 3,  # keep small (1–3) for dev
    project: str = "softmax",  # "softmax" or "euclid"
    # optional LMT-style shift on logits (rare in robust mode, supported)
    use_lmt_logits: bool = False,
    lmt_eps: float = 1.0,
    lmt_alpha: float = 1.0,
    lmt_stop_grad_L: bool = True,
    lmt_conv_mode: str = "tn",
    lmt_conv_tn_iters: int = 8,
    lmt_conv_gram_iters: int = 5,
    lmt_conv_fft_shape: Optional[Tuple[int, int]] = None,
    lmt_conv_input_shape: Optional[Tuple[int, int]] = None,
):
    """Robust loss with m sampled from {1..f} with probability p(m) ∝ C(n,m)."""
    assert (
        isinstance(tries_per_batch, int) and tries_per_batch >= 1
    ), "tries_per_batch must be an integer ≥ 1"
    project = project.lower().strip()
    assert project in {"softmax", "euclid"}

    def _proj_rows(P):
        return (
            jax.nn.softmax(P, axis=-1)
            if project == "softmax"
            else project_rows_to_simplex(P)
        )

    def _maybe_lmt_adjust(model, state, logits, y):
        if not use_lmt_logits:
            return logits
        from quantbayes.stochax.utils.lip_upper import network_lipschitz_upper

        L_hat = network_lipschitz_upper(
            model,
            state=state,
            conv_mode=lmt_conv_mode,
            conv_tn_iters=lmt_conv_tn_iters,
            conv_gram_iters=lmt_conv_gram_iters,
            conv_fft_shape=lmt_conv_fft_shape,
            conv_input_shape=lmt_conv_input_shape,
        )
        if lmt_stop_grad_L:
            L_hat = jax.lax.stop_gradient(L_hat)
        kappa = jnp.asarray(lmt_alpha * lmt_eps, jnp.float32) * jnp.clip(
            L_hat, 1e-12, 1e12
        )
        K = logits.shape[-1]
        onehot = jax.nn.one_hot(y, num_classes=K, dtype=logits.dtype)
        return logits + kappa[..., None] * (1.0 - 2.0 * onehot)

    def _attack_once(model, P0: Array, y: Array, key: PRNG) -> Array:
        """
        One Monte-Carlo draw for a whole batch (B,n,K):
          1) sample m ~ p(m) ∝ C(n,m), m ∈ {1..f}
          2) choose a uniform size-m subset of rows (per example)
          3) run PGD on those m rows under projection (softmax/simplex)
        """
        B, n, K = P0.shape
        slots = int(f)  # static dimension for “max slots” (JAX/XLA friendly)

        # sample permutation per example; ranks identify the last-<something> rows
        k_perm, k_m, k_v = jr.split(key, 3)
        perms = jax.vmap(lambda kk: jr.permutation(kk, n))(jr.split(k_perm, B))  # (B,n)
        ranks = jax.vmap(jnp.argsort)(perms)  # (B,n)

        # sample m ∈ {1..f} with p(m) ∝ C(n,m)
        p_m = choose_m_probs(n, slots)  # length = f
        ms = jr.categorical(k_m, jnp.log(p_m), shape=(B,)) + 1  # (B,), values in [1..f]

        # Build selection tensor T mapping ‘slots’ to last-<slots> positions
        def build_T(rank: Array) -> Array:
            ridx = rank - (n - slots)
            ridx_clipped = jnp.clip(ridx, 0, slots - 1)
            onehot = jax.nn.one_hot(ridx_clipped, slots, dtype=P0.dtype)  # (n,slots)
            in_last = (rank >= (n - slots)).astype(P0.dtype)[:, None]  # (n,1)
            return onehot * in_last

        T = jax.vmap(build_T)(ranks)  # (B,n,slots)

        # masks: which of the slots are active and which rows are truly corrupted
        ar = jnp.arange(slots, dtype=jnp.int32)[None, :]  # (1,slots)
        mask_use_slots = (ar >= (slots - ms[:, None])).astype(P0.dtype)  # (B,slots)
        mask_use_rows = (ranks >= (n - ms[:, None])).astype(P0.dtype)  # (B,n)

        V0 = jr.normal(k_v, shape=(B, slots, K))

        # loss definitions
        def loss_on_P(Pi: Array, yi: jnp.ndarray) -> Array:
            z, _ = model(Pi, None, None)
            return -jnp.log(jax.nn.softmax(z)[yi] + 1e-12)

        def loss_fn_batch(P_batch: Array) -> Array:
            return jnp.mean(jax.vmap(loss_on_P)(P_batch, y.astype(jnp.int32)))

        def step(_, V_cur):
            Vhat = _proj_rows(V_cur)  # (B,slots,K)
            Vsel = Vhat * mask_use_slots[..., None]  # (B,slots,K)
            rows_from_V = jnp.einsum("bnr,brk->bnk", T, Vsel)  # (B,n,K)
            P_tmp = P0 * (1.0 - mask_use_rows[..., None]) + rows_from_V
            gP = jax.grad(loss_fn_batch)(P_tmp)  # (B,n,K)
            gV = jnp.einsum("bnr,bnk->brk", T, gP) * mask_use_slots[..., None]
            return V_cur + pgd_step_size * jnp.sign(gV)

        V_fin = jax.lax.fori_loop(0, pgd_steps, step, V0)
        Vhat = _proj_rows(V_fin)
        Vsel = Vhat * mask_use_slots[..., None]
        rows_from_V = jnp.einsum("bnr,brk->bnk", T, Vsel)
        return P0 * (1.0 - mask_use_rows[..., None]) + rows_from_V

    @eqx.filter_jit
    def loss_fn(model, state, x: Array, y: Array, key: PRNG):
        y = y.astype(jnp.int32)
        keys = jr.split(key, tries_per_batch)

        def one_try(carry, kt):
            best_loss, best_P = carry
            Padv = _attack_once(model, x, y, kt)
            logits = jax.vmap(lambda Pi: model(Pi, None, None)[0])(Padv)
            cur = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
            better = cur > best_loss
            best_loss = jnp.where(better, cur, best_loss)
            best_P = jnp.where(better, Padv, best_P)
            return (best_loss, best_P), None

        (best_loss, best_Padv), _ = jax.lax.scan(
            one_try, (jnp.array(-1e30, dtype=jnp.float32), x), keys
        )
        logits = jax.vmap(lambda Pi: model(Pi, None, None)[0])(best_Padv)
        if use_lmt_logits:
            logits = _maybe_lmt_adjust(model, state, logits, y)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        return loss, state

    return loss_fn


# ------------------------ ERM training (unchanged API + optional ckpts/logs) ------------------------ #


def train_aggregator_erm(
    Ps_tr: Array,
    y_tr: Array,
    Ps_val: Array,
    y_val: Array,
    agg,
    *,
    epochs=10,
    batch_size=256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 3,
    key=None,
    loss_kind: Literal["ce", "lmt"] = "ce",
    lmt_kwargs: Optional[Dict[str, Any]] = None,
    # penalties
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    lambda_specnorm: float = 0.0,
    lambda_sob_jac: float = 0.0,
    lambda_sob_kernel: float = 0.0,
    lambda_liplog: float = 0.0,
    specnorm_conv_mode: str = "tn",
    specnorm_conv_tn_iters: Optional[int] = None,
    specnorm_conv_gram_iters: Optional[int] = None,
    specnorm_conv_fft_shape: Optional[Tuple[int, int]] = None,
    specnorm_conv_input_shape: Optional[Tuple[int, int]] = None,
    lip_conv_mode: str = "tn",
    lip_conv_tn_iters: Optional[int] = None,
    lip_conv_gram_iters: Optional[int] = None,
    lip_conv_fft_shape: Optional[Tuple[int, int]] = None,
    lip_conv_input_shape: Optional[Tuple[int, int]] = None,
    # bound logging (optional)
    log_global_bound_every: Optional[int] = None,
    bound_conv_mode: str = "tn",
    bound_tn_iters: int = 8,
    bound_gram_iters: int = 5,
    bound_fft_shape: Optional[Tuple[int, int]] = None,
    bound_input_shape: Optional[Tuple[int, int]] = None,
    bound_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
    # checkpoints (optional)
    ckpt_dir: Optional[str] = None,
):
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(agg, eqx.is_inexact_array))
    k = jr.PRNGKey(0) if key is None else key

    if loss_kind.lower().strip() == "lmt":
        _lmt = {
            "eps": 1.0,
            "alpha": 1.0,
            "conv_mode": "tn",
            "conv_tn_iters": 8,
            "conv_gram_iters": 5,
            "conv_fft_shape": None,
            "conv_input_shape": None,
            "stop_grad_L": True,
        }
        if lmt_kwargs is not None:
            _lmt.update(lmt_kwargs)
        loss_fn = make_lmt_multiclass_loss(**_lmt)
    else:
        loss_fn = multiclass_loss

    ckpt_path = None
    ckpt_interval = None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "agg_epoch={epoch}.eqx")
        ckpt_interval = 1

    best_model, _, tr, va = train(
        model=agg,
        state=None,
        opt_state=opt_state,
        optimizer=optimizer,
        loss_fn=loss_fn,
        X_train=Ps_tr,
        y_train=y_tr.astype(jnp.int32),
        X_val=Ps_val,
        y_val=y_val.astype(jnp.int32),
        batch_size=batch_size,
        num_epochs=epochs,
        patience=patience,
        key=k,
        # penalties
        lambda_spec=lambda_spec,
        lambda_frob=lambda_frob,
        lambda_specnorm=lambda_specnorm,
        lambda_sob_jac=lambda_sob_jac,
        lambda_sob_kernel=lambda_sob_kernel,
        lambda_liplog=lambda_liplog,
        specnorm_conv_mode=specnorm_conv_mode,  # dense-only aggregator → exact σ per layer
        specnorm_conv_tn_iters=specnorm_conv_tn_iters,
        specnorm_conv_gram_iters=specnorm_conv_gram_iters,
        specnorm_conv_fft_shape=specnorm_conv_fft_shape,
        specnorm_conv_input_shape=specnorm_conv_input_shape,
        lip_conv_mode=lip_conv_mode,
        lip_conv_tn_iters=lip_conv_tn_iters,
        lip_conv_gram_iters=lip_conv_gram_iters,
        lip_conv_fft_shape=lip_conv_fft_shape,
        lip_conv_input_shape=lip_conv_input_shape,
        # bound logging
        log_global_bound_every=log_global_bound_every,
        bound_conv_mode=bound_conv_mode,
        bound_tn_iters=bound_tn_iters,
        bound_gram_iters=bound_gram_iters,
        bound_fft_shape=bound_fft_shape,
        bound_input_shape=bound_input_shape,
        bound_recorder=bound_recorder,
        # checkpoints
        ckpt_path=ckpt_path,
        checkpoint_interval=ckpt_interval,
    )
    return best_model, {"train_loss": tr, "val_loss": va, "ckpt_dir": ckpt_dir}


# ------------------------ robust (RERM) training: paper m-policy ------------------------ #


def train_aggregator_robust(
    Ps_tr: Array,
    y_tr: Array,
    Ps_val: Array,
    y_val: Array,
    agg,
    *,
    f: int,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 5e-5,
    weight_decay: float = 0.0,
    patience: int = 3,
    pgd_steps: int = 20,
    pgd_step_size: float = 5e-2,
    tries_per_batch: int = 3,  # must be int ≥ 1
    project: str = "softmax",
    key=None,
    # optional LMT shift (usually off in robust mode)
    use_lmt_logits: bool = False,
    lmt_eps: float = 1.0,
    lmt_alpha: float = 1.0,
    lmt_stop_grad_L: bool = True,
    lmt_conv_mode: str = "tn",
    lmt_conv_tn_iters: int = 8,
    lmt_conv_gram_iters: int = 5,
    lmt_conv_fft_shape: Optional[Tuple[int, int]] = None,
    lmt_conv_input_shape: Optional[Tuple[int, int]] = None,
    # penalties
    lambda_spec: float = 0.0,
    lambda_frob: float = 0.0,
    lambda_specnorm: float = 0.0,
    lambda_sob_jac: float = 0.0,
    lambda_sob_kernel: float = 0.0,
    lambda_liplog: float = 0.0,
    specnorm_conv_mode: str = "tn",
    specnorm_conv_tn_iters: Optional[int] = None,
    specnorm_conv_gram_iters: Optional[int] = None,
    specnorm_conv_fft_shape: Optional[Tuple[int, int]] = None,
    specnorm_conv_input_shape: Optional[Tuple[int, int]] = None,
    lip_conv_mode: str = "tn",
    lip_conv_tn_iters: Optional[int] = None,
    lip_conv_gram_iters: Optional[int] = None,
    lip_conv_fft_shape: Optional[Tuple[int, int]] = None,
    lip_conv_input_shape: Optional[Tuple[int, int]] = None,
    # bound logging (optional)
    log_global_bound_every: Optional[int] = None,
    bound_conv_mode: str = "tn",
    bound_tn_iters: int = 8,
    bound_gram_iters: int = 5,
    bound_fft_shape: Optional[Tuple[int, int]] = None,
    bound_input_shape: Optional[Tuple[int, int]] = None,
    bound_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
    # checkpoints (optional)
    ckpt_dir: Optional[str] = None,
):
    assert (
        isinstance(tries_per_batch, int) and tries_per_batch >= 1
    ), "tries_per_batch must be an integer ≥ 1"

    loss_fn = make_adv_probit_loss(
        f=f,
        pgd_steps=pgd_steps,
        pgd_step_size=pgd_step_size,
        tries_per_batch=tries_per_batch,
        project=project,
        use_lmt_logits=use_lmt_logits,
        lmt_eps=lmt_eps,
        lmt_alpha=lmt_alpha,
        lmt_stop_grad_L=lmt_stop_grad_L,
        lmt_conv_mode=lmt_conv_mode,
        lmt_conv_tn_iters=lmt_conv_tn_iters,
        lmt_conv_gram_iters=lmt_conv_gram_iters,
        lmt_conv_fft_shape=lmt_conv_fft_shape,
        lmt_conv_input_shape=lmt_conv_input_shape,
    )

    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(agg, eqx.is_inexact_array))
    k = jr.PRNGKey(0) if key is None else key

    ckpt_path = None
    ckpt_interval = None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "agg_epoch={epoch}.eqx")
        ckpt_interval = 1

    best_model, _, tr, va = train(
        model=agg,
        state=None,
        opt_state=opt_state,
        optimizer=optimizer,
        loss_fn=loss_fn,
        X_train=Ps_tr,
        y_train=y_tr.astype(jnp.int32),
        X_val=Ps_val,
        y_val=y_val.astype(jnp.int32),
        batch_size=batch_size,
        num_epochs=epochs,
        patience=patience,
        key=k,
        # penalties
        lambda_spec=lambda_spec,
        lambda_frob=lambda_frob,
        lambda_specnorm=lambda_specnorm,
        lambda_sob_jac=lambda_sob_jac,
        lambda_sob_kernel=lambda_sob_kernel,
        lambda_liplog=lambda_liplog,
        specnorm_conv_mode=specnorm_conv_mode,
        specnorm_conv_tn_iters=specnorm_conv_tn_iters,
        specnorm_conv_gram_iters=specnorm_conv_gram_iters,
        specnorm_conv_fft_shape=specnorm_conv_fft_shape,
        specnorm_conv_input_shape=specnorm_conv_input_shape,
        lip_conv_mode=lip_conv_mode,
        lip_conv_tn_iters=lip_conv_tn_iters,
        lip_conv_gram_iters=lip_conv_gram_iters,
        lip_conv_fft_shape=lip_conv_fft_shape,
        lip_conv_input_shape=lip_conv_input_shape,
        # bound logging
        log_global_bound_every=log_global_bound_every,
        bound_conv_mode=bound_conv_mode,
        bound_tn_iters=bound_tn_iters,
        bound_gram_iters=bound_gram_iters,
        bound_fft_shape=bound_fft_shape,
        bound_input_shape=bound_input_shape,
        bound_recorder=bound_recorder,
        # checkpoints
        ckpt_path=ckpt_path,
        checkpoint_interval=ckpt_interval,
    )
    return best_model, {"train_loss": tr, "val_loss": va, "ckpt_dir": ckpt_dir}
