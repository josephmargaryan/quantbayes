# quantbayes/stochax/robust_inference/clients.py
from __future__ import annotations
from typing import List, Tuple, Optional, Callable, Dict, Any
import os

import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax

from quantbayes.stochax.trainer.train import train, multiclass_loss
from quantbayes.stochax.robust_inference.models import make_client_with_state
from quantbayes.stochax.utils.regularizers import global_spectral_norm_penalty

Array = jnp.ndarray


def _client_logits(model, state, X: Array, *, key: jr.KeyArray) -> Array:
    """Vmapped client forward: (x,key,state)->(logits,state)."""

    def f(x, k):
        return model(x, k, state)[0]

    keys = jr.split(key, X.shape[0])
    return jax.vmap(f)(X, keys)


def _top1_acc(logits: Array, y: Array) -> float:
    return float(jnp.mean((jnp.argmax(logits, axis=-1) == y).astype(jnp.float32)))


def train_clients(
    parts: List[Tuple[jnp.ndarray, jnp.ndarray]],  # [(X_i, y_i)] * n
    *,
    d_in: int,
    k: int,
    width: int = 256,
    epochs: int = 8,
    batch: int = 256,
    lr: float = 1e-3,
    wd: float = 1e-4,
    seed: int = 0,
    X_val: jnp.ndarray | None = None,
    y_val: jnp.ndarray | None = None,
    # ---- optional bound logging from generic trainer ----
    log_client_bounds_every: Optional[int] = None,
    client_bound_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
    # ---- optional per-epoch accuracy/loss via checkpointing ----
    return_history: bool = False,
    save_checkpoints_dir: Optional[str] = None,  # e.g. "runs/.../clients_ckpts"
):
    """
    Train n client models centrally.

    If `return_history=True` and `save_checkpoints_dir` is provided:
      - Save checkpoints at each epoch for each client.
      - Return per-epoch curves:
          [{"train_loss": [...], "val_loss": [...], "train_acc": [...], "val_acc": [...]} ...]
    Otherwise returns only (models, states).

    Bound logging (Lipschitz) is controlled via log_client_bounds_every + recorder.
    """
    models, states = [], []
    histories: List[Dict[str, Any]] = []
    opt = optax.adamw(learning_rate=lr, weight_decay=wd)
    if save_checkpoints_dir:
        os.makedirs(save_checkpoints_dir, exist_ok=True)

    for i, (Xi, yi) in enumerate(parts):
        m0, s0 = make_client_with_state(
            d_in=d_in, width=width, k=k, key=jr.PRNGKey(seed + 10_000 + i)
        )
        opt_state = opt.init(eqx.filter(m0, eqx.is_inexact_array))

        # small val slice if none provided
        if X_val is None or y_val is None:
            Xv, yv = Xi[: min(1024, Xi.shape[0])], yi[: min(1024, yi.shape[0])]
        else:
            Xv, yv = X_val, y_val

        # recorder wrapper: tag client id
        if (log_client_bounds_every is not None) and (
            client_bound_recorder is not None
        ):

            def _rec(rec, c=i, cb=client_bound_recorder):
                r = dict(rec)
                r["client"] = int(c)
                cb(r)

        else:
            _rec = None

        # checkpoints (optional)
        ckpt_template = None
        ckpt_interval = None
        if return_history and save_checkpoints_dir:
            ckpt_template = os.path.join(
                save_checkpoints_dir, f"client_{i}_epoch={{epoch}}.eqx"
            )
            ckpt_interval = 1

        m_best, s_best, tr, va = train(
            model=m0,
            state=s0,
            opt_state=opt_state,
            optimizer=opt,
            loss_fn=multiclass_loss,
            X_train=Xi,
            y_train=yi,
            X_val=Xv,
            y_val=yv,
            batch_size=batch,
            num_epochs=epochs,
            patience=3,
            key=jr.PRNGKey(seed + 20_000 + i),
            # bound logging
            log_global_bound_every=log_client_bounds_every,
            bound_recorder=_rec,
            bound_conv_mode="tn",
            bound_tn_iters=8,
            bound_gram_iters=5,
            # checkpoints
            ckpt_path=ckpt_template,
            checkpoint_interval=ckpt_interval,
        )
        models.append(m_best)
        states.append(s_best)

        # final Σσ (optional)
        if client_bound_recorder is not None:
            sig = float(global_spectral_norm_penalty(m_best, conv_mode="tn"))
            client_bound_recorder({"client": int(i), "sigma_sum_final": sig})

        if return_history:
            hist: Dict[str, Any] = {
                "train_loss": tr,
                "val_loss": va,
                "train_acc": [],
                "val_acc": [],
            }
            if ckpt_template:
                for e in range(1, len(tr) + 1):
                    path = ckpt_template.format(epoch=e)
                    if not os.path.exists(path):
                        break
                    bundle = {"model": m0, "state": s0}
                    bundle = eqx.tree_deserialise_leaves(path, bundle)
                    mt, st = bundle["model"], bundle["state"]
                    log_tr = _client_logits(
                        mt, st, Xi, key=jr.PRNGKey(100 + i * 73 + e)
                    )
                    log_va = _client_logits(
                        mt, st, Xv, key=jr.PRNGKey(200 + i * 97 + e)
                    )
                    hist.setdefault("train_acc", []).append(_top1_acc(log_tr, yi))
                    hist.setdefault("val_acc", []).append(_top1_acc(log_va, yv))
            histories.append(hist)

    # Return 2 or 3 values depending on flag (prevents unpacking errors)
    if return_history:
        return models, states, histories
    return models, states
