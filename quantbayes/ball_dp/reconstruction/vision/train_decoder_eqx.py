# quantbayes/ball_dp/reconstruction/vision/train_decoder_eqx.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

from quantbayes.stochax.trainer.train import train, regression_loss

from .decoders_eqx import DecoderMLP
from .preprocess import clip01


def _train_val_split_idx(
    n: int, val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n))
    rng.shuffle(idx)
    nv = max(1, int(round(val_frac * n)))
    va = idx[:nv]
    tr = idx[nv:]
    if tr.size == 0:
        tr = va
    return tr.astype(np.int64), va.astype(np.int64)


def train_decoder_mlp(
    *,
    E: np.ndarray,  # (N,d_embed)
    X_target_flat: np.ndarray,  # (N,d_img)
    hidden: Tuple[int, ...] = (1024, 1024),
    act: str = "relu",
    out_act: str = "sigmoid",
    lr: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 256,
    seed: int = 0,
    val_frac: float = 0.1,
) -> Dict[str, object]:
    """
    Train a public decoder g(e)->x_flat (post-processing only).
    Uses stochax.train + regression_loss (MSE).
    """
    E = np.asarray(E, dtype=np.float32)
    X = np.asarray(X_target_flat, dtype=np.float32)
    if E.ndim != 2 or X.ndim != 2 or E.shape[0] != X.shape[0]:
        raise ValueError("E must be (N,de), X_target_flat must be (N,dx) with same N")

    tr_idx, va_idx = _train_val_split_idx(E.shape[0], float(val_frac), int(seed))
    Etr, Xtr = E[tr_idx], X[tr_idx]
    Eva, Xva = E[va_idx], X[va_idx]

    Etrj = jnp.asarray(Etr)
    Xtrj = jnp.asarray(Xtr)
    Evaj = jnp.asarray(Eva)
    Xvaj = jnp.asarray(Xva)

    d_in = int(E.shape[1])
    d_out = int(X.shape[1])

    dec = DecoderMLP(
        d_in, d_out, hidden=hidden, key=jr.PRNGKey(int(seed)), act=act, out_act=out_act
    )
    state = None

    opt = optax.adam(float(lr))
    opt_state = opt.init(eqx.filter(dec, eqx.is_inexact_array))

    best_dec, best_state, tr_hist, va_hist = train(
        dec,
        state,
        opt_state,
        opt,
        regression_loss,
        Etrj,
        Xtrj,
        Evaj,
        Xvaj,
        batch_size=int(batch_size),
        num_epochs=int(epochs),
        patience=int(epochs),
        key=jr.PRNGKey(int(seed + 1)),
        augment_fn=None,
    )

    return {
        "decoder": best_dec,
        "train_hist": tr_hist,
        "val_hist": va_hist,
        "d_in": d_in,
        "d_out": d_out,
    }


def decode_images_from_embeddings(
    *,
    decoder: DecoderMLP,
    E: np.ndarray,
    out_shape: Tuple[int, int, int],  # (C,H,W)
) -> np.ndarray:
    """
    Decode embeddings to images.
    Returns (N,C,H,W) float32 in [0,1] if decoder uses sigmoid output.
    """
    E = np.asarray(E, dtype=np.float32)
    Ej = jnp.asarray(E)
    keys = jr.split(jr.PRNGKey(0), Ej.shape[0])
    flat, _ = jax.vmap(lambda e, k: decoder(e, k, None))(Ej, keys)
    flat = np.asarray(flat).astype(np.float32)
    C, H, W = map(int, out_shape)
    imgs = flat.reshape(flat.shape[0], C, H, W)
    return clip01(imgs)


if __name__ == "__main__":
    # smoke: random embeddings -> random targets
    rng = np.random.default_rng(0)
    E = rng.normal(size=(512, 16)).astype(np.float32)
    X = rng.random(size=(512, 64)).astype(np.float32)
    out = train_decoder_mlp(
        E=E, X_target_flat=X, hidden=(64, 64), epochs=2, batch_size=128, seed=0
    )
    dec = out["decoder"]
    imgs = decode_images_from_embeddings(decoder=dec, E=E[:4], out_shape=(1, 8, 8))
    print("decoded imgs:", imgs.shape, imgs.min(), imgs.max())
    print("[OK] train_decoder_eqx smoke.")
