# quantbayes/ball_dp/reconstruction/reconn_eqx.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

from quantbayes.ball_dp.api import eqx_flatten_adapter
from quantbayes.stochax.trainer.train import train, regression_loss, multiclass_loss
from quantbayes.stochax.privacy.dp_train import dp_eqx_train
from quantbayes.stochax.privacy.dp import DPSGDConfig

from quantbayes.ball_dp.reconstruction.nonconvex_head_eqx import EmbeddingMLPClassifier


@dataclass(frozen=True)
class WeightNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, W: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        return (W - self.mean) / (self.std + eps)


class ReconstructorMLP(eqx.Module):
    """
    Weight-vector -> embedding-vector reconstructor.
    Trainer interface: (x, key, state) -> (pred, state)
    """

    layers: Tuple[eqx.nn.Linear, ...]
    act: str = eqx.field(static=True)

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden: Tuple[int, ...],
        *,
        key: jr.PRNGKey,
        act: str = "relu",
    ):
        keys = jr.split(key, len(hidden) + 1)
        ls: List[eqx.nn.Linear] = []
        last = int(d_in)
        for i, h in enumerate(hidden):
            ls.append(eqx.nn.Linear(last, int(h), key=keys[i]))
            last = int(h)
        ls.append(eqx.nn.Linear(last, int(d_out), key=keys[-1]))
        self.layers = tuple(ls)
        self.act = str(act).lower()

    def _phi(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.act == "gelu":
            return jax.nn.gelu(x)
        if self.act == "elu":
            return jax.nn.elu(x)
        return jax.nn.relu(x)

    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey, state):
        h = x
        for L in self.layers[:-1]:
            h = self._phi(L(h))
        out = self.layers[-1](h)
        return out, state


def _train_val_split_idx(
    n: int, val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n))
    rng.shuffle(idx)
    nv = max(1, int(round(val_frac * n)))
    val = idx[:nv]
    tr = idx[nv:]
    if tr.size == 0:
        tr = val
    return tr.astype(np.int64), val.astype(np.int64)


def nearest_neighbor_oracle_l2(z: np.ndarray, pool: np.ndarray) -> float:
    """
    Conservative oracle threshold: min_{p in pool} ||z - p||_2.
    """
    z = np.asarray(z, dtype=np.float32).reshape(1, -1)
    pool = np.asarray(pool, dtype=np.float32)
    d = np.linalg.norm(pool - z, axis=1)
    return float(np.min(d))


def _fit_head_eqx(
    *,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    hidden: int,
    act: str,
    lr: float,
    epochs: int,
    batch_size: int,
    seed_model: int,
    seed_train: int,
    dp_cfg: Optional[DPSGDConfig] = None,
) -> EmbeddingMLPClassifier:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    N, d = X.shape
    model = EmbeddingMLPClassifier(
        d, int(n_classes), int(hidden), key=jr.PRNGKey(int(seed_model)), act=str(act)
    )
    state = None

    Xj = jnp.asarray(X)
    yj = jnp.asarray(y, dtype=jnp.int32)

    # optimizer
    optimizer = optax.adam(float(lr))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    if dp_cfg is None:
        # non-DP train
        best_model, best_state, *_ = train(
            model,
            state,
            opt_state,
            optimizer,
            multiclass_loss,
            Xj,
            yj,
            Xj,
            yj,
            batch_size=int(batch_size),
            num_epochs=int(epochs),
            patience=int(epochs),
            key=jr.PRNGKey(int(seed_train)),
            augment_fn=None,
        )
        return best_model
    else:
        # DP-SGD train (your privacy engine)
        best_model, best_state, *_ = dp_eqx_train(
            model,
            state,
            opt_state,
            optimizer,
            multiclass_loss,
            Xj,
            yj,
            Xj,
            yj,
            dp_config=dp_cfg,
            batch_size=int(batch_size),
            num_epochs=int(epochs),
            patience=int(epochs),
            key=jr.PRNGKey(int(seed_train)),
            shuffle=True,
        )
        return best_model


def train_shadow_models_eqx(
    *,
    Z_fixed: np.ndarray,
    y_fixed: np.ndarray,
    Z_shadow: np.ndarray,
    y_shadow: np.ndarray,
    n_classes: int,
    head_hidden: int = 32,
    head_act: str = "elu",
    head_lr: float = 1e-2,
    head_epochs: int = 30,
    head_batch_size: int = 512,
    known_init: bool = True,
    seed: int = 0,
    dp_cfg: Optional[DPSGDConfig] = None,
) -> Dict[str, object]:
    """
    Build shadow (theta_i, z_i) pairs by training one head per shadow target on D^- ∪ {z_i}.
    Everything is Eqx + your trainers.
    """
    Z_fixed = np.asarray(Z_fixed, dtype=np.float32)
    y_fixed = np.asarray(y_fixed, dtype=np.int64)
    Z_shadow = np.asarray(Z_shadow, dtype=np.float32)
    y_shadow = np.asarray(y_shadow, dtype=np.int64)

    W_list: List[np.ndarray] = []
    for i in range(Z_shadow.shape[0]):
        X = np.concatenate([Z_fixed, Z_shadow[i : i + 1]], axis=0)
        y = np.concatenate([y_fixed, y_shadow[i : i + 1]], axis=0)

        seed_model = int(seed) if known_init else int(seed + 1000 + i)
        seed_train = int(seed)  # keep training seed fixed unless you want to ablate it

        mdl = _fit_head_eqx(
            X=X,
            y=y,
            n_classes=int(n_classes),
            hidden=int(head_hidden),
            act=str(head_act),
            lr=float(head_lr),
            epochs=int(head_epochs),
            batch_size=int(head_batch_size),
            seed_model=seed_model,
            seed_train=seed_train,
            dp_cfg=dp_cfg,
        )
        theta, _ = eqx_flatten_adapter(mdl)  # (p,)
        W_list.append(theta.astype(np.float32, copy=False))

    W = np.stack(W_list, axis=0)  # (k,p)

    mean = W.mean(axis=0)
    std = W.std(axis=0) + 1e-6
    normalizer = WeightNormalizer(mean=mean, std=std)

    return {
        "W_shadow": W,
        "Z_shadow": Z_shadow,
        "normalizer": normalizer,
        "known_init": bool(known_init),
        "dp_used": dp_cfg is not None,
    }


def train_reconstructor_eqx(
    *,
    W_shadow: np.ndarray,
    Z_shadow: np.ndarray,
    hidden: Tuple[int, ...] = (1024, 1024),
    recon_act: str = "relu",
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 128,
    seed: int = 0,
    val_frac: float = 0.1,
    normalizer: Optional[WeightNormalizer] = None,
) -> Dict[str, object]:
    """
    Train ReconstructorMLP on (theta -> z) pairs using your stochax trainer (non-DP).
    """
    W_shadow = np.asarray(W_shadow, dtype=np.float32)
    Z_shadow = np.asarray(Z_shadow, dtype=np.float32)
    if W_shadow.ndim != 2 or Z_shadow.ndim != 2:
        raise ValueError("W_shadow and Z_shadow must be 2D arrays.")
    if W_shadow.shape[0] != Z_shadow.shape[0]:
        raise ValueError("W_shadow and Z_shadow must have same number of rows.")

    if normalizer is None:
        mean = W_shadow.mean(axis=0)
        std = W_shadow.std(axis=0) + 1e-6
        normalizer = WeightNormalizer(mean=mean, std=std)

    Wn = normalizer.normalize(W_shadow).astype(np.float32)

    tr_idx, va_idx = _train_val_split_idx(Wn.shape[0], float(val_frac), seed=int(seed))
    Xtr, Ytr = Wn[tr_idx], Z_shadow[tr_idx]
    Xva, Yva = Wn[va_idx], Z_shadow[va_idx]

    Xtrj = jnp.asarray(Xtr)
    Ytrj = jnp.asarray(Ytr)
    Xvaj = jnp.asarray(Xva)
    Yvaj = jnp.asarray(Yva)

    d_in = int(Wn.shape[1])
    d_out = int(Z_shadow.shape[1])

    model = ReconstructorMLP(
        d_in=d_in,
        d_out=d_out,
        hidden=tuple(int(h) for h in hidden),
        key=jr.PRNGKey(int(seed)),
        act=recon_act,
    )
    state = None

    optimizer = optax.adam(float(lr))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    best_model, best_state, tr_hist, va_hist = train(
        model,
        state,
        opt_state,
        optimizer,
        regression_loss,
        Xtrj,
        Ytrj,
        Xvaj,
        Yvaj,
        batch_size=int(batch_size),
        num_epochs=int(epochs),
        patience=int(epochs),
        key=jr.PRNGKey(int(seed + 1)),
        augment_fn=None,
    )

    return {
        "reconstructor": best_model,
        "normalizer": normalizer,
        "train_hist": tr_hist,
        "val_hist": va_hist,
    }


def reconn_reconstruct_targets_eqx(
    *,
    reconstructor: ReconstructorMLP,
    normalizer: WeightNormalizer,
    Z_fixed: np.ndarray,
    y_fixed: np.ndarray,
    Z_targets: np.ndarray,
    y_targets: np.ndarray,
    n_classes: int,
    head_hidden: int = 32,
    head_act: str = "elu",
    head_lr: float = 1e-2,
    head_epochs: int = 30,
    head_batch_size: int = 512,
    known_init: bool = True,
    seed: int = 0,
    dp_cfg: Optional[DPSGDConfig] = None,
    oracle_pool: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    For each target z:
      - train head on D^- ∪ {z} -> theta
      - predict z_hat = RecoNN(theta)
      - compute ||z_hat - z|| and compare to oracle pool threshold (optional)
    """
    Z_fixed = np.asarray(Z_fixed, dtype=np.float32)
    y_fixed = np.asarray(y_fixed, dtype=np.int64)
    Z_targets = np.asarray(Z_targets, dtype=np.float32)
    y_targets = np.asarray(y_targets, dtype=np.int64)

    errs: List[float] = []
    oracle_thresh: List[float] = []
    success_oracle: List[int] = []

    for i in range(Z_targets.shape[0]):
        z = Z_targets[i]
        y = int(y_targets[i])

        Xfull = np.concatenate([Z_fixed, z[None, :]], axis=0)
        yfull = np.concatenate([y_fixed, np.array([y], dtype=np.int64)], axis=0)

        seed_model = int(seed) if known_init else int(seed + 5000 + i)
        seed_train = int(seed)

        mdl = _fit_head_eqx(
            X=Xfull,
            y=yfull,
            n_classes=int(n_classes),
            hidden=int(head_hidden),
            act=str(head_act),
            lr=float(head_lr),
            epochs=int(head_epochs),
            batch_size=int(head_batch_size),
            seed_model=seed_model,
            seed_train=seed_train,
            dp_cfg=dp_cfg,
        )
        theta, _ = eqx_flatten_adapter(mdl)
        th = normalizer.normalize(theta[None, :]).astype(np.float32)
        thj = jnp.asarray(th)

        zhat, _ = reconstructor(thj[0], jr.PRNGKey(0), None)  # (d,)
        zhat_np = np.asarray(zhat).astype(np.float32)

        e = float(np.linalg.norm(zhat_np - z))
        errs.append(e)

        if oracle_pool is not None:
            t = nearest_neighbor_oracle_l2(z, oracle_pool)
            oracle_thresh.append(t)
            success_oracle.append(int(e < t))

    out = {
        "errors_l2": np.asarray(errs, dtype=np.float32),
        "known_init": bool(known_init),
        "dp_used": dp_cfg is not None,
    }
    if oracle_pool is not None:
        out["oracle_thresh"] = np.asarray(oracle_thresh, dtype=np.float32)
        out["oracle_success"] = np.asarray(success_oracle, dtype=np.int64)
    return out


if __name__ == "__main__":
    # Tiny end-to-end smoke test on synthetic data (small sizes).
    rng = np.random.default_rng(0)
    K, d = 3, 8
    Z = rng.normal(size=(200, d)).astype(np.float32)
    y = rng.integers(low=0, high=K, size=(200,), dtype=np.int64)

    # split
    Z_fix, y_fix = Z[:90], y[:90]
    Z_sh, y_sh = Z[90:150], y[90:150]
    Z_tg, y_tg = Z[150:170], y[150:170]

    sh = train_shadow_models_eqx(
        Z_fixed=Z_fix,
        y_fixed=y_fix,
        Z_shadow=Z_sh,
        y_shadow=y_sh,
        n_classes=K,
        head_hidden=16,
        head_epochs=10,
        head_batch_size=128,
        known_init=True,
        seed=0,
        dp_cfg=None,
    )
    rec = train_reconstructor_eqx(
        W_shadow=sh["W_shadow"],
        Z_shadow=sh["Z_shadow"],
        hidden=(128, 128),
        epochs=10,
        batch_size=32,
        seed=0,
        normalizer=sh["normalizer"],
    )
    ev = reconn_reconstruct_targets_eqx(
        reconstructor=rec["reconstructor"],
        normalizer=rec["normalizer"],
        Z_fixed=Z_fix,
        y_fixed=y_fix,
        Z_targets=Z_tg,
        y_targets=y_tg,
        n_classes=K,
        head_hidden=16,
        head_epochs=10,
        head_batch_size=128,
        known_init=True,
        seed=0,
        dp_cfg=None,
        oracle_pool=np.concatenate([Z_fix, Z_sh], axis=0),
    )
    print("mean recon error:", float(ev["errors_l2"].mean()))
    if "oracle_success" in ev:
        print("oracle success rate:", float(ev["oracle_success"].mean()))
    print("[OK] RecoNN pipeline smoke test completed.")
