# quantbayes/ball_dp/heads/sqhinge_eqx.py
from __future__ import annotations
from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class SquaredHingeSVMEqx(eqx.Module):
    w: jnp.ndarray  # (d,)
    b: jnp.ndarray  # ()

    def __init__(self, d_in: int, *, key: jr.PRNGKey, init_scale: float = 1e-2):
        k1, _ = jr.split(key)
        self.w = init_scale * jr.normal(k1, (d_in,))
        self.b = jnp.array(0.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, self.w) + self.b


def squared_hinge_objective(
    model: SquaredHingeSVMEqx,
    state: Any,
    xb: jnp.ndarray,  # (B,d)
    yb: jnp.ndarray,  # (B,) in {-1,+1}
    key: jr.PRNGKey,
    *,
    lam: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    # vmap fix: use eqx.filter_vmap or jax.vmap
    scores = eqx.filter_vmap(model)(xb)  # (B,)

    margin = 1.0 - yb * scores
    loss = jnp.mean(jnp.maximum(0.0, margin) ** 2)

    # include bias in L2 penalty => strong convexity in full theta=[w,b]
    reg = 0.5 * lam * (jnp.sum(model.w**2) + model.b**2)

    return loss + reg, {"loss": loss, "reg": reg}


if __name__ == "__main__":
    import numpy as np
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx
    from jax.flatten_util import ravel_pytree

    from quantbayes.stochax import train_lbfgs
    from quantbayes.ball_dp.heads.sqhinge_eqx import (
        SquaredHingeSVMEqx,
        squared_hinge_objective,
    )
    from quantbayes.ball_dp.api import dp_release_erm_params_gaussian
    from quantbayes.ball_dp.lz import lz_squared_hinge_bound

    key = jr.PRNGKey(0)
    master_key, k_data, k_model, k_train = jr.split(key, 4)

    def make_blobs(key, n_per_class: int, num_classes: int, d: int, std: float = 0.7):
        keys = jr.split(key, num_classes + 2)
        means = jr.normal(keys[0], (num_classes, d)) * 3.0

        Xs, ys = [], []
        for c in range(num_classes):
            Xc = means[c] + std * jr.normal(keys[c + 1], (n_per_class, d))
            yc = jnp.full((n_per_class,), c, dtype=jnp.int32)
            Xs.append(Xc)
            ys.append(yc)

        X = jnp.concatenate(Xs, axis=0)
        y = jnp.concatenate(ys, axis=0)
        perm = jr.permutation(keys[-1], X.shape[0])
        return X[perm], y[perm]

    num_classes = 2
    input_dim = 4
    n_per_class = 300

    X, y01 = make_blobs(
        k_data, n_per_class=n_per_class, num_classes=num_classes, d=input_dim
    )

    # Enforce public bound B via normalization (recommended for your theory)
    X = X / (jnp.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    B = 1.0

    # Convert labels {0,1} -> {-1,+1}
    y = y01.astype(jnp.float32) * 2.0 - 1.0

    n_train = int(0.8 * X.shape[0])
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    model = SquaredHingeSVMEqx(d_in=X_train.shape[1], key=k_model)
    state = None

    lam = 1e-2  # this must match the training objective

    def loss_fn(mdl, state, xb, yb, key):
        return squared_hinge_objective(mdl, state, xb, yb, key, lam=lam)

    model_trained, state_trained, train_hist, val_hist = train_lbfgs(
        model,
        state,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=int(X_train.shape[0]),  # full-batch for ERM-faithfulness
        num_epochs=50,
        patience=10,
        loss_fn=loss_fn,
        lambda_spec=0.0,
        lambda_frob=0.0,
        lambda_specnorm=0.0,
        lambda_sob_jac=0.0,
        lambda_sob_kernel=0.0,
        lambda_liplog=0.0,
        key=k_train,
    )

    # ---- DP release parameters (example values) ----
    r = 0.5
    eps_dp = 2.0
    delta_dp = 1e-6
    n = int(X_train.shape[0])

    Lz = float(lz_squared_hinge_bound(B=B, lam=lam, include_bias=True))

    params, static = eqx.partition(model_trained, eqx.is_inexact_array)
    theta_jax, unravel = ravel_pytree(params)
    theta_hat = np.asarray(jax.device_get(theta_jax), dtype=np.float32)

    out = dp_release_erm_params_gaussian(
        theta_hat,
        lz=Lz,
        r=r,
        lam=lam,
        n=n,
        eps=eps_dp,
        delta=delta_dp,
        sigma_method="analytic",
    )

    theta_private = out["params_noisy"]
    params_private = unravel(jnp.asarray(theta_private))
    dp_model = eqx.combine(params_private, static)

    print("DP release complete.")
    print("Sensitivity Δ2:", out["Delta"])
    print("Noise σ:", out["sigma"])
