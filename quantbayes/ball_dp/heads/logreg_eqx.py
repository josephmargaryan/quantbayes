# quantbayes/ball_dp/heads/logreg_eqx.py
from __future__ import annotations
from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class LogisticRegressorEqx(eqx.Module):
    w: jnp.ndarray  # (d,)
    b: jnp.ndarray  # ()

    def __init__(self, d_in: int, *, key: jr.PRNGKey, init_scale: float = 1e-2):
        k1, _ = jr.split(key)
        self.w = init_scale * jr.normal(k1, (d_in,))
        self.b = jnp.array(0.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, self.w) + self.b


def logistic_objective(
    model: LogisticRegressorEqx,
    state: Any,
    xb: jnp.ndarray,  # (B,d)
    yb: jnp.ndarray,  # (B,) in {-1,+1}
    key: jr.PRNGKey,
    *,
    lam: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    scores = eqx.filter_vmap(model)(xb)  # (B,)
    # logistic loss: log(1 + exp(-y*s))
    loss = jnp.mean(jnp.logaddexp(0.0, -yb * scores))

    # include bias in L2 penalty => strong convexity in full theta=[w,b]
    reg = 0.5 * lam * (jnp.sum(model.w**2) + model.b**2)
    return loss + reg, {"loss": loss, "reg": reg}


def _binary_accuracy(
    model: LogisticRegressorEqx, X: jnp.ndarray, y_pm1: jnp.ndarray
) -> float:
    scores = jax.vmap(model)(X)
    preds = jnp.where(scores >= 0.0, 1.0, -1.0)
    return float(jnp.mean(preds == y_pm1))


if __name__ == "__main__":
    """
    Demo:
      (1) Baseline binary logistic regression on normalized embeddings
      (2) RFF(embeddings) + binary logistic regression
      (3) DP Gaussian output perturbation for both with Ball-adjacency radius r_policy in embedding space.
          For RFF: r_eff = Lpsi * r_policy with tight Lpsi = sqrt(2/m)*||Omega||_2 (SVD).
    """
    import numpy as np
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx
    from jax.flatten_util import ravel_pytree

    from quantbayes.stochax import train_lbfgs
    from quantbayes.ball_dp.api import dp_release_erm_params_gaussian
    from quantbayes.ball_dp.lz import lz_logistic_binary_bound
    from quantbayes.ball_dp.rff import (
        sample_rff_rbf,
        rff_transform,
        rff_feature_norm_bound,
    )

    # -----------------------------
    # Utilities
    # -----------------------------
    def make_blobs(
        key,
        n_per_class: int,
        num_classes: int,
        d: int,
        std: float = 0.8,
        spread: float = 3.0,
    ):
        keys = jr.split(key, num_classes + 2)
        means = jr.normal(keys[0], (num_classes, d)) * spread

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

    def binary_accuracy(model, X, y_pm1):
        scores = jax.vmap(model)(X)
        preds = jnp.where(scores >= 0.0, 1.0, -1.0)
        return float(jnp.mean(preds == y_pm1))

    def compute_Lpsi_svd(omega: jnp.ndarray, m_rff: int, safety: float = 1e-6) -> float:
        svals = jnp.linalg.svd(omega, compute_uv=False)
        smax = jnp.max(svals)
        return float((1.0 + safety) * jnp.sqrt(2.0 / float(m_rff)) * smax)

    def dp_eval_from_sigma(
        theta_hat: np.ndarray,
        sigma: float,
        unravel,
        static,
        X_eval,
        y_eval,
        num_trials: int = 5,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        accs = []
        for _ in range(num_trials):
            noise = rng.normal(
                loc=0.0, scale=float(sigma), size=theta_hat.shape
            ).astype(theta_hat.dtype)
            theta_noisy = theta_hat + noise
            params_noisy = unravel(jnp.asarray(theta_noisy))
            mdl = eqx.combine(params_noisy, static)
            accs.append(binary_accuracy(mdl, X_eval, y_eval))
        return float(np.mean(accs)), float(np.std(accs))

    # -----------------------------
    # Reproducibility
    # -----------------------------
    key = jr.PRNGKey(0)
    master_key, k_data, k_model0, k_train0, k_rff, k_model1, k_train1 = jr.split(key, 7)

    # -----------------------------
    # Synthetic "embedding" dataset
    # -----------------------------
    num_classes = 2
    d_embed = 64
    n_per_class = 500

    X, y01 = make_blobs(
        k_data, n_per_class=n_per_class, num_classes=num_classes, d=d_embed
    )

    # normalize to enforce public bound ||e||<=1
    X = X / (jnp.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    B_e = 1.0

    # labels {0,1}->{-1,+1}
    y = y01.astype(jnp.float32) * 2.0 - 1.0
    # To go back from {-1,1} -> {0,1} y01 = (y_pm1 + 1.0) / 2.0

    n_train = int(0.8 * X.shape[0])
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    n = int(X_train.shape[0])

    # -----------------------------
    # DP hyperparameters
    # -----------------------------
    r_policy = 0.5
    eps_dp = 2.0
    delta_dp = 1e-6

    lam = 5e-2  # use a reasonable value with conservative bounds

    # ==========================================================
    # (1) Baseline logistic on embeddings
    # ==========================================================
    print("\n=== Baseline: logistic on embeddings ===")
    model0 = LogisticRegressorEqx(d_in=d_embed, key=k_model0)
    state0 = None

    def loss_fn0(mdl, state, xb, yb, key):
        return logistic_objective(mdl, state, xb, yb, key, lam=lam)

    model0_tr, state0_tr, *_ = train_lbfgs(
        model0,
        state0,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=int(X_train.shape[0]),
        num_epochs=80,
        patience=10,
        loss_fn=loss_fn0,
        lambda_spec=0.0,
        lambda_frob=0.0,
        frob_include_bias=True,
        lambda_specnorm=0.0,
        lambda_sob_jac=0.0,
        lambda_sob_kernel=0.0,
        lambda_liplog=0.0,
        key=k_train0,
    )

    acc0 = binary_accuracy(model0_tr, X_val, y_val)
    print(f"Val acc (non-DP): {acc0:.4f}")

    # DP constants (baseline)
    Lz0 = float(lz_logistic_binary_bound(B=B_e, lam=lam, include_bias=True))

    params0, static0 = eqx.partition(model0_tr, eqx.is_inexact_array)
    theta0_jax, unravel0 = ravel_pytree(params0)
    theta0_hat = np.asarray(jax.device_get(theta0_jax), dtype=np.float32)

    out0 = dp_release_erm_params_gaussian(
        theta0_hat,
        lz=Lz0,
        r=r_policy,
        lam=lam,
        n=n,
        eps=eps_dp,
        delta=delta_dp,
        sigma_method="analytic",
    )
    sigma0 = float(out0["sigma"])
    dp_mean0, dp_std0 = dp_eval_from_sigma(
        theta0_hat, sigma0, unravel0, static0, X_val, y_val, num_trials=7, seed=0
    )

    print(f"Val acc (DP, mean±std over noise): {dp_mean0:.4f} ± {dp_std0:.4f}")
    print("Baseline Lz:", Lz0)
    print("Baseline Δ2:", out0["Delta"])
    print("Baseline σ:", sigma0)

    # ==========================================================
    # (2) RFF + logistic
    # ==========================================================
    print("\n=== RFF: logistic on Phi(embeddings) ===")

    m_rff = 2048
    gamma = 1.0
    W_clip = 5.0

    omega, phase = sample_rff_rbf(
        k_rff, d_in=d_embed, m=m_rff, gamma=gamma, clip_omega_norm=W_clip
    )
    X_rff = rff_transform(X, omega, phase)

    Xr_train, yr_train = X_rff[:n_train], y[:n_train]
    Xr_val, yr_val = X_rff[n_train:], y[n_train:]

    model1 = LogisticRegressorEqx(d_in=m_rff, key=k_model1)
    state1 = None

    def loss_fn1(mdl, state, xb, yb, key):
        return logistic_objective(mdl, state, xb, yb, key, lam=lam)

    model1_tr, state1_tr, *_ = train_lbfgs(
        model1,
        state1,
        Xr_train,
        yr_train,
        Xr_val,
        yr_val,
        batch_size=int(Xr_train.shape[0]),
        num_epochs=80,
        patience=10,
        loss_fn=loss_fn1,
        lambda_spec=0.0,
        lambda_frob=0.0,
        frob_include_bias=True,
        lambda_specnorm=0.0,
        lambda_sob_jac=0.0,
        lambda_sob_kernel=0.0,
        lambda_liplog=0.0,
        key=k_train1,
    )

    acc1 = binary_accuracy(model1_tr, Xr_val, yr_val)
    print(f"Val acc (non-DP): {acc1:.4f}")

    # DP constants (RFF)
    B_phi = float(rff_feature_norm_bound())  # sqrt(2)
    Lpsi = compute_Lpsi_svd(omega, m_rff, safety=1e-6)
    r_eff = float(Lpsi) * float(r_policy)

    Lz1 = float(lz_logistic_binary_bound(B=B_phi, lam=lam, include_bias=True))

    params1, static1 = eqx.partition(model1_tr, eqx.is_inexact_array)
    theta1_jax, unravel1 = ravel_pytree(params1)
    theta1_hat = np.asarray(jax.device_get(theta1_jax), dtype=np.float32)

    out1 = dp_release_erm_params_gaussian(
        theta1_hat,
        lz=Lz1,
        r=r_eff,
        lam=lam,
        n=n,
        eps=eps_dp,
        delta=delta_dp,
        sigma_method="analytic",
    )
    sigma1 = float(out1["sigma"])
    dp_mean1, dp_std1 = dp_eval_from_sigma(
        theta1_hat, sigma1, unravel1, static1, Xr_val, yr_val, num_trials=7, seed=1
    )

    print(f"Val acc (DP, mean±std over noise): {dp_mean1:.4f} ± {dp_std1:.4f}")
    print("RFF B_phi:", B_phi)
    print("RFF Lpsi (tight):", Lpsi)
    print("RFF r_eff:", r_eff)
    print("RFF-head Lz:", Lz1)
    print("RFF Δ2:", out1["Delta"])
    print("RFF σ:", sigma1)

    print("\n=== Summary ===")
    print(
        f"Baseline: acc={acc0:.4f}, acc_dp={dp_mean0:.4f}±{dp_std0:.4f}, sigma={sigma0:.6g}"
    )
    print(
        f"RFF:      acc={acc1:.4f}, acc_dp={dp_mean1:.4f}±{dp_std1:.4f}, sigma={sigma1:.6g}"
    )
