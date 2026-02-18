# quantbayes/ball_dp/reconstruction/convex_logreg.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.ball_dp.heads.logreg_eqx import LogisticRegressorEqx
from quantbayes.ball_dp.reconstruction.optim_lbfgs_precise import solve_lbfgs_precise


def to_pm1(y: np.ndarray) -> np.ndarray:
    """
    Convert labels to {-1,+1}. Accepts {-1,+1} or {0,1}.
    """
    y = np.asarray(y).reshape(-1)
    u = set(np.unique(y).tolist())
    if u == {-1, 1}:
        return y.astype(np.int64)
    if u == {0, 1}:
        return (2 * y.astype(np.int64) - 1).astype(np.int64)
    raise ValueError(f"Expected binary labels in {{-1,+1}} or {{0,1}}; got {sorted(u)}")


def logreg_grad_sum_np(
    *,
    w: np.ndarray,
    b: float,
    X: np.ndarray,
    y_pm1: np.ndarray,
) -> np.ndarray:
    """
    Sum of per-example gradients for logistic loss (no regularizer) at (w,b):
      grad_i = -y_i * sigmoid(-y_i*(w^T x_i + b)) * [x_i; 1]
    Returns:
      g_sum: shape (d+1,)
    """
    X = np.asarray(X, dtype=np.float64)
    y = to_pm1(y_pm1).astype(np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    b = float(b)

    t = X @ w + b
    coeff = -y / (1.0 + np.exp(y * t))  # sigma(-y t)
    Xtilde = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    g = (coeff[:, None] * Xtilde).sum(axis=0)
    return g.astype(np.float32)


def reconstruct_missing_binary_logreg_from_release(
    *,
    w_release: np.ndarray,
    b_release: float,
    X_minus: np.ndarray,
    y_minus_pm1: np.ndarray,
    lam: float,
    n_total_full: int,
    enforce_bias_coord: bool = True,
    eps: float = 1e-12,
) -> Dict[str, object]:
    """
    Informed adversary reconstruction from released parameters (w_release,b_release)
    using missing-gradient identity:

      g_missing = -lam*n*theta_release - sum_{D^-} grad_i(theta_release)

    For noiseless theta*, this is exact (up to numerical precision).
    """
    lam = float(lam)
    n_total_full = int(n_total_full)

    g_sum_minus = logreg_grad_sum_np(
        w=np.asarray(w_release, dtype=np.float32),
        b=float(b_release),
        X=X_minus,
        y_pm1=y_minus_pm1,
    ).astype(np.float64)

    theta = np.concatenate(
        [
            np.asarray(w_release, dtype=np.float64).reshape(-1),
            np.array([float(b_release)], dtype=np.float64),
        ]
    )
    g_missing = -lam * float(n_total_full) * theta - g_sum_minus

    a_hat = float(g_missing[-1])  # equals -y*sigma(...)
    if abs(a_hat) <= eps:
        y_hat = 1
    else:
        y_hat = -1 if a_hat > 0 else 1

    x_tilde = g_missing / (a_hat + eps)
    if enforce_bias_coord and abs(x_tilde[-1]) > eps:
        x_tilde = x_tilde / x_tilde[-1]

    x_hat = x_tilde[:-1].astype(np.float32)
    return {
        "x_hat": x_hat,
        "y_hat_pm1": int(y_hat),
        "g_missing": g_missing.astype(np.float32),
        "a_hat": float(a_hat),
    }


def fit_logreg_erm_precise_eqx(
    *,
    X: np.ndarray,
    y_pm1: np.ndarray,
    lam: float,
    key: jr.PRNGKey,
    max_iters: int = 800,
    grad_tol: float = 1e-8,
    memory_size: int = 20,
    verbose: bool = False,
) -> Tuple[LogisticRegressorEqx, Dict[str, object]]:
    """
    High-precision ERM solve for reconstruction experiments.
    """
    X = np.asarray(X, dtype=np.float32)
    y = to_pm1(y_pm1).astype(np.int64)
    lam = float(lam)
    if lam <= 0:
        raise ValueError("lam must be > 0.")

    N, d = X.shape
    mdl0 = LogisticRegressorEqx(d_in=d, key=key)

    Xj = jnp.asarray(X)
    yj = jnp.asarray(y, dtype=jnp.float32)

    def obj(m: LogisticRegressorEqx):
        scores = jax.vmap(m)(Xj)  # (N,)
        loss = jnp.mean(jnp.logaddexp(0.0, -yj * scores))
        reg = 0.5 * lam * (jnp.sum(m.w**2) + m.b**2)
        return loss + reg

    res = solve_lbfgs_precise(
        mdl0,
        obj,
        max_iters=int(max_iters),
        grad_tol=float(grad_tol),
        memory_size=int(memory_size),
        verbose=bool(verbose),
        deterministic_objective=True,
    )
    return res.model, {
        "value": float(res.value),
        "grad_norm": float(res.grad_norm),
        "converged": bool(res.converged),
        "n_iters": int(res.n_iters),
        "n": int(N),
        "d": int(d),
    }


if __name__ == "__main__":
    # Sanity: create a separable-ish dataset so optimum is well-defined and solver converges tightly.
    rng = np.random.default_rng(0)
    N, d = 128, 6
    w_true = rng.normal(size=(d,)).astype(np.float32)
    X = rng.normal(size=(N, d)).astype(np.float32)
    y = np.sign(X @ w_true + 0.1 * rng.normal(size=(N,)).astype(np.float32))
    y = np.where(y >= 0, 1, -1).astype(np.int64)

    X_minus, y_minus = X[:-1], y[:-1]
    x_t, y_t = X[-1], y[-1]
    X_full = np.concatenate([X_minus, x_t[None, :]], axis=0)
    y_full = np.concatenate([y_minus, np.array([y_t], dtype=np.int64)], axis=0)

    lam = 0.2
    mdl, info = fit_logreg_erm_precise_eqx(
        X=X_full,
        y_pm1=y_full,
        lam=lam,
        key=jr.PRNGKey(0),
        max_iters=1000,
        grad_tol=1e-8,
    )
    print("ERM solve:", info)

    rec = reconstruct_missing_binary_logreg_from_release(
        w_release=np.asarray(mdl.w),
        b_release=float(mdl.b),
        X_minus=X_minus,
        y_minus_pm1=y_minus,
        lam=lam,
        n_total_full=X_full.shape[0],
    )
    err = float(np.linalg.norm(rec["x_hat"] - x_t))
    print("logreg recon err:", err, "| y_hat:", rec["y_hat_pm1"], "y_true:", int(y_t))
    print(
        "[NOTE] If recon err isn't near 0, try enabling JAX x64 or decreasing grad_tol further."
    )
