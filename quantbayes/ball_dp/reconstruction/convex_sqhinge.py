# quantbayes/ball_dp/reconstruction/convex_sqhinge.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.ball_dp.heads.sqhinge_eqx import SquaredHingeSVMEqx
from quantbayes.ball_dp.reconstruction.optim_lbfgs_precise import solve_lbfgs_precise


def to_pm1(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    u = set(np.unique(y).tolist())
    if u == {-1, 1}:
        return y.astype(np.int64)
    if u == {0, 1}:
        return (2 * y.astype(np.int64) - 1).astype(np.int64)
    raise ValueError(f"Expected binary labels in {{-1,+1}} or {{0,1}}; got {sorted(u)}")


def sqhinge_grad_sum_np(
    *, w: np.ndarray, b: float, X: np.ndarray, y_pm1: np.ndarray
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = to_pm1(y_pm1).astype(np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    b = float(b)

    s = X @ w + b
    margin = 1.0 - y * s
    active = (margin > 0).astype(np.float64)
    coeff = -2.0 * y * margin * active

    Xtilde = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    return (coeff[:, None] * Xtilde).sum(axis=0).astype(np.float32)


def reconstruct_missing_sqhinge_from_release(
    *,
    w_release: np.ndarray,
    b_release: float,
    X_minus: np.ndarray,
    y_minus_pm1: np.ndarray,
    lam: float,
    n_total_full: int,
    eps: float = 1e-12,
    zero_tol: float = 1e-10,
) -> Dict[str, object]:
    lam = float(lam)
    n_total_full = int(n_total_full)

    w = np.asarray(w_release, dtype=np.float64).reshape(-1)
    b = float(b_release)
    theta = np.concatenate([w, np.array([b], dtype=np.float64)], axis=0)

    g_sum_minus = sqhinge_grad_sum_np(w=w, b=b, X=X_minus, y_pm1=y_minus_pm1).astype(
        np.float64
    )
    g_missing = -lam * float(n_total_full) * theta - g_sum_minus

    gnorm = float(np.linalg.norm(g_missing))
    if gnorm <= float(zero_tol):
        return {
            "support_vector": False,
            "g_missing_norm": gnorm,
            "g_missing": g_missing.astype(np.float32),
        }

    a_hat = float(g_missing[-1])
    if abs(a_hat) <= eps:
        return {
            "support_vector": False,
            "g_missing_norm": gnorm,
            "g_missing": g_missing.astype(np.float32),
            "note": "a_hat ~ 0",
        }

    y_hat = -1 if a_hat > 0 else 1
    x_tilde = g_missing / (a_hat + eps)
    if abs(x_tilde[-1]) > eps:
        x_tilde = x_tilde / x_tilde[-1]

    return {
        "support_vector": True,
        "x_hat": x_tilde[:-1].astype(np.float32),
        "y_hat_pm1": int(y_hat),
        "g_missing_norm": gnorm,
        "a_hat": a_hat,
    }


def fit_sqhinge_erm_precise_eqx(
    *,
    X: np.ndarray,
    y_pm1: np.ndarray,
    lam: float,
    key: jr.PRNGKey,
    max_iters: int = 800,
    grad_tol: float = 1e-7,
    stall_patience: int = 50,
    stall_grad_tol: float = 1e-6,
    memory_size: int = 20,
    verbose: bool = False,
    print_every: int = 25,
    warmup_compile: bool = True,
) -> Tuple[SquaredHingeSVMEqx, Dict[str, object]]:
    X = np.asarray(X, dtype=np.float32)
    y = to_pm1(y_pm1).astype(np.int64)
    lam = float(lam)

    N, d = X.shape
    mdl0 = SquaredHingeSVMEqx(d_in=d, key=key)

    Xj = jnp.asarray(X)
    yj = jnp.asarray(y, dtype=jnp.float32)

    def obj(m: SquaredHingeSVMEqx):
        scores = jax.vmap(m)(Xj)
        margin = 1.0 - yj * scores
        loss = jnp.mean(jnp.maximum(0.0, margin) ** 2)
        reg = 0.5 * lam * (jnp.sum(m.w**2) + m.b**2)
        return loss + reg

    if warmup_compile:
        print("[sqhinge] compiling objective/grad...")
        vg0 = jax.jit(jax.value_and_grad(lambda mm: obj(mm)))
        v0, _ = vg0(mdl0)
        _ = jax.block_until_ready(v0)
        print("[sqhinge] compile done. Solving...")

    res = solve_lbfgs_precise(
        mdl0,
        obj,
        max_iters=int(max_iters),
        grad_tol=float(grad_tol),
        memory_size=int(memory_size),
        verbose=bool(verbose),
        print_every=int(print_every),
        min_iters=25,
        stall_patience=int(stall_patience),
        stall_grad_tol=float(stall_grad_tol),
        value_tol=1e-12,
    )

    return res.model, {
        "value": float(res.value),
        "grad_norm": float(res.grad_norm),
        "converged": bool(res.converged),
        "stop_reason": res.stop_reason,
        "n_iters": int(res.n_iters),
        "n": int(N),
        "d": int(d),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, d = 120, 6
    X = rng.normal(size=(N, d)).astype(np.float32)
    y = to_pm1((rng.random(size=(N,)) > 0.5).astype(np.int64))

    X_minus, y_minus = X[:-1], y[:-1]
    x_t, y_t = X[-1], y[-1]
    X_full = np.concatenate([X_minus, x_t[None, :]], axis=0)
    y_full = np.concatenate([y_minus, np.array([y_t], dtype=np.int64)], axis=0)

    lam = 10.0  # makes it more likely target is a support vector
    mdl, info = fit_sqhinge_erm_precise_eqx(
        X=X_full, y_pm1=y_full, lam=lam, key=jr.PRNGKey(0), verbose=True, max_iters=800
    )
    print("ERM solve:", info)

    rec = reconstruct_missing_sqhinge_from_release(
        w_release=np.asarray(mdl.w),
        b_release=float(mdl.b),
        X_minus=X_minus,
        y_minus_pm1=y_minus,
        lam=lam,
        n_total_full=X_full.shape[0],
    )
    print("support_vector:", rec["support_vector"])
    if rec["support_vector"]:
        err = float(np.linalg.norm(rec["x_hat"] - x_t))
        print(
            "sqhinge recon err:", err, "| y_hat:", rec["y_hat_pm1"], "y_true:", int(y_t)
        )
    else:
        print(
            "[NOTE] Not a support vector => exact reconstruction impossible (by theorem)."
        )
