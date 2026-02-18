# quantbayes/ball_dp/reconstruction/convex_softmax.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.ball_dp.heads.softmax_eqx import SoftmaxLinearEqx
from quantbayes.ball_dp.reconstruction.optim_lbfgs_precise import solve_lbfgs_precise


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def softmax_grad_sum_np(
    *, W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    N, d = X.shape
    K = W.shape[0]

    Xtilde = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)
    Wtilde = np.concatenate([W, b[:, None]], axis=1)

    logits = Xtilde @ Wtilde.T
    P = _softmax_np(logits)

    Y = np.zeros((N, K), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    G_sum = (P - Y).T @ Xtilde
    return G_sum.astype(np.float32)


def reconstruct_missing_softmax_from_release(
    *,
    W_release: np.ndarray,
    b_release: np.ndarray,
    X_minus: np.ndarray,
    y_minus: np.ndarray,
    lam: float,
    n_total_full: int,
    enforce_bias_coord: bool = True,
    eps: float = 1e-12,
) -> Dict[str, object]:
    lam = float(lam)
    n_total_full = int(n_total_full)

    W = np.asarray(W_release, dtype=np.float32)
    b = np.asarray(b_release, dtype=np.float32).reshape(-1)

    G_sum_minus = softmax_grad_sum_np(W=W, b=b, X=X_minus, y=y_minus).astype(np.float64)
    Wtilde = np.concatenate(
        [W.astype(np.float64), b.astype(np.float64)[:, None]], axis=1
    )
    G_missing = -lam * float(n_total_full) * Wtilde - G_sum_minus

    a_hat = G_missing[:, -1].copy()
    y_hat = int(np.argmin(a_hat))

    rows = G_missing / (a_hat[:, None] + eps)
    wts = np.abs(a_hat) + eps
    x_tilde = (wts[:, None] * rows).sum(axis=0) / wts.sum()

    if enforce_bias_coord and abs(x_tilde[-1]) > eps:
        x_tilde = x_tilde / x_tilde[-1]

    return {
        "x_hat": x_tilde[:-1].astype(np.float32),
        "y_hat": int(y_hat),
        "a_hat": a_hat.astype(np.float32),
        "G_missing": G_missing.astype(np.float32),
    }


def fit_softmax_erm_precise_eqx(
    *,
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    lam: float,
    key: jr.PRNGKey,
    # production defaults (fast + stable)
    max_iters: int = 600,
    grad_tol: float = 1e-7,
    stall_patience: int = 50,
    stall_grad_tol: float = 1e-6,
    memory_size: int = 20,
    verbose: bool = False,
    print_every: int = 25,
    warmup_compile: bool = True,
) -> Tuple[SoftmaxLinearEqx, Dict[str, object]]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    K = int(num_classes)
    lam = float(lam)

    N, d = X.shape
    mdl0 = SoftmaxLinearEqx(d_in=d, n_classes=K, key=key)

    Xj = jnp.asarray(X)
    yj = jnp.asarray(y, dtype=jnp.int32)

    def obj(m: SoftmaxLinearEqx):
        logits = jax.vmap(m)(Xj)
        logZ = jax.nn.logsumexp(logits, axis=1)
        logp_y = logits[jnp.arange(N), yj] - logZ
        loss = -jnp.mean(logp_y)
        reg = 0.5 * lam * (jnp.sum(m.W**2) + jnp.sum(m.b**2))
        return loss + reg

    if warmup_compile:
        print("[softmax] compiling objective/grad (first run can take a while)...")
        vg0 = jax.jit(jax.value_and_grad(lambda mm: obj(mm)))
        v0, _ = vg0(mdl0)
        _ = jax.block_until_ready(v0)
        print("[softmax] compile done. Solving...")

    res = solve_lbfgs_precise(
        mdl0,
        obj,
        max_iters=int(max_iters),
        grad_tol=float(grad_tol),
        memory_size=int(memory_size),
        verbose=bool(verbose),
        print_every=int(print_every),
        # stall settings
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
        "K": int(K),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, d, K = 200, 6, 4
    W_true = rng.normal(size=(K, d)).astype(np.float32)
    b_true = rng.normal(size=(K,)).astype(np.float32)
    X = rng.normal(size=(N, d)).astype(np.float32)
    logits = X @ W_true.T + b_true[None, :]
    y = np.argmax(logits, axis=1).astype(np.int64)

    X_minus, y_minus = X[:-1], y[:-1]
    x_t, y_t = X[-1], y[-1]
    X_full = np.concatenate([X_minus, x_t[None, :]], axis=0)
    y_full = np.concatenate([y_minus, np.array([y_t], dtype=np.int64)], axis=0)

    lam = 0.2
    mdl, info = fit_softmax_erm_precise_eqx(
        X=X_full,
        y=y_full,
        num_classes=K,
        lam=lam,
        key=jr.PRNGKey(0),
        verbose=True,
        max_iters=600,
        grad_tol=1e-7,
        stall_patience=50,
        stall_grad_tol=1e-6,
    )
    print("ERM solve:", info)

    rec = reconstruct_missing_softmax_from_release(
        W_release=np.asarray(mdl.W),
        b_release=np.asarray(mdl.b),
        X_minus=X_minus,
        y_minus=y_minus,
        lam=lam,
        n_total_full=X_full.shape[0],
    )
    err = float(np.linalg.norm(rec["x_hat"] - x_t))
    print("softmax recon err:", err, "| y_hat:", rec["y_hat"], "y_true:", int(y_t))
