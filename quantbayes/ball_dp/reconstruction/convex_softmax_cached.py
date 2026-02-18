# quantbayes/ball_dp/reconstruction/convex_softmax_cached.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from .optim_lbfgs_array import solve_lbfgs_array


def _pack_Wb(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return np.concatenate([W, b[:, None]], axis=1).reshape(-1).astype(np.float32)


def _unpack_theta(theta: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=np.float32).reshape(-1)
    d1 = int(d) + 1
    K = int(theta.size // d1)
    Wtilde = theta.reshape(K, d1)
    W = Wtilde[:, :d]
    b = Wtilde[:, d]
    return W.astype(np.float32), b.astype(np.float32)


def _add_bias_col(X: jnp.ndarray) -> jnp.ndarray:
    ones = jnp.ones((X.shape[0], 1), dtype=X.dtype)
    return jnp.concatenate([X, ones], axis=1)


def softmax_obj_theta(
    theta: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, lam: jnp.ndarray
) -> jnp.ndarray:
    """
    theta: (K*(d+1),)
    X: (N,d)
    y: (N,) int
    lam: scalar
    """
    Xtilde = _add_bias_col(X)  # (N,d+1)
    d1 = Xtilde.shape[1]
    K = theta.shape[0] // d1
    Wtilde = theta.reshape((K, d1))  # (K,d+1)

    logits = Xtilde @ Wtilde.T  # (N,K)
    logZ = jax.nn.logsumexp(logits, axis=1)
    logp_y = logits[jnp.arange(logits.shape[0]), y] - logZ
    loss = -jnp.mean(logp_y)
    reg = 0.5 * lam * jnp.sum(Wtilde * Wtilde)
    return loss + reg


_softmax_obj_jit = jax.jit(softmax_obj_theta)
_softmax_vg_jit = jax.jit(jax.value_and_grad(softmax_obj_theta))


def fit_softmax_erm_cached(
    *,
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    lam: float,
    theta_init: Optional[np.ndarray] = None,
    seed: int = 0,
    # solver knobs
    max_iters: int = 120,
    grad_tol: float = 1e-7,
    stall_patience: int = 30,
    stall_grad_tol: float = 1e-6,
    memory_size: int = 20,
    verbose: bool = False,
    print_every: int = 10,
) -> Dict[str, object]:
    """
    Cached/JIT-stable convex softmax ERM solve.

    IMPORTANT: This does NOT create new jitted functions per call, so it is fast in loops.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    N, d = X.shape
    K = int(num_classes)
    lam_j = jnp.asarray(float(lam), dtype=jnp.float32)

    if theta_init is None:
        rng = np.random.default_rng(int(seed))
        W0 = (1e-2 * rng.normal(size=(K, d))).astype(np.float32)
        b0 = (1e-2 * rng.normal(size=(K,))).astype(np.float32)
        theta0 = _pack_Wb(W0, b0)
    else:
        theta0 = np.asarray(theta_init, dtype=np.float32).reshape(-1)

    Xj = jnp.asarray(X)
    yj = jnp.asarray(y, dtype=jnp.int32)

    def vg(th):
        return _softmax_vg_jit(th, Xj, yj, lam_j)

    def f(th):
        return _softmax_obj_jit(th, Xj, yj, lam_j)

    # warm compile once per shape
    if verbose:
        v0, g0 = vg(jnp.asarray(theta0))
        _ = jax.block_until_ready(v0)

    res = solve_lbfgs_array(
        jnp.asarray(theta0),
        value_and_grad_fn=vg,
        value_fn=f,
        max_iters=int(max_iters),
        grad_tol=float(grad_tol),
        stall_patience=int(stall_patience),
        stall_grad_tol=float(stall_grad_tol),
        memory_size=int(memory_size),
        verbose=bool(verbose),
        print_every=int(print_every),
    )

    theta_hat = np.asarray(res.theta).astype(np.float32)
    W_hat, b_hat = _unpack_theta(theta_hat, d=d)

    return {
        "theta": theta_hat,
        "W": W_hat,
        "b": b_hat,
        "info": {
            "value": float(res.value),
            "grad_norm": float(res.grad_norm),
            "stop_reason": res.stop_reason,
            "n_iters": int(res.n_iters),
            "converged": bool(res.converged),
            "N": int(N),
            "d": int(d),
            "K": int(K),
        },
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, d, K = 200, 6, 4
    W_true = rng.normal(size=(K, d)).astype(np.float32)
    b_true = rng.normal(size=(K,)).astype(np.float32)
    X = rng.normal(size=(N, d)).astype(np.float32)
    y = np.argmax(X @ W_true.T + b_true[None, :], axis=1).astype(np.int64)

    out = fit_softmax_erm_cached(X=X, y=y, num_classes=K, lam=0.2, seed=0, verbose=True)
    print(out["info"])
    print("[OK] convex_softmax_cached smoke.")
