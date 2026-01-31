# quantbayes/stochax/robust_inference/graph_smooth.py
from __future__ import annotations
import jax
import jax.numpy as jnp
from quantbayes.stochax.distributed_training.spectral import (
    T_K_scalar,
    xi_from_interval,
)
from quantbayes.stochax.robust_inference.simplex import (
    project_rows_to_simplex,
)  # single source of truth


def chebyshev_filter_matrix(
    W: jnp.ndarray, K: int, lam_min: float, lam_max: float
) -> jnp.ndarray:
    """
    Build S = T_K(Z(W)) / T_K(ξ) explicitly for small n (server-side).
    W must be symmetric DS; lam_min/max are the disagreement interval.
    """
    n = W.shape[0]
    if K <= 0:
        return jnp.eye(n)
    width = lam_max - lam_min
    a = 2.0 / width
    b = (lam_max + lam_min) / width
    # Recurrence: T_0=I, T_1=Z, T_{k+1}=2 Z T_k - T_{k-1}
    Z = a * W - b * jnp.eye(n)
    T0 = jnp.eye(n)
    if K == 1:
        TK = Z
    else:
        T1 = Z
        Tk_2, Tk_1 = T0, T1
        for _ in range(2, K + 1):
            Tk = 2.0 * (Z @ Tk_1) - Tk_2
            Tk_2, Tk_1 = Tk_1, Tk
        TK = Tk_1
    xi = xi_from_interval(lam_min, lam_max)
    scale = 1.0 / T_K_scalar(K, float(xi))
    return scale * TK


def smooth_probits(P: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    # P: (n,K), S: (n,n) row-stochastic -> returns (n,K)
    return S @ P


def smooth_probits_project(P: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    """P: (n,K), S: (n,n) -- returns (n,K) with each row in Δ^K."""
    Z = S @ P
    return project_rows_to_simplex(Z)  # row-wise Euclidean projection to simplex
