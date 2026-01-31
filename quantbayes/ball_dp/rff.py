# quantbayes/ball_dp/rff.py
from __future__ import annotations
from typing import Literal, Optional, Tuple
import math
import jax.numpy as jnp
import jax.random as jr


def sample_rff_rbf(
    key: jr.PRNGKey,
    *,
    d_in: int,
    m: int,
    gamma: float = 1.0,
    clip_omega_norm: Optional[float] = None,
    dtype=jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Random Fourier Features for the RBF kernel k(x,x') = exp(-gamma ||x-x'||^2).

    Sampling:
      omega_k ~ N(0, 2*gamma I)
      b_k ~ Uniform[0, 2pi)

    If clip_omega_norm is set, each row omega_k is norm-clipped to <= clip_omega_norm.
    This gives a deterministic Lipschitz bound Lpsi <= sqrt(2)*clip_omega_norm.
    """
    if d_in <= 0 or m <= 0:
        raise ValueError("d_in and m must be positive.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    k1, k2 = jr.split(key, 2)
    omega = jr.normal(k1, (m, d_in), dtype=dtype) * jnp.sqrt(
        jnp.asarray(2.0 * gamma, dtype=dtype)
    )
    if clip_omega_norm is not None:
        W = jnp.asarray(float(clip_omega_norm), dtype=dtype)
        row_norms = jnp.linalg.norm(omega, axis=1, keepdims=True) + jnp.asarray(
            1e-12, dtype=dtype
        )
        omega = omega * jnp.minimum(1.0, W / row_norms)

    phase = jr.uniform(k2, (m,), minval=0.0, maxval=2.0 * jnp.pi, dtype=dtype)
    return omega, phase


def rff_transform(
    X: jnp.ndarray, omega: jnp.ndarray, phase: jnp.ndarray
) -> jnp.ndarray:
    """
    X:     (N, d_in)
    omega: (m, d_in)
    phase: (m,)
    returns:
      Phi(X): (N, m) with Phi(x)=sqrt(2/m)*cos(omega x + phase).
    """
    m = omega.shape[0]
    proj = X @ omega.T + phase  # (N, m)
    return jnp.sqrt(2.0 / float(m)) * jnp.cos(proj)


def rff_feature_norm_bound() -> float:
    """For Phi(x)=sqrt(2/m)*cos(...), we always have ||Phi(x)||_2 <= sqrt(2)."""
    return math.sqrt(2.0)


def spectral_norm_svd(omega: jnp.ndarray, *, safety: float = 1e-6) -> float:
    """
    Return an (essentially) exact spectral norm ||Omega||_2 via SVD.
    safety: multiply by (1+safety) to avoid tiny FP underestimation.
    """
    s = jnp.linalg.svd(omega, compute_uv=False)
    smax = float(jnp.max(s))
    return float((1.0 + safety) * smax)


def spectral_norm_power_iteration(
    omega: jnp.ndarray,
    *,
    key: jr.PRNGKey,
    num_iters: int = 60,
    eps: float = 1e-12,
) -> float:
    """
    Approximate ||Omega||_2 by power iteration.
    WARNING: may underestimate. Use only for diagnostics or add conservative safeguards.
    """
    v = jr.normal(key, (omega.shape[1],), dtype=omega.dtype)
    v = v / (jnp.linalg.norm(v) + eps)
    for _ in range(num_iters):
        u = omega @ v
        u = u / (jnp.linalg.norm(u) + eps)
        v = omega.T @ u
        v = v / (jnp.linalg.norm(v) + eps)
    return float(jnp.linalg.norm(omega @ v))


def rff_lipschitz_bound(
    omega: jnp.ndarray,
    *,
    method: Literal["clip", "fro", "svd", "power"] = "fro",
    clip_omega_norm: Optional[float] = None,
    key: Optional[jr.PRNGKey] = None,
    safety: float = 1e-6,
) -> float:
    """
    Upper bound Lpsi such that ||Phi(e)-Phi(e')|| <= Lpsi ||e-e'||.

    - "clip": deterministic Lpsi <= sqrt(2)*W (requires clip_omega_norm). Safe but loose.
    - "fro":  safe Lpsi <= sqrt(2/m)*||Omega||_F. Usually looser than SVD but always safe.
    - "svd":  tight Lpsi = sqrt(2/m)*||Omega||_2 computed by SVD. Best for certificates.
    - "power": approximate Lpsi via power iteration. Not automatically safe unless you guard it.
    """
    m = int(omega.shape[0])

    if method == "clip":
        if clip_omega_norm is None:
            raise ValueError("clip_omega_norm must be provided when method='clip'.")
        return math.sqrt(2.0) * float(clip_omega_norm)

    if method == "fro":
        fro = float(jnp.linalg.norm(omega))  # ||Omega||_F
        return math.sqrt(2.0 / float(m)) * fro

    if method == "svd":
        smax = spectral_norm_svd(omega, safety=safety)
        return math.sqrt(2.0 / float(m)) * smax

    if method == "power":
        if key is None:
            raise ValueError("key must be provided when method='power'.")
        s_est = spectral_norm_power_iteration(omega, key=key)
        # NOT guaranteed safe. If you want a certificate, combine with a safe upper bound:
        fro = float(jnp.linalg.norm(omega))
        s_safe = min(
            fro, (1.0 + 1e-2) * s_est
        )  # heuristic guard; prefer "svd" for certificates
        return math.sqrt(2.0 / float(m)) * s_safe

    raise ValueError(f"Unknown method: {method}")
