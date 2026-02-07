# quantbayes/stochax/vae/pk/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional

import equinox as eqx
import jax.numpy as jnp


class FeatureMap(Protocol):
    out_dim: int

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """Map z -> u. For batched z (B,D), return (B,M)."""
        ...

    def vjp(self, z: jnp.ndarray, g_u: jnp.ndarray) -> jnp.ndarray:
        """Compute J_F(z)^T g_u. Shapes: z(B,D), g_u(B,M) -> (B,D)."""
        ...

    def prior_score_u(self, u: jnp.ndarray) -> Optional[jnp.ndarray]:
        """If available, analytic score of u under z~N(0,I) pushforward. Return (B,M) or None."""
        ...


@dataclass
class IdentityFeatureMap(eqx.Module):
    latent_dim: int = eqx.static_field()

    @property
    def out_dim(self) -> int:
        return int(self.latent_dim)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z)
        if z.ndim == 1:
            return z[None, :]
        return z

    def vjp(self, z: jnp.ndarray, g_u: jnp.ndarray) -> jnp.ndarray:
        # For identity, J^T g = g
        g_u = jnp.asarray(g_u)
        if g_u.ndim == 1:
            return g_u[None, :]
        return g_u

    def prior_score_u(self, u: jnp.ndarray) -> jnp.ndarray:
        # u ~ N(0,I) if z~N(0,I) and u=z
        u = jnp.asarray(u)
        if u.ndim == 1:
            u = u[None, :]
        return -u


@dataclass
class LinearFeatureMap(eqx.Module):
    """
    u = z @ A^T + b
    A: (M,D), b: (M,)
    Under z~N(0,I), u ~ N(b, A A^T). Analytic score available (precomputes Sigma^{-1}).
    """

    A: jnp.ndarray  # (M,D)
    b: jnp.ndarray  # (M,)
    eps: float = 1e-6

    _Sigma_inv: jnp.ndarray = eqx.field(init=False)  # (M,M)

    def __post_init__(self):
        A = jnp.asarray(self.A)
        M = A.shape[0]
        Sigma = A @ A.T + self.eps * jnp.eye(M, dtype=A.dtype)
        self._Sigma_inv = jnp.linalg.inv(Sigma)

    @property
    def out_dim(self) -> int:
        return int(self.A.shape[0])

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z)
        if z.ndim == 1:
            z = z[None, :]
        return z @ self.A.T + self.b[None, :]

    def vjp(self, z: jnp.ndarray, g_u: jnp.ndarray) -> jnp.ndarray:
        # J is A, so J^T g = g @ A
        z = jnp.asarray(z)
        g_u = jnp.asarray(g_u)
        if z.ndim == 1:
            z = z[None, :]
        if g_u.ndim == 1:
            g_u = g_u[None, :]
        return g_u @ self.A  # (B,D)

    def prior_score_u(self, u: jnp.ndarray) -> jnp.ndarray:
        u = jnp.asarray(u)
        if u.ndim == 1:
            u = u[None, :]
        # score of N(b, Sigma): -Sigma^{-1}(u-b)
        return -(u - self.b[None, :]) @ self._Sigma_inv.T


@dataclass
class NormFeatureMap(eqx.Module):
    """
    u = ||z||_2 (scalar). No analytic pushforward score supplied (non-Gaussian).
    """

    eps: float = 1e-8

    @property
    def out_dim(self) -> int:
        return 1

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z)
        if z.ndim == 1:
            z = z[None, :]
        n = jnp.sqrt(jnp.sum(z * z, axis=-1, keepdims=True) + self.eps)  # (B,1)
        return n

    def vjp(self, z: jnp.ndarray, g_u: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z)
        if z.ndim == 1:
            z = z[None, :]
        g_u = jnp.asarray(g_u)
        if g_u.ndim == 1:
            g_u = g_u.reshape((1, 1))
        n = jnp.sqrt(jnp.sum(z * z, axis=-1, keepdims=True) + self.eps)
        return g_u * (z / n)

    def prior_score_u(self, u: jnp.ndarray):
        return None
