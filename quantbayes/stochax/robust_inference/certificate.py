# quantbayes/stochax/robust_inference/certificate.py
from __future__ import annotations
import jax.numpy as jnp
from quantbayes.stochax.robust_inference.probits import sigma_x, margin


def kappa_const(n: int, f: int) -> float:
    return (6.0 * f / (n - 2.0 * f)) * (1.0 + f / (n - 2.0 * f))


def certified_top1_preserved(P: jnp.ndarray, f: int) -> bool:
    """
    P: (n,K) honest client probits for a single x.
    Returns True if Theorem 1 condition certifies top-1 under up to f corruptions.
    """
    n = P.shape[0]
    Pbar = jnp.mean(P, axis=0)
    sig = sigma_x(P)
    kap = kappa_const(n, f)
    thresh = 2.0 * (jnp.sqrt(kap * n / (n - f)) + jnp.sqrt(f / (n - f))) * sig
    return (margin(Pbar) > thresh).item()


def certified_top1_with_smoothing(
    P: jnp.ndarray,
    f: int,
    *,
    rho: float,
    alpha: float = 0.0,
    include_bias: bool = True,
) -> bool:
    """
    Optional smoothing-aware certificate (your extension):
      • variance contracts by ρ = ||S - 11^T/n||_2
      • bias term from mixing bounded by 2*sqrt(2)*alpha
    Inputs:
      P: honest probits (n,K) before smoothing
      rho: spectral radius of S - 11^T/n  (0≤rho<1 for DS S)
      alpha: mixing weight if you write S=(1-α)I+αW (set 0 if unknown)
    """
    n = P.shape[0]
    Pbar = jnp.mean(P, axis=0)
    sig = sigma_x(P)
    kap = kappa_const(n, f)
    thresh_var = (
        2.0 * (jnp.sqrt(kap * n / (n - f)) * float(rho) + jnp.sqrt(f / (n - f))) * sig
    )
    bias = 2.0 * jnp.sqrt(2.0) * float(alpha) if include_bias else 0.0
    return (margin(Pbar) > (thresh_var + bias)).item()
