# norms.py
"""
Exact or high-accuracy spectral and Frobenius norms for any Equinox model.

Public API
----------
spectral_frobenius_lists(
    model: eqx.Module,
    *,
    n_iter: int = 25,
    key: Optional[jax.random.PRNGKey] = None
) -> tuple[list[float], list[float], list[float]]
    Returns (sigmas, frobs, dev21) in layer-order.

    • sigmas[i] = σ₂(W_i)
    • frobs[i]  = ‖W_i‖_F
    • dev21[i]  = ‖Wᵀ‖_{2,1} or same as frob when not used
"""
from __future__ import annotations
from typing import Any, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def _power_iteration(
    W: jnp.ndarray,
    n_iter: int,
    key: jr.PRNGKey,
) -> float:
    """Approximate the top singular value of W via n_iter power-iterations."""
    v = jr.normal(key, (W.shape[1],))
    v = v / (jnp.linalg.norm(v) + 1e-12)
    for _ in range(n_iter):
        u = W @ v
        u = u / (jnp.linalg.norm(u) + 1e-12)
        v = W.T @ u
        v = v / (jnp.linalg.norm(v) + 1e-12)
    return float(jnp.dot(u, W @ v))


def _layer_norms(
    layer: Any,
    *,
    n_iter: int,
    key: jr.PRNGKey,
) -> Optional[Tuple[float, float, float]]:
    """
    For recognized linear modules, returns (σ₂, Frobenius, 2,1-deviation).
    Otherwise returns None.
    """
    from quantbayes.stochax.layers import (
        SpectralDense,
        AdaptiveSpectralDense,
        SpectralCirculantLayer,
        AdaptiveSpectralCirculantLayer,
        SpectralCirculantLayer2d,
        AdaptiveSpectralCirculantLayer2d,
        SpectralConv2d,
        AdaptiveSpectralConv2d,
    )

    # 1) SVD‐param dense
    if isinstance(layer, (SpectralDense, AdaptiveSpectralDense)):
        s = layer.s
        sigma = float(jnp.max(jnp.abs(s)))
        frob = float(jnp.linalg.norm(s))
        return sigma, frob, frob

    # 2) 1-D circulant
    if isinstance(layer, (SpectralCirculantLayer, AdaptiveSpectralCirculantLayer)):
        fft_full = layer.get_fourier_coeffs()
        sigma = float(jnp.max(jnp.abs(fft_full)))
        half = layer.k_half
        vec = jnp.concatenate([layer.w_real, layer.w_imag])
        frob = float(jnp.sqrt(layer.padded_dim / half) * jnp.linalg.norm(vec))
        return sigma, frob, frob

    # 3) 2-D circulant
    if isinstance(layer, (SpectralCirculantLayer2d, AdaptiveSpectralCirculantLayer2d)):
        fftk = layer.get_fft_kernel()  # (C_out, C_in, H, W)
        sigma = float(jnp.max(jnp.abs(fftk)))
        vec = jnp.concatenate(
            [
                layer.w_real.reshape(-1),
                layer.w_imag.reshape(-1),
            ]
        )
        frob = float(jnp.linalg.norm(vec))
        return sigma, frob, frob

    # 4) SVD‐param conv2d
    if isinstance(layer, (SpectralConv2d, AdaptiveSpectralConv2d)):
        s = layer.s
        sigma = float(jnp.max(jnp.abs(s)))
        frob = float(jnp.linalg.norm(s))
        return sigma, frob, frob

    # 5) eqx.nn.Linear
    if isinstance(layer, eqx.nn.Linear):
        W = layer.weight
        sigma = _power_iteration(W, n_iter=n_iter, key=key)
        frob = float(jnp.linalg.norm(W))
        dev21 = float(jnp.linalg.norm(W, axis=0).sum())
        return sigma, frob, dev21

    # 6) eqx.nn.Conv2d fallback
    if isinstance(layer, eqx.nn.Conv2d):
        W = layer.weight.reshape(layer.out_channels, -1)
        sigma = _power_iteration(W, n_iter=n_iter, key=key)
        frob = float(jnp.linalg.norm(W))
        dev21 = float(jnp.linalg.norm(W, axis=0).sum())
        return sigma, frob, dev21

    return None


def spectral_frobenius_lists(
    model: eqx.Module,
    *,
    n_iter: int = 25,
    key: Optional[jr.PRNGKey] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Walk model fields and collect (σ₂, Frobenius, 2,1-dev) for each
    recognized linear-like layer, in declaration order.
    """
    if key is None:
        key = jr.PRNGKey(0)

    sigmas: List[float] = []
    frobs: List[float] = []
    dev21: List[float] = []

    def _scan(mdl: Any):
        out = _layer_norms(mdl, n_iter=n_iter, key=key)
        if out is not None:
            σ, f, d = out
            sigmas.append(σ)
            frobs.append(f)
            dev21.append(d)
        if isinstance(mdl, eqx.Module):
            for v in vars(mdl).values():
                _scan(v)
        elif isinstance(mdl, (list, tuple)):
            for v in mdl:
                _scan(v)
        elif isinstance(mdl, dict):
            for v in mdl.values():
                _scan(v)

    _scan(model)
    if not sigmas:
        raise ValueError("spectral_frobenius_lists: no recognized linear layers found")
    return sigmas, frobs, dev21
