# theory_tools.py
# All-in-one: per-layer spectral/Frobenius norms, network Lipschitz, margin loss,
# PAC-Bayes-style bound evaluation, and safe extractors for our spectral layers.
#
# This file avoids training utilities; it only computes quantities used in theory
# sections / diagnostic experiments.

"""
sigmas, frobs = spectral_frobenius_lists(model, method2d="power", n_iter=20)
L_net = lipschitz_serial(sigmas)
bound = compute_bound_on_train(model, logits_fn, X_train, y_train, gamma=1.0)

"""

from __future__ import annotations
from typing import Any, List, Optional, Sequence, Tuple, Literal

import math
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


# ----------------------------- margin loss -------------------------------------


def margin_loss(logits: jnp.ndarray, y: jnp.ndarray, gamma: float) -> float:
    """
    Empirical γ-margin loss.
      • multiclass: frac_i [ f(x_i)_y_i ≤ max_{j≠y_i} f(x_i)_j + γ ]
      • binary:     frac_i [ (2y_i−1)·logit_i ≤ γ ]
    Accepts:
      - (N,) or (N,1) → binary
      - (N,K) with K>1 → multiclass
    """
    z = logits
    if z.ndim == 2 and z.shape[-1] == 1:
        z = z.reshape(-1)

    if z.ndim == 2 and z.shape[-1] > 1:
        corr = z[jnp.arange(y.shape[0]), y]
        other = jnp.where(
            jnp.arange(z.shape[-1])[None, :] == y[:, None], -jnp.inf, z
        ).max(axis=-1)
        vio = corr <= (other + gamma)
    elif z.ndim == 1:
        signed = jnp.where(y == 1, z, -z)
        vio = signed <= gamma
    else:
        raise ValueError(f"Unsupported logits shape {logits.shape}")
    return float(jnp.mean(vio))


# --------------------- per-layer spectral / frobenius --------------------------


def _max_abs(z: jnp.ndarray) -> float:
    return float(jnp.max(jnp.abs(z)))


def layer_operator_norm(
    layer: Any,
    *,
    method2d: Literal["svd", "power"] = "power",
    n_iter: int = 20,
    key: Any = None,
) -> float:
    """
    Operator norm ||W||_2 for recognized spectral layers.

    - RFFTCirculant1D:  max|H_half|
    - SpectralTokenMixer: max|H_half| * max|gate|
    - RFFTCirculant2D: max_{u,v} σ_max(K[u,v]) via SVD or power-iteration (across all freqs)
    - SpectralDense / SpectralConv2d / Adaptive variants: max|s|
    - Legacy SpectralCirculantLayer / 2d: use get_fourier_coeffs / get_fft_kernel

    Returns float. Raises on unknown layers.
    """
    name = type(layer).__name__
    # New 1D
    if name == "RFFTCirculant1D":
        return _max_abs(layer.H_half)
    # Token mixer
    if name == "SpectralTokenMixer":
        return _max_abs(layer.H_half) * (
            float(jnp.max(jnp.abs(layer.gate))) if hasattr(layer, "gate") else 1.0
        )
    # New 2D
    if name == "RFFTCirculant2D":
        K = layer.K_half  # (Cout,Cin,H,W_half)
        H, W = int(K.shape[-2]), int(K.shape[-1])
        mats = jnp.transpose(K, (2, 3, 0, 1)).reshape(H * W, K.shape[0], K.shape[1])
        if method2d == "svd":
            smax = jax.vmap(lambda M: jnp.linalg.svd(M, compute_uv=False)[0])(mats)
            return float(jnp.max(smax))
        else:
            # batched power iteration
            key = jr.PRNGKey(0) if key is None else key
            key_u, key_v = jr.split(key, 2)
            u = jr.normal(key_u, (mats.shape[0], mats.shape[1]))
            v = jr.normal(key_v, (mats.shape[0], mats.shape[2]))
            u = u / (jnp.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
            v = v / (jnp.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

            def one_iter(uv, _):
                uu, vv = uv
                uu = jax.vmap(lambda M, vv: M @ vv)(mats, vv)
                uu = uu / (jnp.linalg.norm(uu, axis=1, keepdims=True) + 1e-12)
                vv = jax.vmap(
                    lambda M, uu: (M.conj().T if jnp.iscomplexobj(M) else M.T) @ uu
                )(mats, uu)
                vv = vv / (jnp.linalg.norm(vv, axis=1, keepdims=True) + 1e-12)
                return (uu, vv), None

            (u, v), _ = jax.lax.scan(one_iter, (u, v), None, length=n_iter)
            rq = jax.vmap(lambda M, uu, vv: jnp.vdot(uu, M @ vv))(mats, u, v)
            return float(jnp.max(jnp.abs(rq)))

    # SVD-parameterized
    if name in [
        "SpectralDense",
        "AdaptiveSpectralDense",
        "SpectralConv2d",
        "AdaptiveSpectralConv2d",
    ]:
        return float(jnp.max(jnp.abs(layer.s)))

    # Legacy 1D
    if name in ["SpectralCirculantLayer", "AdaptiveSpectralCirculantLayer"]:
        H = layer.get_fourier_coeffs()
        return _max_abs(H)
    # Legacy 2D
    if name in ["SpectralCirculantLayer2d", "AdaptiveSpectralCirculantLayer2d"]:
        K_full = layer.get_fft_kernel()
        K_half = jnp.fft.rfft(K_full, axis=-1, norm="ortho")
        H, W = int(K_half.shape[-2]), int(K_half.shape[-1])
        mats = jnp.transpose(K_half, (2, 3, 0, 1)).reshape(
            H * W, K_half.shape[0], K_half.shape[1]
        )
        smax = jax.vmap(lambda M: jnp.linalg.svd(M, compute_uv=False)[0])(mats)
        return float(jnp.max(smax))

    raise ValueError(f"Unknown layer type for operator norm: {name}")


def layer_frobenius_norm(layer: Any) -> float:
    """
    Frobenius norm of the *linear operator* (not just parameters).
    For circulant layers we use the diagonalization identity:
      ‖W‖_F^2 = sum_freq ||K(freq)||_F^2         (unitary FFT, block-diagonal in freq)
    Implemented efficiently using spatial kernels for 1D/2D circulant:
      ‖W‖_F^2 = N * sum_{h} |k[h]|^2     (1D single-channel)
      ‖W‖_F^2 = H*W * sum_{o,i,h,w} |k_{o,i}[h,w]|^2   (2D multi-channel)
    For SVD-param layers, ‖W‖_F = ||s||_2.
    """
    name = type(layer).__name__
    if name == "RFFTCirculant1D":
        N = int(layer.padded_dim)
        k_spatial = jnp.fft.irfft(layer.H_half, n=N, norm="ortho").real  # (N,)
        return float(jnp.sqrt(N * jnp.sum(k_spatial**2)))

    if name == "RFFTCirculant2D":
        H, W = int(layer.H_pad), int(layer.W_pad)
        k_spatial = jnp.fft.irfft2(
            layer.K_half, s=(H, W), norm="ortho"
        ).real  # (Cout,Cin,H,W)
        return float(jnp.sqrt(H * W * jnp.sum(k_spatial**2)))

    if name in [
        "SpectralDense",
        "AdaptiveSpectralDense",
        "SpectralConv2d",
        "AdaptiveSpectralConv2d",
    ]:
        return float(jnp.linalg.norm(layer.s))

    if name in ["SpectralTokenMixer"]:
        # Conservative: treat per-channel gating as separate linear scaling (max gate factor)
        return float(_max_abs(layer.H_half)) * float(jnp.linalg.norm(layer.gate))

    if name in ["SpectralCirculantLayer", "AdaptiveSpectralCirculantLayer"]:
        N = int(layer.padded_dim)
        H = layer.get_fourier_coeffs()  # full complex
        k_spatial = jnp.fft.ifft(H, n=N, norm="ortho").real
        return float(jnp.sqrt(N * jnp.sum(k_spatial**2)))

    if name in ["SpectralCirculantLayer2d", "AdaptiveSpectralCirculantLayer2d"]:
        K_full = layer.get_fft_kernel()
        H, W = int(K_full.shape[-2]), int(K_full.shape[-1])
        k_spatial = jnp.fft.ifftn(K_full, axes=(-2, -1), norm="ortho").real
        return float(jnp.sqrt(H * W * jnp.sum(k_spatial**2)))

    # Fallback: eqx.nn.Linear
    if isinstance(layer, eqx.nn.Linear):
        W = layer.weight
        return float(jnp.linalg.norm(W))

    # Fallback: eqx.nn.Conv2d (flatten weights)
    if isinstance(layer, eqx.nn.Conv2d):
        W = layer.weight.reshape(layer.out_channels, -1)
        return float(jnp.linalg.norm(W))

    raise ValueError(f"Unknown layer type for Frobenius norm: {name}")


# -------------------- model-wide extraction & Lipschitz ------------------------


def collect_layers(model: Any) -> List[Any]:
    """Preorder scan collecting all leaf modules (heuristic: eqx.Module instances)."""
    layers = []

    def _scan(m):
        if isinstance(m, eqx.Module):
            # record leaf modules only
            children = [v for v in vars(m).values() if isinstance(v, eqx.Module)]
            if not children:
                layers.append(m)
            else:
                for v in vars(m).values():
                    _scan(v)
        elif isinstance(m, (list, tuple)):
            for v in m:
                _scan(v)
        elif isinstance(m, dict):
            for v in m.values():
                _scan(v)

    _scan(model)
    return layers


def spectral_frobenius_lists(
    model: Any,
    *,
    method2d: Literal["svd", "power"] = "power",
    n_iter: int = 20,
    key: Any = None,
) -> Tuple[List[float], List[float]]:
    """Return (sigmas, frobs) over recognized layers, in declaration order."""
    sigmas, frobs = [], []
    for lyr in collect_layers(model):
        try:
            σ = layer_operator_norm(lyr, method2d=method2d, n_iter=n_iter, key=key)
            F = layer_frobenius_norm(lyr)
            sigmas.append(float(σ))
            frobs.append(float(F))
        except Exception:
            # silently skip unrecognized
            pass
    if not sigmas:
        raise ValueError("No recognized linear/spectral layers found.")
    return sigmas, frobs


def lipschitz_serial(sigmas: Sequence[float]) -> float:
    """Lipschitz of serial composition: product of operator norms."""
    p = 1.0
    for s in sigmas:
        p *= max(0.0, float(s))
    return p


def lipschitz_residual(block_sigma: float) -> float:
    """Upper bound for residual y = x + f(x): <= 1 + ||f||_Lip."""
    return 1.0 + max(0.0, float(block_sigma))


# ------------------------- PAC-Bayes-style bound -------------------------------


def pac_bayes_excess(
    sigmas: np.ndarray,
    frobs: np.ndarray,
    *,
    B: float,
    gamma: float,
    m: int,
    delta: float,
) -> float:
    """
    A Neyshabur-style excess term:
      R = sqrt( [ B^2 * d^2 * ∏ σ_i^2 * ∑ (||W_i||_F^2 / σ_i^2) + ln(m/δ) ] / (γ^2 m) )
    where d = number of layers included.
    """
    d = len(sigmas)
    prod_sigma_sq = float(np.prod(sigmas**2))
    sum_ratio = float(np.sum((frobs**2) / (sigmas**2 + 1e-12)))
    numer = (B**2) * (d**2) * prod_sigma_sq * sum_ratio + math.log(
        max(m, 2) / max(delta, 1e-12)
    )
    denom = (gamma**2) * max(m, 1)
    return math.sqrt(max(numer, 0.0) / max(denom, 1e-12))


def compute_bound_on_train(
    model: Any,
    logits_fn,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    *,
    gamma: float = 1.0,
    delta: float = 0.05,
    B: Optional[float] = None,
    method2d: Literal["svd", "power"] = "power",
    n_iter: int = 20,
    key: Any = None,
) -> float:
    """
    Compute min(1, L̂_γ + R) on the *training* set.
      - logits_fn(model, X) -> (N,) or (N,1) or (N,K)
      - B: data radius in input norm; if None, use max ||x||_2 over train.
    """
    logits = logits_fn(model, X_train)
    L_hat = margin_loss(logits, y_train, gamma)

    sigmas, frobs = spectral_frobenius_lists(
        model, method2d=method2d, n_iter=n_iter, key=key
    )
    sigmas = np.asarray(sigmas, dtype=float)
    frobs = np.asarray(frobs, dtype=float)

    if B is None:
        B = float(
            jnp.max(jnp.linalg.norm(X_train.reshape(X_train.shape[0], -1), axis=1))
        )

    R = pac_bayes_excess(
        sigmas, frobs, B=B, gamma=gamma, m=X_train.shape[0], delta=delta
    )
    return float(min(1.0, L_hat + R))
