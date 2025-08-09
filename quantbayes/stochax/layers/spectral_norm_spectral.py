# spectral_norm_spectral.py
# Stateful spectral normalisation tailored to custom spectral layers.
#
# Features
# --------
# - Equinox-style State handling (matching eqx.nn.SpectralNorm API).
# - Handles:
#     • 1D spectral-circulant (RFFTCirculant1D): exact σ = max|H_half|.
#     • 2D spectral-circulant (RFFTCirculant2D): σ = max_{u,v} σ_max(K[u,v]),
#         with two strategies:
#         (a) sampled power iteration across a fixed set of frequencies (stateful),
#         (b) exact per-frequency SVD (small channel counts; no state updates).
#     • Token mixer (SpectralTokenMixer): σ = max|H_half| * max|gate|.
#     • SVD-parameterised (SpectralDense/SpectralConv2d/Adaptive variants): σ = max|s|.
#     • Legacy real-imag circulant layers: use get_fourier_coeffs()/get_fft_kernel().
# - Targeted normalisation: divide the minimal set of parameters to enforce ||W||₂ ≤ τ.
#
# Notes
# -----
# - For sampled PI, we keep (u,v) indices fixed across training and store (u,v) vectors
#   in the State. This keeps updates fast and stable.
# - For exact='svd', we compute σ per frequency by SVD (recommended only for small
#   channels or validation-time checks).
#
"""
sn = SpectralNormSpectral(my_layer, target=1.0, method2d="pi-sampled",
                          num_power_iterations=1, num_freq_samples=64,
                          key=jr.PRNGKey(0))
out, state = sn(x, state)  # during training; respects eqx.nn.inference_mode

"""
from __future__ import annotations
from typing import Any, Optional, Literal, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from equinox import tree_at
from equinox.nn import StatefulLayer
from equinox.nn._stateful import State, StateIndex
from equinox.nn._misc import named_scope


# ---- layer type markers (import lazily by name to avoid hard deps) ----
def _isinstance_by_name(obj, names):
    return any(type(obj).__name__ == n for n in names)


def _max_abs(z: jnp.ndarray) -> jnp.ndarray:
    return jnp.max(jnp.abs(z))


def _pi_step(W: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, eps: float):
    # One power iteration for a single matrix W (complex or real)
    # Returns updated (u, v). (Shapes: u: (Cout,), v: (Cin,))
    u = W @ v
    u = u / (jnp.linalg.norm(u) + eps)
    v = (W.conj().T if jnp.iscomplexobj(W) else W.T) @ u
    v = v / (jnp.linalg.norm(v) + eps)
    return u, v


def _gather_freq_mats(K_half: jnp.ndarray, freq_idx: jnp.ndarray) -> jnp.ndarray:
    # K_half: (Cout, Cin, H_pad, W_half) complex
    # freq_idx: (M,2) with [u,v] indices
    def one(idx):
        u, v = idx[0], idx[1]
        return K_half[:, :, u, v]  # (Cout, Cin)

    return jax.vmap(one)(freq_idx)  # (M, Cout, Cin)


class SpectralNormSpectral(StatefulLayer):
    """
    Spectral normalisation wrapper for spectral layers.

    Args:
      layer: an instance of one of the supported spectral layers.
      target: enforce ||W||₂ ≤ target by dividing parameters by scale = max(1, σ/target).
      method2d: strategy for 2D circulant multi-channel layers:
          - "pi-sampled": fixed M frequency samples with power iteration (stateful).
          - "svd": exact per-frequency SVD (no state updates; slower).
      num_power_iterations: power-iteration steps per call (training mode).
      num_freq_samples: M for "pi-sampled". If None with "pi-sampled", defaults to
          min(H_pad * W_half, 64).
      eps: numerical epsilon.
      inference: initial inference flag (toggle with eqx.nn.inference_mode at call).
      key: PRNGKey to sample frequency indices and initialise u/v when needed.

    __call__(x, state, *, key=None, inference=None) -> (y, new_state)
    """

    layer: eqx.Module
    target: float = eqx.field(static=True)
    method2d: Literal["pi-sampled", "svd"] = eqx.field(static=True)
    num_power_iterations: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    inference: bool

    # 2D circulant state (only when method2d == "pi-sampled")
    freq_idx: Optional[jnp.ndarray] = eqx.field(static=True)  # (M,2)
    uv_index: Optional[StateIndex[Tuple[jnp.ndarray, jnp.ndarray]]]  # (u,v) stacked

    def __init__(
        self,
        layer: eqx.Module,
        *,
        target: float = 1.0,
        method2d: Literal["pi-sampled", "svd"] = "pi-sampled",
        num_power_iterations: int = 1,
        num_freq_samples: Optional[int] = None,
        eps: float = 1e-12,
        inference: bool = False,
        key: Any,
    ):
        self.layer = layer
        self.target = float(target)
        self.method2d = method2d
        self.num_power_iterations = int(num_power_iterations)
        self.eps = float(eps)
        self.inference = bool(inference)

        self.freq_idx = None
        self.uv_index = None

        # Setup state if this is a 2D multi-channel circulant
        if _isinstance_by_name(
            layer,
            [
                "RFFTCirculant2D",
                "SpectralCirculantLayer2d",
                "AdaptiveSpectralCirculantLayer2d",
            ],
        ):
            # Determine shapes
            if hasattr(layer, "K_half"):
                C_out, C_in = int(layer.C_out), int(layer.C_in)
                H_pad, W_half = int(layer.H_pad), int(layer.W_half)

                def all_pairs():
                    uu, vv = jnp.meshgrid(
                        jnp.arange(H_pad), jnp.arange(W_half), indexing="ij"
                    )
                    return jnp.stack([uu.reshape(-1), vv.reshape(-1)], axis=1)

            else:
                # legacy: w_real/w_imag + FFT kernel builder
                fftk = layer.get_fft_kernel()  # (C_out,C_in,H_pad,W_pad) complex FULL
                C_out, C_in, H_pad, W_pad = map(int, fftk.shape)
                W_half = (W_pad // 2) + 1
                # Convert to half-plane for indexing convenience
                K_half = jnp.fft.rfft(fftk, axis=-1, norm="ortho")

                def all_pairs():
                    uu, vv = jnp.meshgrid(
                        jnp.arange(H_pad), jnp.arange(W_half), indexing="ij"
                    )
                    return jnp.stack([uu.reshape(-1), vv.reshape(-1)], axis=1)

            Mmax = H_pad * W_half
            if self.method2d == "pi-sampled":
                M = (
                    Mmax
                    if num_freq_samples is None
                    else int(min(num_freq_samples, Mmax))
                )
                # Fix a set of indices (static) and initialise u,v
                all_idx = all_pairs()  # (Mmax,2)
                if M == Mmax:
                    chosen = all_idx
                else:
                    key, sub = jr.split(key)
                    perm = jr.permutation(sub, Mmax)
                    chosen = all_idx[perm[:M]]

                self.freq_idx = chosen  # static
                # init u,v (M,Cout) and (M,Cin)
                key_u, key_v = jr.split(key, 2)
                u0 = jr.normal(key_u, (chosen.shape[0], C_out))
                v0 = jr.normal(key_v, (chosen.shape[0], C_in))
                # normalise
                u0 = u0 / (jnp.linalg.norm(u0, axis=1, keepdims=True) + eps)
                v0 = v0 / (jnp.linalg.norm(v0, axis=1, keepdims=True) + eps)
                self.uv_index = StateIndex((u0, v0))
            else:
                self.freq_idx = None
                self.uv_index = None

    # --------- utilities to compute σ(W) per layer ----------
    def _sigma_vector(self, vec: jnp.ndarray) -> float:
        return float(_max_abs(vec))

    def _sigma_token_mixer(self, layer) -> float:
        # ||Mixer|| ≤ max_g max_k |H_g[k]|  *  max_c |gate[c]|
        sig_h = float(_max_abs(layer.H_half))
        sig_g = float(jnp.max(jnp.abs(layer.gate))) if hasattr(layer, "gate") else 1.0
        return sig_h * sig_g

    def _sigma_2d_svd(self, K_half: jnp.ndarray) -> float:
        # exact: max over (u,v) of top singular value via SVD
        def top_sv(M):
            # M: (Cout,Cin) complex
            # jnp.linalg.svd handles complex. Return s_max.
            s = jnp.linalg.svd(M, compute_uv=False)
            return s[0]

        # vmap over (H_pad,W_half)
        H = K_half.shape[-2]
        W = K_half.shape[-1]
        mats = jnp.transpose(K_half, (2, 3, 0, 1)).reshape(
            H * W, K_half.shape[0], K_half.shape[1]
        )
        svals = jax.vmap(top_sv)(mats)
        return float(jnp.max(svals))

    def _sigma_2d_pi(self, layer, state: State) -> Tuple[float, State]:
        assert self.freq_idx is not None and self.uv_index is not None
        u, v = state.get(self.uv_index)
        # Extract matrices for sampled freqs
        if hasattr(layer, "K_half"):
            K_half = layer.K_half  # (Cout,Cin,H_pad,W_half)
        else:
            K_full = layer.get_fft_kernel()  # (Cout,Cin,H_pad,W_pad)
            # convert to half-plane along last axis
            K_half = jnp.fft.rfft(K_full, axis=-1, norm="ortho")
        mats = _gather_freq_mats(K_half, self.freq_idx)  # (M,Cout,Cin)

        # Run PI updates (training only)
        if not self.inference:

            def step(carry, _):
                uu, vv = carry
                uu, vv = jax.vmap(_pi_step, in_axes=(0, 0, 0, None))(
                    mats, uu, vv, self.eps
                )
                return (uu, vv), None

            (u_new, v_new), _ = jax.lax.scan(
                step, (u, v), None, length=self.num_power_iterations
            )
            state = state.set(self.uv_index, (u_new, v_new))
            u, v = u_new, v_new

        # Rayleigh quotient estimate
        def rq(M, uu, vv):
            return jnp.vdot(uu, M @ vv)

        est = jax.vmap(rq)(mats, u, v)
        sigma = float(jnp.max(jnp.abs(est)))
        return sigma, state

    # --------- apply scaling ----------
    def _apply_scale(self, scale: float):
        # divide named params by `scale` (>=1)
        lyr = self.layer
        name = type(lyr).__name__

        if _isinstance_by_name(lyr, ["RFFTCirculant1D"]):
            new = tree_at(lambda m: m.H_half, lyr, lyr.H_half / scale)
            self.layer = new
            return

        if _isinstance_by_name(lyr, ["SpectralTokenMixer"]):
            new = tree_at(lambda m: m.H_half, lyr, lyr.H_half / scale)
            if hasattr(lyr, "gate"):
                new = tree_at(lambda m: m.gate, new, lyr.gate / scale)
            self.layer = new
            return

        if _isinstance_by_name(lyr, ["RFFTCirculant2D"]):
            new = tree_at(lambda m: m.K_half, lyr, lyr.K_half / scale)
            self.layer = new
            return

        if _isinstance_by_name(
            lyr,
            [
                "SpectralDense",
                "AdaptiveSpectralDense",
                "SpectralConv2d",
                "AdaptiveSpectralConv2d",
            ],
        ):
            new = tree_at(lambda m: m.s, lyr, lyr.s / scale)
            self.layer = new
            return

        # Legacy real-imag circulant layers
        if _isinstance_by_name(
            lyr, ["SpectralCirculantLayer", "AdaptiveSpectralCirculantLayer"]
        ):
            new = tree_at(lambda m: m.w_real, lyr, lyr.w_real / scale)
            new = tree_at(lambda m: m.w_imag, new, lyr.w_imag / scale)
            self.layer = new
            return

        if _isinstance_by_name(
            lyr, ["SpectralCirculantLayer2d", "AdaptiveSpectralCirculantLayer2d"]
        ):
            new = tree_at(lambda m: m.w_real, lyr, lyr.w_real / scale)
            new = tree_at(lambda m: m.w_imag, new, lyr.w_imag / scale)
            self.layer = new
            return

        raise ValueError(f"Unsupported layer type for SpectralNormSpectral: {name}")

    # --------- forward ----------
    @named_scope("SpectralNormSpectral")
    def __call__(
        self,
        x: jnp.ndarray,
        state: State,
        *,
        key: Any = None,
        inference: Optional[bool] = None,
    ):
        if inference is not None:
            self.inference = bool(inference)

        # 1) compute σ
        lyr = self.layer
        if _isinstance_by_name(lyr, ["RFFTCirculant1D"]):
            sigma = self._sigma_vector(lyr.H_half)

        elif _isinstance_by_name(lyr, ["SpectralTokenMixer"]):
            sigma = self._sigma_token_mixer(lyr)

        elif _isinstance_by_name(lyr, ["RFFTCirculant2D"]):
            if self.method2d == "svd":
                sigma = self._sigma_2d_svd(lyr.K_half)
            else:
                sigma, state = self._sigma_2d_pi(lyr, state)

        elif _isinstance_by_name(
            lyr,
            [
                "SpectralDense",
                "AdaptiveSpectralDense",
                "SpectralConv2d",
                "AdaptiveSpectralConv2d",
            ],
        ):
            sigma = float(jnp.max(jnp.abs(lyr.s)))

        elif _isinstance_by_name(
            lyr, ["SpectralCirculantLayer", "AdaptiveSpectralCirculantLayer"]
        ):
            H = lyr.get_fourier_coeffs()
            sigma = float(_max_abs(H))

        elif _isinstance_by_name(
            lyr, ["SpectralCirculantLayer2d", "AdaptiveSpectralCirculantLayer2d"]
        ):
            K_full = lyr.get_fft_kernel()  # complex (Cout,Cin,H,W)
            if self.method2d == "svd":
                # convert to half-plane along last axis
                K_half = jnp.fft.rfft(K_full, axis=-1, norm="ortho")
                sigma = self._sigma_2d_svd(K_half)
            else:
                # build half-plane for sampled PI
                H_pad = K_full.shape[-2]
                W_pad = K_full.shape[-1]
                K_half = jnp.fft.rfft(
                    K_full, axis=-1, norm="ortho"
                )  # (Cout,Cin,H_pad,W_half)
                if self.freq_idx is None:
                    # initialise indices and state on first call
                    W_half = K_half.shape[-1]
                    uu, vv = jnp.meshgrid(
                        jnp.arange(H_pad), jnp.arange(W_half), indexing="ij"
                    )
                    all_idx = jnp.stack([uu.reshape(-1), vv.reshape(-1)], axis=1)
                    M = min(all_idx.shape[0], 64)
                    key_u, key_v = (
                        jr.split(jr.PRNGKey(0), 2) if key is None else jr.split(key, 2)
                    )
                    perm = jr.permutation(key_u, all_idx.shape[0])
                    chosen = all_idx[perm[:M]]
                    self.freq_idx = chosen
                    C_out, C_in = int(K_full.shape[0]), int(K_full.shape[1])
                    u0 = jr.normal(key_v, (M, C_out))
                    v0 = jr.normal(key_v, (M, C_in))
                    u0 = u0 / (jnp.linalg.norm(u0, axis=1, keepdims=True) + self.eps)
                    v0 = v0 / (jnp.linalg.norm(v0, axis=1, keepdims=True) + self.eps)
                    self.uv_index = StateIndex((u0, v0))

                # fabricate a proxy layer-like object to reuse the PI path
                class _Proxy:
                    pass

                proxy = _Proxy()
                proxy.K_half = K_half
                sigma, state = self._sigma_2d_pi(proxy, state)

        else:
            raise ValueError(f"Unsupported layer type: {type(lyr).__name__}")

        # 2) scale if needed
        scale = max(1.0, float(sigma) / self.target)
        if scale > 1.0:
            self._apply_scale(scale)

        # 3) forward through wrapped layer (pass state through unchanged)
        out = self.layer(x)
        return out, state
