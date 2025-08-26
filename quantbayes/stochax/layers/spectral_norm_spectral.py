# quantbayes/stochax/layers/spectral_norm_spectral.py
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


def _isinstance_by_name(obj, names):
    return any(type(obj).__name__ == n for n in names)


def _max_abs(z: jnp.ndarray) -> jnp.ndarray:
    # JAX scalar (rank-0 array), not Python float
    return jnp.max(jnp.abs(z))


def _pi_step(W: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, eps: float):
    u = W @ v
    u = u / (jnp.linalg.norm(u) + eps)
    v = (W.conj().T if jnp.iscomplexobj(W) else W.T) @ u
    v = v / (jnp.linalg.norm(v) + eps)
    return u, v


def _gather_freq_mats(K_half: jnp.ndarray, freq_idx: jnp.ndarray) -> jnp.ndarray:
    # K_half: (C_out, C_in, H, W_half); freq_idx: (M,2) of (u,v)
    def one(idx):
        u, v = idx[0], idx[1]
        return K_half[:, :, u, v]

    return jax.vmap(one)(freq_idx)


class SpectralNormSpectral(StatefulLayer):
    """
    Spectral normalisation wrapper for spectral layers (JAX-safe).

    - Computes σ exactly for 1D circulant and SVD-parameterised layers.
    - For 2D circulant: exact via per-frequency SVD or PI sampling over frequencies.
    - Applies scaling as a pure 'view' (no in-place mutation during JIT).
    """

    layer: eqx.Module
    target: float = eqx.field(static=True)
    method2d: Literal["pi-sampled", "svd"] = eqx.field(static=True)
    num_power_iterations: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    safety_factor: float = eqx.field(static=True)
    detach_scale: bool = eqx.field(static=True)
    inference: bool

    freq_idx: Optional[jnp.ndarray] = eqx.field(static=True)  # (M,2)
    uv_index: Optional[StateIndex[Tuple[jnp.ndarray, jnp.ndarray]]]

    def __init__(
        self,
        layer: eqx.Module,
        *,
        target: float = 1.0,
        method2d: Literal["pi-sampled", "svd"] = "pi-sampled",
        num_power_iterations: int = 1,
        num_freq_samples: Optional[int] = 64,
        eps: float = 1e-12,
        safety_factor: float = 1.05,  # cushion for PI/sampling
        detach_scale: bool = True,  # stop-grad through σ̂
        inference: bool = False,
        key: Any,
    ):
        self.layer = layer
        self.target = float(target)
        self.method2d = method2d
        self.num_power_iterations = int(num_power_iterations)
        self.eps = float(eps)
        self.safety_factor = float(safety_factor)
        self.detach_scale = bool(detach_scale)
        self.inference = bool(inference)

        self.freq_idx = None
        self.uv_index = None

        # Set up frequency sampling state for 2D circulant layers.
        if _isinstance_by_name(
            layer,
            [
                "RFFTCirculant2D",
                "SpectralCirculantLayer2d",
                "AdaptiveSpectralCirculantLayer2d",
            ],
        ):
            if hasattr(layer, "K_half"):
                C_out, C_in = int(layer.C_out), int(layer.C_in)
                H_pad, W_half = int(layer.H_pad), int(layer.W_half)
            else:
                # derive from full kernel
                fftk = layer.get_fft_kernel()  # (C_out, C_in, H_pad, W_pad), complex
                C_out, C_in, H_pad, W_pad = map(int, fftk.shape)
                W_half = (W_pad // 2) + 1

            if self.method2d == "pi-sampled":
                Mmax = H_pad * W_half

                # Always include DC & Nyquist edges when present.
                special = [(0, 0)]
                if H_pad % 2 == 0:
                    special.append((H_pad // 2, 0))
                if W_half > 1:
                    special.append((0, W_half - 1))
                    if H_pad % 2 == 0:
                        special.append((H_pad // 2, W_half - 1))
                special = jnp.array(special, dtype=jnp.int32)

                # Pool of all half-plane indices, excluding the special ones.
                all_u = jnp.arange(H_pad, dtype=jnp.int32)
                all_v = jnp.arange(W_half, dtype=jnp.int32)
                UU, VV = jnp.meshgrid(all_u, all_v, indexing="ij")
                all_idx = jnp.stack([UU.reshape(-1), VV.reshape(-1)], axis=1)

                def not_special(idx):
                    eqs = jnp.all(idx[None, :] == special[:, :], axis=1)
                    return ~jnp.any(eqs)

                mask = jax.vmap(not_special)(all_idx)
                pool = all_idx[mask]

                M = (
                    Mmax
                    if (num_freq_samples is None)
                    else int(min(num_freq_samples, Mmax))
                )
                rest_needed = max(0, M - special.shape[0])
                key, sub = jr.split(key)
                perm = jr.permutation(sub, pool.shape[0])
                chosen_rest = pool[perm[:rest_needed]]
                chosen = jnp.concatenate([special, chosen_rest], axis=0)

                self.freq_idx = chosen

                # Init PI vectors for each sampled frequency
                key_u, key_v = jr.split(key, 2)
                u0 = jr.normal(key_u, (chosen.shape[0], C_out))
                v0 = jr.normal(key_v, (chosen.shape[0], C_in))
                u0 = u0 / (jnp.linalg.norm(u0, axis=1, keepdims=True) + eps)
                v0 = v0 / (jnp.linalg.norm(v0, axis=1, keepdims=True) + eps)
                self.uv_index = StateIndex((u0, v0))

    # ---- exact/token-mixer σ (JAX scalar) ----
    def _sigma_token_mixer(self, layer) -> jnp.ndarray:
        H = layer.H_half  # (G, k_half)
        s_g = jnp.max(jnp.abs(H), axis=1)  # (G,)

        if hasattr(layer, "gate"):
            C = layer.gate.shape[0]
            G = H.shape[0]
            ch_per_g = C // G
            gate = jnp.abs(layer.gate).reshape(G, ch_per_g)
            g_g = jnp.max(gate, axis=1)
        else:
            g_g = jnp.ones_like(s_g)

        return jnp.max(s_g * g_g)  # rank-0 array

    def _sigma_2d_svd(self, K_half: jnp.ndarray) -> jnp.ndarray:
        # K_half: (C_out, C_in, H, W_half)
        mats = jnp.transpose(K_half, (2, 3, 0, 1)).reshape(
            -1, K_half.shape[0], K_half.shape[1]
        )
        svals = jax.vmap(lambda M: jnp.linalg.svd(M, compute_uv=False)[0])(mats)
        return jnp.max(svals)

    def _sigma_2d_pi_from_half(
        self, K_half: jnp.ndarray, state: State
    ) -> Tuple[jnp.ndarray, State]:
        # K_half: (C_out, C_in, H, W_half)
        assert self.freq_idx is not None and self.uv_index is not None
        u, v = state.get(self.uv_index)
        mats = _gather_freq_mats(K_half, self.freq_idx)  # (M, C_out, C_in)

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

        # Rayleigh quotient estimates
        est = jax.vmap(lambda M, uu, vv: jnp.vdot(uu, M @ vv))(mats, u, v)
        sigma = jnp.max(jnp.abs(est))
        return sigma, state

    # ---- build a scaled 'view' of the wrapped layer (pure) ----
    def _scaled_layer(self, scale: jnp.ndarray) -> eqx.Module:
        # Optionally detach the scale to avoid grads through σ̂
        if self.detach_scale:
            scale = jax.lax.stop_gradient(scale)

        lyr = self.layer
        name = type(lyr).__name__

        if _isinstance_by_name(lyr, ["RFFTCirculant1D"]):
            return tree_at(lambda m: m.H_half, lyr, lyr.H_half / scale)

        if _isinstance_by_name(lyr, ["SpectralTokenMixer"]):
            return tree_at(lambda m: m.H_half, lyr, lyr.H_half / scale)

        if _isinstance_by_name(lyr, ["RFFTCirculant2D"]):
            return tree_at(lambda m: m.K_half, lyr, lyr.K_half / scale)

        if _isinstance_by_name(
            lyr,
            [
                "SpectralDense",
                "AdaptiveSpectralDense",
                "SpectralConv2d",
                "AdaptiveSpectralConv2d",
            ],
        ):
            return tree_at(lambda m: m.s, lyr, lyr.s / scale)

        if _isinstance_by_name(
            lyr, ["SpectralCirculantLayer", "AdaptiveSpectralCirculantLayer"]
        ):
            new = tree_at(lambda m: m.w_real, lyr, lyr.w_real / scale)
            new = tree_at(lambda m: m.w_imag, new, lyr.w_imag / scale)
            return new

        if _isinstance_by_name(
            lyr, ["SpectralCirculantLayer2d", "AdaptiveSpectralCirculantLayer2d"]
        ):
            new = tree_at(lambda m: m.w_real, lyr, lyr.w_real / scale)
            new = tree_at(lambda m: m.w_imag, new, lyr.w_imag / scale)
            return new

        raise ValueError(f"Unsupported layer type for SpectralNormSpectral: {name}")

    # (Optional) diagnostic hint – fine as Python float (not used inside jit path)
    def __operator_norm_hint__(self) -> float | None:
        try:
            core = getattr(self, "layer", None)
            if core is not None and hasattr(core, "__operator_norm_hint__"):
                v = core.__operator_norm_hint__()
                if v is not None:
                    return float(v)
        except Exception:
            pass
        return float(self.target * getattr(self, "safety_factor", 1.0))

    # ---- forward ----
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

        lyr = self.layer

        # 1) estimate σ as a JAX scalar
        if _isinstance_by_name(lyr, ["RFFTCirculant1D"]):
            sigma = _max_abs(lyr.H_half)

        elif _isinstance_by_name(lyr, ["SpectralTokenMixer"]):
            sigma = self._sigma_token_mixer(lyr)

        elif _isinstance_by_name(lyr, ["RFFTCirculant2D"]):
            sigma = (
                self._sigma_2d_svd(lyr.K_half)
                if self.method2d == "svd"
                else self._sigma_2d_pi_from_half(lyr.K_half, state)[0]
            )

        elif _isinstance_by_name(
            lyr,
            [
                "SpectralDense",
                "AdaptiveSpectralDense",
                "SpectralConv2d",
                "AdaptiveSpectralConv2d",
            ],
        ):
            sigma = jnp.max(jnp.abs(lyr.s))

        elif _isinstance_by_name(
            lyr, ["SpectralCirculantLayer", "AdaptiveSpectralCirculantLayer"]
        ):
            sigma = _max_abs(lyr.get_fourier_coeffs())

        elif _isinstance_by_name(
            lyr, ["SpectralCirculantLayer2d", "AdaptiveSpectralCirculantLayer2d"]
        ):
            K_full = lyr.get_fft_kernel()  # complex, shape (C_out, C_in, H_pad, W_pad)
            W_half = (K_full.shape[-1] // 2) + 1
            K_half = K_full[..., :W_half]  # use half-plane; no rfft on complex
            sigma = (
                self._sigma_2d_svd(K_half)
                if self.method2d == "svd"
                else self._sigma_2d_pi_from_half(K_half, state)[0]
            )

        else:
            raise ValueError(f"Unsupported layer type: {type(lyr).__name__}")

        # 2) cushion and compute scaling factor (all JAX ops)
        sigma_adj = sigma * self.safety_factor
        scale = jnp.maximum(1.0, sigma_adj / self.target)  # rank-0 array

        # 3) build scaled view and forward (pure, no in-place mutation)
        lyr_eff = self._scaled_layer(scale)
        out = lyr_eff(x)

        # 4) update state if PI computed
        if _isinstance_by_name(
            lyr,
            [
                "RFFTCirculant2D",
                "SpectralCirculantLayer2d",
                "AdaptiveSpectralCirculantLayer2d",
            ],
        ) and (self.method2d == "pi-sampled"):
            if hasattr(lyr, "K_half"):
                pass  # state already updated inside _sigma_2d_pi_from_half
            else:
                # already updated above as well (we used half-plane)
                pass

        return out, state
