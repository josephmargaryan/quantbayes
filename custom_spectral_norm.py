import equinox as eqx
import jax
import jax.numpy as jnp
from equinox._tree import tree_at

from quantbayes.stochax.layers import (
    SpectralCirculantLayer,
    AdaptiveSpectralCirculantLayer,
    SpectralCirculantLayer2d,
    AdaptiveSpectralCirculantLayer2d,
    SpectralDense,
    AdaptiveSpectralDense,
    SpectralConv2d,
    AdaptiveSpectralConv2d,
)


def _power_iteration_singular_values(
    mats: jnp.ndarray, iters: int  # (n_freq, Cout, Cin), real or complex
) -> jnp.ndarray:
    """Approximate top‐singular of each mats[i] via a few power‐it steps."""

    def one(W):
        # W: (Cout, Cin)
        v = jnp.ones((W.shape[-1],), W.dtype)
        v = v / (jnp.linalg.norm(v) + 1e-12)
        for _ in range(iters):
            u = W @ v
            u = u / (jnp.linalg.norm(u) + 1e-12)
            v = (W.conj().T if ~jnp.isrealobj(W) else W.T) @ u
            v = v / (jnp.linalg.norm(v) + 1e-12)
        σ = jnp.vdot(u, W @ v)
        return jnp.abs(σ)

    return jax.vmap(one)(mats)


class SpectralNorm(eqx.Module):
    """
    Wrap _any_ of your spectral layers and enforce ‖W‖₂ ≤ 1 at every forward
    by dividing out the stopped‐gradient σ estimate.
    """

    layer: eqx.Module
    eps: float = eqx.field(static=True)
    n_power_iters: int = eqx.field(static=True)

    # internal, set automatically in __init__:
    _weight_fn: callable = eqx.field(static=True)
    _param_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        layer: eqx.Module,
        *,
        n_power_iters: int = 0,
        eps: float = 1e-6,
    ):
        self.layer = layer
        self.eps = eps
        self.n_power_iters = n_power_iters

        # --- pick the right extractor + param list ---
        if isinstance(layer, (SpectralCirculantLayer, AdaptiveSpectralCirculantLayer)):
            # 1D circulant → extract full FFT‐vector
            self._weight_fn = lambda l: l.get_fourier_coeffs()
            self._param_names = ("w_real", "w_imag")

        elif isinstance(
            layer, (SpectralCirculantLayer2d, AdaptiveSpectralCirculantLayer2d)
        ):
            # 2D circulant → extract per‐freq matrices, shape (H*W, Cout, Cin)
            def extract2d(l):
                fftk = l.get_fft_kernel()  # (Cout,Cin,H,W)
                # move spatial dims to front, flatten them
                mats = jnp.transpose(fftk, (2, 3, 0, 1))  # (H,W,Cout,Cin)
                return mats.reshape(-1, l.C_out, l.C_in)  # (H*W,Cout,Cin)

            self._weight_fn = extract2d
            self._param_names = ("w_real", "w_imag")

        elif isinstance(
            layer,
            (
                SpectralDense,
                AdaptiveSpectralDense,
                SpectralConv2d,
                AdaptiveSpectralConv2d,
            ),
        ):
            # SVD‐parametrized → operator norm = max(|s|)
            self._weight_fn = lambda l: l.s  # vector
            self._param_names = ("s",)

        else:
            raise ValueError(f"SpectralNorm: unsupported layer type {type(layer)}")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        W = self._weight_fn(self.layer)

        # 1) estimate σ
        if self.n_power_iters > 0 and W.ndim == 3:
            # freq‐wise power iteration
            sigmas = _power_iteration_singular_values(W, self.n_power_iters)
            σ = jnp.max(sigmas)
        else:
            # exact via max absolute entry
            σ = jnp.max(jnp.abs(W))

        # 2) stop_gradient & clamp
        σ = jax.lax.stop_gradient(σ)
        scale = jnp.maximum(σ, self.eps)

        # 3) divide out of each named param
        new_layer = self.layer
        for name in self._param_names:
            w = getattr(self.layer, name) / scale
            new_layer = tree_at(lambda m: getattr(m, name), new_layer, w)

        # 4) forward
        return new_layer(x)
