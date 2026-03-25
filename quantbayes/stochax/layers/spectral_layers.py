from __future__ import annotations
from typing import Any, Optional, Sequence, Callable, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

__all__ = [
    "SpectralCirculantLayer",
    "AdaptiveSpectralCirculantLayer",
    "SpectralCirculantLayer2d",
    "AdaptiveSpectralCirculantLayer2d",
    "SpectralDense",
    "AdaptiveSpectralDense",
    "SpectralConv2d",
    "AdaptiveSpectralConv2d",
    "RFFTCirculant1D",
    "RFFTCirculant2D",
    "SpectralTokenMixer",
    "SVDDense",
    "GraphChebSobolev",
]


@jax.custom_jvp
def spectral_circulant_matmul(x: jnp.ndarray, fft_full: jnp.ndarray) -> jnp.ndarray:
    """
    y = IFFT( FFT(x_pad) * fft_full ), along the last dimension.
    Uses orthonormal normalization to keep scaling consistent.
    - x: (..., d_in) real
    - fft_full: (padded_dim,) complex; full Hermitian spectrum
    returns: (..., padded_dim) real
    """
    padded_dim = int(fft_full.shape[0])
    tail = x.shape[:-1]
    d_in = int(x.shape[-1])

    if d_in < padded_dim:
        pad = [(0, 0)] * x.ndim
        pad[-1] = (0, padded_dim - d_in)
        x_pad = jnp.pad(x, pad_width=pad)
    else:
        x_pad = x[..., :padded_dim]

    Xf = jnp.fft.fft(x_pad, axis=-1, norm="ortho")
    yf = Xf * jnp.reshape(fft_full, (1,) * (x.ndim - 1) + (padded_dim,))
    y = jnp.fft.ifft(yf, axis=-1, norm="ortho").real
    return y


@spectral_circulant_matmul.defjvp
def _spectral_circulant_matmul_jvp(primals, tangents):
    x, fft_full = primals
    dx, dfft = tangents

    padded_dim = int(fft_full.shape[0])
    d_in = int(x.shape[-1])

    if d_in < padded_dim:
        pad = [(0, 0)] * x.ndim
        pad[-1] = (0, padded_dim - d_in)
        x_pad = jnp.pad(x, pad)
        dx_pad = None if dx is None else jnp.pad(dx, pad)
    else:
        x_pad = x[..., :padded_dim]
        dx_pad = None if dx is None else dx[..., :padded_dim]

    Xf = jnp.fft.fft(x_pad, axis=-1, norm="ortho")
    y_primal = jnp.fft.ifft(Xf * fft_full, axis=-1, norm="ortho").real

    dXf = 0.0 if dx_pad is None else jnp.fft.fft(dx_pad, axis=-1, norm="ortho")
    dF = 0.0 if dfft is None else dfft
    dyf = dXf * fft_full + Xf * dF
    dy = jnp.fft.ifft(dyf, axis=-1, norm="ortho").real
    return y_primal, dy


def _enforce_hermitian(fft2d: jnp.ndarray) -> jnp.ndarray:
    """
    Project complex (..., H, W) onto Hermitian subspace (real spatial IFFT).
    """
    H, W = fft2d.shape[-2:]
    conj_flip = jnp.flip(jnp.conj(fft2d), axis=(-2, -1))
    herm = 0.5 * (fft2d + conj_flip)

    herm = herm.at[..., 0, 0].set(jnp.real(herm[..., 0, 0]))
    if H % 2 == 0:
        herm = herm.at[..., H // 2, :].set(jnp.real(herm[..., H // 2, :]))
    if W % 2 == 0:
        herm = herm.at[..., :, W // 2].set(jnp.real(herm[..., :, W // 2]))
    return herm


@jax.custom_jvp
def spectral_circulant_conv2d(x: jnp.ndarray, fft_kernel: jnp.ndarray) -> jnp.ndarray:
    """
    Circular 2D convolution via FFT:
      - x: (..., C_in, H_in, W_in) real
      - fft_kernel: (C_out, C_in, H_pad, W_pad) complex (Hermitian)
      -> (..., C_out, H_pad, W_pad) real
    """
    C_out, C_in, H_pad, W_pad = map(int, fft_kernel.shape)
    single = x.ndim == 3
    if single:
        x = x[None, ...]  # (1, C_in, H_in, W_in)

    *lead, C_in_x, H_in, W_in = x.shape
    if C_in_x != C_in:
        raise ValueError(f"Cin mismatch: kernel={C_in}, x={C_in_x}")

    pad_h = max(0, H_pad - H_in)
    pad_w = max(0, W_pad - W_in)
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[..., :H_pad, :W_pad]

    Xf = jnp.fft.fftn(x_pad, axes=(-2, -1), norm="ortho")
    Yf = jnp.einsum("oihw,bihw->bohw", fft_kernel, Xf)
    y = jnp.fft.ifftn(Yf, axes=(-2, -1), norm="ortho").real

    return y[0] if single else y


@spectral_circulant_conv2d.defjvp
def _spectral_circulant_conv2d_jvp(primals, tangents):
    x, fft_kernel = primals
    dx, dk = tangents

    C_out, C_in, H_pad, W_pad = map(int, fft_kernel.shape)
    single = x.ndim == 3
    if single:
        x = x[None, ...]
        dx = None if dx is None else dx[None, ...]

    *lead, C_in_x, H_in, W_in = x.shape
    if C_in_x != C_in:
        raise ValueError(f"Cin mismatch: kernel={C_in}, x={C_in_x}")

    pad_h = max(0, H_pad - H_in)
    pad_w = max(0, W_pad - W_in)
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[..., :H_pad, :W_pad]
    dx_pad = (
        None
        if dx is None
        else jnp.pad(dx, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[..., :H_pad, :W_pad]
    )

    Xf = jnp.fft.fftn(x_pad, axes=(-2, -1), norm="ortho")
    Yf = jnp.einsum("oihw,bihw->bohw", fft_kernel, Xf)
    y = jnp.fft.ifftn(Yf, axes=(-2, -1), norm="ortho").real

    dXf = 0.0 if dx_pad is None else jnp.fft.fftn(dx_pad, axes=(-2, -1), norm="ortho")
    dKf = 0.0 if dk is None else dk
    dYf = jnp.einsum("oihw,bihw->bohw", fft_kernel, dXf) + jnp.einsum(
        "oihw,bihw->bohw", dKf, Xf
    )
    dy = jnp.fft.ifftn(dYf, axes=(-2, -1), norm="ortho").real

    return (y[0], dy[0]) if single else (y, dy)


# ---- shared spectral mixin ---------------------------------------------------


class _SpectralMixin:
    """Provides quadratic spectral penalty: sum(β * θ^2).
    Also exposes an operator-norm hint for diagnostics.
    """

    def _spectral_weights(self) -> jnp.ndarray:
        raise NotImplementedError

    def _spectral_scale(self) -> jnp.ndarray:
        raise NotImplementedError

    def __spectral_penalty__(self) -> jnp.ndarray:
        θ = self._spectral_weights()
        β = self._spectral_scale()
        return jnp.sum(β * (θ**2))

    # NEW: diagnostics hook — return a float (operator norm) or None if unknown.
    def __operator_norm_hint__(self) -> float | None:
        return None


def _alpha_bounded(alpha_raw, lo=0.1, hi=4.0):
    a = jnp.asarray(alpha_raw)
    t = jnp.tanh(a)
    return lo + (hi - lo) * (t + 1) / 2.0


class SpectralCirculantLayer(eqx.Module, _SpectralMixin):
    """
    1D spectral-circulant linear map with trainable α.
    Bias: scalar by default (equivariance-preserving). Set vector_bias=True ONLY
    if you intentionally want per-position bias (breaks shift equivariance).
    """

    in_features: int = eqx.field(static=True)
    padded_dim: int = eqx.field(static=True)
    K: int = eqx.field(static=True)
    k_half: int = eqx.field(static=True)
    crop_output: bool = eqx.field(static=True)
    vector_bias: bool = eqx.field(static=True)

    alpha_raw: jnp.ndarray
    w_real: jnp.ndarray
    w_imag: jnp.ndarray
    bias: jnp.ndarray  # () if scalar, (padded_dim,) if vector_bias

    def __init__(
        self,
        in_features: int,
        *,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        crop_output: bool = True,
        alpha_init: float = 1.0,
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        vector_bias: bool = False,
        dtype=jnp.float32,
    ):
        self.in_features = int(in_features)
        self.padded_dim = int(padded_dim or in_features)
        self.k_half = self.padded_dim // 2 + 1
        self.K = int(self.k_half if (K is None or K > self.k_half) else K)
        self.crop_output = bool(crop_output)
        self.vector_bias = bool(vector_bias)

        self.alpha_raw = jnp.asarray(alpha_init, dtype)

        k = jnp.arange(self.k_half, dtype=dtype)
        k_norm = k / jnp.maximum(self.k_half - 1, 1)
        α0 = _alpha_bounded(self.alpha_raw)
        std0 = 1.0 / jnp.sqrt(1.0 + k_norm**α0)

        k1, k2, k3 = jr.split(key, 3)
        w_r = jr.normal(k1, (self.k_half,), dtype) * (init_scale * std0)
        w_i = jr.normal(k2, (self.k_half,), dtype) * (init_scale * std0)
        w_i = w_i.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            w_i = w_i.at[-1].set(0.0)
        self.w_real = w_r
        self.w_imag = w_i
        self.bias = (
            jr.normal(k3, (self.padded_dim,), dtype) * bias_scale
            if self.vector_bias
            else jr.normal(k3, (), dtype) * bias_scale
        )

    def get_fourier_coeffs(self) -> jnp.ndarray:
        mask = (jnp.arange(self.k_half) < self.K).astype(self.w_real.dtype)
        half = (self.w_real * mask) + 1j * (self.w_imag * mask)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            nyq = half[-1].real[None]
            full = jnp.concatenate([half[:-1], nyq, jnp.conj(half[1:-1])[::-1]])
        else:
            full = jnp.concatenate([half, jnp.conj(half[1:])[::-1]])
        return full

    def _spectral_weights(self) -> jnp.ndarray:
        return jnp.concatenate([self.w_real, self.w_imag])

    def _spectral_scale(self) -> jnp.ndarray:
        α = _alpha_bounded(self.alpha_raw)
        k = jnp.arange(self.k_half, dtype=self.w_real.dtype)
        k_norm = k / jnp.maximum(self.k_half - 1, 1)
        β = 1.0 + k_norm**α
        return jnp.concatenate([β, β])

    def __call__(
        self, x: jnp.ndarray, *args, key: Any | None = None, **kwargs
    ) -> jnp.ndarray:
        y = spectral_circulant_matmul(x, self.get_fourier_coeffs())
        if self.vector_bias:
            y = y + (self.bias if y.ndim == 1 else self.bias[None, :])
        else:
            y = y + self.bias  # scalar broadcast
        if not self.crop_output or (self.padded_dim == self.in_features):
            return y
        return (
            y[..., : self.in_features]
            if self.crop_output and (self.padded_dim != self.in_features)
            else y
        )

    # NEW (exact for circulant)
    def __operator_norm_hint__(self) -> float:
        H = self.get_fourier_coeffs()  # (padded_dim,)
        return float(jnp.max(jnp.abs(H)))


class AdaptiveSpectralCirculantLayer(eqx.Module, _SpectralMixin):
    """
    1D spectral-circulant with per-frequency extra exponent δ (softplus).
    Bias: scalar by default; set vector_bias=True only if you want positional bias.
    """

    in_features: int = eqx.field(static=True)
    padded_dim: int = eqx.field(static=True)
    K: int = eqx.field(static=True)
    k_half: int = eqx.field(static=True)
    crop_output: bool = eqx.field(static=True)
    vector_bias: bool = eqx.field(static=True)

    alpha_raw: jnp.ndarray
    delta_z: jnp.ndarray
    w_real: jnp.ndarray
    w_imag: jnp.ndarray
    bias: jnp.ndarray  # () or (padded_dim,)

    def __init__(
        self,
        in_features: int,
        *,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        crop_output: bool = True,
        alpha_init: float = 1.0,
        delta_init: float = 0.1,
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        vector_bias: bool = False,
        dtype=jnp.float32,
    ):
        self.in_features = int(in_features)
        self.padded_dim = int(padded_dim or in_features)
        self.k_half = self.padded_dim // 2 + 1
        self.K = int(self.k_half if (K is None or K > self.k_half) else K)
        self.crop_output = bool(crop_output)
        self.vector_bias = bool(vector_bias)

        self.alpha_raw = jnp.asarray(alpha_init, dtype)
        z0 = jnp.log(jnp.exp(delta_init) - 1.0)
        self.delta_z = jnp.full((self.k_half,), z0, dtype)

        k = jnp.arange(self.k_half, dtype=dtype)
        k_norm = k / jnp.maximum(self.k_half - 1, 1)
        α0 = _alpha_bounded(self.alpha_raw)
        std0 = 1.0 / jnp.sqrt(1.0 + k_norm**α0)

        k1, k2, k3 = jr.split(key, 3)
        w_r = jr.normal(k1, (self.k_half,), dtype) * (init_scale * std0)
        w_i = jr.normal(k2, (self.k_half,), dtype) * (init_scale * std0)
        w_i = w_i.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            w_i = w_i.at[-1].set(0.0)
        self.w_real, self.w_imag = w_r, w_i
        self.bias = (
            jr.normal(k3, (self.padded_dim,), dtype) * bias_scale
            if self.vector_bias
            else jr.normal(k3, (), dtype) * bias_scale
        )

    @property
    def delta_alpha(self) -> jnp.ndarray:
        return jax.nn.softplus(self.delta_z)

    def _spectral_weights(self) -> jnp.ndarray:
        return jnp.concatenate([self.w_real, self.w_imag])

    def _spectral_scale(self) -> jnp.ndarray:
        δ = self.delta_alpha
        α_total = _alpha_bounded(self.alpha_raw + δ)
        k = jnp.arange(self.k_half, dtype=self.w_real.dtype)
        k_norm = k / jnp.maximum(self.k_half - 1, 1)
        β = 1.0 + k_norm**α_total
        return jnp.concatenate([β, β])

    def get_fourier_coeffs(self) -> jnp.ndarray:
        mask = (jnp.arange(self.k_half) < self.K).astype(self.w_real.dtype)
        half = (self.w_real * mask) + 1j * (self.w_imag * mask)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            nyq = half[-1].real[None]
            full = jnp.concatenate([half[:-1], nyq, jnp.conj(half[1:-1])[::-1]])
        else:
            full = jnp.concatenate([half, jnp.conj(half[1:])[::-1]])
        return full

    def __call__(
        self, x: jnp.ndarray, *args, key: Any | None = None, **kwargs
    ) -> jnp.ndarray:
        y = spectral_circulant_matmul(x, self.get_fourier_coeffs())
        if self.vector_bias:
            y = y + (self.bias if y.ndim == 1 else self.bias[None, :])
        else:
            y = y + self.bias
        if not self.crop_output or (self.padded_dim == self.in_features):
            return y
        return (
            y[..., : self.in_features]
            if self.crop_output and (self.padded_dim != self.in_features)
            else y
        )

    # NEW (exact for circulant)
    def __operator_norm_hint__(self) -> float:
        H = self.get_fourier_coeffs()
        return float(jnp.max(jnp.abs(H)))


# ======================= 2D spectral-circulant convolution ====================


class SpectralCirculantLayer2d(eqx.Module, _SpectralMixin):
    """
    2D spectral-circulant conv with scalar α and optional low-mode mask.
    Bias: per-output-channel (C_out,) to preserve translation equivariance.
    """

    C_in: int = eqx.field(static=True)
    C_out: int = eqx.field(static=True)
    H_in: int = eqx.field(static=True)
    W_in: int = eqx.field(static=True)
    H_pad: int = eqx.field(static=True)
    W_pad: int = eqx.field(static=True)
    crop_output: bool = eqx.field(static=True)
    K_rad: Optional[float] = eqx.field(static=True)
    use_soft_mask: bool = eqx.field(static=True)
    mask_steepness: float = eqx.field(static=True)

    alpha_raw: jnp.ndarray
    w_real: jnp.ndarray
    w_imag: jnp.ndarray
    bias: jnp.ndarray  # (C_out,)

    def __init__(
        self,
        C_in: int,
        C_out: int,
        H_in: int,
        W_in: int,
        *,
        H_pad: Optional[int] = None,
        W_pad: Optional[int] = None,
        alpha_init: float = 1.0,
        K_rad: Optional[float] = None,
        use_soft_mask: bool = True,
        mask_steepness: float = 40.0,
        crop_output: bool = True,
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        dtype=jnp.float32,
    ):
        self.C_in, self.C_out = int(C_in), int(C_out)
        self.H_in, self.W_in = int(H_in), int(W_in)
        self.H_pad = int(H_pad or H_in)
        self.W_pad = int(W_pad or W_in)
        self.crop_output = bool(crop_output)
        self.K_rad = None if K_rad is None else float(jnp.clip(K_rad, 0.0, 1.0))
        self.use_soft_mask = bool(use_soft_mask)
        self.mask_steepness = float(mask_steepness)

        self.alpha_raw = jnp.asarray(alpha_init, dtype)

        # normalized radial grid
        u = jnp.fft.fftfreq(self.H_pad).astype(dtype) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad).astype(dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        R_norm = R / jnp.maximum(jnp.max(R), 1.0)

        α0 = _alpha_bounded(self.alpha_raw)
        std0 = 1.0 / jnp.sqrt(1.0 + R_norm**α0)

        k1, k2, k3, k4 = jr.split(key, 4)
        self.w_real = jr.normal(
            k1, (self.C_out, self.C_in, self.H_pad, self.W_pad), dtype
        ) * (init_scale * std0)
        wi = jr.normal(k2, (self.C_out, self.C_in, self.H_pad, self.W_pad), dtype) * (
            init_scale * std0
        )
        wi = wi.at[..., 0, 0].set(0.0)
        if self.H_pad % 2 == 0:
            wi = wi.at[..., self.H_pad // 2, :].set(0.0)
        if self.W_pad % 2 == 0:
            wi = wi.at[..., :, self.W_pad // 2].set(0.0)
        self.w_imag = wi
        self.bias = jr.normal(k4, (self.C_out,), dtype) * bias_scale

    def _spectral_weights(self) -> jnp.ndarray:
        return jnp.concatenate([self.w_real.reshape(-1), self.w_imag.reshape(-1)])

    def _spectral_scale(self) -> jnp.ndarray:
        u = jnp.fft.fftfreq(self.H_pad).astype(self.w_real.dtype) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad).astype(self.w_real.dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        R_norm = R / jnp.maximum(jnp.max(R), 1.0)
        α = _alpha_bounded(self.alpha_raw)
        β2d = 1.0 + R_norm**α
        flat = jnp.broadcast_to(β2d, self.w_real.shape).reshape(-1)
        return jnp.concatenate([flat, flat])

    def _lowpass_mask(self) -> Optional[jnp.ndarray]:
        if self.K_rad is None:
            return None
        u = jnp.fft.fftfreq(self.H_pad).astype(self.w_real.dtype) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad).astype(self.w_real.dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        R_norm = R / jnp.maximum(jnp.max(R), 1.0)
        if self.use_soft_mask:
            return jax.nn.sigmoid(self.mask_steepness * (self.K_rad - R_norm))
        else:
            return (R_norm <= self.K_rad).astype(self.w_real.dtype)

    def get_fft_kernel(self) -> jnp.ndarray:
        kernel = _enforce_hermitian(self.w_real + 1j * self.w_imag)
        m = self._lowpass_mask()
        if m is not None:
            kernel = kernel * m[None, None, :, :]
        return kernel

    def __call__(
        self, x: jnp.ndarray, *args, key: Any | None = None, **kwargs
    ) -> jnp.ndarray:
        y = spectral_circulant_conv2d(x, self.get_fft_kernel())
        if y.ndim == 3:
            y = y + self.bias[:, None, None]
        else:
            y = y + self.bias[None, :, None, None]
        if not self.crop_output or (
            self.H_pad == self.H_in and self.W_pad == self.W_in
        ):
            return y
        return (
            y[..., : self.H_in, : self.W_in]
            if self.crop_output and (self.H_pad != self.H_in or self.W_pad != self.W_in)
            else y
        )

    # NEW (exact for circular conv: sup_ω σ_max(K(ω)))
    def __operator_norm_hint__(self) -> float:
        K = self.get_fft_kernel()  # (C_out, C_in, H_pad, W_pad), complex
        HW = self.H_pad * self.W_pad
        Khw = jnp.transpose(
            K.reshape(self.C_out, self.C_in, HW), (2, 0, 1)
        )  # (HW, C_out, C_in)
        svals = jnp.linalg.svd(Khw, compute_uv=False)  # (HW, min(C_out, C_in))
        return float(jnp.max(svals))


class AdaptiveSpectralCirculantLayer2d(SpectralCirculantLayer2d):
    """
    2D variant with per-frequency extra exponent δ (H_pad x W_pad).
    Inherits per-channel bias from the base class.
    """

    delta_z: jnp.ndarray

    def __init__(
        self,
        C_in: int,
        C_out: int,
        H_in: int,
        W_in: int,
        *,
        H_pad: Optional[int] = None,
        W_pad: Optional[int] = None,
        alpha_init: float = 1.0,
        delta_init: float = 0.1,
        K_rad: Optional[float] = None,
        use_soft_mask: bool = True,
        mask_steepness: float = 40.0,
        crop_output: bool = True,
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        dtype=jnp.float32,
    ):
        super().__init__(
            C_in=C_in,
            C_out=C_out,
            H_in=H_in,
            W_in=W_in,
            H_pad=H_pad,
            W_pad=W_pad,
            alpha_init=alpha_init,
            K_rad=K_rad,
            use_soft_mask=use_soft_mask,
            mask_steepness=mask_steepness,
            crop_output=crop_output,
            key=key,
            init_scale=init_scale,
            bias_scale=bias_scale,
            dtype=dtype,
        )
        z0 = jnp.log(jnp.exp(delta_init) - 1.0)
        self.delta_z = jnp.full((self.H_pad, self.W_pad), z0, dtype)

    @property
    def delta_alpha(self) -> jnp.ndarray:
        return jax.nn.softplus(self.delta_z)

    def _spectral_scale(self) -> jnp.ndarray:
        u = jnp.fft.fftfreq(self.H_pad).astype(self.w_real.dtype) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad).astype(self.w_real.dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        R_norm = R / jnp.maximum(jnp.max(R), 1.0)
        α_total = _alpha_bounded(self.alpha_raw + self.delta_alpha)
        β2d = 1.0 + R_norm**α_total
        flat = jnp.broadcast_to(β2d, self.w_real.shape).reshape(-1)
        return jnp.concatenate([flat, flat])

    def __operator_norm_hint__(self) -> float:
        # identical logic — kernel already includes masking and α+δ
        K = self.get_fft_kernel()
        HW = self.H_pad * self.W_pad
        Khw = jnp.transpose(K.reshape(self.C_out, self.C_in, HW), (2, 0, 1))
        svals = jnp.linalg.svd(Khw, compute_uv=False)
        return float(jnp.max(svals))


# ============================ Spectral SVD (Dense) ============================


class SpectralDense(eqx.Module, _SpectralMixin):
    """
    Rectangular spectral dense: W = U diag(s) V^T with fixed U∈R^{out×r}, V∈R^{in×r}.
    """

    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)

    U: jnp.ndarray  # (out, r)
    V: jnp.ndarray  # (in,  r)
    s: jnp.ndarray  # (r,)
    bias: jnp.ndarray  # (out,)
    alpha_raw: jnp.ndarray  # scalar

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: Optional[int] = None,
        alpha_init: float = 1.0,
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        dtype=jnp.float32,
    ):
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        r = min(self.in_features, self.out_features) if rank is None else int(rank)
        self.rank = r

        kU, kV, ks, kb = jr.split(key, 4)
        U0, _ = jnp.linalg.qr(
            jr.normal(kU, (self.out_features, self.out_features), dtype)
        )
        V0, _ = jnp.linalg.qr(
            jr.normal(kV, (self.in_features, self.in_features), dtype)
        )
        self.U = jax.lax.stop_gradient(U0[:, :r])
        self.V = jax.lax.stop_gradient(V0[:, :r])

        self.s = jr.normal(ks, (r,), dtype) * init_scale
        self.bias = jr.normal(kb, (self.out_features,), dtype) * bias_scale
        self.alpha_raw = jnp.asarray(alpha_init, dtype)

    def _spectral_weights(self) -> jnp.ndarray:
        return self.s

    def _spectral_scale(self) -> jnp.ndarray:
        r = self.rank
        k = jnp.arange(r, dtype=self.s.dtype)
        k_norm = k / jnp.maximum(r - 1, 1)
        α = _alpha_bounded(self.alpha_raw)
        return 1.0 + k_norm**α

    def __call__(self, x, *args, key=None, **kwargs):
        z = x @ self.V  # (B,in)@(in,r) -> (B,r)
        z = z * self.s  # (B,r) ⊙ (r,)
        y = z @ self.U.T  # (B,r)@(r,out) -> (B,out)
        return y + self.bias

    def __operator_norm_hint__(self):
        # exact: max |s| ; return JAX scalar (jit-safe)
        return jnp.max(jnp.abs(self.s)).astype(jnp.float32)


class AdaptiveSpectralDense(SpectralDense):
    """
    Dense with per-mode extra exponent δ (softplus). Exposes delta_alpha.
    """

    delta_z: jnp.ndarray  # (rank,)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: Optional[int] = None,
        alpha_init: float = 1.0,
        delta_init: float = 0.1,
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        dtype=jnp.float32,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha_init=alpha_init,
            key=key,
            init_scale=init_scale,
            bias_scale=bias_scale,
            dtype=dtype,
        )
        z0 = jnp.log(jnp.exp(delta_init) - 1.0)
        self.delta_z = jnp.full((self.rank,), z0, dtype)

    @property
    def delta_alpha(self) -> jnp.ndarray:
        return jax.nn.softplus(self.delta_z)

    def _spectral_scale(self) -> jnp.ndarray:
        r = self.rank
        k = jnp.arange(r, dtype=self.s.dtype)
        k_norm = k / jnp.maximum(r - 1, 1)
        α_total = _alpha_bounded(self.alpha_raw + self.delta_alpha)
        return 1.0 + k_norm**α_total

    def __operator_norm_hint__(self):
        # exact: max |s| ; return JAX scalar (jit-safe)
        return jnp.max(jnp.abs(self.s)).astype(jnp.float32)


# ============================ Spectral SVD (Conv2d) ===========================


class SpectralConv2d(eqx.Module, _SpectralMixin):
    """
    Conv kernel parameterized via SVD of the flattened matrix:
      W_mat (C_out × [C_in*H_k*W_k]) = U diag(s) V^T, with fixed U,V; s trainable.

    Accepts either a single image (C_in, H, W) or a batch (N, C_in, H, W).
    """

    C_in: int = eqx.field(static=True)
    C_out: int = eqx.field(static=True)
    H_k: int = eqx.field(static=True)
    W_k: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    strides: Sequence[int] = eqx.field(static=True)
    padding: Any = eqx.field(static=True)

    U: jnp.ndarray  # (C_out, r)
    V: jnp.ndarray  # (C_in*H_k*W_k, r)
    s: jnp.ndarray  # (r,)
    bias: jnp.ndarray  # (C_out,)
    alpha_raw: jnp.ndarray

    def __init__(
        self,
        C_in: int,
        C_out: int,
        H_k: int,
        W_k: int,
        *,
        rank: Optional[int] = None,
        strides: Sequence[int] = (1, 1),
        padding: Any = "SAME",
        alpha_init: float = 1.0,
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        dtype=jnp.float32,
    ):
        self.C_in, self.C_out, self.H_k, self.W_k = (
            int(C_in),
            int(C_out),
            int(H_k),
            int(W_k),
        )
        self.strides = tuple(int(s) for s in strides)
        self.padding = padding

        in_dim = self.C_in * self.H_k * self.W_k
        out_dim = self.C_out
        r = min(in_dim, out_dim) if rank is None else int(rank)
        self.rank = r

        kU, kV, ks, kb = jr.split(key, 4)
        U0, _ = jnp.linalg.qr(jr.normal(kU, (out_dim, out_dim), dtype))
        V0, _ = jnp.linalg.qr(jr.normal(kV, (in_dim, in_dim), dtype))
        self.U = jax.lax.stop_gradient(U0[:, :r])
        self.V = jax.lax.stop_gradient(V0[:, :r])

        self.s = jr.normal(ks, (r,), dtype) * init_scale
        self.bias = jr.normal(kb, (self.C_out,), dtype) * bias_scale
        self.alpha_raw = jnp.asarray(alpha_init, dtype)

    def _spectral_weights(self) -> jnp.ndarray:
        return self.s

    def _spectral_scale(self) -> jnp.ndarray:
        r = self.rank
        k = jnp.arange(r, dtype=self.s.dtype)
        k_norm = k / jnp.maximum(r - 1, 1)
        α = _alpha_bounded(self.alpha_raw)
        return 1.0 + k_norm**α

    def __call__(
        self, x: jnp.ndarray, *args, key: Any | None = None, **kwargs
    ) -> jnp.ndarray:
        single = x.ndim == 3
        if single:
            x = x[None, ...]
        W_mat = self.U @ (self.s[:, None] * self.V.T)
        W = jnp.reshape(W_mat, (self.C_out, self.C_in, self.H_k, self.W_k))
        y = jax.lax.conv_general_dilated(
            x,
            W,
            window_strides=self.strides,
            padding=self.padding,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        y = y + self.bias[None, :, None, None]
        return y[0] if single else y

    # NEW (upper bound for non-circulant conv; equals σ_max of the flattened map)
    def __operator_norm_hint__(self) -> float:
        """
        Diagnostic-only Schur bound on the **flattened kernel matrix** (not the
        convolution operator). Specifically, this returns
            sqrt( ‖W_flat‖_1 · ‖W_flat‖_∞ ),
        where W_flat ∈ ℝ^{C_out × (C_in·H_k·W_k)}.
        For certified bounds on the *convolution* operator, use
        `regularizers.global_spectral_norm_penalty(..., conv_mode=...)`.
        """
        W_mat = self.U @ (self.s[:, None] * self.V.T)
        W = jnp.reshape(W_mat, (self.C_out, self.C_in, self.H_k, self.W_k))
        absW = jnp.abs(W).astype(jnp.float32)
        row_sum_max = jnp.max(jnp.sum(absW, axis=(1, 2, 3)))
        col_sum_max = jnp.max(jnp.sum(absW, axis=(0, 2, 3)))
        return float(jnp.sqrt(row_sum_max * col_sum_max))


class AdaptiveSpectralConv2d(SpectralConv2d):
    """
    Conv2d SVD with per-mode extra exponent δ (softplus). Exposes delta_alpha.
    """

    delta_z: jnp.ndarray  # (rank,)

    def __init__(
        self,
        C_in: int,
        C_out: int,
        H_k: int,
        W_k: int,
        *,
        rank: Optional[int] = None,
        strides: Sequence[int] = (1, 1),
        padding: Any = "SAME",
        alpha_init: float = 1.0,
        delta_init: float = 0.1,
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        dtype=jnp.float32,
    ):
        super().__init__(
            C_in=C_in,
            C_out=C_out,
            H_k=H_k,
            W_k=W_k,
            rank=rank,
            strides=strides,
            padding=padding,
            alpha_init=alpha_init,
            key=key,
            init_scale=init_scale,
            bias_scale=bias_scale,
            dtype=dtype,
        )
        z0 = jnp.log(jnp.exp(delta_init) - 1.0)
        self.delta_z = jnp.full((self.rank,), z0, dtype)

    @property
    def delta_alpha(self) -> jnp.ndarray:
        return jax.nn.softplus(self.delta_z)

    def _spectral_scale(self) -> jnp.ndarray:
        r = self.rank
        k = jnp.arange(r, dtype=self.s.dtype)
        k_norm = k / jnp.maximum(r - 1, 1)
        α_total = _alpha_bounded(self.alpha_raw + self.delta_alpha)
        return 1.0 + k_norm**α_total

    # NEW
    def __operator_norm_hint__(self) -> float:
        """
        Diagnostic-only Schur bound on the **flattened kernel matrix** (not the
        convolution operator). Specifically, this returns
            sqrt( ‖W_flat‖_1 · ‖W_flat‖_∞ ),
        where W_flat ∈ ℝ^{C_out × (C_in·H_k·W_k)}.
        For certified bounds on the *convolution* operator, use
        `regularizers.global_spectral_norm_penalty(..., conv_mode=...)`.
        """
        W_mat = self.U @ (self.s[:, None] * self.V.T)
        W = jnp.reshape(W_mat, (self.C_out, self.C_in, self.H_k, self.W_k))
        absW = jnp.abs(W).astype(jnp.float32)
        row_sum_max = jnp.max(jnp.sum(absW, axis=(1, 2, 3)))
        col_sum_max = jnp.max(jnp.sum(absW, axis=(0, 2, 3)))
        return float(jnp.sqrt(row_sum_max * col_sum_max))


class RFFTCirculant1D(eqx.Module):
    """1D spectral-circulant with RFFT storage (half-spectrum)."""

    in_features: int = eqx.field(static=True)
    padded_dim: int = eqx.field(static=True)
    k_half: int = eqx.field(static=True)
    K: int = eqx.field(static=True)
    crop_output: bool = eqx.field(static=True)
    vector_bias: bool = eqx.field(static=True)

    alpha_raw: jnp.ndarray
    H_half: jnp.ndarray  # (k_half,) complex
    bias: jnp.ndarray  # () or (padded_dim,)

    def __init__(
        self,
        in_features: int,
        *,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        crop_output: bool = True,
        alpha_init: float = 1.0,
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        vector_bias: bool = False,
        dtype=jnp.float32,
    ):
        self.in_features = int(in_features)
        self.padded_dim = int(padded_dim or in_features)
        self.k_half = self.padded_dim // 2 + 1
        self.K = int(self.k_half if (K is None or K > self.k_half) else K)
        self.crop_output = bool(crop_output)
        self.vector_bias = bool(vector_bias)

        self.alpha_raw = jnp.asarray(alpha_init, dtype)

        # Frequency-dependent std for init
        k = jnp.arange(self.k_half, dtype=dtype)
        k_norm = k / jnp.maximum(self.k_half - 1, 1)
        α0 = _alpha_bounded(self.alpha_raw)
        std0 = 1.0 / jnp.sqrt(1.0 + k_norm**α0)

        k1, k2, k3 = jr.split(key, 3)
        w_r = jr.normal(k1, (self.k_half,), dtype) * (init_scale * std0)
        w_i = jr.normal(k2, (self.k_half,), dtype) * (init_scale * std0)

        # Ensure real self-conjugate bins
        w_i = w_i.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            w_i = w_i.at[-1].set(0.0)

        complex_dtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
        self.H_half = (w_r + 1j * w_i).astype(complex_dtype)

        self.bias = (
            jr.normal(k3, (self.padded_dim,), dtype) * bias_scale
            if self.vector_bias
            else jr.normal(k3, (), dtype) * bias_scale
        )

    # ---- regularization hooks (optional) -------------------------------------
    def _spectral_weights(self) -> jnp.ndarray:
        return jnp.concatenate([self.H_half.real, self.H_half.imag])

    def _spectral_scale(self) -> jnp.ndarray:
        α = _alpha_bounded(self.alpha_raw)
        k = jnp.arange(self.k_half, dtype=self.H_half.real.dtype)
        k_norm = k / jnp.maximum(self.k_half - 1, 1)
        β = 1.0 + k_norm**α
        return jnp.concatenate([β, β])

    # ---- helpers --------------------------------------------------------------
    def _masked_half(self) -> jnp.ndarray:
        if self.K >= self.k_half:
            return self.H_half
        mask = (jnp.arange(self.k_half, dtype=self.H_half.real.dtype) < self.K).astype(
            self.H_half.real.dtype
        )
        mask = jax.lax.stop_gradient(mask)
        return (self.H_half.real * mask + 1j * self.H_half.imag * mask).astype(
            self.H_half.dtype
        )

    def _half_clean(self) -> jnp.ndarray:
        Hh = self._masked_half()
        Hh = Hh.at[0].set(jnp.real(Hh[0]) + 0.0j)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            Hh = Hh.at[-1].set(jnp.real(Hh[-1]) + 0.0j)
        return Hh

    # ---- forward --------------------------------------------------------------
    def __call__(self, x: jnp.ndarray, *_, key: Any | None = None, **__) -> jnp.ndarray:
        d_in = x.shape[-1]
        if d_in < self.padded_dim:
            pad = [(0, 0)] * x.ndim
            pad[-1] = (0, self.padded_dim - d_in)
            x_pad = jnp.pad(x, pad)
        else:
            x_pad = x[..., : self.padded_dim]

        Xf = jnp.fft.rfft(x_pad, axis=-1, norm="ortho")
        Yf = Xf * self._half_clean()
        y = jnp.fft.irfft(Yf, n=self.padded_dim, axis=-1, norm="ortho").real

        if self.vector_bias:
            y = y + (self.bias if y.ndim == 1 else self.bias[None, :])
        else:
            y = y + self.bias

        return (
            y
            if (not self.crop_output or self.padded_dim == self.in_features)
            else y[..., : self.in_features]
        )

    def __operator_norm_hint__(self):
        # exact: max |H_half|
        return jnp.max(jnp.abs(self._half_clean())).astype(jnp.float32)


class RFFTCirculant2D(eqx.Module):
    """2D spectral-circulant conv with RFFT2 storage (half-plane)."""

    C_in: int = eqx.field(static=True)
    C_out: int = eqx.field(static=True)
    H_in: int = eqx.field(static=True)
    W_in: int = eqx.field(static=True)
    H_pad: int = eqx.field(static=True)
    W_pad: int = eqx.field(static=True)
    W_half: int = eqx.field(static=True)
    crop_output: bool = eqx.field(static=True)
    K_rad: Optional[float] = eqx.field(static=True)
    use_soft_mask: bool = eqx.field(static=True)
    mask_steepness: float = eqx.field(static=True)

    alpha_raw: jnp.ndarray
    K_half: jnp.ndarray  # (C_out, C_in, H_pad, W_half) complex
    bias: jnp.ndarray  # (C_out,)

    def __init__(
        self,
        C_in: int,
        C_out: int,
        H_in: int,
        W_in: int,
        *,
        H_pad: Optional[int] = None,
        W_pad: Optional[int] = None,
        alpha_init: float = 1.0,
        K_rad: Optional[float] = None,
        use_soft_mask: bool = True,
        mask_steepness: float = 20.0,
        crop_output: bool = True,
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        dtype=jnp.float32,
    ):
        self.C_in, self.C_out = int(C_in), int(C_out)
        self.H_in, self.W_in = int(H_in), int(W_in)
        self.H_pad = int(H_pad or H_in)
        self.W_pad = int(W_pad or W_in)
        self.W_half = self.W_pad // 2 + 1
        self.crop_output = bool(crop_output)
        self.K_rad = None if K_rad is None else float(jnp.clip(K_rad, 0.0, 1.0))
        self.use_soft_mask = bool(use_soft_mask)
        self.mask_steepness = float(mask_steepness)

        self.alpha_raw = jnp.asarray(alpha_init, dtype)

        # radial grid (half-plane)
        u = jnp.fft.fftfreq(self.H_pad).astype(dtype) * self.H_pad
        v = jnp.fft.rfftfreq(self.W_pad).astype(dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        R_norm = R / jnp.maximum(jnp.max(R), 1.0)

        α0 = _alpha_bounded(self.alpha_raw)
        std0 = 1.0 / jnp.sqrt(1.0 + R_norm**α0)

        k1, k2, k3 = jr.split(key, 3)
        wr = jr.normal(k1, (self.C_out, self.C_in, self.H_pad, self.W_half), dtype) * (
            init_scale * std0[None, None, :, :]
        )
        wi = jr.normal(k2, (self.C_out, self.C_in, self.H_pad, self.W_half), dtype) * (
            init_scale * std0[None, None, :, :]
        )

        # enforce real self-conjugate bins
        wi = wi.at[..., 0, 0].set(0.0)
        if self.H_pad % 2 == 0:
            wi = wi.at[..., self.H_pad // 2, 0].set(0.0)
        if self.W_pad % 2 == 0:
            wi = wi.at[..., 0, self.W_half - 1].set(0.0)
            if self.H_pad % 2 == 0:
                wi = wi.at[..., self.H_pad // 2, self.W_half - 1].set(0.0)

        complex_dtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
        self.K_half = (wr + 1j * wi).astype(complex_dtype)
        self.bias = jr.normal(k3, (self.C_out,), dtype) * bias_scale

    # ---- regularization hooks (optional) -------------------------------------
    def _spectral_weights(self) -> jnp.ndarray:
        return jnp.concatenate([self.K_half.real.ravel(), self.K_half.imag.ravel()])

    def _spectral_scale(self) -> jnp.ndarray:
        u = jnp.fft.fftfreq(self.H_pad).astype(self.K_half.real.dtype) * self.H_pad
        v = jnp.fft.rfftfreq(self.W_pad).astype(self.K_half.real.dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        R_norm = R / jnp.maximum(jnp.max(R), 1.0)
        α = _alpha_bounded(self.alpha_raw)
        β2d = 1.0 + R_norm**α
        base = jnp.broadcast_to(β2d, self.K_half.real.shape[2:]).ravel()
        base = jnp.tile(base, self.C_out * self.C_in)
        return jnp.concatenate([base, base])

    # ---- helpers --------------------------------------------------------------
    def _lowpass_mask_half(self) -> Optional[jnp.ndarray]:
        if self.K_rad is None:
            return None
        u = jnp.fft.fftfreq(self.H_pad).astype(self.K_half.real.dtype) * self.H_pad
        v = jnp.fft.rfftfreq(self.W_pad).astype(self.K_half.real.dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        R_norm = R / jnp.maximum(jnp.max(R), 1.0)
        if self.use_soft_mask:
            return jax.nn.sigmoid(self.mask_steepness * (self.K_rad - R_norm))
        else:
            return (R_norm <= self.K_rad).astype(self.K_half.real.dtype)

    def _clean_halfplane(self, K: jnp.ndarray) -> jnp.ndarray:
        Kc = K
        Kc = Kc.at[..., 0, 0].set(jnp.real(Kc[..., 0, 0]) + 0.0j)
        if self.H_pad % 2 == 0:
            Kc = Kc.at[..., self.H_pad // 2, 0].set(
                jnp.real(Kc[..., self.H_pad // 2, 0]) + 0.0j
            )
        if self.W_pad % 2 == 0:
            Kc = Kc.at[..., 0, self.W_half - 1].set(
                jnp.real(Kc[..., 0, self.W_half - 1]) + 0.0j
            )
            if self.H_pad % 2 == 0:
                Kc = Kc.at[..., self.H_pad // 2, self.W_half - 1].set(
                    jnp.real(Kc[..., self.H_pad // 2, self.W_half - 1]) + 0.0j
                )
        return Kc

    # ---- forward --------------------------------------------------------------
    @classmethod
    def from_conv2d(
        cls,
        conv: eqx.nn.Conv2d,
        *,
        H_in: int,
        W_in: int,
        H_pad: int,
        W_pad: int,
        crop_output: bool = True,
        use_soft_mask: bool = True,
        mask_steepness: float = 20.0,
        key: Any,
        alpha_init: float = 1.0,
    ) -> "RFFTCirculant2D":
        C_out, C_in, _, _ = conv.weight.shape
        return cls(
            C_in=C_in,
            C_out=C_out,
            H_in=H_in,
            W_in=W_in,
            H_pad=H_pad,
            W_pad=W_pad,
            crop_output=crop_output,
            use_soft_mask=use_soft_mask,
            mask_steepness=mask_steepness,
            alpha_init=alpha_init,
            key=key,
        )

    def __call__(self, x: jnp.ndarray, *_, key: Any | None = None, **__) -> jnp.ndarray:
        # Accept (C,H,W) or (B,C,H,W)
        single = x.ndim == 3
        if single:
            x = x[None, ...]

        pad_h = max(0, self.H_pad - x.shape[-2])
        pad_w = max(0, self.W_pad - x.shape[-1])
        x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[
            ..., : self.H_pad, : self.W_pad
        ]

        Xf = jnp.fft.rfft2(x_pad, axes=(-2, -1), norm="ortho")

        K = self.K_half
        m = self._lowpass_mask_half()
        if m is not None:
            K = K * jax.lax.stop_gradient(m)[None, None, :, :]
        K = self._clean_halfplane(K)

        Yf = jnp.einsum("oihw,bihw->bohw", K, Xf)
        y = jnp.fft.irfft2(
            Yf, s=(self.H_pad, self.W_pad), axes=(-2, -1), norm="ortho"
        ).real

        y = y + self.bias[None, :, None, None]

        if self.crop_output and (self.H_pad != self.H_in or self.W_pad != self.W_in):
            y = y[..., : self.H_in, : self.W_in]

        return y[0] if single else y

    def __operator_norm_hint__(self):
        K = self.K_half
        m = self._lowpass_mask_half()
        if m is not None:
            K = K * jax.lax.stop_gradient(m)[None, None, :, :]
        K = self._clean_halfplane(K)
        HW = self.H_pad * self.W_half
        Khw = jnp.transpose(
            K.reshape(self.C_out, self.C_in, HW), (2, 0, 1)
        )  # (HW, Co, Ci)
        svals = jax.vmap(lambda M: jnp.linalg.svd(M, compute_uv=False)[0])(Khw)
        # exact per-frequency SVD over half-plane; return JAX scalar (jit-safe)
        return jnp.max(jnp.real(svals)).astype(jnp.float32)


class SpectralTokenMixer(eqx.Module, _SpectralMixin):
    """
    Spectral Token Mixer for sequences:
      x ∈ R^{B,N,C} or R^{N,C} → y = iRFFT( RFFT(x along N) ⊙ H_half ).
    Grouped: share one spectral filter per group of channels (C/G groups).
    If use_residual=True, returns x + mix(x).
    """

    n_tokens: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    groups: int = eqx.field(static=True)
    k_half: int = eqx.field(static=True)
    use_residual: bool = eqx.field(static=True)

    alpha_raw: jnp.ndarray  # scalar
    H_half: jnp.ndarray  # (groups, k_half) complex
    gate: jnp.ndarray  # (channels,) real scalar per-channel

    def __init__(
        self,
        n_tokens: int,
        channels: int,
        *,
        groups: int = 1,
        alpha_init: float = 1.0,
        key: Any,
        init_scale: float = 0.1,
        use_residual: bool = True,
        dtype=jnp.float32,
    ):
        assert channels % groups == 0, "channels must be divisible by groups"
        self.n_tokens = int(n_tokens)
        self.channels = int(channels)
        self.groups = int(groups)
        self.k_half = self.n_tokens // 2 + 1
        self.use_residual = bool(use_residual)

        self.alpha_raw = jnp.asarray(alpha_init, dtype)

        k = jnp.arange(self.k_half, dtype=dtype)
        k_norm = k / jnp.maximum(self.k_half - 1, 1)
        α0 = _alpha_bounded(self.alpha_raw)
        std0 = 1.0 / jnp.sqrt(1.0 + k_norm**α0)

        k1, k2 = jr.split(key, 2)
        wr = jr.normal(k1, (self.groups, self.k_half), dtype) * (
            init_scale * std0[None, :]
        )
        wi = jr.normal(k2, (self.groups, self.k_half), dtype) * (
            init_scale * std0[None, :]
        )
        wi = wi.at[:, 0].set(0.0)
        if self.n_tokens % 2 == 0:
            wi = wi.at[:, -1].set(0.0)
        self.H_half = wr + 1j * wi

        self.gate = jnp.ones((self.channels,), dtype=dtype)

    def _spectral_weights(self) -> jnp.ndarray:
        return jnp.concatenate(
            [self.H_half.real.reshape(-1), self.H_half.imag.reshape(-1)]
        )

    def _spectral_scale(self) -> jnp.ndarray:
        α = _alpha_bounded(self.alpha_raw)
        k = jnp.arange(self.k_half, dtype=self.H_half.real.dtype)
        k_norm = k / jnp.maximum(self.k_half - 1, 1)
        β = 1.0 + k_norm**α
        flat = jnp.tile(β, self.groups)
        return jnp.concatenate([flat, flat])

    def __call__(
        self, x: jnp.ndarray, *args, key: Any | None = None, **kwargs
    ) -> jnp.ndarray:
        # Accept (N,C) or (B,N,C); return matching batch structure.
        if x.ndim == 2:
            x = x[None, ...]  # (1,N,C)
            single = True
        elif x.ndim == 3:
            single = False
        else:
            raise ValueError(f"Expected (N,C) or (B,N,C), got {x.shape}")

        B, N, C = x.shape
        if (N != self.n_tokens) or (C != self.channels):
            raise ValueError(
                f"Expected (*,{self.n_tokens},{self.channels}), got {x.shape}"
            )

        Xf = jnp.fft.rfft(x, axis=1, norm="ortho")  # (B, k_half, C)

        G = self.groups
        ch_per_g = C // G
        H = self.H_half  # (G, k_half)

        # Apply per-group filter to tokens
        pieces = []
        for g in range(G):
            sl = slice(g * ch_per_g, (g + 1) * ch_per_g)
            pieces.append(Xf[:, :, sl] * H[g][None, :, None])
        Xf_out = jnp.concatenate(pieces, axis=-1)

        y = jnp.fft.irfft(Xf_out, n=N, axis=1, norm="ortho").real  # (B,N,C)

        y = y * self.gate[None, None, :]
        y = x + y if self.use_residual else y

        return y[0] if single else y

    # NEW (conservative; residual adds ≤1)
    def __operator_norm_hint__(self):
        # exact with gate; +1 if residual
        g = jnp.max(jnp.abs(self.gate))
        h = jnp.max(jnp.abs(self.H_half))
        base = (g * h).astype(jnp.float32)
        return base + jnp.array(1.0, jnp.float32) if self.use_residual else base


class SVDDense(eqx.Module, _SpectralMixin):
    """Rectangular SVD dense: W = U diag(s) V^T, U∈R^{out×r}, V∈R^{in×r}."""

    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    U: jnp.ndarray  # (out,r)
    V: jnp.ndarray  # (in,r)
    s: jnp.ndarray  # (r,)
    bias: jnp.ndarray  # (out,)
    alpha_raw: jnp.ndarray

    def __init__(self, U, V, s, bias, alpha_init=1.0):
        r = s.shape[0]
        assert U.shape == (bias.shape[0], r)
        assert V.shape[1] == r
        self.in_features = V.shape[0]
        self.out_features = U.shape[0]
        self.U, self.V, self.s = U, V, s
        self.bias = bias
        self.alpha_raw = jnp.asarray(alpha_init, jnp.float32)

    def _spectral_weights(self):
        return self.s

    def _spectral_scale(self):
        r = self.s.shape[0]
        k = jnp.arange(r, dtype=self.s.dtype)
        k_norm = k / jnp.maximum(r - 1, 1)
        α = _alpha_bounded(self.alpha_raw)
        return 1.0 + k_norm**α

    def __call__(
        self, x: jnp.ndarray, *args, key: Any | None = None, **kwargs
    ) -> jnp.ndarray:
        z = x @ self.V
        z = z * self.s
        y = z @ self.U.T
        return y + self.bias

    def __operator_norm_hint__(self):
        # exact: max |s| ; return JAX scalar (jit-safe)
        return jnp.max(jnp.abs(self.s)).astype(jnp.float32)

    @classmethod
    def from_linear(
        cls, lin: eqx.nn.Linear, *, rank: int | None = None, alpha_init: float = 1.0
    ):
        """Exact SVD warm-start from an eqx.nn.Linear."""
        W = lin.weight  # (out, in)
        U, s, Vh = jnp.linalg.svd(
            W, full_matrices=False
        )  # U:(out,r), s:(r,), Vh:(r,in)
        if rank is not None:
            r = int(min(rank, s.shape[0]))
            U, s, Vh = U[:, :r], s[:r], Vh[:r, :]
        V = Vh.T
        bias = (
            lin.bias
            if getattr(lin, "bias", None) is not None
            else jnp.zeros((W.shape[0],), W.dtype)
        )
        return cls(U=U, V=V, s=s, bias=bias, alpha_init=alpha_init)


# GraphChebSobolev implements a Chebyshev spectral graph filter with a Sobolev-style
# spectral penalty applied to polynomial coefficients. It supports:
#  - learnable bounded smoothness exponent alpha in β_k = 1 + (k/K)^alpha
#  - order-K Chebyshev filter via recurrence (no eigendecomposition)
#  - optional bias
#  - works with dense Laplacian or a user-supplied matvec for sparse graphs
#
# Notation
# --------
#  Given normalized Laplacian L (symmetric PSD), rescale to L̃ = (2/λ_max) L - I,
#  so that spec(L̃) ⊆ [-1, 1]. Then T_0(X)=X, T_1(X)=L̃X, T_k(X)=2 L̃ T_{k-1}(X) - T_{k-2}(X).
#  Output: Y = sum_{k=0..K} T_k(X) @ W_k + bias.
#
# Penalty
# -------
#  Spectral penalty on weights: sum_k β_k ||W_k||_F^2 with β_k = 1 + (k/K)^alpha.
#  This is a discrete Sobolev envelope on polynomial degree. It integrates with your
#  global_spectral_penalty (via _SpectralMixin).
#
# Shapes
# ------
#  X: (N, Fin)  or (B, N, Fin)
#  W_k: (Fin, Fout)  for each k = 0..K
#  Y: (N, Fout) or (B, N, Fout)


def _power_iter_sym(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    dim: int,
    n_iter: int = 50,
    key: Any = None,
) -> float:
    """Power iteration to approximate largest eigenvalue of symmetric PSD operator."""
    if key is None:
        key = jr.PRNGKey(0)
    v = jr.normal(key, (dim,))
    v = v / (jnp.linalg.norm(v) + 1e-12)

    def body(_, v):
        w = matvec(v)
        w_norm = jnp.linalg.norm(w) + 1e-12
        return w / w_norm

    v = jax.lax.fori_loop(0, n_iter, body, v)
    λ = jnp.dot(v, matvec(v))
    return float(jnp.maximum(λ, 0.0))


def _make_dense_matvec(A: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return lambda x: A @ x


def _batchify_mv(matvec: Callable[[jnp.ndarray], jnp.ndarray]):
    """Lift a vector matvec to matrix/batched: applies along the last dim."""

    def mv(X):
        # Accept (..., N, Fin) and do matvec over N dimension for each feature column
        # Here matvec is defined over vectors in R^N; so we vmap over feature columns.
        if X.ndim == 2:
            # (N, Fin)
            return jax.vmap(matvec, in_axes=1, out_axes=1)(X)
        elif X.ndim == 3:
            # (B, N, Fin)
            return jax.vmap(lambda Z: jax.vmap(matvec, in_axes=1, out_axes=1)(Z))(X)
        else:
            raise ValueError(f"Expected (N,Fin) or (B,N,Fin), got {X.shape}")

    return mv


class GraphChebSobolev(eqx.Module, _SpectralMixin):
    """
    Chebyshev graph filter with Sobolev penalty on polynomial degree.

    Args:
      N: number of nodes.
      Fin: input features.
      Fout: output features.
      K: Chebyshev order (>=0).
      L: (optional) dense normalized Laplacian in R^{N×N}. If given, we estimate
         λ_max from L and internally build L̃ matvec.
      matvec: (optional) callable v -> L v for sparse/implicit graphs.
      lmax: (optional) largest eigenvalue of L, if you already know it (speeds init).
      bias: include bias term per output feature.
      alpha_init: initial smoothness exponent in β_k = 1 + (k/K)^α.
      key: PRNGKey for init.

    Either provide L (dense) or provide matvec (+ optionally lmax).

    Shapes:
      X: (N,Fin) or (B,N,Fin); returns (N,Fout) or (B,N,Fout).
    """

    N: int = eqx.field(static=True)
    Fin: int = eqx.field(static=True)
    Fout: int = eqx.field(static=True)
    K: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    # Operator state (static)
    _Ltilde_mv: Callable[[jnp.ndarray], jnp.ndarray] = eqx.field(static=True)

    # Parameters
    alpha_raw: jnp.ndarray  # scalar
    W: jnp.ndarray  # (K+1, Fin, Fout)
    bias: Optional[jnp.ndarray]  # (Fout,)

    def __init__(
        self,
        N: int,
        Fin: int,
        Fout: int,
        K: int,
        *,
        L: Optional[jnp.ndarray] = None,
        matvec: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        lmax: Optional[float] = None,
        bias: bool = True,
        alpha_init: float = 1.0,
        key: Any,
        init_scale: float = 0.1,
        dtype=jnp.float32,
    ):
        assert K >= 0
        self.N, self.Fin, self.Fout, self.K = int(N), int(Fin), int(Fout), int(K)
        self.use_bias = bool(bias)

        # Build L̃ matvec
        if (L is None) == (matvec is None):
            raise ValueError(
                "Provide exactly one of `L` (dense) or `matvec` (callable)."
            )

        if L is not None:
            if L.shape != (N, N):
                raise ValueError(f"L shape must be {(N,N)}, got {L.shape}")
            mv = _make_dense_matvec(L)
            if lmax is None:
                lmax = _power_iter_sym(mv, N, n_iter=60)
            lscale = 2.0 / max(lmax, 1e-12)
            Ltilde_mv = lambda x: lscale * (L @ x) - x
        else:
            assert matvec is not None
            mv = matvec
            if lmax is None:
                lmax = _power_iter_sym(mv, N, n_iter=60)
            lscale = 2.0 / max(lmax, 1e-12)
            Ltilde_mv = lambda x: lscale * matvec(x) - x

        # Store batched matvec over node-dimension.
        self._Ltilde_mv = _batchify_mv(Ltilde_mv)

        # Parameters
        self.alpha_raw = jnp.asarray(alpha_init, dtype)
        k1, k2 = jr.split(key, 2)
        self.W = jr.normal(k1, (K + 1, Fin, Fout), dtype) * init_scale
        self.bias = (jr.normal(k2, (Fout,), dtype) * init_scale) if bias else None

    # ---- spectral penalty hooks ----
    def _spectral_weights(self) -> jnp.ndarray:
        return self.W.reshape(-1)

    def _spectral_scale(self) -> jnp.ndarray:
        α = _alpha_bounded(self.alpha_raw)
        k = jnp.arange(self.K + 1, dtype=self.W.dtype)
        k_norm = jnp.where(self.K == 0, 0.0, k / self.K)
        β = 1.0 + (k_norm**α)  # (K+1,)
        # replicate per parameter in W_k
        per_k = self.Fin * self.Fout
        β_full = jnp.repeat(β, per_k)
        return β_full

    # ---- forward ----
    def __call__(
        self, X: jnp.ndarray, *args, key: Any | None = None, **kwargs
    ) -> jnp.ndarray:
        """
        X: (N,Fin) or (B,N,Fin). Returns Y with same batch structure and Fout channels.
        """
        if X.ndim == 2:
            X = X[None, ...]  # (1,N,Fin)

        B, N, Fin = X.shape
        if N != self.N or Fin != self.Fin:
            raise ValueError(
                f"X shape mismatch; expected (*,{self.N},{self.Fin}), got {X.shape}"
            )

        # Chebyshev stack T_k X via recurrence
        T0 = X  # (B,N,Fin)
        out = T0 @ self.W[0]  # (B,N,Fout)
        if self.K >= 1:
            T1 = self._Ltilde_mv(X)  # (B,N,Fin)
            out = out + (T1 @ self.W[1])
            Tk_2, Tk_1 = T0, T1
            for k in range(2, self.K + 1):
                Tk = 2.0 * self._Ltilde_mv(Tk_1) - Tk_2
                out = out + (Tk @ self.W[k])
                Tk_2, Tk_1 = Tk_1, Tk

        if self.use_bias and self.bias is not None:
            out = out + self.bias[None, None, :]

        return out[0] if B == 1 else out
