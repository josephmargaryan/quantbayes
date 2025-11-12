# numpyro_spectral_layers.py
# -------------------------------------------------------------------------
# Probabilistic analogues of your Equinox "spectral" layers implemented in NumPyro.
# Shapes, FFT conventions, and equivariance-preserving biases are matched to the
# Equinox versions. FFTs use norm="ortho" for consistent scaling.
# -------------------------------------------------------------------------

from __future__ import annotations
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


__all__ = [
    # helpers
    "spectral_circulant_matmul",
    "spectral_circulant_conv2d",
    # 1D circulant (full FFT)
    "SpectralCirculantLayer",
    "AdaptiveSpectralCirculantLayer",
    # 2D circulant (full FFT)
    "SpectralCirculantLayer2d",
    "AdaptiveSpectralCirculantLayer2d",
    # 1D & 2D circulant (RFFT storage)
    "RFFTCirculant1D",
    "RFFTCirculant2D",
    "RFFTCirculant2D_Sparse",
    # SVD-parametrized dense & conv
    "SpectralDense",
    "AdaptiveSpectralDense",
    "SVDDense",
    "SpectralConv2d",
    "AdaptiveSpectralConv2d",
    # Token mixer (sequence-wise spectral filtering)
    "SpectralTokenMixer",
]


# ------------------------------- utilities ----------------------------------


def _alpha_bounded(
    alpha_raw: jnp.ndarray, lo: float = 0.1, hi: float = 3.0
) -> jnp.ndarray:
    """Clamp alpha into [lo, hi] via a sigmoid map (keeps gradients healthy)."""
    return lo + (hi - lo) * jax.nn.sigmoid(alpha_raw)


def _half_to_full_hermitian(half: jnp.ndarray, padded_dim: int) -> jnp.ndarray:
    """Build full complex spectrum from RFFT-style half (length k_half)."""
    k_half = padded_dim // 2 + 1
    if padded_dim % 2 == 0 and k_half > 1:
        nyq = half[-1].real[None]
        full = jnp.concatenate([half[:-1], nyq, jnp.conj(half[1:-1])[::-1]])
    else:
        full = jnp.concatenate([half, jnp.conj(half[1:])[::-1]])
    return full


def _enforce_hermitian_2d(fft2d: jnp.ndarray) -> jnp.ndarray:
    """
    Project complex (..., H, W) onto Hermitian subspace (real IFFT2 output).
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


def _bilinear_upsample(grid: jnp.ndarray, H_pad: int, W_pad: int) -> jnp.ndarray:
    """Bilinearly upsample a coarse (gh, gw) grid to (H_pad, W_pad)."""
    gh, gw = grid.shape
    y = jnp.linspace(0.0, 1.0, H_pad)
    x = jnp.linspace(0.0, 1.0, W_pad)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    yy = yy * (gh - 1)
    xx = xx * (gw - 1)
    y0, x0 = jnp.floor(yy).astype(int), jnp.floor(xx).astype(int)
    y1 = jnp.clip(y0 + 1, 0, gh - 1)
    x1 = jnp.clip(x0 + 1, 0, gw - 1)
    wy, wx = yy - y0, xx - x0

    v00 = grid[y0, x0]
    v01 = grid[y0, x1]
    v10 = grid[y1, x0]
    v11 = grid[y1, x1]

    return (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11)


def _default_prior(scale: jnp.ndarray) -> dist.Distribution:
    """Default zero-mean Gaussian prior with elementwise scale."""
    return dist.Normal(0.0, jnp.asarray(scale))


# ------------------------ FFT-based primitive ops ----------------------------


@jax.custom_jvp
def spectral_circulant_matmul(x: jnp.ndarray, fft_full: jnp.ndarray) -> jnp.ndarray:
    """
    y = IFFT( FFT(x_pad) * fft_full ), along the last dimension.
    Orthonormal normalization to keep scaling consistent.
      - x: (..., d_in) real
      - fft_full: (padded_dim,) complex; full Hermitian spectrum
      -> (..., padded_dim) real
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
        x = x[None, ...]  # (1, C_in, H, W)

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


# -------------------------- 1D spectral-circulant ----------------------------


class SpectralCirculantLayer:
    """
    1D spectral-circulant linear map with Sobolev-style prior on Fourier modes.
    Bias: scalar by default (equivariance-preserving). Set vector_bias=True only
    if you intentionally want per-position bias (breaks shift equivariance).
    """

    def __init__(
        self,
        in_features: int,
        *,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        crop_output: bool = True,
        alpha: Optional[float] = None,  # if None, sample it
        name: str = "spec1d",
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        vector_bias: bool = False,
        dtype=jnp.float32,
    ):
        self.in_features = int(in_features)
        self.padded_dim = int(padded_dim or in_features)
        self.k_half = self.padded_dim // 2 + 1
        self.K = int(self.k_half if (K is None or K > self.k_half) else K)
        self.crop_output = bool(crop_output)
        self.alpha = alpha
        self.name = name
        self.prior_fn = prior_fn or _default_prior
        self.vector_bias = bool(vector_bias)
        self.dtype = dtype
        self._last_fft_full = None  # for inspection

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # α
        if self.alpha is None:
            alpha_z = numpyro.sample(f"{self.name}_alpha_z", dist.Normal(0.0, 1.0))
            alpha = _alpha_bounded(alpha_z)
        else:
            alpha = jnp.asarray(self.alpha, self.dtype)

        k = jnp.arange(self.k_half, dtype=self.dtype)
        k_norm = k / jnp.maximum(self.k_half - 1, 1)
        std = 1.0 / jnp.sqrt(1.0 + k_norm**alpha)

        active_idx = jnp.arange(self.K)
        std_active = std[active_idx]

        real = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(std_active).to_event(1),
        )
        imag = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(std_active).to_event(1),
        )

        full_r = jnp.zeros((self.k_half,), self.dtype).at[active_idx].set(real)
        full_i = jnp.zeros((self.k_half,), self.dtype).at[active_idx].set(imag)
        full_i = full_i.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            full_i = full_i.at[-1].set(0.0)

        half = full_r + 1j * full_i
        fft_full = _half_to_full_hermitian(half, self.padded_dim)
        self._last_fft_full = jax.lax.stop_gradient(fft_full)

        if self.vector_bias:
            bias = numpyro.sample(
                f"{self.name}_bias_vec",
                dist.Normal(0.0, 1.0).expand([self.padded_dim]).to_event(1),
            )
        else:
            bias = numpyro.sample(f"{self.name}_bias", dist.Normal(0.0, 1.0))

        y = spectral_circulant_matmul(x, fft_full)
        y = y + (bias if self.vector_bias else bias)
        if not self.crop_output or (self.padded_dim == self.in_features):
            return y
        return y[..., : self.in_features]

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self._last_fft_full is None:
            raise RuntimeError("Call the layer once to build its spectrum.")
        return self._last_fft_full


class AdaptiveSpectralCirculantLayer(SpectralCirculantLayer):
    """
    1D spectral-circulant with per-frequency extra exponent δ_k (softplus).
    """

    def __init__(
        self,
        in_features: int,
        *,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        crop_output: bool = True,
        alpha_global: float = 1.0,
        name: str = "adap_spec1d",
        alpha_prior: dist.Distribution = dist.HalfNormal(1.0),
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        vector_bias: bool = False,
        dtype=jnp.float32,
    ):
        super().__init__(
            in_features=in_features,
            padded_dim=padded_dim,
            K=K,
            crop_output=crop_output,
            alpha=None,
            name=name,
            prior_fn=prior_fn,
            vector_bias=vector_bias,
            dtype=dtype,
        )
        self.alpha_global = float(alpha_global)
        self.alpha_prior = alpha_prior

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        delta_z = numpyro.sample(
            f"{self.name}_delta",
            self.alpha_prior.expand([self.k_half]).to_event(1),
        )
        alpha_k = _alpha_bounded(self.alpha_global + jax.nn.softplus(delta_z))

        k = jnp.arange(self.k_half, dtype=self.dtype)
        k_norm = k / jnp.maximum(self.k_half - 1, 1)
        std = 1.0 / jnp.sqrt(1.0 + k_norm**alpha_k)

        active_idx = jnp.arange(self.K)
        std_active = std[active_idx]

        real = numpyro.sample(
            f"{self.name}_real",
            (self.prior_fn(std_active)).to_event(1),
        )
        imag = numpyro.sample(
            f"{self.name}_imag",
            (self.prior_fn(std_active)).to_event(1),
        )

        full_r = jnp.zeros((self.k_half,), self.dtype).at[active_idx].set(real)
        full_i = jnp.zeros((self.k_half,), self.dtype).at[active_idx].set(imag)
        full_i = full_i.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            full_i = full_i.at[-1].set(0.0)

        half = full_r + 1j * full_i
        fft_full = _half_to_full_hermitian(half, self.padded_dim)
        self._last_fft_full = jax.lax.stop_gradient(fft_full)

        if self.vector_bias:
            bias = numpyro.sample(
                f"{self.name}_bias_vec",
                dist.Normal(0.0, 1.0).expand([self.padded_dim]).to_event(1),
            )
        else:
            bias = numpyro.sample(f"{self.name}_bias", dist.Normal(0.0, 1.0))

        y = spectral_circulant_matmul(x, fft_full)
        y = y + (bias if self.vector_bias else bias)
        if not self.crop_output or (self.padded_dim == self.in_features):
            return y
        return y[..., : self.in_features]


# -------------------------- 2D spectral-circulant ----------------------------


class SpectralCirculantLayer2d:
    """
    2D spectral-circulant conv with scalar α and optional low-mode mask.
    Bias: per-output-channel (C_out,) to preserve translation equivariance.
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        H_in: int,
        W_in: int,
        *,
        H_pad: Optional[int] = None,
        W_pad: Optional[int] = None,
        alpha: Optional[float] = None,
        K_rad: Optional[float] = None,  # normalized cutoff radius in [0,1]
        use_soft_mask: bool = True,
        mask_steepness: float = 40.0,
        crop_output: bool = True,
        name: str = "spec2d",
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        dtype=jnp.float32,
    ):
        self.C_in, self.C_out = int(C_in), int(C_out)
        self.H_in, self.W_in = int(H_in), int(W_in)
        self.H_pad = int(H_pad or H_in)
        self.W_pad = int(W_pad or W_in)
        self.alpha = alpha
        self.K_rad = None if K_rad is None else float(jnp.clip(K_rad, 0.0, 1.0))
        self.use_soft_mask = bool(use_soft_mask)
        self.mask_steepness = float(mask_steepness)
        self.crop_output = bool(crop_output)
        self.name = name
        self.prior_fn = prior_fn or _default_prior
        self.dtype = dtype
        self._fft_kernel = None

    def _lowpass_mask(self) -> Optional[jnp.ndarray]:
        if self.K_rad is None:
            return None
        u = jnp.fft.fftfreq(self.H_pad, dtype=self.dtype) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad, dtype=self.dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        Rn = R / jnp.maximum(jnp.max(R), 1.0)
        if self.use_soft_mask:
            return jax.nn.sigmoid(self.mask_steepness * (self.K_rad - Rn))
        else:
            return (Rn <= self.K_rad).astype(self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # α
        if self.alpha is None:
            alpha_z = numpyro.sample(f"{self.name}_alpha_z", dist.Normal(0.0, 1.0))
            alpha = _alpha_bounded(alpha_z)
        else:
            alpha = jnp.asarray(self.alpha, self.dtype)

        u = jnp.fft.fftfreq(self.H_pad, dtype=self.dtype) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad, dtype=self.dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        Rn = R / jnp.maximum(jnp.max(R), 1.0)
        std2d = 1.0 / jnp.sqrt(1.0 + Rn**alpha)
        std = std2d[None, None, :, :].astype(self.dtype)

        real = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(std)
            .expand((self.C_out, self.C_in, self.H_pad, self.W_pad))
            .to_event(4),
        )
        imag = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(std)
            .expand((self.C_out, self.C_in, self.H_pad, self.W_pad))
            .to_event(4),
        )

        K = _enforce_hermitian_2d(real + 1j * imag)
        m = self._lowpass_mask()
        if m is not None:
            K = K * m[None, None, :, :]

        self._fft_kernel = jax.lax.stop_gradient(K)

        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0.0, 1.0).expand([self.C_out]).to_event(1),
        )

        y = spectral_circulant_conv2d(x, K)
        y = y + (bias[:, None, None] if y.ndim == 3 else bias[None, :, None, None])

        if self.crop_output and (self.H_pad != self.H_in or self.W_pad != self.W_in):
            y = y[..., : self.H_in, : self.W_in]
        return y

    def get_fft_kernel(self) -> jnp.ndarray:
        if self._fft_kernel is None:
            raise RuntimeError("Call the layer once to build its kernel.")
        return self._fft_kernel


class AdaptiveSpectralCirculantLayer2d(SpectralCirculantLayer2d):
    """
    2D variant with per-frequency extra exponent δ(u,v) (softplus).
    You can optionally sample a coarse grid and bilinearly upsample to (H_pad,W_pad).
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        H_in: int,
        W_in: int,
        *,
        H_pad: Optional[int] = None,
        W_pad: Optional[int] = None,
        alpha_global: float = 1.0,
        alpha_coarse_shape: Tuple[int, int] = (8, 8),
        name: str = "adap_spec2d",
        alpha_prior: dist.Distribution = dist.HalfNormal(1.0),
        K_rad: Optional[float] = None,
        use_soft_mask: bool = True,
        mask_steepness: float = 40.0,
        crop_output: bool = True,
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        dtype=jnp.float32,
    ):
        super().__init__(
            C_in=C_in,
            C_out=C_out,
            H_in=H_in,
            W_in=W_in,
            H_pad=H_pad,
            W_pad=W_pad,
            alpha=None,
            K_rad=K_rad,
            use_soft_mask=use_soft_mask,
            mask_steepness=mask_steepness,
            crop_output=crop_output,
            name=name,
            prior_fn=prior_fn,
            dtype=dtype,
        )
        self.alpha_global = float(alpha_global)
        self.alpha_coarse_shape = tuple(alpha_coarse_shape)
        self.alpha_prior = alpha_prior

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Sample coarse δ-grid and upsample
        delta_coarse = numpyro.sample(
            f"{self.name}_delta_coarse",
            self.alpha_prior.expand(self.alpha_coarse_shape).to_event(2),
        )
        delta_full = jax.nn.softplus(
            _bilinear_upsample(delta_coarse, self.H_pad, self.W_pad)
        )
        alpha_map = _alpha_bounded(self.alpha_global + delta_full)

        u = jnp.fft.fftfreq(self.H_pad, dtype=self.dtype) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad, dtype=self.dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        Rn = R / jnp.maximum(jnp.max(R), 1.0)
        std2d = 1.0 / jnp.sqrt(1.0 + Rn**alpha_map)
        std = std2d[None, None, :, :].astype(self.dtype)

        real = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(std)
            .expand((self.C_out, self.C_in, self.H_pad, self.W_pad))
            .to_event(4),
        )
        imag = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(std)
            .expand((self.C_out, self.C_in, self.H_pad, self.W_pad))
            .to_event(4),
        )

        K = _enforce_hermitian_2d(real + 1j * imag)
        m = self._lowpass_mask()
        if m is not None:
            K = K * m[None, None, :, :]

        self._fft_kernel = jax.lax.stop_gradient(K)

        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0.0, 1.0).expand([self.C_out]).to_event(1),
        )
        y = spectral_circulant_conv2d(x, K)
        y = y + (bias[:, None, None] if y.ndim == 3 else bias[None, :, None, None])

        if self.crop_output and (self.H_pad != self.H_in or self.W_pad != self.W_in):
            y = y[..., : self.H_in, : self.W_in]
        return y


# ----------------------------- RFFT variants ---------------------------------


class RFFTCirculant1D:
    """
    1D spectral-circulant with RFFT storage (half-spectrum).
    Bias: scalar by default; set vector_bias=True to allow positional bias.
    """

    def __init__(
        self,
        in_features: int,
        *,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        crop_output: bool = True,
        alpha: Optional[float] = None,
        name: str = "rfft1d",
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        vector_bias: bool = False,
        dtype=jnp.float32,
    ):
        self.in_features = int(in_features)
        self.padded_dim = int(padded_dim or in_features)
        self.k_half = self.padded_dim // 2 + 1
        self.K = int(self.k_half if (K is None or K > self.k_half) else K)
        self.crop_output = bool(crop_output)
        self.alpha = alpha
        self.name = name
        self.prior_fn = prior_fn or _default_prior
        self.vector_bias = bool(vector_bias)
        self.dtype = dtype
        self._last_half = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # α
        if self.alpha is None:
            az = numpyro.sample(f"{self.name}_alpha_z", dist.Normal(0.0, 1.0))
            alpha = _alpha_bounded(az)
        else:
            alpha = jnp.asarray(self.alpha, self.dtype)

        k = jnp.arange(self.k_half, dtype=self.dtype)
        k_norm = k / jnp.maximum(self.k_half - 1, 1)
        std = 1.0 / jnp.sqrt(1.0 + k_norm**alpha)

        act = jnp.arange(self.K)
        std_act = std[act]

        wr = numpyro.sample(f"{self.name}_real", self.prior_fn(std_act).to_event(1))
        wi = numpyro.sample(f"{self.name}_imag", self.prior_fn(std_act).to_event(1))

        H_r = jnp.zeros((self.k_half,), self.dtype).at[act].set(wr)
        H_i = jnp.zeros((self.k_half,), self.dtype).at[act].set(wi)
        H_i = H_i.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            H_i = H_i.at[-1].set(0.0)

        H_half = H_r + 1j * H_i
        self._last_half = jax.lax.stop_gradient(H_half)

        # pad/truncate
        d_in = x.shape[-1]
        if d_in < self.padded_dim:
            pad = [(0, 0)] * x.ndim
            pad[-1] = (0, self.padded_dim - d_in)
            x_pad = jnp.pad(x, pad)
        else:
            x_pad = x[..., : self.padded_dim]

        Xf = jnp.fft.rfft(x_pad, axis=-1, norm="ortho")
        Yf = Xf * H_half
        y = jnp.fft.irfft(Yf, n=self.padded_dim, axis=-1, norm="ortho").real

        if self.vector_bias:
            b = numpyro.sample(
                f"{self.name}_bias_vec",
                dist.Normal(0.0, 1.0).expand([self.padded_dim]).to_event(1),
            )
            y = y + (b if y.ndim == 1 else b[None, :])
        else:
            b = numpyro.sample(f"{self.name}_bias", dist.Normal(0.0, 1.0))
            y = y + b

        if not self.crop_output or (self.padded_dim == self.in_features):
            return y
        return y[..., : self.in_features]

    def get_half_spectrum(self) -> jnp.ndarray:
        if self._last_half is None:
            raise RuntimeError("Call the layer once to build its half spectrum.")
        return self._last_half


class RFFTCirculant2D:
    """
    2D spectral-circulant conv with RFFT2 storage (half-plane).
    Bias: per-output-channel (C_out,) to preserve translation equivariance.
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        H_in: int,
        W_in: int,
        *,
        H_pad: Optional[int] = None,
        W_pad: Optional[int] = None,
        alpha: Optional[float] = None,
        K_rad: Optional[float] = None,
        use_soft_mask: bool = True,
        mask_steepness: float = 40.0,
        crop_output: bool = True,
        name: str = "rfft2d",
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        dtype=jnp.float32,
    ):
        self.C_in, self.C_out = int(C_in), int(C_out)
        self.H_in, self.W_in = int(H_in), int(W_in)
        self.H_pad = int(H_pad or H_in)
        self.W_pad = int(W_pad or W_in)
        self.W_half = self.W_pad // 2 + 1
        self.alpha = alpha
        self.K_rad = None if K_rad is None else float(jnp.clip(K_rad, 0.0, 1.0))
        self.use_soft_mask = bool(use_soft_mask)
        self.mask_steepness = float(mask_steepness)
        self.crop_output = bool(crop_output)
        self.name = name
        self.prior_fn = prior_fn or _default_prior
        self.dtype = dtype
        self._K_half = None

    def _lowpass_mask_half(self) -> Optional[jnp.ndarray]:
        if self.K_rad is None:
            return None
        u = jnp.fft.fftfreq(self.H_pad, dtype=self.dtype) * self.H_pad
        v = jnp.fft.rfftfreq(self.W_pad, dtype=self.dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        Rn = R / jnp.maximum(jnp.max(R), 1.0)
        if self.use_soft_mask:
            return jax.nn.sigmoid(self.mask_steepness * (self.K_rad - Rn))
        else:
            return (Rn <= self.K_rad).astype(self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # α
        if self.alpha is None:
            az = numpyro.sample(f"{self.name}_alpha_z", dist.Normal(0.0, 1.0))
            alpha = _alpha_bounded(az)
        else:
            alpha = jnp.asarray(self.alpha, self.dtype)

        u = jnp.fft.fftfreq(self.H_pad, dtype=self.dtype) * self.H_pad
        v = jnp.fft.rfftfreq(self.W_pad, dtype=self.dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        Rn = R / jnp.maximum(jnp.max(R), 1.0)
        std2d = 1.0 / jnp.sqrt(1.0 + Rn**alpha)  # (H_pad, W_half)
        std = std2d[None, None, :, :].astype(self.dtype)

        wr = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(std)
            .expand((self.C_out, self.C_in, self.H_pad, self.W_half))
            .to_event(4),
        )
        wi = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(std)
            .expand((self.C_out, self.C_in, self.H_pad, self.W_half))
            .to_event(4),
        )
        # impose purely real on DC (+ Nyquist column if even W_pad)
        wi = wi.at[..., :, 0].set(0.0)
        if (self.W_pad % 2 == 0) and (self.W_half > 1):
            wi = wi.at[..., :, -1].set(0.0)

        K_half = wr + 1j * wi
        m = self._lowpass_mask_half()
        if m is not None:
            K_half = K_half * m[None, None, :, :]

        self._K_half = jax.lax.stop_gradient(K_half)

        # pad/truncate input then RFFT2 conv
        single = x.ndim == 3
        if single:
            x = x[None, ...]

        pad_h = max(0, self.H_pad - x.shape[-2])
        pad_w = max(0, self.W_pad - x.shape[-1])
        x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[
            ..., : self.H_pad, : self.W_pad
        ]

        Xf = jnp.fft.rfft2(x_pad, axes=(-2, -1), norm="ortho")
        Yf = jnp.einsum("oihw,bihw->bohw", K_half, Xf)
        y = jnp.fft.irfft2(
            Yf, s=(self.H_pad, self.W_pad), axes=(-2, -1), norm="ortho"
        ).real

        b = numpyro.sample(
            f"{self.name}_bias", dist.Normal(0.0, 1.0).expand([self.C_out]).to_event(1)
        )
        y = y + (b[None, :, None, None])

        if self.crop_output and (self.H_pad != self.H_in or self.W_pad != self.W_in):
            y = y[..., : self.H_in, : self.W_in]

        return y[0] if single else y

    def get_half_kernel(self) -> jnp.ndarray:
        if self._K_half is None:
            raise RuntimeError("Call the layer once to build its half-plane kernel.")
        return self._K_half


class RFFTCirculant2D_Sparse(RFFTCirculant2D):
    def __init__(self, *args, active_idx=None, M=None, radial=True, **kw):
        super().__init__(*args, **kw)
        # Build active index set 𝒜 of size M on the half-plane (H_pad, W_half)
        if active_idx is None:
            Hh, Wh = self.H_pad, self.W_half
            uu = jnp.fft.fftfreq(self.H_pad) * self.H_pad
            vv = jnp.fft.rfftfreq(self.W_pad) * self.W_pad
            U, V = jnp.meshgrid(uu, vv, indexing="ij")
            R = jnp.sqrt(U**2 + V**2)
            # rank by radius then lexicographic; skip DC/Nyquist handling as desired
            order = jnp.argsort(R.reshape(-1), kind="stable")
            if M is None:
                raise ValueError("Provide M or active_idx.")
            flat_idx = order[:M]
            i = flat_idx // Wh
            j = flat_idx % Wh
            self.active_idx = jnp.stack([i, j], axis=-1)  # (M,2)
        else:
            self.active_idx = jnp.asarray(active_idx, jnp.int32)  # (M,2)

    def __call__(self, x):
        # sample only on active indices (real+imag), apply mask by scatter_add
        if self.alpha is None:
            az = numpyro.sample(f"{self.name}_alpha_z", dist.Normal(0.0, 1.0))
            alpha = _alpha_bounded(az)
        else:
            alpha = jnp.asarray(self.alpha, self.dtype)

        # std at active freqs
        uu = jnp.fft.fftfreq(self.H_pad, dtype=self.dtype) * self.H_pad
        vv = jnp.fft.rfftfreq(self.W_pad, dtype=self.dtype) * self.W_pad
        U, V = jnp.meshgrid(uu, vv, indexing="ij")
        Rn = jnp.sqrt(U**2 + V**2) / jnp.maximum(jnp.max(jnp.sqrt(U**2 + V**2)), 1.0)
        std_full = (1.0 / jnp.sqrt(1.0 + Rn**alpha)).astype(self.dtype)
        ai = self.active_idx
        std = std_full[ai[:, 0], ai[:, 1]][None, None, :]  # (1,1,M)

        wr = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(std).expand((self.C_out, self.C_in, ai.shape[0])).to_event(3),
        )
        wi = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(std).expand((self.C_out, self.C_in, ai.shape[0])).to_event(3),
        )

        # scatter into half-plane
        K_r = jnp.zeros((self.C_out, self.C_in, self.H_pad, self.W_half), self.dtype)
        K_i = jnp.zeros_like(K_r)
        K_r = K_r.at[..., ai[:, 0], ai[:, 1]].set(wr)
        K_i = K_i.at[..., ai[:, 0], ai[:, 1]].set(wi)

        # enforce real bins
        K_i = K_i.at[..., 0, 0].set(0.0)
        if (self.W_pad % 2 == 0) and (self.W_half > 1):
            K_i = K_i.at[..., :, -1].set(0.0)

        K_half = K_r + 1j * K_i
        m = self._lowpass_mask_half()
        if m is not None:
            K_half = K_half * jax.lax.stop_gradient(m)[None, None, :, :]

        self._K_half = jax.lax.stop_gradient(K_half)

        # forward as before
        single = x.ndim == 3
        if single:
            x = x[None, ...]
        pad_h = max(0, self.H_pad - x.shape[-2])
        pad_w = max(0, self.W_pad - x.shape[-1])
        x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[
            ..., : self.H_pad, : self.W_pad
        ]
        Xf = jnp.fft.rfft2(x_pad, axes=(-2, -1), norm="ortho")
        Yf = jnp.einsum("oihw,bihw->bohw", K_half, Xf)
        y = jnp.fft.irfft2(
            Yf, s=(self.H_pad, self.W_pad), axes=(-2, -1), norm="ortho"
        ).real
        b = numpyro.sample(
            f"{self.name}_bias", dist.Normal(0.0, 1.0).expand([self.C_out]).to_event(1)
        )
        y = y + b[None, :, None, None]
        if self.crop_output and (self.H_pad != self.H_in or self.W_pad != self.W_in):
            y = y[..., : self.H_in, : self.W_in]
        return y[0] if single else y


# ----------------------------- SVD (Dense) -----------------------------------


class SpectralDense:
    """
    Rectangular spectral dense: W = U diag(s) V^T with fixed U∈R^{out×r}, V∈R^{in×r}.
    Here we *construct* U,V internally via QR from standard Normal (stop-gradient).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        name: str = "fc_spec",
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        dtype=jnp.float32,
    ):
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        r = min(self.in_features, self.out_features) if rank is None else int(rank)
        self.rank = r
        self.alpha = alpha
        self.init_scale = float(init_scale)
        self.bias_scale = float(bias_scale)
        self.name = name
        self.prior_fn = prior_fn or _default_prior
        self.dtype = dtype

        # Construct fixed orthonormal U,V (statelessly, with PRNG during inference)
        # We avoid random ops here; instead, caller can supply fixed U,V via SVDDense.
        # For reproducibility, you can pass precomputed U,V to SVDDense below.

    def _build_UV(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Deterministic orthonormal bases from SVD of identity paddings
        # (No randomness; produces a valid orthonormal pair)
        U = jnp.eye(self.out_features, dtype=self.dtype)[:, : self.rank]
        V = jnp.eye(self.in_features, dtype=self.dtype)[:, : self.rank]
        return U, V

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        U, V = self._build_UV()  # (out,r), (in,r)

        # α
        if self.alpha is None:
            az = numpyro.sample(f"{self.name}_alpha_z", dist.Normal(0.0, 1.0))
            alpha = _alpha_bounded(az)
        else:
            alpha = jnp.asarray(self.alpha, self.dtype)

        k = jnp.arange(self.rank, dtype=self.dtype)
        k_norm = k / jnp.maximum(self.rank - 1, 1)
        std = self.init_scale / jnp.sqrt(1.0 + k_norm**alpha)

        s = numpyro.sample(f"{self.name}_s", self.prior_fn(std).to_event(1))  # (r,)
        b = numpyro.sample(
            f"{self.name}_b",
            dist.Normal(0.0, self.bias_scale).expand([self.out_features]).to_event(1),
        )  # (out,)

        W = U @ (s * V.T)  # (out, in)
        return x @ W.T + b


class AdaptiveSpectralDense(SpectralDense):
    """
    Dense with per-mode extra exponent δ_k (softplus):
      alpha_k = alpha_global + δ_k  ->  bounded to [0.1, 3.0].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: Optional[int] = None,
        alpha_global: float = 1.0,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        name: str = "adap_fc_spec",
        alpha_prior: dist.Distribution = dist.HalfNormal(1.0),
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        dtype=jnp.float32,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha_global,
            init_scale=init_scale,
            bias_scale=bias_scale,
            name=name,
            prior_fn=prior_fn,
            dtype=dtype,
        )
        self.alpha_prior = alpha_prior

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        U, V = self._build_UV()
        delta = numpyro.sample(
            f"{self.name}_delta", self.alpha_prior.expand([self.rank]).to_event(1)
        )
        alpha_k = _alpha_bounded(
            jnp.asarray(self.alpha, self.dtype) + jax.nn.softplus(delta)
        )

        k = jnp.arange(self.rank, dtype=self.dtype)
        k_norm = k / jnp.maximum(self.rank - 1, 1)
        std = self.init_scale / jnp.sqrt(1.0 + k_norm**alpha_k)

        s = numpyro.sample(f"{self.name}_s", self.prior_fn(std).to_event(1))
        b = numpyro.sample(
            f"{self.name}_b",
            dist.Normal(0.0, self.bias_scale).expand([self.out_features]).to_event(1),
        )
        W = U @ (s * V.T)
        return x @ W.T + b


class SVDDense:
    """
    SVD dense with *provided* orthonormal factors U∈R^{out×r}, V∈R^{in×r}.
    Samples singular values s and bias b with Sobolev-style prior on modes.
    """

    def __init__(
        self,
        U: jnp.ndarray,
        V: jnp.ndarray,
        *,
        alpha: Optional[float] = None,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        name: str = "svd_dense",
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        dtype=jnp.float32,
    ):
        assert U.ndim == 2 and V.ndim == 2
        assert U.shape[1] == V.shape[1]
        self.U = U  # (out, r)
        self.V = V  # (in,  r)
        self.rank = int(U.shape[1])
        self.out_features = int(U.shape[0])
        self.alpha = alpha
        self.init_scale = float(init_scale)
        self.bias_scale = float(bias_scale)
        self.name = name
        self.prior_fn = prior_fn or _default_prior
        self.dtype = dtype

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # α
        if self.alpha is None:
            az = numpyro.sample(f"{self.name}_alpha_z", dist.Normal(0.0, 1.0))
            alpha = _alpha_bounded(az)
        else:
            alpha = jnp.asarray(self.alpha, self.dtype)

        k = jnp.arange(self.rank, dtype=self.dtype)
        k_norm = k / jnp.maximum(self.rank - 1, 1)
        std = self.init_scale / jnp.sqrt(1.0 + k_norm**alpha)

        s = numpyro.sample(f"{self.name}_s", self.prior_fn(std).to_event(1))  # (r,)
        b = numpyro.sample(
            f"{self.name}_b",
            dist.Normal(0.0, self.bias_scale).expand([self.out_features]).to_event(1),
        )  # (out,)

        W = self.U @ (s * self.V.T)  # (out,in)
        return x @ W.T + b


# ----------------------------- SVD (Conv2d) ----------------------------------


class SpectralConv2d:
    """
    Conv kernel parameterized via SVD of flattened matrix:
      W_mat (C_out × [C_in*H_k*W_k]) = U diag(s) V^T, with fixed U,V; s ~ prior.
    """

    def __init__(
        self,
        U: jnp.ndarray,
        V: jnp.ndarray,
        C_in: int,
        C_out: int,
        H_k: int,
        W_k: int,
        *,
        strides: Sequence[int] = (1, 1),
        padding: str = "SAME",
        alpha: Optional[float] = None,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        name: str = "spec_conv2d",
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        dtype=jnp.float32,
    ):
        assert U.ndim == 2 and V.ndim == 2
        assert U.shape[1] == V.shape[1]
        self.U = U  # (C_out, r)
        self.V = V  # (C_in*H_k*W_k, r)
        self.rank = int(U.shape[1])
        self.C_in, self.C_out = int(C_in), int(C_out)
        self.H_k, self.W_k = int(H_k), int(W_k)
        self.strides = tuple(int(s) for s in strides)
        self.padding = padding
        self.alpha = alpha
        self.init_scale = float(init_scale)
        self.bias_scale = float(bias_scale)
        self.name = name
        self.prior_fn = prior_fn or _default_prior
        self.dtype = dtype

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # α
        if self.alpha is None:
            az = numpyro.sample(f"{self.name}_alpha_z", dist.Normal(0.0, 1.0))
            alpha = _alpha_bounded(az)
        else:
            alpha = jnp.asarray(self.alpha, self.dtype)

        k = jnp.arange(self.rank, dtype=self.dtype)
        k_norm = k / jnp.maximum(self.rank - 1, 1)
        std = self.init_scale / jnp.sqrt(1.0 + k_norm**alpha)

        s = numpyro.sample(f"{self.name}_s", self.prior_fn(std).to_event(1))  # (r,)
        b = numpyro.sample(
            f"{self.name}_b",
            dist.Normal(0.0, self.bias_scale).expand([self.C_out]).to_event(1),
        )  # (C_out,)

        W_mat = self.U @ (s * self.V.T)  # (C_out, C_in*H_k*W_k)
        W = W_mat.reshape(self.C_out, self.C_in, self.H_k, self.W_k)

        # (N,C_in,H,W) -> (N,C_out,H’,W’)
        y = jax.lax.conv_general_dilated(
            x,
            W,
            window_strides=self.strides,
            padding=self.padding,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        return y + b[None, :, None, None]


class AdaptiveSpectralConv2d(SpectralConv2d):
    """
    SVD-Conv2d with per-mode δ_k (softplus) on α_k.
    """

    def __init__(
        self,
        U: jnp.ndarray,
        V: jnp.ndarray,
        C_in: int,
        C_out: int,
        H_k: int,
        W_k: int,
        *,
        strides: Sequence[int] = (1, 1),
        padding: str = "SAME",
        alpha_global: float = 1.0,
        alpha_prior: dist.Distribution = dist.HalfNormal(1.0),
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        name: str = "adap_spec_conv2d",
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        dtype=jnp.float32,
    ):
        super().__init__(
            U=U,
            V=V,
            C_in=C_in,
            C_out=C_out,
            H_k=H_k,
            W_k=W_k,
            strides=strides,
            padding=padding,
            alpha=alpha_global,
            init_scale=init_scale,
            bias_scale=bias_scale,
            name=name,
            prior_fn=prior_fn,
            dtype=dtype,
        )
        self.alpha_prior = alpha_prior

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        delta = numpyro.sample(
            f"{self.name}_delta", self.alpha_prior.expand([self.rank]).to_event(1)
        )
        alpha_k = _alpha_bounded(
            jnp.asarray(self.alpha, self.dtype) + jax.nn.softplus(delta)
        )

        k = jnp.arange(self.rank, dtype=self.dtype)
        k_norm = k / jnp.maximum(self.rank - 1, 1)
        std = self.init_scale / jnp.sqrt(1.0 + k_norm**alpha_k)

        s = numpyro.sample(f"{self.name}_s", self.prior_fn(std).to_event(1))
        b = numpyro.sample(
            f"{self.name}_b",
            dist.Normal(0.0, self.bias_scale).expand([self.C_out]).to_event(1),
        )

        W_mat = self.U @ (s * self.V.T)
        W = W_mat.reshape(self.C_out, self.C_in, self.H_k, self.W_k)
        y = jax.lax.conv_general_dilated(
            x,
            W,
            window_strides=self.strides,
            padding=self.padding,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        return y + b[None, :, None, None]


# --------------------------- Spectral Token Mixer ----------------------------


class SpectralTokenMixer:
    """
    Spectral Token Mixer for sequences:
      x ∈ R^{B,N,C} or R^{N,C} → y = iRFFT( RFFT(x along N) ⊙ H_half ).
    Grouped: share one spectral filter per group of channels (C/G groups).
    If use_residual=True, returns x + mix(x).
    """

    def __init__(
        self,
        n_tokens: int,
        channels: int,
        *,
        groups: int = 1,
        alpha: Optional[float] = None,
        name: str = "spec_mixer",
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        use_residual: bool = True,
        learn_gate: bool = False,
        gate_scale: float = 0.1,
        dtype=jnp.float32,
    ):
        assert channels % groups == 0, "channels must be divisible by groups"
        self.n_tokens = int(n_tokens)
        self.channels = int(channels)
        self.groups = int(groups)
        self.k_half = self.n_tokens // 2 + 1
        self.alpha = alpha
        self.name = name
        self.prior_fn = prior_fn or _default_prior
        self.use_residual = bool(use_residual)
        self.learn_gate = bool(learn_gate)
        self.gate_scale = float(gate_scale)
        self.dtype = dtype

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # α
        if self.alpha is None:
            az = numpyro.sample(f"{self.name}_alpha_z", dist.Normal(0.0, 1.0))
            alpha = _alpha_bounded(az)
        else:
            alpha = jnp.asarray(self.alpha, self.dtype)

        k = jnp.arange(self.k_half, dtype=self.dtype)
        k_norm = k / jnp.maximum(self.k_half - 1, 1)
        std = 1.0 / jnp.sqrt(1.0 + k_norm**alpha)  # (k_half,)

        # grouped half-spectrum filters
        wr = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(jnp.broadcast_to(std, (self.groups, self.k_half))).to_event(
                2
            ),
        )
        wi = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(jnp.broadcast_to(std, (self.groups, self.k_half))).to_event(
                2
            ),
        )
        wi = wi.at[:, 0].set(0.0)
        if self.n_tokens % 2 == 0:
            wi = wi.at[:, -1].set(0.0)
        H_half = wr + 1j * wi  # (G,k_half)

        if x.ndim == 2:
            x = x[None, ...]
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
        pieces = []
        for g in range(G):
            sl = slice(g * ch_per_g, (g + 1) * ch_per_g)
            pieces.append(Xf[:, :, sl] * H_half[g][None, :, None])
        Xf_out = jnp.concatenate(pieces, axis=-1)

        y = jnp.fft.irfft(Xf_out, n=N, axis=1, norm="ortho").real  # (B,N,C)

        if self.learn_gate:
            gate = numpyro.sample(
                f"{self.name}_gate",
                dist.Normal(1.0, self.gate_scale).expand([self.channels]).to_event(1),
            )
            y = y * gate[None, None, :]

        y = x + y if self.use_residual else y
        return y[0] if single else y
