from __future__ import annotations
from typing import Any, Tuple, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

__all__ = [
    "circulant_matmul",
    "Circulant",
    "Circulant2d",
    "block_circulant_matmul_custom",
    "BlockCirculant",
    "BlockCirculantProcess",
    "spectral_circulant_matmul",
    "SpectralCirculantLayer",
    "AdaptiveSpectralCirculantLayer",
    "circulant_conv2d_fft",
    "SpectralCirculantLayer2d",
    "AdaptiveSpectralCirculantLayer2d",
    "SpectralDense",
    "AdaptiveSpectralDense",
    "SpectralConv2d",
    "AdaptiveSpectralConv2d",
    "GibbsKernel",
    "GibbsKernel2d",
    "InputWarping",
    "InputWarping2d",
]


@jax.custom_jvp
def circulant_matmul(x: jnp.ndarray, first_row: jnp.ndarray) -> jnp.ndarray:
    """
    Compute y = C x where C is the circulant matrix defined by first_row,
    using real‐FFT to halve frequency‐domain memory.

    x         : (..., n) real
    first_row : (n,)     real
    returns   : (..., n) real
    """
    n = first_row.shape[-1]
    first_col = jnp.roll(jnp.flip(first_row), shift=1)
    fft_first_col = jnp.fft.rfft(first_col, axis=-1)
    fft_x = jnp.fft.rfft(x, axis=-1)
    y = jnp.fft.irfft(fft_x * fft_first_col, n=n, axis=-1)
    return y


@circulant_matmul.defjvp
def circulant_matmul_jvp(primals, tangents):
    x, first_row = primals
    dx, dfirst_row = tangents
    n = first_row.shape[-1]

    first_col = jnp.roll(jnp.flip(first_row), shift=1)
    fft_first_col = jnp.fft.rfft(first_col, axis=-1)
    fft_x = jnp.fft.rfft(x, axis=-1)
    y = jnp.fft.irfft(fft_x * fft_first_col, n=n, axis=-1)

    dfft_x = jnp.fft.rfft(dx, axis=-1) if dx is not None else jnp.zeros_like(fft_x)
    dy_dx = jnp.fft.irfft(dfft_x * fft_first_col, n=n, axis=-1)

    if dfirst_row is not None:
        dfirst_col = jnp.roll(jnp.flip(dfirst_row), shift=1)
        dfft_first = jnp.fft.rfft(dfirst_col, axis=-1)
        dy_df = jnp.fft.irfft(fft_x * dfft_first, n=n, axis=-1)
    else:
        dy_df = jnp.zeros_like(y)

    return y, dy_dx + dy_df


class Circulant(eqx.Module):
    """
    A production‐quality circulant layer:
      y = circulant_matmul(x_padded, first_row) + bias

    Stores only first_row (n,) and bias (n,); pads input if in_features != out_features.
    """

    first_row: jnp.ndarray
    bias: jnp.ndarray
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        padded_dim: Optional[int] = None,
        *,
        key,
        init_scale: float = 1.0,
    ):
        self.in_features = in_features
        self.out_features = padded_dim if padded_dim is not None else in_features

        k1, k2 = jr.split(key, 2)
        self.first_row = jr.normal(k1, (self.out_features,)) * init_scale
        self.bias = jr.normal(k2, (self.out_features,)) * init_scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.out_features != self.in_features:
            pad_width = [(0, 0)] * (x.ndim - 1) + [
                (0, self.out_features - self.in_features)
            ]
            x = jnp.pad(x, pad_width)
        y = circulant_matmul(x, self.first_row)
        return y + self.bias


@jax.custom_jvp
def block_circulant_matmul_custom(
    W: jnp.ndarray, x: jnp.ndarray, d_bernoulli: jnp.ndarray
):
    """
    Compute the block-circulant matrix multiplication:
       out = B x
    where B is defined via blocks of circulant matrices.

    - W has shape (k_out, k_in, b): each row W[i, j, :] is the first row of a b×b circulant block.
    - x is the input, of shape (batch, in_features) or (in_features,).
    - d_bernoulli is an optional diagonal (of shape (in_features,)) of ±1 entries.

    This function performs the following:
      1. (Optionally) multiplies x elementwise by d_bernoulli.
      2. Zero-pads x so that its length equals k_in * b.
      3. Reshapes x into (batch, k_in, b).
      4. For each output block i, sums over j the circulant multiplication via FFT:
           FFT(x_block) * conj(FFT(W[i,j]))  → summed over j, then inverse FFT.
    """
    single_example = x.ndim == 1
    if single_example:
        x = x[None, :]
    batch_size = x.shape[0]
    d_in = x.shape[-1]
    k_in = W.shape[1]
    b = W.shape[-1]

    if d_bernoulli is not None:
        x_d = x * d_bernoulli[None, :]
    else:
        x_d = x

    pad_len = k_in * b - d_in
    if pad_len > 0:
        x_d = jnp.pad(x_d, ((0, 0), (0, pad_len)), mode="constant", constant_values=0.0)

    X_blocks = x_d.reshape(batch_size, k_in, b)
    X_fft = jnp.fft.fft(X_blocks, axis=-1)
    W_fft = jnp.fft.fft(W, axis=-1)

    def compute_block_row(i):
        prod = X_fft * jnp.conjugate(W_fft[i, :, :])[None, :, :]
        sum_over_j = jnp.sum(prod, axis=1)
        return jnp.fft.ifft(sum_over_j, axis=-1).real

    block_out = jax.vmap(compute_block_row)(jnp.arange(W.shape[0]))
    out = jnp.transpose(block_out, (1, 0, 2)).reshape(batch_size, W.shape[0] * b)
    if single_example:
        out = out[0]
    return out


@block_circulant_matmul_custom.defjvp
def block_circulant_matmul_jvp(primals, tangents):
    W, x, d_bernoulli = primals
    dW, dx, dd = tangents
    single_example = x.ndim == 1
    if single_example:
        x = x[None, :]
    batch_size = x.shape[0]
    d_in = x.shape[-1]
    k_in = W.shape[1]
    b = W.shape[-1]

    if d_bernoulli is not None:
        x_d = x * d_bernoulli[None, :]
    else:
        x_d = x

    pad_len = k_in * b - d_in
    if pad_len > 0:
        x_d = jnp.pad(x_d, ((0, 0), (0, pad_len)), mode="constant", constant_values=0.0)
    X_blocks = x_d.reshape(batch_size, k_in, b)
    X_fft = jnp.fft.fft(X_blocks, axis=-1)
    W_fft = jnp.fft.fft(W, axis=-1)

    def compute_block_row(i):
        prod = X_fft * jnp.conjugate(W_fft[i, :, :])[None, :, :]
        sum_over_j = jnp.sum(prod, axis=1)
        return jnp.fft.ifft(sum_over_j, axis=-1).real

    block_out = jax.vmap(compute_block_row)(jnp.arange(W.shape[0]))
    out = jnp.transpose(block_out, (1, 0, 2)).reshape(batch_size, W.shape[0] * b)
    if single_example:
        out = out[0]

    if d_bernoulli is not None:
        dx_d = dx * d_bernoulli[None, :] + (x * dd[None, :] if dd is not None else 0.0)
    else:
        dx_d = dx

    if pad_len > 0:
        dx_d = jnp.pad(
            dx_d, ((0, 0), (0, pad_len)), mode="constant", constant_values=0.0
        )
    dX_blocks = dx_d.reshape(batch_size, k_in, b)
    dX_fft = jnp.fft.fft(dX_blocks, axis=-1)

    if dW is not None:
        dW_fft = jnp.fft.fft(dW, axis=-1)
    else:
        dW_fft = 0.0

    def compute_block_row_tangent(i):
        term1 = dX_fft * jnp.conjugate(W_fft[i, :, :])[None, :, :]
        term2 = X_fft * (
            jnp.conjugate(dW_fft[i, :, :])[None, :, :] if dW is not None else 0.0
        )
        sum_over_j = jnp.sum(term1 + term2, axis=1)
        return jnp.fft.ifft(sum_over_j, axis=-1).real

    dblock_out = jax.vmap(compute_block_row_tangent)(jnp.arange(W.shape[0]))
    d_out = jnp.transpose(dblock_out, (1, 0, 2)).reshape(batch_size, W.shape[0] * b)
    if single_example:
        d_out = d_out[0]
    return out, d_out


class BlockCirculant(eqx.Module):
    """
    Equinox module implementing a block-circulant layer that uses a custom JVP rule.

    Parameters:
      - W has shape (k_out, k_in, b), where each W[i,j,:] is the first row of a circulant block.
      - D_bernoulli is an optional diagonal of ±1 entries (shape: (in_features,)).
      - bias is added after the block-circulant multiplication.
      - in_features and out_features are the overall dimensions (they may be padded up to a multiple of b).

    The forward pass computes:
         out = block_circulant_matmul_custom(W, x, D_bernoulli) + bias
    and the custom JVP rule reuses FFT computations to accelerate gradient evaluation.
    """

    W: jnp.ndarray
    D_bernoulli: jnp.ndarray
    bias: jnp.ndarray
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    k_in: int = eqx.field(static=True)
    k_out: int = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int,
        *,
        key,
        init_scale: float = 0.1,
        use_bernoulli_diag: bool = True,
        use_bias: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        k_in = (in_features + block_size - 1) // block_size
        k_out = (out_features + block_size - 1) // block_size
        object.__setattr__(self, "k_in", k_in)
        object.__setattr__(self, "k_out", k_out)

        k1, k2, k3 = jr.split(key, 3)
        self.W = jr.normal(k1, (k_out, k_in, block_size)) * init_scale

        if use_bernoulli_diag:
            diag_signs = jnp.where(
                jr.bernoulli(k2, p=0.5, shape=(in_features,)), 1.0, -1.0
            )
        else:
            diag_signs = jnp.ones((in_features,))
        object.__setattr__(self, "D_bernoulli", diag_signs)
        if use_bias:
            self.bias = jr.normal(k3, (out_features,)) * init_scale
        else:
            self.bias = jnp.zeros((out_features,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = block_circulant_matmul_custom(self.W, x, self.D_bernoulli)
        k_out_b = self.k_out * self.block_size
        if k_out_b > self.out_features:
            out = out[..., : self.out_features]
        return out + self.bias[None, :] if out.ndim > 1 else out + self.bias


@jax.custom_jvp
def circulant_conv2d_fft(kernel: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    kernel : (C_out, C_in, H_pad, W_pad)   real
    x      : (..., C_in,  H_in,  W_in)     real
    returns: (..., C_out, H_pad, W_pad)    real
    """
    C_out, C_in, H_pad, W_pad = kernel.shape
    single = x.ndim == 3
    if single:
        x = x[None, ...]

    *lead, C_in_x, H_in, W_in = x.shape
    if C_in_x != C_in:
        raise ValueError(f"Cin mismatch: kernel={C_in}, x={C_in_x}")

    pad_h = max(0, H_pad - H_in)
    pad_w = max(0, W_pad - W_in)
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
    x_pad = x_pad[..., :H_pad, :W_pad]

    Kf = jnp.fft.rfftn(kernel, axes=(-2, -1))
    Xf = jnp.fft.rfftn(x_pad, axes=(-2, -1))
    Yf = jnp.einsum("oihw,bihw->bohw", Kf, Xf)
    y = jnp.fft.irfftn(Yf, s=(H_pad, W_pad), axes=(-2, -1))

    return y[0] if single else y


@circulant_conv2d_fft.defjvp
def _circulant_conv2d_fft_jvp(primals, tangents):
    kernel, x = primals
    dk, dx = tangents

    C_out, C_in, H_pad, W_pad = kernel.shape
    single = x.ndim == 3
    if single:
        x = x[None, ...]
        dx = None if dx is None else dx[None, ...]

    pad_h = max(0, H_pad - x.shape[-2])
    pad_w = max(0, W_pad - x.shape[-1])
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[..., :H_pad, :W_pad]
    dx_pad = None
    if dx is not None:
        dx_pad = jnp.pad(dx, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[
            ..., :H_pad, :W_pad
        ]

    Kf = jnp.fft.rfftn(kernel, axes=(-2, -1))
    Xf = jnp.fft.rfftn(x_pad, axes=(-2, -1))
    Yf = jnp.einsum("oihw,bihw->bohw", Kf, Xf)
    y = jnp.fft.irfftn(Yf, s=(H_pad, W_pad), axes=(-2, -1))

    dXf = (
        jnp.fft.rfftn(dx_pad, axes=(-2, -1))
        if dx_pad is not None
        else jnp.zeros_like(Xf)
    )
    dKf = jnp.fft.rfftn(dk, axes=(-2, -1)) if dk is not None else jnp.zeros_like(Kf)

    dYf = jnp.einsum("oihw,bihw->bohw", Kf, dXf) + jnp.einsum(
        "oihw,bihw->bohw", dKf, Xf
    )
    dy = jnp.fft.irfftn(dYf, s=(H_pad, W_pad), axes=(-2, -1))

    if single:
        return y[0], dy[0]
    else:
        return y, dy


class Circulant2d(eqx.Module):
    """Real weight-space wrap-around convolution."""

    kernel: jnp.ndarray
    bias: jnp.ndarray
    C_in: int = eqx.field(static=True)
    C_out: int = eqx.field(static=True)
    H_pad: int = eqx.field(static=True)
    W_pad: int = eqx.field(static=True)

    def __init__(
        self,
        C_in: int,
        C_out: int,
        H_in: int,
        W_in: int,
        *,
        H_pad: Optional[int] = None,
        W_pad: Optional[int] = None,
        key,
        init_scale: float = 1.0,
        bias_scale: float = 1.0,
        dtype=jnp.float32,
    ):
        self.C_in, self.C_out = C_in, C_out
        self.H_pad = int(H_pad or H_in)
        self.W_pad = int(W_pad or W_in)

        k1, k2 = jr.split(key, 2)
        self.kernel = (
            jr.normal(k1, (C_out, C_in, self.H_pad, self.W_pad), dtype) * init_scale
        )
        self.bias = jr.normal(k2, (C_out, self.H_pad, self.W_pad), dtype) * bias_scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = circulant_conv2d_fft(self.kernel, x)
        return y + (self.bias if y.ndim == 3 else self.bias[None, :])


class BlockCirculantProcess(eqx.Module):
    """
    Block-circulant layer with custom JVP.

    This layer partitions the weight matrix into blocks where each block is circulant.
    The forward pass computes:
       out = block_circulant_matmul_custom(W, x, D_bernoulli) + bias
    without mutating internal state. The Fourier coefficients for each block are
    computed on the fly in get_fourier_coeffs().
    """

    W: jnp.ndarray
    D_bernoulli: Optional[jnp.ndarray]
    bias: jnp.ndarray
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    k_in: int = eqx.field(static=True)
    k_out: int = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int,
        *,
        key,
        init_scale: float = 0.1,
        use_diag: bool = True,
        use_bias: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        k_in = (in_features + block_size - 1) // block_size
        k_out = (out_features + block_size - 1) // block_size
        object.__setattr__(self, "k_in", k_in)
        object.__setattr__(self, "k_out", k_out)

        k1, k2, k3 = jr.split(key, 3)
        self.W = jr.normal(k1, (k_out, k_in, block_size)) * init_scale
        if use_diag:
            diag = jr.bernoulli(k2, p=0.5, shape=(in_features,))
            self.D_bernoulli = jnp.where(diag, 1.0, -1.0)
        else:
            self.D_bernoulli = None
        if use_bias:
            self.bias = jr.normal(k3, (out_features,)) * init_scale
        else:
            self.bias = jnp.zeros((out_features,))

    def get_fourier_coeffs(self) -> jnp.ndarray:
        """
        Compute and return the Fourier coefficients for each circulant block.
        Expected shape: (k_out, k_in, block_size)
        """
        return jnp.fft.fft(self.W, axis=-1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = block_circulant_matmul_custom(self.W, x, self.D_bernoulli)
        if self.k_out * self.block_size > self.out_features:
            out = out[..., : self.out_features]
        if out.ndim == 1:
            return out + self.bias
        else:
            return out + self.bias[None, :]


@jax.custom_jvp
def spectral_circulant_matmul(x: jnp.ndarray, fft_full: jnp.ndarray) -> jnp.ndarray:
    """
    Compute y = IFFT( FFT(x_pad) * fft_full ), with x zero‑padded to length padded_dim.
    x can be (batch, d_in) or just (d_in,). Operation is along the last dimension.
    """
    padded_dim = fft_full.shape[0]
    single = x.ndim == 1
    if single:
        x = x[None, :]
    d_in = x.shape[-1]
    if d_in < padded_dim:
        x_pad = jnp.pad(x, ((0, 0), (0, padded_dim - d_in)))
    else:
        x_pad = x[..., :padded_dim]
    Xf = jnp.fft.fft(x_pad, axis=-1)
    yf = Xf * fft_full[None, :]
    y = jnp.fft.ifft(yf, axis=-1).real
    return y[0] if single else y


@spectral_circulant_matmul.defjvp
def spectral_circulant_matmul_jvp(primals, tangents):
    x, fft_full = primals
    dx, dfft = tangents
    padded_dim = fft_full.shape[0]
    single = x.ndim == 1
    if single:
        x = x[None, :]
        if dx is not None:
            dx = dx[None, :]
    d_in = x.shape[-1]
    if d_in < padded_dim:
        x_pad = jnp.pad(x, ((0, 0), (0, padded_dim - d_in)))
        dx_pad = (
            jnp.pad(dx, ((0, 0), (0, padded_dim - d_in))) if dx is not None else None
        )
    else:
        x_pad = x[..., :padded_dim]
        dx_pad = dx[..., :padded_dim] if dx is not None else None

    Xf = jnp.fft.fft(x_pad, axis=-1)
    y_primal = jnp.fft.ifft(Xf * fft_full[None, :], axis=-1).real

    dXf = jnp.fft.fft(dx_pad, axis=-1) if dx_pad is not None else 0.0
    term2 = Xf * dfft[None, :] if dfft is not None else 0.0
    dyf = dXf * fft_full[None, :] + term2
    dy = jnp.fft.ifft(dyf, axis=-1).real

    if single:
        return y_primal[0], dy[0]
    return y_primal, dy


class _SpectralMixin:
    """Provides a quadratic Sobolev penalty to the optimiser."""

    alpha: jnp.ndarray | float

    def _spectral_weights(self) -> jnp.ndarray:
        raise NotImplementedError

    def _spectral_scale(self) -> jnp.ndarray:
        raise NotImplementedError

    def __spectral_penalty__(self) -> jnp.ndarray:
        θ = self._spectral_weights()
        β = self._spectral_scale()
        return jnp.sum(β * (θ**2))


class SpectralCirculantLayer(eqx.Module, _SpectralMixin):
    """
    Deterministic spectral‑circulant layer with a trainable decay exponent α.
    Uses the same custom‑JVP FFT mat‑mul defined above.
    """

    in_features: int = eqx.field(static=True)
    padded_dim: int = eqx.field(static=True)
    K: int = eqx.field(static=True)
    k_half: int = eqx.field(static=True)

    alpha: jnp.ndarray
    w_real: jnp.ndarray
    w_imag: jnp.ndarray
    bias: jnp.ndarray

    def __init__(
        self,
        in_features: int,
        padded_dim: int = None,
        alpha_init: float = 1.0,
        K: int = None,
        *,
        key,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
    ):
        self.in_features = in_features
        self.padded_dim = padded_dim if padded_dim is not None else in_features
        self.k_half = (self.padded_dim // 2) + 1
        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        self.alpha = jnp.array(alpha_init, dtype=jnp.float32)

        key_r, key_i, key_b = jr.split(key, 3)
        k_idx = jnp.arange(self.k_half, dtype=jnp.float32)
        prior_s = 1.0 / jnp.sqrt(1.0 + k_idx**alpha_init)

        w_r = jr.normal(key_r, (self.k_half,)) * (init_scale * prior_s)
        w_i = jr.normal(key_i, (self.k_half,)) * (init_scale * prior_s)

        w_i = w_i.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            w_i = w_i.at[-1].set(0.0)

        self.w_real = w_r
        self.w_imag = w_i
        self.bias = jr.normal(key_b, (self.padded_dim,)) * bias_scale

    def get_fourier_coeffs(self) -> jnp.ndarray:
        """
        Build the full-length FFT coefficients via Hermitian symmetry,
        masking out frequencies >= K.
        """
        mask = jnp.arange(self.k_half) < self.K
        half = (self.w_real * mask) + 1j * (self.w_imag * mask)

        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            nyq = half[-1].real[None]
            full = jnp.concatenate([half[:-1], nyq, jnp.conj(half[1:-1])[::-1]])
        else:
            full = jnp.concatenate([half, jnp.conj(half[1:])[::-1]])
        return full

    def _spectral_weights(self):
        return jnp.concatenate([self.w_real, self.w_imag])

    def _spectral_scale(self) -> jnp.ndarray:
        k = jnp.arange(self.k_half, dtype=self.alpha.dtype)
        k_norm = k / (self.k_half - 1)
        β = 1.0 + k_norm**self.alpha
        return jnp.concatenate([β, β])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        y = spectral_circulant_matmul(x, F) + bias
        """
        fft_full = self.get_fourier_coeffs()
        y = spectral_circulant_matmul(x, fft_full)
        if y.ndim == 1:
            return y + self.bias
        else:
            return y + self.bias[None, :]


class AdaptiveSpectralCirculantLayer(eqx.Module, _SpectralMixin):
    """
    1-D spectral circulant layer with trainable per-frequency exponent via
    softplus(delta_z), so δ starts positive and has nonzero gradient.
    """

    in_features: int = eqx.field(static=True)
    padded_dim: int = eqx.field(static=True)
    K: int = eqx.field(static=True)
    k_half: int = eqx.field(static=True)

    alpha: jnp.ndarray
    delta_z: jnp.ndarray
    w_real: jnp.ndarray
    w_imag: jnp.ndarray
    bias: jnp.ndarray

    def __init__(
        self,
        in_features: int,
        padded_dim: Optional[int] = None,
        alpha_init: float = 1.0,
        K: Optional[int] = None,
        *,
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        delta_init: float = 0.1,
        dtype=jnp.float32,
    ):
        self.in_features = in_features
        self.padded_dim = padded_dim or in_features
        self.k_half = (self.padded_dim // 2) + 1
        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        self.alpha = jnp.asarray(alpha_init, dtype)
        z0 = jnp.log(jnp.exp(delta_init) - 1.0)
        self.delta_z = jnp.full((self.k_half,), z0, dtype)
        key_r, key_i, key_b = jr.split(key, 3)
        k_idx = jnp.arange(self.k_half, dtype=dtype)
        base_std = 1.0 / jnp.sqrt(1.0 + k_idx**alpha_init)

        w_r = jr.normal(key_r, (self.k_half,), dtype) * (init_scale * base_std)
        w_i = jr.normal(key_i, (self.k_half,), dtype) * (init_scale * base_std)
        w_i = w_i.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            w_i = w_i.at[-1].set(0.0)

        self.w_real = w_r
        self.w_imag = w_i
        self.bias = jr.normal(key_b, (self.padded_dim,), dtype) * bias_scale

    def _spectral_weights(self) -> jnp.ndarray:
        return jnp.concatenate([self.w_real, self.w_imag])

    def _spectral_scale(self) -> jnp.ndarray:
        δ = jax.nn.softplus(self.delta_z)
        z = self.alpha + δ
        α_min, α_max = 0.1, 3.0
        raw = α_min + (α_max - α_min) * jax.nn.sigmoid(z)

        k = jnp.arange(self.k_half, dtype=raw.dtype)
        k_norm = k / jnp.maximum(self.k_half - 1, 1)
        β = 1.0 + k_norm**raw
        return jnp.concatenate([β, β])

    def get_fourier_coeffs(self) -> jnp.ndarray:
        mask = jnp.arange(self.k_half) < self.K
        half = (self.w_real * mask) + 1j * (self.w_imag * mask)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            nyq = half[-1].real[None]
            full = jnp.concatenate([half[:-1], nyq, jnp.conj(half[1:-1])[::-1]])
        else:
            full = jnp.concatenate([half, jnp.conj(half[1:])[::-1]])
        return full

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        fft_full = self.get_fourier_coeffs()
        y = spectral_circulant_matmul(x, fft_full)
        return y + (self.bias if y.ndim == 1 else self.bias[None, :])


def _enforce_hermitian(fft2d: jnp.ndarray) -> jnp.ndarray:
    """
    Project complex tensor (..., H, W) onto the Hermitian subspace so that the
    spatial inverse FFT yields a real array.

    Works on an arbitrary leading batch / channel shape; only the last two axes
    are treated as frequency dimensions.
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
    Circular (wrap-around) convolution via direct multiplication in
    Fourier space.

        fft_kernel : (C_out, C_in, H_pad, W_pad)   — complex Hermitian
        x          : (..., C_in, H_in,  W_in)      — real
        returns    : (..., C_out, H_pad, W_pad)    — real
    """
    C_out, C_in, H_pad, W_pad = fft_kernel.shape
    single = x.ndim == 3
    if single:
        x = x[None, ...]

    *lead, C_in_x, H_in, W_in = x.shape
    if C_in_x != C_in:
        raise ValueError(f"Cin mismatch: kernel={C_in}, x={C_in_x}")

    pad_h, pad_w = max(0, H_pad - H_in), max(0, W_pad - W_in)
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
    x_pad = x_pad[..., :H_pad, :W_pad]

    Xf = jnp.fft.fftn(x_pad, axes=(-2, -1))
    Yf = jnp.einsum("oihw,bihw->bohw", fft_kernel, Xf)
    y = jnp.fft.ifftn(Yf, axes=(-2, -1)).real

    if single:
        y = y[0]
    return y


@spectral_circulant_conv2d.defjvp
def _spectral_circulant_conv2d_jvp(primals, tangents):
    x, fft_kernel = primals
    dx, dk = tangents

    C_out, C_in, H_pad, W_pad = fft_kernel.shape
    single = x.ndim == 3
    if single:
        x = x[None, ...]
        dx = None if dx is None else dx[None, ...]

    pad_h = max(0, H_pad - x.shape[-2])
    pad_w = max(0, W_pad - x.shape[-1])
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[..., :H_pad, :W_pad]
    dx_pad = None
    if dx is not None:
        dx_pad = jnp.pad(dx, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[
            ..., :H_pad, :W_pad
        ]

    Xf = jnp.fft.fftn(x_pad, axes=(-2, -1))
    Yf = jnp.einsum("oihw,bihw->bohw", fft_kernel, Xf)
    y = jnp.fft.ifftn(Yf, axes=(-2, -1)).real

    dXf = 0.0 if dx_pad is None else jnp.fft.fftn(dx_pad, axes=(-2, -1))
    dKf = 0.0 if dk is None else dk

    dYf = jnp.einsum("oihw,bihw->bohw", fft_kernel, dXf) + jnp.einsum(
        "oihw,bihw->bohw", dKf, Xf
    )
    dy = jnp.fft.ifftn(dYf, axes=(-2, -1)).real

    if single:
        return y[0], dy[0]
    else:
        return y, dy


class SpectralCirculantLayer2d(eqx.Module, _SpectralMixin):
    alpha: jnp.ndarray
    w_real: jnp.ndarray
    w_imag: jnp.ndarray
    bias: jnp.ndarray

    H_pad: int = eqx.field(static=True)
    W_pad: int = eqx.field(static=True)

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
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        dtype=jnp.float32,
    ):
        self.H_pad = int(H_pad or H_in)
        self.W_pad = int(W_pad or W_in)
        self.alpha = jnp.asarray(alpha_init, dtype)

        u = jnp.fft.fftfreq(self.H_pad, dtype=dtype) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad, dtype=dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)

        std0 = (1.0 / jnp.sqrt(1.0 + R**alpha_init))[None, None, :, :]

        k1, k2, k3 = jr.split(key, 3)
        self.w_real = jr.normal(k1, (C_out, C_in, self.H_pad, self.W_pad), dtype) * (
            init_scale * std0
        )

        wi = jr.normal(k2, (C_out, C_in, self.H_pad, self.W_pad), dtype) * (
            init_scale * std0
        )
        wi = wi.at[..., 0, 0].set(0.0)
        if self.H_pad % 2 == 0:
            wi = wi.at[..., self.H_pad // 2, :].set(0.0)
        if self.W_pad % 2 == 0:
            wi = wi.at[..., :, self.W_pad // 2].set(0.0)
        self.w_imag = wi

        self.bias = jr.normal(k3, (C_out, self.H_pad, self.W_pad), dtype) * bias_scale

    def get_fft_kernel(self) -> jnp.ndarray:
        return _enforce_hermitian(self.w_real + 1j * self.w_imag)

    def _spectral_weights(self):
        wr = self.w_real.reshape(-1)
        wi = self.w_imag.reshape(-1)
        return jnp.concatenate([wr, wi])

    def _spectral_scale(self) -> jnp.ndarray:
        u = jnp.fft.fftfreq(self.H_pad, dtype=self.w_real.dtype) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad, dtype=self.w_real.dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        α_min, α_max = 0.1, 3.0
        α = α_min + (α_max - α_min) * jax.nn.sigmoid(self.alpha)
        R_norm = R / jnp.max(R)
        β2d = 1.0 + R_norm**α
        β_full = jnp.broadcast_to(β2d, self.w_real.shape)
        flat = β_full.reshape(-1)
        return jnp.concatenate([flat, flat])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        fft_kernel = self.get_fft_kernel()
        y = spectral_circulant_conv2d(x, fft_kernel)
        return y + (self.bias if y.ndim == 3 else self.bias[None, :])


class AdaptiveSpectralCirculantLayer2d(eqx.Module, _SpectralMixin):
    C_in: int = eqx.field(static=True)
    C_out: int = eqx.field(static=True)
    H_pad: int = eqx.field(static=True)
    W_pad: int = eqx.field(static=True)
    alpha: jnp.ndarray
    delta_z: jnp.ndarray
    w_real: jnp.ndarray
    w_imag: jnp.ndarray
    bias: jnp.ndarray

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
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        dtype=jnp.float32,
    ):
        self.C_in, self.C_out = C_in, C_out
        self.H_pad = int(H_pad or H_in)
        self.W_pad = int(W_pad or W_in)
        self.alpha = jnp.asarray(alpha_init, dtype)

        z0 = jnp.log(jnp.exp(delta_init) - 1.0)
        self.delta_z = jnp.full((self.H_pad, self.W_pad), z0, dtype)

        u = jnp.fft.fftfreq(self.H_pad, dtype=dtype) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad, dtype=dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        std0 = (1.0 / jnp.sqrt(1.0 + R**alpha_init))[None, None, :, :]

        k1, k2, k3 = jr.split(key, 3)
        self.w_real = jr.normal(k1, (C_out, C_in, self.H_pad, self.W_pad), dtype) * (
            init_scale * std0
        )

        wi = jr.normal(k2, (C_out, C_in, self.H_pad, self.W_pad), dtype) * (
            init_scale * std0
        )
        wi = wi.at[..., 0, 0].set(0.0)
        if self.H_pad % 2 == 0:
            wi = wi.at[..., self.H_pad // 2, :].set(0.0)
        if self.W_pad % 2 == 0:
            wi = wi.at[..., :, self.W_pad // 2].set(0.0)
        self.w_imag = wi

        self.bias = jr.normal(k3, (C_out, self.H_pad, self.W_pad), dtype) * bias_scale

    def _spectral_weights(self) -> jnp.ndarray:
        return jnp.concatenate(
            [
                self.w_real.reshape(-1),
                self.w_imag.reshape(-1),
            ]
        )

    def _spectral_scale(self) -> jnp.ndarray:
        u = jnp.fft.fftfreq(self.H_pad, dtype=self.alpha.dtype) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad, dtype=self.alpha.dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        δ = jax.nn.softplus(self.delta_z)
        z = self.alpha + δ
        α_min, α_max = 0.1, 3.0
        raw = α_min + (α_max - α_min) * jax.nn.sigmoid(z)

        R_norm = R / jnp.max(R)
        β2d = 1.0 + R_norm**raw
        flat = jnp.broadcast_to(β2d, self.w_real.shape).reshape(-1)
        return jnp.concatenate([flat, flat])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        fft_kernel = _enforce_hermitian(self.w_real + 1j * self.w_imag)
        y = spectral_circulant_conv2d(x, fft_kernel)
        return y + (self.bias if y.ndim == 3 else self.bias[None, :])


class SpectralDense(eqx.Module, _SpectralMixin):
    """
    Square spectral dense layer: x -> W x + b with W = U diag(s) V^T.
    U, V are fixed random orthonormal matrices. Applies normalized Sobolev
    penalty via exponent alpha (clamped to [0.1,3]).

    Example usage:
        key, k1, k2 = jr.split(key, 3)
        d = in_features  # for a “square” dense layer

        U_full, _ = jnp.linalg.qr(jr.normal(k1, (d, d)))
        V_full, _ = jnp.linalg.qr(jr.normal(k2, (d, d)))
        U0, V0 = U_full, V_full 
    """

    in_features: int = eqx.field(static=True)
    U: jnp.ndarray
    V: jnp.ndarray
    s: jnp.ndarray
    bias: jnp.ndarray
    alpha: jnp.ndarray

    def __init__(
        self,
        d: int,
        *,
        alpha_init: float = 1.0,
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
    ):
        self.in_features = d
        ku, kv, ks, kb = jr.split(key, 4)
        U0, _ = jnp.linalg.qr(jr.normal(ku, (d, d)))
        V0, _ = jnp.linalg.qr(jr.normal(kv, (d, d)))
        self.U = jax.lax.stop_gradient(U0)
        self.V = jax.lax.stop_gradient(V0)
        self.s = jr.normal(ks, (d,)) * init_scale
        self.bias = jr.normal(kb, (d,)) * bias_scale
        self.alpha = jnp.asarray(alpha_init, jnp.float32)

    def _spectral_weights(self) -> jnp.ndarray:
        return self.s

    def _spectral_scale(self) -> jnp.ndarray:
        d = self.s.shape[0]
        k = jnp.arange(d, dtype=self.s.dtype)
        k_norm = k / jnp.maximum(d - 1, 1)
        α_min, α_max = 0.1, 3.0
        α = α_min + (α_max - α_min) * jax.nn.sigmoid(self.alpha)
        return 1.0 + k_norm**α

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        W = self.U @ jnp.diag(self.s) @ self.V.T
        return x @ W.T + self.bias


class AdaptiveSpectralDense(SpectralDense):
    """
    Adaptive spectral dense: learns per-frequency smoothness exponent via
    delta_z (softplus reparameterization). Adds one extra parameter per freq.

    Example usage:
        key, k1, k2 = jr.split(key, 3)
        d = in_features  # for a “square” dense layer

        U_full, _ = jnp.linalg.qr(jr.normal(k1, (d, d)))
        V_full, _ = jnp.linalg.qr(jr.normal(k2, (d, d)))
        U0, V0 = U_full, V_full  
    """

    delta_z: jnp.ndarray

    def __init__(
        self,
        d: int,
        *,
        alpha_init: float = 1.0,
        delta_init: float = 0.1,
        key: Any,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
    ):
        super().__init__(
            d=d,
            alpha_init=alpha_init,
            key=key,
            init_scale=init_scale,
            bias_scale=bias_scale,
        )
        z0 = jnp.log(jnp.exp(delta_init) - 1.0)
        self.delta_z = jnp.full((d,), z0, dtype=jnp.float32)

    def _spectral_weights(self) -> jnp.ndarray:
        return self.s

    def _spectral_scale(self) -> jnp.ndarray:
        d = self.s.shape[0]
        k = jnp.arange(d, dtype=self.s.dtype)
        k_norm = k / jnp.maximum(d - 1, 1)
        δ = jax.nn.softplus(self.delta_z)
        z = self.alpha + δ
        α_min, α_max = 0.1, 3.0
        raw = α_min + (α_max - α_min) * jax.nn.sigmoid(z)

        return 1.0 + k_norm**raw


class SpectralConv2d(eqx.Module, _SpectralMixin):
    """
    SVD-parameterized 2D conv:
      - Flatten kernel to W_mat ∈ R[C_out, C_in*H_k*W_k]
      - Factor W_mat = U @ diag(s) @ V^T, with fixed random U,V
      - Reshape diag(s) back into a (C_out,C_in,H_k,W_k) conv kernel
      - Sobolev regularization via _SpectralMixin on s

    Example usage:
        key, k1, k2 = jr.split(key, 3)
        C_out, C_in, H_k, W_k = 7, 1, 3, 3
        U_full, _ = jnp.linalg.qr(jr.normal(k1, (C_out, C_out)))
        V_full, _ = jnp.linalg.qr(jr.normal(k2, (C_in*H_k*W_k, C_in*H_k*W_k)))
        d = min(C_out, C_in*H_k*W_k)
        U0, V0 = U_full[:, :d], V_full[:, :d]
    """

    C_in: int = eqx.field(static=True)
    C_out: int = eqx.field(static=True)
    H_k: int = eqx.field(static=True)
    W_k: int = eqx.field(static=True)
    strides: Sequence[int] = eqx.field(static=True)
    padding: Sequence[Sequence[int]] = eqx.field(static=True)

    U: jnp.ndarray
    V: jnp.ndarray
    s: jnp.ndarray
    bias: jnp.ndarray
    alpha: jnp.ndarray

    def __init__(
        self,
        C_in: int,
        C_out: int,
        H_k: int,
        W_k: int,
        *,
        key: Any,
        strides: Sequence[int] = (1, 1),
        padding: Optional[str] = "SAME",
        alpha_init: float = 1.0,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        dtype=jnp.float32,
    ):
        self.C_in, self.C_out, self.H_k, self.W_k = C_in, C_out, H_k, W_k
        self.strides = strides
        if isinstance(padding, str):
            self.padding = padding
        else:
            self.padding = padding

        d = min(C_out, C_in * H_k * W_k)
        k1, k2, k3, k4 = jr.split(key, 4)

        U0, _ = jnp.linalg.qr(jr.normal(k1, (C_out, C_out), dtype))
        V0, _ = jnp.linalg.qr(
            jr.normal(k2, (C_in * H_k * W_k, C_in * H_k * W_k), dtype)
        )
        self.U = jax.lax.stop_gradient(U0[:, :d])
        self.V = jax.lax.stop_gradient(V0[:, :d])

        self.s = jr.normal(k3, (d,), dtype) * init_scale
        self.bias = jr.normal(k4, (C_out,), dtype) * bias_scale
        self.alpha = jnp.asarray(alpha_init, dtype)

    def _spectral_weights(self) -> jnp.ndarray:
        return self.s

    def _spectral_scale(self) -> jnp.ndarray:
        d = self.s.shape[0]
        k = jnp.arange(d, dtype=self.s.dtype)
        k_norm = k / jnp.maximum(d - 1, 1)
        α_min, α_max = 0.1, 3.0
        α = α_min + (α_max - α_min) * jax.nn.sigmoid(self.alpha)
        return 1.0 + k_norm**α

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        was_single = x.ndim == 3
        if was_single:
            x = x[None, ...]

        W_mat = self.U @ jnp.diag(self.s) @ self.V.T
        W = W_mat.reshape(self.C_out, self.C_in, self.H_k, self.W_k)

        y = jax.lax.conv_general_dilated(
            x,
            W,
            window_strides=self.strides,
            padding=self.padding,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

        y = y + self.bias[None, :, None, None]

        if was_single:
            y = jnp.squeeze(y, axis=0)

        return y


class AdaptiveSpectralConv2d(SpectralConv2d):
    """
    Adaptive per-frequency Sobolev exponent:
      Adds delta_z ∈ R[d] to allow each singular mode its own α.

    Example usage:
        key, k1, k2 = jr.split(key, 3)
        C_out, C_in, H_k, W_k = 7, 1, 3, 3
        U_full, _ = jnp.linalg.qr(jr.normal(k1, (C_out, C_out)))
        V_full, _ = jnp.linalg.qr(jr.normal(k2, (C_in*H_k*W_k, C_in*H_k*W_k)))
        d = min(C_out, C_in*H_k*W_k)
        U0, V0 = U_full[:, :d], V_full[:, :d]
    """

    delta_z: jnp.ndarray

    def __init__(
        self,
        C_in: int,
        C_out: int,
        H_k: int,
        W_k: int,
        *,
        key: Any,
        alpha_init: float = 1.0,
        delta_init: float = 0.1,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        strides: Sequence[int] = (1, 1),
        padding: Optional[str] = "SAME",
        dtype=jnp.float32,
    ):
        super().__init__(
            C_in=C_in,
            C_out=C_out,
            H_k=H_k,
            W_k=W_k,
            key=key,
            strides=strides,
            padding=padding,
            alpha_init=alpha_init,
            init_scale=init_scale,
            bias_scale=bias_scale,
            dtype=dtype,
        )
        d = self.s.shape[0]
        z0 = jnp.log(jnp.exp(delta_init) - 1.0)
        self.delta_z = jnp.full((d,), z0, dtype)

    def _spectral_scale(self) -> jnp.ndarray:
        d = self.s.shape[0]
        k = jnp.arange(d, dtype=self.s.dtype)
        k_norm = k / jnp.maximum(d - 1, 1)
        δ = jax.nn.softplus(self.delta_z)
        raw = self.alpha + δ
        α_min, α_max = 0.1, 3.0
        α = α_min + (α_max - α_min) * jax.nn.sigmoid(raw)
        return 1.0 + k_norm**α


def graph_laplacian(adj: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return eigenvectors Q and eigenvalues λ of the combinatorial Laplacian."""
    deg = jnp.diag(adj.sum(axis=1))
    L = deg - adj
    λ, Q = jnp.linalg.eigh(L)
    return Q, λ


class GraphSpectralDense(eqx.Module, _SpectralMixin):
    """
    Dense layer with W = Q diag(s) Qᵀ  using Laplacian eigen-basis of `adj`.
    Requires adj to be square; output = input = d.
    """

    in_features: int = eqx.field(static=True)
    Q: jnp.ndarray
    λ: jnp.ndarray
    s: jnp.ndarray
    bias: jnp.ndarray
    alpha: jnp.ndarray

    def __init__(
        self,
        adj: jnp.ndarray,
        *,
        alpha: float = 1.0,
        key: Any,
        init_scale=0.1,
        bias_scale=0.1,
    ):
        Q, lam = graph_laplacian(adj)
        d = adj.shape[0]
        self.in_features = d
        self.Q = jax.lax.stop_gradient(Q)
        self.λ = jax.lax.stop_gradient(lam)
        ks, kb = jr.split(key, 2)
        self.s = jr.normal(ks, (d,)) * init_scale
        self.bias = jr.normal(kb, (d,)) * bias_scale
        self.alpha = jnp.asarray(alpha, jnp.float32)

    def _spectral_weights(self):
        return self.s

    def _spectral_scale(self):
        return 1.0 + self.λ**self.alpha

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        W = self.Q @ jnp.diag(self.s) @ self.Q.T
        return x @ W.T + self.bias


class AdaptiveGraphSpectralDense(GraphSpectralDense):
    delta_alpha: jnp.ndarray

    def __init__(
        self,
        adj: jnp.ndarray,
        *,
        alpha_global=1.0,
        key: Any,
        init_scale=0.1,
        bias_scale=0.1,
        delta_init=0.0,
    ):
        super().__init__(
            adj,
            alpha=alpha_global,
            key=key,
            init_scale=init_scale,
            bias_scale=bias_scale,
        )
        self.delta_alpha = jnp.full(self.λ.shape, delta_init)

    def _spectral_scale(self):
        α = self.alpha + self.delta_alpha
        return 1.0 + self.λ**α


class GibbsKernel(eqx.Module):
    N: int = eqx.field(static=True)
    lengthscale_net: callable

    def __init__(self, in_features: int, lengthscale_net, *, key=None):
        self.N = in_features
        self.lengthscale_net = lengthscale_net

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, N = x.shape
        assert N == self.N
        pos = jnp.linspace(0.0, 1.0, N)
        log_l = self.lengthscale_net(pos)
        l = jnp.exp(log_l)

        l_i = l[:, None]
        l_j = l[None, :]
        denom = l_i**2 + l_j**2 + 1e-6

        sq = jnp.sqrt(2 * (l_i * l_j) / denom)
        dmat = (jnp.arange(N)[:, None] - jnp.arange(N)[None, :]) ** 2
        exp = jnp.exp(-dmat / denom)

        K = sq * exp
        return x @ K.T


class InputWarping(eqx.Module):
    N: int = eqx.field(static=True)
    warp_net: callable
    base_psd: callable

    def __init__(self, in_features: int, warp_net, base_psd, *, key=None):
        self.N = in_features
        self.warp_net = warp_net
        self.base_psd = base_psd

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, N = x.shape
        assert N == self.N
        pos = jnp.linspace(0.0, 1.0, N)
        delta_u = self.warp_net(pos)
        u = pos + delta_u

        def warp_one(seq):
            return jnp.interp(pos, u, seq)

        x_warp = jax.vmap(warp_one)(x)

        freqs = jnp.fft.fftfreq(N)
        S = self.base_psd(freqs)
        Xf = jnp.fft.fft(x_warp, axis=-1)
        y = jnp.fft.ifft(Xf * S[None, :], axis=-1).real
        return y


class GibbsKernel2d(eqx.Module):
    H: int = eqx.field(static=True)
    W: int = eqx.field(static=True)
    net_x: callable
    net_y: callable

    def __init__(
        self, H: int, W: int, lengthscale_net_x, lengthscale_net_y, *, key=None
    ):
        self.H, self.W = H, W
        self.net_x = lengthscale_net_x
        self.net_y = lengthscale_net_y

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W = x.shape
        assert (H, W) == (self.H, self.W)
        pos_x = jnp.linspace(0.0, 1.0, W)
        pos_y = jnp.linspace(0.0, 1.0, H)
        lx = jnp.exp(self.net_x(pos_x))
        ly = jnp.exp(self.net_y(pos_y))

        def gibbs1d(pos, ell):
            d = pos[:, None] - pos[None, :]
            l_i = ell[:, None]
            l_j = ell[None, :]
            denom = l_i**2 + l_j**2 + 1e-6
            sq = jnp.sqrt(2 * l_i * l_j / denom)
            return sq * jnp.exp(-(d**2) / denom)

        Kx = gibbs1d(pos_x, lx)
        Ky = gibbs1d(pos_y, ly)

        x1 = jnp.einsum("bhw,wd->bhd", x, Kx)
        y = jnp.einsum("bhd,hk->bkd", x1, Ky)
        return y


class InputWarping2d(eqx.Module):
    H: int = eqx.field(static=True)
    W: int = eqx.field(static=True)
    warp_x: callable
    warp_y: callable
    base_psd: callable

    def __init__(self, H: int, W: int, warp_net_x, warp_net_y, base_psd, *, key=None):
        self.H, self.W = H, W
        self.warp_x, self.warp_y = warp_net_x, warp_net_y
        self.base_psd = base_psd

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W = x.shape
        pos_x = jnp.linspace(0.0, 1.0, W)
        pos_y = jnp.linspace(0.0, 1.0, H)

        def make_u(net, pos):
            raw = net(pos)
            inc = jax.nn.softplus(raw)
            cs = jnp.cumsum(inc)
            return (cs - cs[0]) / (cs[-1] - cs[0])

        ux = make_u(self.warp_x, pos_x)
        uy = make_u(self.warp_y, pos_y)

        def warp_rows(img):
            return jnp.stack([jnp.interp(pos_x, ux, row) for row in img], axis=0)

        x1 = jax.vmap(warp_rows)(x)

        def warp_cols(img):
            return jnp.stack([jnp.interp(pos_y, uy, col) for col in img.T], axis=1)

        x2 = jax.vmap(warp_cols)(x1)

        fy, fx = jnp.fft.fftfreq(H), jnp.fft.fftfreq(W)
        FY, FX = jnp.meshgrid(fy, fx, indexing="ij")
        S = self.base_psd(FY, FX)
        Xf = jnp.fft.fftn(x2, axes=(1, 2))
        y = jnp.fft.ifftn(Xf * S[None, :, :], axes=(1, 2)).real
        return y


@jax.custom_jvp
def _specmix_patch_1d(x: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    """Compute IFFT( FFT(x) * S ) along the last axis."""
    Xf = jnp.fft.fft(x, axis=-1)
    return jnp.fft.ifft(Xf * S, axis=-1).real


@_specmix_patch_1d.defjvp
def _specmix_patch_1d_jvp(primals, tangents):
    x, S = primals
    dx, dS = tangents
    Xf = jnp.fft.fft(x, axis=-1)
    dXf = jnp.fft.fft(dx, axis=-1) if dx is not None else 0.0
    Yf = Xf * S
    dYf = dXf * S + Xf * dS
    return jnp.fft.ifft(Yf, axis=-1).real, jnp.fft.ifft(dYf, axis=-1).real


class PatchWiseSpectralMixture(eqx.Module):
    """
    Partition length L into P patches of length l.
    Each patch uses a Q‑component spectral mixture PSD.
    """

    L: int = eqx.field(static=True)
    l: int = eqx.field(static=True)
    P: int = eqx.field(static=True)
    Q: int = eqx.field(static=True)
    jitter: float = eqx.field(static=True)

    f: jnp.ndarray  # (l,)
    logits: jnp.ndarray  # (P, Q)
    mu: jnp.ndarray  # (P, Q)
    sigma: jnp.ndarray  # (P, Q)
    bias: jnp.ndarray  # (P, l)

    def __init__(
        self, L: int, patch_len: int, Q: int = 3, jitter: float = 1e-6, *, key
    ):
        assert L % patch_len == 0, "L must be divisible by patch_len"
        self.L, self.l = L, patch_len
        self.P = L // patch_len
        self.Q = Q
        self.jitter = jitter

        self.f = jnp.fft.fftfreq(self.l)

        k1, k2, k3, k4 = jr.split(key, 4)
        self.logits = jr.normal(k1, (self.P, self.Q))
        self.mu = jr.normal(k2, (self.P, self.Q)) * 0.3
        self.sigma = jnp.exp(jr.normal(k3, (self.P, self.Q)) * 0.3)
        self.bias = jr.normal(k4, (self.P, self.l))

    def make_S(
        self, w_p: jnp.ndarray, mu_p: jnp.ndarray, sig_p: jnp.ndarray
    ) -> jnp.ndarray:
        """Construct the length-l PSD for one patch."""
        g1 = jnp.exp(-0.5 * ((self.f[None, :] - mu_p[:, None]) / sig_p[:, None]) ** 2)
        g2 = jnp.exp(-0.5 * ((self.f[None, :] + mu_p[:, None]) / sig_p[:, None]) ** 2)
        return (w_p[:, None] * (g1 + g2)).sum(0) + self.jitter

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x : (B, L) -> returns (B, L)
        """
        B, L = x.shape
        assert L == self.L
        w = jax.nn.softmax(self.logits, axis=-1)
        S = jax.vmap(self.make_S)(w, self.mu, self.sigma)
        # split & apply
        x_split = x.reshape(B, self.P, self.l)
        y_split = jax.vmap(
            lambda xp, Sp: _specmix_patch_1d(xp, Sp), in_axes=(1, 0), out_axes=1
        )(x_split, S)
        return (y_split + self.bias[None, :, :]).reshape(B, self.L)


@jax.custom_jvp
def _specmix_patch_2d(x: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    """Compute IFFTn( FFTn(x) * S ) across last two axes."""
    Xf = jnp.fft.fftn(x, axes=(-2, -1))
    return jnp.fft.ifftn(Xf * S, axes=(-2, -1)).real


@_specmix_patch_2d.defjvp
def _specmix_patch_2d_jvp(primals, tangents):
    x, S = primals
    dx, dS = tangents
    Xf = jnp.fft.fftn(x, axes=(-2, -1))
    dXf = jnp.fft.fftn(dx, axes=(-2, -1)) if dx is not None else 0.0
    Yf = Xf * S
    dYf = dXf * S + Xf * dS
    return jnp.fft.ifftn(Yf, axes=(-2, -1)).real, jnp.fft.ifftn(dYf, axes=(-2, -1)).real


class PatchWiseSpectralMixture2d(eqx.Module):
    """
    Non-overlapping h×w patches; Q-component spectral mixture PSD per patch.
    """

    H: int = eqx.field(static=True)
    W: int = eqx.field(static=True)
    ph: int = eqx.field(static=True)
    pw: int = eqx.field(static=True)
    nh: int = eqx.field(static=True)
    nw: int = eqx.field(static=True)
    P: int = eqx.field(static=True)
    Q: int = eqx.field(static=True)
    jitter: float = eqx.field(static=True)

    fy: jnp.ndarray
    fx: jnp.ndarray
    logits: jnp.ndarray
    mu: jnp.ndarray
    sigma: jnp.ndarray
    bias: jnp.ndarray

    def __init__(
        self,
        H: int,
        W: int,
        patch_h: int,
        patch_w: int,
        Q: int = 3,
        jitter: float = 1e-6,
        *,
        key,
    ):
        assert H % patch_h == 0 and W % patch_w == 0
        self.H, self.W = H, W
        self.ph, self.pw = patch_h, patch_w
        self.nh, self.nw = H // patch_h, W // patch_w
        self.P = self.nh * self.nw
        self.Q = Q
        self.jitter = jitter

        self.fy = jnp.fft.fftfreq(self.ph)[:, None]
        self.fx = jnp.fft.fftfreq(self.pw)[None, :]

        k1, k2, k3, k4 = jr.split(key, 4)
        self.logits = jr.normal(k1, (self.P, self.Q))
        self.mu = jr.normal(k2, (self.P, self.Q, 2)) * 0.3
        self.sigma = jnp.exp(jr.normal(k3, (self.P, self.Q, 2)) * 0.3)
        self.bias = jr.normal(k4, (self.P, self.ph, self.pw))

    def make_S(
        self, w_p: jnp.ndarray, mu_p: jnp.ndarray, sig_p: jnp.ndarray
    ) -> jnp.ndarray:
        mu_y = mu_p[:, 0][:, None, None]
        mu_x = mu_p[:, 1][:, None, None]
        sig_y = sig_p[:, 0][:, None, None]
        sig_x = sig_p[:, 1][:, None, None]
        fy = self.fy[None, :, :]
        fx = self.fx[None, :, :]
        g1 = jnp.exp(-0.5 * (((fy - mu_y) / sig_y) ** 2 + ((fx - mu_x) / sig_x) ** 2))
        g2 = jnp.exp(-0.5 * (((fy + mu_y) / sig_y) ** 2 + ((fx + mu_x) / sig_x) ** 2))
        return (w_p[:, None, None] * (g1 + g2)).sum(0) + self.jitter

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (B, H, W) -> (B, H, W)"""
        B, H, W = x.shape
        assert (H, W) == (self.H, self.W)

        w = jax.nn.softmax(self.logits, axis=-1)
        S = jax.vmap(self.make_S)(w, self.mu, self.sigma)

        x_p = x.reshape(B, self.nh, self.ph, self.nw, self.pw)
        x_p = x_p.transpose(1, 3, 0, 2, 4).reshape(self.P, B, self.ph, self.pw)

        def apply_patch(xp, Sp, bp):
            return _specmix_patch_2d(xp, Sp) + bp

        y_p = jax.vmap(apply_patch, in_axes=(0, 0, 0))(x_p, S, self.bias)

        y = y_p.reshape(self.nh, self.nw, B, self.ph, self.pw)
        y = y.transpose(2, 0, 3, 1, 4).reshape(B, self.H, self.W)
        return y
