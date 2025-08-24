from __future__ import annotations
from typing import Any, Tuple, Optional, Sequence

import equinox as eqx
from equinox import Module, field
from jaxtyping import Array, PRNGKeyArray
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
