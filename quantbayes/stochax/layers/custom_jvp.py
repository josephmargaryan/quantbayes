from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

__all__ = [
    "Circulant",
    "BlockCirculant",
    "SpectralCirculantLayer",
    "SpectralCirculantLayer2d",
    "AdaptiveSpectralCirculantLayer2d",
    "AdaptiveSpectralCirculantLayer" "GibbsKernel1D",
    "GibbsKernel2D",
    "InputWarping1D",
    "InputWarping2D",
    "PatchWiseSpectralMixture1D",
    "PatchWiseSpectralMixture2D",
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

    dfft_x = jnp.fft.rfft(dx, axis=-1) if dx is not None else 0.0
    dy_dx = jnp.fft.irfft(dfft_x * fft_first_col, n=n, axis=-1)

    if dfirst_row is not None:
        dfirst_col = jnp.roll(jnp.flip(dfirst_row), shift=1)
        dfft_first = jnp.fft.rfft(dfirst_col, axis=-1)
        dy_df = jnp.fft.irfft(fft_x * dfft_first, n=n, axis=-1)
    else:
        dy_df = 0.0

    return y, dy_dx + dy_df


class Circulant(eqx.Module):
    """
    A production‐quality circulant layer:
      y = circulant_matmul(x_padded, first_row) + bias

    Stores only first_row (n,) and bias (n,); pads input if in_features != out_features.
    """

    first_row: jnp.ndarray
    bias: jnp.ndarray
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()

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
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()
    block_size: int = eqx.static_field()
    k_in: int = eqx.static_field()
    k_out: int = eqx.static_field()

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
    Wrap-around 2D convolution by FFT:
      y = ifft2( fft2(kernel) * fft2(x_padded) ).real
    kernel: (H_pad, W_pad)
    x:      (..., H_in, W_in)
    returns (..., H_pad, W_pad)
    """
    H_pad, W_pad = kernel.shape
    single = x.ndim == 2
    if single:
        x = x[None, ...]
    H_in, W_in = x.shape[-2:]

    pad_h = max(0, H_pad - H_in)
    pad_w = max(0, W_pad - W_in)
    x_pad = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w)))
    x_pad = x_pad[..., :H_pad, :W_pad]

    Kf = jnp.fft.fftn(kernel, axes=(0, 1))
    Xf = jnp.fft.fftn(x_pad, axes=(-2, -1))
    Yf = Kf[None, :, :] * Xf
    y = jnp.fft.ifftn(Yf, axes=(-2, -1)).real

    return y[0] if single else y


@circulant_conv2d_fft.defjvp
def circulant_conv2d_fft_jvp(primals, tangents):
    kernel, x = primals
    dk, dx = tangents
    H_pad, W_pad = kernel.shape
    single = x.ndim == 2

    if single:
        x = x[None, ...]
        dx = None if dx is None else dx[None, ...]

    H_in, W_in = x.shape[-2:]
    pad_h = max(0, H_pad - H_in)
    pad_w = max(0, W_pad - W_in)
    x_pad = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w)))[..., :H_pad, :W_pad]
    dx_pad = (
        None
        if dx is None
        else jnp.pad(dx, ((0, 0), (0, pad_h), (0, pad_w)))[..., :H_pad, :W_pad]
    )

    Kf = jnp.fft.fftn(kernel, axes=(0, 1))
    Xf = jnp.fft.fftn(x_pad, axes=(-2, -1))
    Yf = Kf[None, :, :] * Xf
    y = jnp.fft.ifftn(Yf, axes=(-2, -1)).real

    dKf = jnp.fft.fftn(dk, axes=(0, 1)) if dk is not None else 0.0
    dXf = jnp.fft.fftn(dx_pad, axes=(-2, -1)) if dx_pad is not None else 0.0
    dYf = dKf[None, :, :] * Xf + Kf[None, :, :] * dXf
    dy = jnp.fft.ifftn(dYf, axes=(-2, -1)).real

    if single:
        return y[0], dy[0]
    else:
        return y, dy


class Circulant2d(eqx.Module):
    """
    2-D circulant (wrap-around) convolution layer via FFT.

    Attributes:
      kernel:   real weights of shape (H_pad, W_pad)
      bias:     real bias of same shape
      H_in, W_in:   input spatial dims
      H_pad, W_pad: padded (and output) spatial dims
    """

    kernel: jnp.ndarray
    bias: jnp.ndarray
    H_in: int = eqx.static_field()
    W_in: int = eqx.static_field()
    H_pad: int = eqx.static_field()
    W_pad: int = eqx.static_field()

    def __init__(
        self,
        H_in: int,
        W_in: int,
        H_pad: Optional[int] = None,
        W_pad: Optional[int] = None,
        *,
        key,
        init_scale: float = 1.0,
        bias_scale: float = 1.0,
    ):
        self.H_in = H_in
        self.W_in = W_in
        self.H_pad = H_pad or H_in
        self.W_pad = W_pad or W_in

        k1, k2 = jr.split(key, 2)
        self.kernel = jr.normal(k1, (self.H_pad, self.W_pad)) * init_scale
        self.bias = jr.normal(k2, (self.H_pad, self.W_pad)) * bias_scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (..., H_in, W_in)
        returns (..., H_pad, W_pad)
        """
        out = circulant_conv2d_fft(self.kernel, x)
        return out + self.bias


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


class SpectralCirculantLayer(eqx.Module):
    """
    Deterministic spectral‑circulant layer with a trainable decay exponent α.
    Uses the same custom‑JVP FFT mat‑mul defined above.
    """

    in_features: int = eqx.static_field()
    padded_dim: int = eqx.static_field()
    K: int = eqx.static_field()
    k_half: int = eqx.static_field()

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


class AdaptiveSpectralCirculantLayer(eqx.Module):
    in_features: int = eqx.static_field()
    padded_dim: int = eqx.static_field()
    K: int = eqx.static_field()
    k_half: int = eqx.static_field()

    alpha_global: float
    delta_alpha: jnp.ndarray
    w_real: jnp.ndarray
    w_imag: jnp.ndarray
    bias: jnp.ndarray

    def __init__(
        self,
        in_features: int,
        padded_dim: int = None,
        alpha_global: float = 1.0,
        K: int = None,
        *,
        key,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        delta_init: float = 0.0,
    ):
        self.in_features = in_features
        self.padded_dim = padded_dim if padded_dim is not None else in_features
        self.k_half = (self.padded_dim // 2) + 1
        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        self.alpha_global = alpha_global
        self.delta_alpha = jnp.full((self.k_half,), delta_init)

        key_r, key_i, key_b = jr.split(key, 3)
        k_idx = jnp.arange(self.k_half, dtype=jnp.float32)
        S0 = 1.0 / jnp.sqrt(1.0 + k_idx**alpha_global)
        self.w_real = jr.normal(key_r, (self.k_half,)) * (init_scale * S0)
        w_i = jr.normal(key_i, (self.k_half,)) * (init_scale * S0)
        w_i = w_i.at[0].set(0.0)
        if self.padded_dim % 2 == 0 and self.k_half > 1:
            w_i = w_i.at[-1].set(0.0)
        self.w_imag = w_i

        self.bias = jr.normal(key_b, (self.padded_dim,)) * bias_scale

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
        if y.ndim == 1:
            return y + self.bias
        else:
            return y + self.bias[None, :]


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
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()
    block_size: int = eqx.static_field()
    k_in: int = eqx.static_field()
    k_out: int = eqx.static_field()

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


def upsample_bilinear(grid, H_pad, W_pad):
    """Bilinearly up‑sample a coarse grid (Bilinear resize, no grad issues)."""
    gh, gw = grid.shape
    y = jnp.linspace(0.0, 1.0, H_pad)
    x = jnp.linspace(0.0, 1.0, W_pad)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    yy = yy * (gh - 1)
    xx = xx * (gw - 1)
    y0, x0 = jnp.floor(yy).astype(int), jnp.floor(xx).astype(int)
    y1, x1 = jnp.clip(y0 + 1, 0, gh - 1), jnp.clip(x0 + 1, 0, gw - 1)
    wy, wx = yy - y0, xx - x0
    v00, v01 = grid[y0, x0], grid[y0, x1]
    v10, v11 = grid[y1, x0], grid[y1, x1]
    return (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11)


def _enforce_hermitian(fft2d: jnp.ndarray) -> jnp.ndarray:
    H, W = fft2d.shape
    conj_flip = jnp.flip(jnp.conj(fft2d), (0, 1))
    herm = 0.5 * (fft2d + conj_flip)
    herm = herm.at[0, 0].set(jnp.real(herm[0, 0]))
    if H % 2 == 0:
        herm = herm.at[H // 2, :].set(jnp.real(herm[H // 2, :]))
    if W % 2 == 0:
        herm = herm.at[:, W // 2].set(jnp.real(herm[:, W // 2]))
    return herm


@jax.custom_jvp
def spectral_circulant_conv2d(x: jnp.ndarray, fft_kernel: jnp.ndarray) -> jnp.ndarray:
    H_pad, W_pad = fft_kernel.shape
    single = x.ndim == 2
    if single:
        x = x[None, ...]
    H_in, W_in = x.shape[-2:]
    pad_h = max(0, H_pad - H_in)
    pad_w = max(0, W_pad - W_in)
    x_pad = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w)))
    x_pad = x_pad[..., :H_pad, :W_pad]
    Xf = jnp.fft.fftn(x_pad, axes=(-2, -1))
    Yf = Xf * fft_kernel[None, :, :]
    y = jnp.fft.ifftn(Yf, axes=(-2, -1)).real
    return y[0] if single else y


@spectral_circulant_conv2d.defjvp
def _spectral_circulant_conv2d_jvp(primals, tangents):
    x, fft_kernel = primals
    dx, dk = tangents
    H_pad, W_pad = fft_kernel.shape
    single = x.ndim == 2
    if single:
        x = x[None, ...]
        dx = None if dx is None else dx[None, ...]
    H_in, W_in = x.shape[-2:]
    pad_h = max(0, H_pad - H_in)
    pad_w = max(0, W_pad - W_in)
    x_pad = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w)))
    x_pad = x_pad[..., :H_pad, :W_pad]
    dx_pad = None
    if dx is not None:
        dx_pad = jnp.pad(dx, ((0, 0), (0, pad_h), (0, pad_w)))
        dx_pad = dx_pad[..., :H_pad, :W_pad]
    Xf = jnp.fft.fftn(x_pad, axes=(-2, -1))
    Yf = Xf * fft_kernel[None, :, :]
    y = jnp.fft.ifftn(Yf, axes=(-2, -1)).real
    dXf = 0.0 if dx_pad is None else jnp.fft.fftn(dx_pad, axes=(-2, -1))
    dKf = 0.0 if dk is None else dk
    dYf = dXf * fft_kernel[None, :, :] + Xf * dKf[None, :, :]
    dy = jnp.fft.ifftn(dYf, axes=(-2, -1)).real
    if single:
        return y[0], dy[0]
    return y, dy


class SpectralCirculantLayer2d(eqx.Module):
    H_pad: int = eqx.static_field()
    W_pad: int = eqx.static_field()
    alpha: jnp.ndarray  # scalar decay exponent
    w_real: jnp.ndarray  # shape (H_pad, W_pad)
    w_imag: jnp.ndarray  # shape (H_pad, W_pad)
    bias: jnp.ndarray  # shape (H_pad, W_pad)

    def __init__(
        self,
        H_in: int,
        W_in: int,
        H_pad: int = None,
        W_pad: int = None,
        alpha_init: float = 1.0,
        *,
        key,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
    ):
        self.H_pad = H_pad or H_in
        self.W_pad = W_pad or W_in
        self.alpha = jnp.array(alpha_init, dtype=jnp.float32)
        key_r, key_i, key_b = jr.split(key, 3)
        u = jnp.fft.fftfreq(self.H_pad) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        std0 = 1.0 / jnp.sqrt(1.0 + R**alpha_init)
        self.w_real = jr.normal(key_r, (self.H_pad, self.W_pad)) * (init_scale * std0)
        w_i = jr.normal(key_i, (self.H_pad, self.W_pad)) * (init_scale * std0)
        w_i = w_i.at[0, 0].set(0.0)
        if self.H_pad % 2 == 0:
            w_i = w_i.at[self.H_pad // 2, :].set(0.0)
        if self.W_pad % 2 == 0:
            w_i = w_i.at[:, self.W_pad // 2].set(0.0)
        self.w_imag = w_i
        self.bias = jr.normal(key_b, (self.H_pad, self.W_pad)) * bias_scale

    def get_fft_kernel(self) -> jnp.ndarray:
        return _enforce_hermitian(self.w_real + 1j * self.w_imag)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        fft2d = self.get_fft_kernel()
        y = spectral_circulant_conv2d(x, fft2d)
        if y.ndim == 2:
            return y + self.bias
        else:
            return y + self.bias[None, :, :]


class AdaptiveSpectralCirculantLayer2d(eqx.Module):
    H_pad: int = eqx.static_field()
    W_pad: int = eqx.static_field()
    alpha_global: float
    alpha_coarse_shape: tuple[int, int] = eqx.static_field()
    delta_alpha_coarse: jnp.ndarray
    w_real: jnp.ndarray
    w_imag: jnp.ndarray
    bias: jnp.ndarray

    def __init__(
        self,
        H_in: int,
        W_in: int,
        H_pad: int = None,
        W_pad: int = None,
        alpha_global: float = 1.0,
        alpha_coarse_shape: tuple[int, int] = (8, 8),
        *,
        key,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        delta_init: float = 0.0,
    ):
        self.H_pad = H_pad or H_in
        self.W_pad = W_pad or W_in
        self.alpha_global = alpha_global
        self.alpha_coarse_shape = alpha_coarse_shape
        self.delta_alpha_coarse = jnp.full(alpha_coarse_shape, delta_init)
        key_r, key_i, key_b = jr.split(key, 3)
        u = jnp.fft.fftfreq(self.H_pad) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        std0 = 1.0 / jnp.sqrt(1.0 + R**alpha_global)
        self.w_real = jr.normal(key_r, (self.H_pad, self.W_pad)) * (init_scale * std0)
        w_i = jr.normal(key_i, (self.H_pad, self.W_pad)) * (init_scale * std0)
        w_i = w_i.at[0, 0].set(0.0)
        if self.H_pad % 2 == 0:
            w_i = w_i.at[self.H_pad // 2, :].set(0.0)
        if self.W_pad % 2 == 0:
            w_i = w_i.at[:, self.W_pad // 2].set(0.0)
        self.w_imag = w_i
        self.bias = jr.normal(key_b, (self.H_pad, self.W_pad)) * bias_scale

    def get_spectral_alpha(self) -> jnp.ndarray:
        delta_full = upsample_bilinear(self.delta_alpha_coarse, self.H_pad, self.W_pad)
        return jnp.clip(self.alpha_global + delta_full, 0.1, 5.0)

    def get_fft_kernel(self) -> jnp.ndarray:
        return _enforce_hermitian(self.w_real + 1j * self.w_imag)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        fft2d = self.get_fft_kernel()
        y = spectral_circulant_conv2d(x, fft2d)
        if y.ndim == 2:
            return y + self.bias
        else:
            return y + self.bias[None, :, :]


class GibbsKernel(eqx.Module):
    N: int = eqx.static_field()
    lengthscale_net: callable

    def __init__(self, in_features: int, lengthscale_net, *, key=None):
        self.N = in_features
        self.lengthscale_net = lengthscale_net

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, N)
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
    N: int = eqx.static_field()
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
    H: int = eqx.static_field()
    W: int = eqx.static_field()
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
    H: int = eqx.static_field()
    W: int = eqx.static_field()
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

    L: int = eqx.static_field()
    l: int = eqx.static_field()
    P: int = eqx.static_field()
    Q: int = eqx.static_field()
    jitter: float = eqx.static_field()

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

    H: int = eqx.static_field()
    W: int = eqx.static_field()
    ph: int = eqx.static_field()
    pw: int = eqx.static_field()
    nh: int = eqx.static_field()
    nw: int = eqx.static_field()
    P: int = eqx.static_field()
    Q: int = eqx.static_field()
    jitter: float = eqx.static_field()

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
