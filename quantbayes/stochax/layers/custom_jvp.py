from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

__all__ = [
    "Circulant",
    "BlockCirculant",
    "CirculantProcess",
    "BlockCirculantProcess",
    "SpectralConv1d",
    "SpectralConv2d",
]


@jax.custom_jvp
def circulant_matmul(x: jnp.ndarray, first_row: jnp.ndarray) -> jnp.ndarray:
    """
    Compute y = C x, where C is a circulant matrix defined by its first row.
    Instead of forming C explicitly, we compute via FFT:
      - first, compute first_col = roll(flip(first_row), shift=1)
      - then, y = IFFT( FFT(x) * FFT(first_col) )
    """
    # Compute first_col from first_row
    first_col = jnp.roll(jnp.flip(first_row), shift=1)
    fft_first_col = jnp.fft.fft(first_col)
    fft_x = jnp.fft.fft(x, axis=-1)
    y = jnp.fft.ifft(fft_x * fft_first_col, axis=-1).real
    return y


@circulant_matmul.defjvp
def circulant_matmul_jvp(primals, tangents):
    x, first_row = primals
    dx, dfirst_row = tangents
    # Forward pass (same as in circulant_matmul)
    first_col = jnp.roll(jnp.flip(first_row), shift=1)
    fft_first_col = jnp.fft.fft(first_col)
    fft_x = jnp.fft.fft(x, axis=-1)
    y = jnp.fft.ifft(fft_x * fft_first_col, axis=-1).real
    # Tangent contribution from x:
    dfft_x = jnp.fft.fft(dx, axis=-1)
    dy_dx = jnp.fft.ifft(dfft_x * fft_first_col, axis=-1).real
    # Tangent contribution from first_row:
    dfirst_col = jnp.roll(jnp.flip(dfirst_row), shift=1)
    dfft_first_col = jnp.fft.fft(dfirst_col)
    dy_df = jnp.fft.ifft(fft_x * dfft_first_col, axis=-1).real
    return y, dy_dx + dy_df


class Circulant(eqx.Module):
    """
    A circulant layer that uses a circulant weight matrix defined by its first row.
    The layer stores only the first row (a vector of shape (n,)) and a bias vector (shape (n,)).
    The forward pass computes:
        y = circulant_matmul(x_padded, first_row) + bias
    where circulant_matmul is accelerated via a custom JVP rule.

    If `padded_dim` is provided, the first row and bias are of length `padded_dim`, and
    the input is padded with zeros on the right to match that size.
    """

    first_row: jnp.ndarray  # shape (n,)
    bias: jnp.ndarray  # shape (n,)
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
        # Use padded_dim if provided; otherwise, no padding (square matrix of in_features).
        self.out_features = padded_dim if padded_dim is not None else in_features

        key1, key2 = jr.split(key)
        self.first_row = jr.normal(key1, (self.out_features,)) * init_scale
        self.bias = jr.normal(key2, (self.out_features,)) * init_scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # If the layer uses padding, pad the input x along its last dimension.
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
    # Ensure x has a batch dimension.
    single_example = x.ndim == 1
    if single_example:
        x = x[None, :]
    batch_size = x.shape[0]
    d_in = x.shape[-1]
    k_in = W.shape[1]
    b = W.shape[-1]

    # (1) Multiply by Bernoulli diagonal if provided.
    if d_bernoulli is not None:
        x_d = x * d_bernoulli[None, :]
    else:
        x_d = x

    # (2) Zero-pad x_d to length k_in * b.
    pad_len = k_in * b - d_in
    if pad_len > 0:
        x_d = jnp.pad(x_d, ((0, 0), (0, pad_len)), mode="constant", constant_values=0.0)

    # (3) Reshape into blocks: (batch, k_in, b)
    X_blocks = x_d.reshape(batch_size, k_in, b)
    # Compute FFT along the block dimension.
    X_fft = jnp.fft.fft(X_blocks, axis=-1)  # shape: (batch, k_in, b)

    # (4) Compute FFT for the circulant blocks in W.
    W_fft = jnp.fft.fft(W, axis=-1)  # shape: (k_out, k_in, b)

    # (5) For each output block row, sum over input blocks.
    def compute_block_row(i):
        # Multiply: for each input block j,
        #   multiply X_fft[:, j, :] with conj(W_fft[i, j, :])
        prod = X_fft * jnp.conjugate(W_fft[i, :, :])[None, :, :]  # (batch, k_in, b)
        sum_over_j = jnp.sum(prod, axis=1)  # (batch, b)
        # Inverse FFT to get the circulant product (real-valued).
        return jnp.fft.ifft(sum_over_j, axis=-1).real  # (batch, b)

    # Compute for all block rows (vmap over i=0,...,k_out-1).
    block_out = jax.vmap(compute_block_row)(jnp.arange(W.shape[0]))  # (k_out, batch, b)
    # Reshape: transpose to (batch, k_out, b) then flatten last two dims.
    out = jnp.transpose(block_out, (1, 0, 2)).reshape(batch_size, W.shape[0] * b)
    if single_example:
        out = out[0]
    return out


@block_circulant_matmul_custom.defjvp
def block_circulant_matmul_jvp(primals, tangents):
    W, x, d_bernoulli = primals
    dW, dx, dd = tangents  # dd corresponds to the tangent for d_bernoulli
    # ----- Forward Pass -----
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
    X_fft = jnp.fft.fft(X_blocks, axis=-1)  # (batch, k_in, b)
    W_fft = jnp.fft.fft(W, axis=-1)  # (k_out, k_in, b)

    def compute_block_row(i):
        prod = X_fft * jnp.conjugate(W_fft[i, :, :])[None, :, :]
        sum_over_j = jnp.sum(prod, axis=1)
        return jnp.fft.ifft(sum_over_j, axis=-1).real  # (batch, b)

    block_out = jax.vmap(compute_block_row)(jnp.arange(W.shape[0]))
    out = jnp.transpose(block_out, (1, 0, 2)).reshape(batch_size, W.shape[0] * b)
    if single_example:
        out = out[0]

    # ----- Tangent (JVP) Computation -----
    # First, differentiate through the input multiplication by d_bernoulli.
    if d_bernoulli is not None:
        # d(x_d) = (dx * d_bernoulli) + (x * dd)
        dx_d = dx * d_bernoulli[None, :] + (x * dd[None, :] if dd is not None else 0.0)
    else:
        dx_d = dx

    if pad_len > 0:
        dx_d = jnp.pad(
            dx_d, ((0, 0), (0, pad_len)), mode="constant", constant_values=0.0
        )
    dX_blocks = dx_d.reshape(batch_size, k_in, b)
    dX_fft = jnp.fft.fft(dX_blocks, axis=-1)  # (batch, k_in, b)

    # For dW, if provided compute its FFT; otherwise, treat as zero.
    if dW is not None:
        dW_fft = jnp.fft.fft(dW, axis=-1)  # (k_out, k_in, b)
    else:
        dW_fft = 0.0

    # For each output block row, the tangent contribution is given by:
    #   ifft( sum_j ( dX_fft[:, j, :] * conj(W_fft[i, j, :])
    #                + X_fft[:, j, :] * conj(dW_fft[i, j, :]) ) )
    def compute_block_row_tangent(i):
        term1 = dX_fft * jnp.conjugate(W_fft[i, :, :])[None, :, :]
        term2 = X_fft * (
            jnp.conjugate(dW_fft[i, :, :])[None, :, :] if dW is not None else 0.0
        )
        sum_over_j = jnp.sum(term1 + term2, axis=1)
        return jnp.fft.ifft(sum_over_j, axis=-1).real  # (batch, b)

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

    W: jnp.ndarray  # shape: (k_out, k_in, b)
    D_bernoulli: jnp.ndarray  # shape: (in_features,)
    bias: jnp.ndarray  # shape: (out_features,)
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

        # Determine the number of blocks along the input and output dimensions.
        k_in = (in_features + block_size - 1) // block_size
        k_out = (out_features + block_size - 1) // block_size
        object.__setattr__(self, "k_in", k_in)
        object.__setattr__(self, "k_out", k_out)

        # Initialize W with shape (k_out, k_in, block_size)
        k1, k2, k3 = jr.split(key, 3)
        self.W = jr.normal(k1, (k_out, k_in, block_size)) * init_scale

        # Initialize the Bernoulli diagonal if enabled.
        if use_bernoulli_diag:
            diag_signs = jnp.where(
                jr.bernoulli(k2, p=0.5, shape=(in_features,)), 1.0, -1.0
            )
        else:
            diag_signs = jnp.ones((in_features,))
        object.__setattr__(self, "D_bernoulli", diag_signs)

        # Initialize bias if requested.
        if use_bias:
            self.bias = jr.normal(k3, (out_features,)) * init_scale
        else:
            self.bias = jnp.zeros((out_features,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Compute block-circulant multiplication using the custom JVP rule.
        out = block_circulant_matmul_custom(self.W, x, self.D_bernoulli)
        # (Optional) slice the output if the padded output dimension is larger than out_features.
        k_out_b = self.k_out * self.block_size
        if k_out_b > self.out_features:
            out = out[..., : self.out_features]
        # Add bias.
        return out + self.bias[None, :] if out.ndim > 1 else out + self.bias


@jax.custom_jvp
def spectral_circulant_matmul(x: jnp.ndarray, fft_full: jnp.ndarray) -> jnp.ndarray:
    """
    Compute y = IFFT( FFT(x_pad) * fft_full ), with x zero-padded to length padded_dim.
    x can be (batch, d_in) or just (d_in,). Operation is along the last dimension.
    """
    padded_dim = fft_full.shape[0]
    single_example = x.ndim == 1
    if single_example:
        x = x[None, :]
    d_in = x.shape[-1]
    if d_in < padded_dim:
        pad_len = padded_dim - d_in
        x_pad = jnp.pad(x, ((0, 0), (0, pad_len)))
    elif d_in > padded_dim:
        # truncate if input is bigger than padded_dim
        x_pad = x[..., :padded_dim]
    else:
        x_pad = x
    X_fft = jnp.fft.fft(x_pad, axis=-1)
    y_fft = X_fft * fft_full[None, :]
    y = jnp.fft.ifft(y_fft, axis=-1).real
    return y[0] if single_example else y


@spectral_circulant_matmul.defjvp
def spectral_circulant_matmul_jvp(primals, tangents):
    x, fft_full = primals
    dx, dfft = tangents
    padded_dim = fft_full.shape[0]
    single_example = x.ndim == 1
    if single_example:
        x = x[None, :]
        if dx is not None:
            dx = dx[None, :]
    d_in = x.shape[-1]
    if d_in < padded_dim:
        pad_len = padded_dim - d_in
        x_pad = jnp.pad(x, ((0, 0), (0, pad_len)))
        dx_pad = jnp.pad(dx, ((0, 0), (0, pad_len))) if dx is not None else None
    elif d_in > padded_dim:
        x_pad = x[..., :padded_dim]
        dx_pad = dx[..., :padded_dim] if dx is not None else None
    else:
        x_pad = x
        dx_pad = dx

    X_fft = jnp.fft.fft(x_pad, axis=-1)
    primal_y_fft = X_fft * fft_full[None, :]
    primal_y = jnp.fft.ifft(primal_y_fft, axis=-1).real

    # Compute tangents
    if dx_pad is None:
        dX_fft = 0.0
    else:
        dX_fft = jnp.fft.fft(dx_pad, axis=-1)

    if dfft is None:
        term2 = 0.0
    else:
        term2 = X_fft * dfft[None, :]

    dY_fft = dX_fft * fft_full[None, :] + term2
    dY = jnp.fft.ifft(dY_fft, axis=-1).real

    if single_example:
        return primal_y[0], dY[0]
    return primal_y, dY


class CirculantProcess(eqx.Module):
    """
    Spectral circulant layer with custom JVP.

    This layer parameterizes the circulant weight matrix via a half-spectrum
    (w_real and w_imag) to enforce Hermitian symmetry (ensuring real outputs)
    and imposes a spectral prior. The Fourier coefficients are computed on the fly
    in get_fourier_coeffs(), so __call__ remains pure and immutable.
    """

    in_features: int
    padded_dim: int
    alpha: float
    K: int
    k_half: int = eqx.static_field()

    w_real: jnp.ndarray  # shape: (k_half,)
    w_imag: jnp.ndarray  # shape: (k_half,)
    bias: jnp.ndarray  # shape: (padded_dim,)

    def __init__(
        self,
        in_features: int,
        padded_dim: int = None,
        alpha: float = 1.0,
        K: int = None,
        *,
        key,
        init_scale: float = 0.1,
        bias_init_scale: float = 0.1,
    ):
        self.in_features = in_features
        self.padded_dim = padded_dim if padded_dim is not None else in_features
        self.alpha = alpha
        self.k_half = (self.padded_dim // 2) + 1
        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        key_r, key_i, key_b = jr.split(key, 3)
        w_r = jr.normal(key_r, (self.k_half,)) * init_scale
        w_i = jr.normal(key_i, (self.k_half,)) * init_scale
        # Enforce DC (index 0) and Nyquist (if applicable) to be purely real.
        w_i = w_i.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            w_i = w_i.at[-1].set(0.0)
        self.w_real = w_r
        self.w_imag = w_i
        self.bias = jr.normal(key_b, (self.padded_dim,)) * bias_init_scale

    def get_fourier_coeffs(self) -> jnp.ndarray:
        """
        Compute and return the full Fourier coefficients from the half-spectrum.
        """
        freq_mask = jnp.arange(self.k_half) < self.K
        half_complex = (self.w_real * freq_mask) + 1j * (self.w_imag * freq_mask)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            nyquist = half_complex[-1].real[None]
            fft_full = jnp.concatenate(
                [half_complex[:-1], nyquist, jnp.conjugate(half_complex[1:-1])[::-1]]
            )
        else:
            fft_full = jnp.concatenate(
                [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
            )
        return fft_full

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        fft_full = self.get_fourier_coeffs()
        out = spectral_circulant_matmul(x, fft_full)
        if out.ndim == 2:
            out = out + self.bias[None, :]
        else:
            out = out + self.bias
        return out


class BlockCirculantProcess(eqx.Module):
    """
    Block-circulant layer with custom JVP.

    This layer partitions the weight matrix into blocks where each block is circulant.
    The forward pass computes:
       out = block_circulant_matmul_custom(W, x, D_bernoulli) + bias
    without mutating internal state. The Fourier coefficients for each block are
    computed on the fly in get_fourier_coeffs().
    """

    W: jnp.ndarray  # shape: (k_out, k_in, block_size)
    D_bernoulli: Optional[jnp.ndarray]  # shape: (in_features,) or None
    bias: jnp.ndarray  # shape: (out_features,)
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


@jax.custom_jvp
def spectral_conv1d(x: jnp.ndarray, fft_kernel_1d: jnp.ndarray) -> jnp.ndarray:
    """
    x: shape (batch, length) or (length,)
    fft_kernel_1d: shape (padded_len,) – the 1D FFT of the kernel
    Returns shape (batch, padded_len) or (padded_len,) by default.
    """
    padded_len = fft_kernel_1d.shape[0]
    single_example = x.ndim == 1
    if single_example:
        x = x[None, :]  # add batch dim

    length = x.shape[-1]
    # Zero-pad or truncate
    if length < padded_len:
        pad_len = padded_len - length
        x_padded = jnp.pad(x, ((0, 0), (0, pad_len)))
    elif length > padded_len:
        x_padded = x[..., :padded_len]
    else:
        x_padded = x

    # FFT
    X_fft = jnp.fft.fft(x_padded, axis=-1)  # shape (batch, padded_len)
    # Multiply
    Y_fft = X_fft * fft_kernel_1d[None, :]
    # iFFT
    y = jnp.fft.ifft(Y_fft, axis=-1).real

    if single_example:
        y = y[0]
    return y


@spectral_conv1d.defjvp
def spectral_conv1d_jvp(primals, tangents):
    x, fft_kernel_1d = primals
    dx, dkernel = tangents

    padded_len = fft_kernel_1d.shape[0]
    single_example = x.ndim == 1
    if single_example:
        x = x[None, :]
        dx = None if dx is None else dx[None, :]

    length = x.shape[-1]
    if length < padded_len:
        pad_len = padded_len - length
        x_padded = jnp.pad(x, ((0, 0), (0, pad_len)))
        dx_padded = None if dx is None else jnp.pad(dx, ((0, 0), (0, pad_len)))
    elif length > padded_len:
        x_padded = x[..., :padded_len]
        dx_padded = None if dx is None else dx[..., :padded_len]
    else:
        x_padded = x
        dx_padded = dx

    X_fft = jnp.fft.fft(x_padded, axis=-1)
    primal_Y_fft = X_fft * fft_kernel_1d[None, :]
    primal_y = jnp.fft.ifft(primal_Y_fft, axis=-1).real

    dX_fft = jnp.fft.fft(dx_padded, axis=-1) if dx_padded is not None else 0.0
    dK_fft = dkernel if dkernel is not None else 0.0

    dY_fft = dX_fft * fft_kernel_1d[None, :] + X_fft * dK_fft[None, :]
    dY = jnp.fft.ifft(dY_fft, axis=-1).real

    if single_example:
        return primal_y[0], dY[0]
    return primal_y, dY


class SpectralConv1d(eqx.Module):
    """
    1D "spectral convolution" in Equinox (deterministic).
    Single-channel for simplicity.
    """

    in_size: int
    padded_len: int
    alpha: float
    K: int
    k_half: int = eqx.static_field()

    w_real: jnp.ndarray  # shape (k_half,)
    w_imag: jnp.ndarray  # shape (k_half,)
    bias: jnp.ndarray  # shape (padded_len,)

    def __init__(
        self,
        in_size: int,
        padded_len: int = None,
        alpha: float = 1.0,
        K: int = None,
        *,
        key,
        init_scale: float = 0.1,
        bias_init_scale: float = 0.1,
    ):
        """
        :param in_size: nominal length of the input signal
        :param padded_len: If provided, we pad/truncate to this dimension; else = in_size
        :param alpha: spectral decay exponent (for reference). Not strictly used here unless you want to custom init
        :param K: number of active frequencies. If None, use full half-spectrum
        :param key: PRNGKey for initialization
        """
        self.in_size = in_size
        self.padded_len = padded_len if padded_len is not None else in_size
        self.alpha = alpha
        self.k_half = (self.padded_len // 2) + 1
        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        # Initialize frequency parameters
        key_r, key_i, key_b = jr.split(key, 3)
        w_r = jr.normal(key_r, (self.k_half,)) * init_scale
        w_i = jr.normal(key_i, (self.k_half,)) * init_scale

        # enforce purely real DC and Nyquist if even
        w_i = w_i.at[0].set(0.0)
        if (self.padded_len % 2 == 0) and (self.k_half > 1):
            w_i = w_i.at[-1].set(0.0)

        # store
        self.w_real = w_r
        self.w_imag = w_i
        self.bias = jr.normal(key_b, (self.padded_len,)) * bias_init_scale

    def get_fourier_coeffs_1d(self) -> jnp.ndarray:
        """
        Build the full spectrum (size padded_len) from half-spectrum w_real + i*w_imag
        with Hermitian symmetry.
        """
        freq_mask = jnp.arange(self.k_half) < self.K
        half_cplx = (self.w_real * freq_mask) + 1j * (self.w_imag * freq_mask)
        if (self.padded_len % 2 == 0) and (self.k_half > 1):
            nyquist = half_cplx[-1].real[None]  # shape (1,)
            fft_full = jnp.concatenate(
                [half_cplx[:-1], nyquist, jnp.conjugate(half_cplx[1:-1])[::-1]]
            )
        else:
            fft_full = jnp.concatenate([half_cplx, jnp.conjugate(half_cplx[1:])[::-1]])
        return fft_full

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: shape (batch, length) or (length,).
        returns shape (batch, padded_len) or (padded_len,).
        """
        fft_full = self.get_fourier_coeffs_1d()
        y = spectral_conv1d(x, fft_full)
        # add bias
        if y.ndim == 2:
            return y + self.bias[None, :]
        else:
            return y + self.bias


@jax.custom_jvp
def spectral_conv2d(x: jnp.ndarray, fft_kernel_2d: jnp.ndarray) -> jnp.ndarray:
    """
    x: shape (batch, H, W) or (H, W)
    fft_kernel_2d: shape (H_pad, W_pad)
    Return shape (batch, H_pad, W_pad) or (H_pad, W_pad).
    """
    H_pad, W_pad = fft_kernel_2d.shape
    single_example = x.ndim == 2
    if single_example:
        x = x[None, ...]  # shape (1,H,W)

    H_in, W_in = x.shape[-2], x.shape[-1]

    # Pad/truncate
    pad_h = max(0, H_pad - H_in)
    pad_w = max(0, W_pad - W_in)
    x_padded = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w)))
    x_padded = x_padded[..., :H_pad, :W_pad]  # if needed for oversize

    # 2D FFT
    X_fft = jnp.fft.fftn(x_padded, axes=(-2, -1))
    # multiply
    Y_fft = X_fft * fft_kernel_2d[None, :, :]
    # iFFT
    y = jnp.fft.ifftn(Y_fft, axes=(-2, -1)).real

    if single_example:
        y = y[0]
    return y


@spectral_conv2d.defjvp
def spectral_conv2d_jvp(primals, tangents):
    x, fft_kernel_2d = primals
    dx, dkernel = tangents

    H_pad, W_pad = fft_kernel_2d.shape
    single_example = x.ndim == 2

    if single_example:
        x = x[None, ...]
        dx = None if dx is None else dx[None, ...]

    H_in, W_in = x.shape[-2], x.shape[-1]
    pad_h = max(0, H_pad - H_in)
    pad_w = max(0, W_pad - W_in)

    x_padded = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w)))
    x_padded = x_padded[..., :H_pad, :W_pad]

    X_fft = jnp.fft.fftn(x_padded, axes=(-2, -1))
    primal_Y_fft = X_fft * fft_kernel_2d[None, :, :]
    primal_y = jnp.fft.ifftn(primal_Y_fft, axes=(-2, -1)).real

    dX_fft = (
        0.0
        if dx is None
        else jnp.fft.fftn(
            jnp.pad(dx, ((0, 0), (0, pad_h), (0, pad_w)))[..., :H_pad, :W_pad],
            axes=(-2, -1),
        )
    )
    dK_fft = 0.0 if dkernel is None else dkernel

    dY_fft = dX_fft * fft_kernel_2d[None, :, :] + X_fft * dK_fft[None, :, :]
    dY = jnp.fft.ifftn(dY_fft, axes=(-2, -1)).real

    if single_example:
        return primal_y[0], dY[0]
    return primal_y, dY


class SpectralConv2d(eqx.Module):
    """
    2D "spectral convolution" in Equinox (deterministic).
    Single-channel for simplicity.
    """

    in_height: int
    in_width: int
    H_pad: int
    W_pad: int
    alpha: float

    w_real: jnp.ndarray  # shape (H_pad, W_pad)
    w_imag: jnp.ndarray  # shape (H_pad, W_pad)
    bias: jnp.ndarray  # shape (H_pad, W_pad)

    def __init__(
        self,
        in_height: int,
        in_width: int,
        H_pad: int = None,
        W_pad: int = None,
        alpha: float = 1.0,
        *,
        key,
        init_scale: float = 0.1,
        bias_init_scale: float = 0.1,
    ):
        self.in_height = in_height
        self.in_width = in_width
        self.H_pad = H_pad if (H_pad is not None) else in_height
        self.W_pad = W_pad if (W_pad is not None) else in_width
        self.alpha = alpha

        key_r, key_i, key_b = jr.split(key, 3)
        wr = jr.normal(key_r, (self.H_pad, self.W_pad)) * init_scale
        wi = jr.normal(key_i, (self.H_pad, self.W_pad)) * init_scale

        # example: force DC = real
        wr = wr.at[0, 0].set(wr[0, 0])
        wi = wi.at[0, 0].set(0.0)

        self.w_real = wr
        self.w_imag = wi
        self.bias = jr.normal(key_b, (self.H_pad, self.W_pad)) * bias_init_scale

    def get_fourier_coeffs_2d(self) -> jnp.ndarray:
        """
        Combine w_real + i*w_imag into a single complex array of shape (H_pad, W_pad).
        You *can* manually enforce full Hermitian symmetry if you want a guaranteed real
        convolution kernel in the spatial domain.
        """
        return self.w_real + 1j * self.w_imag

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: shape (batch, H_in, W_in) or (H_in, W_in)
        returns shape (batch, H_pad, W_pad) or (H_pad, W_pad).
        """
        fft2d = self.get_fourier_coeffs_2d()
        y = spectral_conv2d(x, fft2d)
        if y.ndim == 3:
            return y + self.bias[None, :, :]
        else:
            return y + self.bias
