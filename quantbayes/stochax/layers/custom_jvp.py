from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

__all__ = [
    "JVPCirculant",
    "JVPBlockCirculant",
    "JVPCirculantProcess",
    "JVPBlockCirculantProcess",
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


class JVPCirculant(eqx.Module):
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


class JVPBlockCirculant(eqx.Module):
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
    Compute y = IFFT( FFT(x_pad) * fft_full ) with x zero-padded to length padded_dim,
    where fft_full is the full (complex) Fourier mask.

    x can be a 1D array or batched with shape (..., in_features).
    The operation is performed along the last axis.
    """
    padded_dim = fft_full.shape[0]
    single_example = False
    if x.ndim == 1:
        single_example = True
        x = x[None, :]
    d_in = x.shape[-1]
    if d_in < padded_dim:
        pad_len = padded_dim - d_in
        x_pad = jnp.pad(x, ((0, 0), (0, pad_len)))
    elif d_in > padded_dim:
        x_pad = x[..., :padded_dim]
    else:
        x_pad = x
    X_fft = jnp.fft.fft(x_pad, axis=-1)
    y_fft = X_fft * fft_full[None, :]
    y = jnp.fft.ifft(y_fft, axis=-1).real
    if single_example:
        return y[0]
    return y


@spectral_circulant_matmul.defjvp
def spectral_circulant_matmul_jvp(primals, tangents):
    x, fft_full = primals
    dx, dfft = tangents

    padded_dim = fft_full.shape[0]
    single_example = False
    if x.ndim == 1:
        single_example = True
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
    y_fft = X_fft * fft_full[None, :]
    y = jnp.fft.ifft(y_fft, axis=-1).real

    # Compute tangent contributions.
    if dx is None:
        dX_fft = 0.0
    else:
        dX_fft = jnp.fft.fft(dx_pad, axis=-1)
    term1 = dX_fft * fft_full[None, :]
    if dfft is None:
        term2 = 0.0
    else:
        term2 = X_fft * dfft[None, :]
    dy_fft = term1 + term2
    dy = jnp.fft.ifft(dy_fft, axis=-1).real

    if single_example:
        return y[0], dy[0]
    return y, dy


class JVPCirculantProcess(eqx.Module):
    """
    Equinox-based spectral circulant layer with custom JVP.

    The layer stores trainable Fourier coefficients for the half-spectrum.
    In the forward pass it reconstructs the full Fourier mask and then calls a
    custom_jvp-decorated function for the FFT-based multiplication.
    """

    in_features: int
    padded_dim: int
    alpha: float
    K: int
    k_half: int = eqx.static_field()
    w_real: jnp.ndarray  # shape (k_half,)
    w_imag: jnp.ndarray  # shape (k_half,)

    # Store the last computed full FFT mask for retrieval.
    _last_fft_full: jnp.ndarray = eqx.field(default=None, repr=False)

    def __init__(
        self,
        in_features: int,
        padded_dim: Optional[int] = None,
        alpha: float = 1.0,
        K: int = None,
        *,
        key,
    ):
        self.in_features = in_features
        self.padded_dim = padded_dim if padded_dim is not None else in_features
        self.alpha = alpha
        self.k_half = self.padded_dim // 2 + 1

        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        key_r, key_i = jr.split(key)
        self.w_real = jr.normal(key_r, (self.k_half,)) * 0.1
        self.w_imag = jr.normal(key_i, (self.k_half,)) * 0.1

        # Ensure the DC component is real.
        self.w_imag = self.w_imag.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            self.w_imag = self.w_imag.at[-1].set(0.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Create a mask for frequencies beyond K.
        freq_mask = jnp.arange(self.k_half) < self.K
        half_complex = (self.w_real * freq_mask) + 1j * (self.w_imag * freq_mask)
        # Reconstruct the full Fourier spectrum.
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            nyquist = half_complex[-1].real[None]
            fft_full = jnp.concatenate(
                [half_complex[:-1], nyquist, jnp.conjugate(half_complex[1:-1])[::-1]]
            )
        else:
            fft_full = jnp.concatenate(
                [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
            )
        # Store for later retrieval.
        object.__setattr__(self, "_last_fft_full", fft_full)
        return spectral_circulant_matmul(x, fft_full)

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self._last_fft_full is None:
            raise ValueError(
                "No Fourier coefficients available. Call the layer on some input first."
            )
        return self._last_fft_full


class JVPBlockCirculantProcess(eqx.Module):
    """
    Equinox module for a block-circulant layer that uses a custom JVP rule.

    The layer stores a block of circulant parameters (each block is defined by its first row)
    and (optionally) a Bernoulli diagonal. The forward pass computes:

         out = block_circulant_matmul_custom(W, x, D_bernoulli) + bias

    where the custom JVP rule for block_circulant_matmul_custom reuses FFT computations.
    """

    W: jnp.ndarray  # shape (k_out, k_in, block_size)
    D_bernoulli: Optional[jnp.ndarray]  # shape (in_features,) or None
    bias: jnp.ndarray  # shape (out_features,)
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()
    block_size: int = eqx.static_field()
    k_in: int = eqx.static_field()
    k_out: int = eqx.static_field()

    # Store the last computed full block FFT coefficients.
    _last_block_fft: jnp.ndarray = eqx.field(default=None, repr=False)

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
        self.k_in = (in_features + block_size - 1) // block_size
        self.k_out = (out_features + block_size - 1) // block_size

        k1, k2, k3 = jr.split(key, 3)
        self.W = jr.normal(k1, (self.k_out, self.k_in, block_size)) * init_scale

        if use_diag:
            diag = jr.bernoulli(k2, p=0.5, shape=(in_features,))
            self.D_bernoulli = jnp.where(diag, 1.0, -1.0)
        else:
            self.D_bernoulli = None

        if use_bias:
            self.bias = jr.normal(k3, (out_features,)) * init_scale
        else:
            self.bias = jnp.zeros((out_features,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Assume block_circulant_matmul_custom is defined elsewhere with a custom JVP rule.
        out = block_circulant_matmul_custom(self.W, x, self.D_bernoulli)
        if self.k_out * self.block_size > self.out_features:
            out = out[..., : self.out_features]
        if out.ndim == 1:
            out = out + self.bias
        else:
            out = out + self.bias[None, :]
        return out

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self._last_block_fft is None:
            raise ValueError(
                "No Fourier coefficients available. Call the layer on some input first."
            )
        return self._last_block_fft
