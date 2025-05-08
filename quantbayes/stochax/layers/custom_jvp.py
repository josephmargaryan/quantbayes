from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

__all__ = [
    "Circulant",
    "BlockCirculant",
    "SpectralCirculantLayer",
    "BilinearSpectralCirculantLayer",
    "BlockCirculantProcess",
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


# ---------------------------------------------------------------------------
# 1) custom‐JVP FFT mat‐mul
# ---------------------------------------------------------------------------
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

    # directional derivative
    dXf = jnp.fft.fft(dx_pad, axis=-1) if dx_pad is not None else 0.0
    term2 = Xf * dfft[None, :] if dfft is not None else 0.0
    dyf = dXf * fft_full[None, :] + term2
    dy = jnp.fft.ifft(dyf, axis=-1).real

    if single:
        return y_primal[0], dy[0]
    return y_primal, dy


# ---------------------------------------------------------------------------
# 2) Deterministic CirculantProcess with learnable α
# ---------------------------------------------------------------------------
class SpectralCirculantLayer(eqx.Module):
    """
    Deterministic spectral‑circulant layer with a trainable decay exponent α.
    Uses the same custom‑JVP FFT mat‑mul defined above.
    """

    in_features: int
    padded_dim: int
    K: int
    k_half: int = eqx.static_field()

    # trainable parameters
    alpha: jnp.ndarray  # scalar decay exponent
    w_real: jnp.ndarray  # shape (k_half,)
    w_imag: jnp.ndarray  # shape (k_half,)
    bias: jnp.ndarray  # shape (padded_dim,)

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
        # dims
        self.in_features = in_features
        self.padded_dim = padded_dim if padded_dim is not None else in_features
        self.k_half = (self.padded_dim // 2) + 1
        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        # trainable decay exponent
        # (will only affect training via your spectral_penalty)
        self.alpha = jnp.array(alpha_init, dtype=jnp.float32)

        # initialize real & imag weights (full half‑spectrum)
        key_r, key_i, key_b = jr.split(key, 3)
        # compute initial spectral scale S_k = 1/√(1 + k^α_init)
        k_idx = jnp.arange(self.k_half, dtype=jnp.float32)
        prior_s = 1.0 / jnp.sqrt(1.0 + k_idx**alpha_init)

        w_r = jr.normal(key_r, (self.k_half,)) * (init_scale * prior_s)
        w_i = jr.normal(key_i, (self.k_half,)) * (init_scale * prior_s)

        # enforce real DC and Nyquist
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
        # mask active freqs
        mask = jnp.arange(self.k_half) < self.K
        half = (self.w_real * mask) + 1j * (self.w_imag * mask)

        # reflect for full circulant spectrum
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
        # add bias (handles both [d] and [batch, d])
        if y.ndim == 1:
            return y + self.bias
        else:
            return y + self.bias[None, :]


class BilinearSpectralCirculantLayer(eqx.Module):
    in_features: int
    padded_dim: int
    K: int
    k_half: int = eqx.static_field()

    # trainable parameters
    alpha_global: float  # fixed baseline
    delta_alpha: jnp.ndarray  # shape (k_half,)
    w_real: jnp.ndarray  # shape (k_half,)
    w_imag: jnp.ndarray  # shape (k_half,)
    bias: jnp.ndarray  # shape (padded_dim,)

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

        # store the global α and a trainable offset Δα
        self.alpha_global = alpha_global
        self.delta_alpha = jnp.full((self.k_half,), delta_init)

        # init w_real, w_imag just as before
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
        # exactly the same Hermitian build as before,
        # *independent* of α:
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


def upsample_bilinear(grid, H_pad, W_pad):
    """Bilinearly up‑sample a coarse grid (Bilinear resize, no grad issues)."""
    gh, gw = grid.shape
    # Normalised coordinates in [0,1]
    y = jnp.linspace(0.0, 1.0, H_pad)
    x = jnp.linspace(0.0, 1.0, W_pad)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    # Map to coarse‑grid indices
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


# Placeholder for the 2D FFT conv with custom JVP (as defined earlier)
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


# ---------------------------------------------------------------------------
# 1) Deterministic single-α 2D spectral layer
# ---------------------------------------------------------------------------
class SpectralPowerLawConv2dDet(eqx.Module):
    H_pad: int
    W_pad: int
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
        # trainable scalar α
        self.alpha = jnp.array(alpha_init, dtype=jnp.float32)
        # rng split
        key_r, key_i, key_b = jr.split(key, 3)
        # radial frequency grid
        u = jnp.fft.fftfreq(self.H_pad) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        # initialize weights with same spectral scaling
        std0 = 1.0 / jnp.sqrt(1.0 + R**alpha_init)
        self.w_real = jr.normal(key_r, (self.H_pad, self.W_pad)) * (init_scale * std0)
        w_i = jr.normal(key_i, (self.H_pad, self.W_pad)) * (init_scale * std0)
        # enforce real DC/Nyquist
        w_i = w_i.at[0, 0].set(0.0)
        if self.H_pad % 2 == 0:
            w_i = w_i.at[self.H_pad // 2, :].set(0.0)
        if self.W_pad % 2 == 0:
            w_i = w_i.at[:, self.W_pad // 2].set(0.0)
        self.w_imag = w_i
        # bias
        self.bias = jr.normal(key_b, (self.H_pad, self.W_pad)) * bias_scale

    def get_fft_kernel(self) -> jnp.ndarray:
        # enforce Hermitian symmetry
        return _enforce_hermitian(self.w_real + 1j * self.w_imag)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        fft2d = self.get_fft_kernel()
        y = spectral_circulant_conv2d(x, fft2d)
        # add bias (handles [H,W] or [batch,H,W])
        if y.ndim == 2:
            return y + self.bias
        else:
            return y + self.bias[None, :, :]


# ---------------------------------------------------------------------------
# 2) Deterministic bilinear (nonstationary) 2D spectral layer
# ---------------------------------------------------------------------------
class BilinearSpectralPowerLawConv2dDet(eqx.Module):
    H_pad: int
    W_pad: int
    alpha_global: float
    alpha_coarse_shape: tuple[int, int]
    delta_alpha_coarse: jnp.ndarray  # shape alpha_coarse_shape
    w_real: jnp.ndarray  # shape (H_pad, W_pad)
    w_imag: jnp.ndarray  # shape (H_pad, W_pad)
    bias: jnp.ndarray  # shape (H_pad, W_pad)

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
        # trainable coarse offsets Δα
        self.delta_alpha_coarse = jnp.full(alpha_coarse_shape, delta_init)
        # rng split for w_real/imag/bias
        key_r, key_i, key_b = jr.split(key, 3)
        # init weights like stationary with α_global
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
        # bilinearly upsample and clamp
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


# -----------------------------------------------------------------------------
# Custom-JVP primitives for patch-wise spectral mixture
# -----------------------------------------------------------------------------
@jax.custom_jvp
def patchwise_specmix_1d(x: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    """
    1D FFT-based convolution with custom JVP:
      y = irfft(rfft(x) * S)
    x: (B, P), S: (P//2+1,)
    returns y: (B, P)
    """
    Xf = jnp.fft.rfft(x, axis=-1)
    return jnp.fft.irfft(Xf * S, n=x.shape[-1], axis=-1)


@patchwise_specmix_1d.defjvp
def patchwise_specmix_1d_jvp(primals, tangents):
    x, S = primals
    dx, dS = tangents
    Xf = jnp.fft.rfft(x, axis=-1)
    dXf = jnp.fft.rfft(dx, axis=-1) if dx is not None else 0.0
    Yf = Xf * S
    dYf = dXf * S + Xf * dS
    y = jnp.fft.irfft(Yf, n=x.shape[-1], axis=-1)
    dy = jnp.fft.irfft(dYf, n=x.shape[-1], axis=-1)
    return y, dy


@jax.custom_jvp
def patchwise_specmix_2d(x: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    """
    2D FFT-based convolution with custom JVP:
      y = ifftn(fftn(x) * S)
    x: (B, ph, pw), S: (ph, pw)
    returns y: (B, ph, pw)
    """
    Xf = jnp.fft.fftn(x, axes=(-2, -1))
    return jnp.fft.ifftn(Xf * S[None, :, :], axes=(-2, -1)).real


@patchwise_specmix_2d.defjvp
def patchwise_specmix_2d_jvp(primals, tangents):
    x, S = primals
    dx, dS = tangents
    Xf = jnp.fft.fftn(x, axes=(-2, -1))
    dXf = jnp.fft.fftn(dx, axes=(-2, -1)) if dx is not None else 0.0
    Yf = Xf * S[None, :, :]
    dYf = dXf * S[None, :, :] + Xf * dS[None, :, :]
    y = jnp.fft.ifftn(Yf, axes=(-2, -1)).real
    dy = jnp.fft.ifftn(dYf, axes=(-2, -1)).real
    return y, dy


# -----------------------------------------------------------------------------
# 1) GibbsKernel1DDet
# -----------------------------------------------------------------------------
class GibbsKernel1DDet(eqx.Module):
    N: int
    lengthscale_net: callable

    def __init__(self, in_features: int, lengthscale_net, *, key=None):
        self.N = in_features
        self.lengthscale_net = lengthscale_net

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, N)
        B, N = x.shape
        assert N == self.N
        pos = jnp.linspace(0.0, 1.0, N)
        log_l = self.lengthscale_net(pos)  # (N,)
        l = jnp.exp(log_l)  # (N,)

        l_i = l[:, None]  # (N,1)
        l_j = l[None, :]  # (1,N)
        denom = l_i**2 + l_j**2 + 1e-6

        sq = jnp.sqrt(2 * (l_i * l_j) / denom)
        dmat = (jnp.arange(N)[:, None] - jnp.arange(N)[None, :]) ** 2
        exp = jnp.exp(-dmat / denom)

        K = sq * exp  # (N,N)
        return x @ K.T  # (B,N)


# -----------------------------------------------------------------------------
# 2) InputWarping1DDet
# -----------------------------------------------------------------------------
class InputWarping1DDet(eqx.Module):
    N: int
    warp_net: callable
    base_psd: callable

    def __init__(self, in_features: int, warp_net, base_psd, *, key=None):
        self.N = in_features
        self.warp_net = warp_net
        self.base_psd = base_psd

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, N)
        B, N = x.shape
        assert N == self.N
        pos = jnp.linspace(0.0, 1.0, N)
        delta_u = self.warp_net(pos)  # (N,)
        u = pos + delta_u  # (N,)

        # warp each sequence
        def warp_one(seq):
            return jnp.interp(pos, u, seq)

        x_warp = jax.vmap(warp_one)(x)  # (B, N)

        # stationary FFT conv in warped space
        freqs = jnp.fft.fftfreq(N)  # (N,)
        S = self.base_psd(freqs)  # (N,)
        Xf = jnp.fft.fft(x_warp, axis=-1)
        y = jnp.fft.ifft(Xf * S[None, :], axis=-1).real
        return y


# -----------------------------------------------------------------------------
# 3) GibbsKernel2DDet (separable)
# -----------------------------------------------------------------------------
class GibbsKernel2DDet(eqx.Module):
    H: int
    W: int
    net_x: callable
    net_y: callable

    def __init__(
        self, H: int, W: int, lengthscale_net_x, lengthscale_net_y, *, key=None
    ):
        self.H, self.W = H, W
        self.net_x = lengthscale_net_x
        self.net_y = lengthscale_net_y

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W)
        B, H, W = x.shape
        assert (H, W) == (self.H, self.W)
        pos_x = jnp.linspace(0.0, 1.0, W)
        pos_y = jnp.linspace(0.0, 1.0, H)
        lx = jnp.exp(self.net_x(pos_x))  # (W,)
        ly = jnp.exp(self.net_y(pos_y))  # (H,)

        def gibbs1d(pos, ell):
            d = pos[:, None] - pos[None, :]
            l_i = ell[:, None]
            l_j = ell[None, :]
            denom = l_i**2 + l_j**2 + 1e-6
            sq = jnp.sqrt(2 * l_i * l_j / denom)
            return sq * jnp.exp(-(d**2) / denom)

        Kx = gibbs1d(pos_x, lx)  # (W,W)
        Ky = gibbs1d(pos_y, ly)  # (H,H)

        # separable conv: width then height
        x1 = jnp.einsum("bhw,wd->bhd", x, Kx)
        y = jnp.einsum("bhd,hk->bkd", x1, Ky)
        return y  # (B, H, W)


# -----------------------------------------------------------------------------
# 4) InputWarping2DDet (separable)
# -----------------------------------------------------------------------------
class InputWarping2DDet(eqx.Module):
    H: int
    W: int
    warp_x: callable
    warp_y: callable
    base_psd: callable

    def __init__(self, H: int, W: int, warp_net_x, warp_net_y, base_psd, *, key=None):
        self.H, self.W = H, W
        self.warp_x, self.warp_y = warp_net_x, warp_net_y
        self.base_psd = base_psd

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W)
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

        # 2D FFT conv
        fy, fx = jnp.fft.fftfreq(H), jnp.fft.fftfreq(W)
        FY, FX = jnp.meshgrid(fy, fx, indexing="ij")
        S = self.base_psd(FY, FX)  # (H,W)
        Xf = jnp.fft.fftn(x2, axes=(1, 2))
        y = jnp.fft.ifftn(Xf * S[None, :, :], axes=(1, 2)).real
        return y


# -----------------------------------------------------------------------------
# 5) PatchWiseSpectralMixture1DDet with per‑patch bias
# -----------------------------------------------------------------------------
class PatchWiseSpectralMixture1DDet(eqx.Module):
    N: int
    P: int
    M: int
    logits_w: jnp.ndarray  # (n_patches, M)
    raw_mu: jnp.ndarray  # (n_patches, M)
    raw_sigma: jnp.ndarray  # (n_patches, M)
    bias: jnp.ndarray  # (n_patches, P)

    def __init__(self, in_features: int, patch_size: int, n_mixtures: int = 3, *, key):
        assert (
            in_features % patch_size == 0
        ), "in_features must be divisible by patch_size"
        self.N, self.P, self.M = in_features, patch_size, n_mixtures
        n_patches = self.N // self.P
        k1, k2, k3, k4 = jr.split(key, 4)
        # initialize mixture params
        self.logits_w = jnp.zeros((n_patches, n_mixtures))
        self.raw_mu = jr.normal(k1, (n_patches, n_mixtures)) * 0.01
        self.raw_sigma = jnp.zeros((n_patches, n_mixtures))
        # initialize per-patch bias vectors to zero
        self.bias = jnp.zeros((n_patches, self.P))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, N = x.shape
        assert N == self.N, f"Expected input features {self.N}, got {N}"

        w = jax.nn.softmax(self.logits_w, axis=-1)  # (n_patches, M)
        mu = 0.5 * jax.nn.sigmoid(self.raw_mu)  # (n_patches, M)
        sigma = jax.nn.softplus(self.raw_sigma) + 1e-3  # (n_patches, M)
        freqs = jnp.fft.rfftfreq(self.P)  # (P//2+1,)

        outputs = []
        for p in range(self.N // self.P):
            xp = x[:, p * self.P : (p + 1) * self.P]  # (B, P)
            # build PSD S
            S = jnp.zeros_like(freqs)
            for q in range(self.M):
                S = S + w[p, q] * (
                    jnp.exp(-0.5 * ((freqs - mu[p, q]) / sigma[p, q]) ** 2)
                    + jnp.exp(-0.5 * ((freqs + mu[p, q]) / sigma[p, q]) ** 2)
                )
            S = S + 1e-6
            # convolution via custom JVP primitive
            y_p = patchwise_specmix_1d(xp, S)  # (B, P)
            # add per-patch bias vector
            y_p = y_p + self.bias[p]  # broadcast to (B, P)
            outputs.append(y_p)
        return jnp.concatenate(outputs, axis=-1)  # (B, N)


# -----------------------------------------------------------------------------
# 6) PatchWiseSpectralMixture2DDet with per‑patch bias
# -----------------------------------------------------------------------------
class PatchWiseSpectralMixture2DDet(eqx.Module):
    H: int
    W: int
    ph: int
    pw: int
    nh: int
    nw: int
    M: int
    logits_w: jnp.ndarray  # (npatch, M)
    raw_mu: jnp.ndarray  # (npatch, M, 2)
    raw_sigma: jnp.ndarray  # (npatch, M, 2)
    bias: jnp.ndarray  # (npatch, ph, pw)

    def __init__(
        self, H: int, W: int, patch_h: int, patch_w: int, n_mixtures: int = 3, *, key
    ):
        assert (
            H % patch_h == 0 and W % patch_w == 0
        ), "H,W must be divisible by patch_h,patch_w"
        self.H, self.W = H, W
        self.ph, self.pw = patch_h, patch_w
        self.nh, self.nw = H // patch_h, W // patch_w
        self.M = n_mixtures
        npatch = self.nh * self.nw
        k1, k2, k3, k4 = jr.split(key, 4)
        self.logits_w = jnp.zeros((npatch, self.M))
        self.raw_mu = jr.normal(k1, (npatch, self.M, 2)) * 0.01
        self.raw_sigma = jnp.zeros((npatch, self.M, 2))
        # bias maps init to zero
        self.bias = jnp.zeros((npatch, self.ph, self.pw))
        # freq grids
        fy = jnp.fft.fftfreq(self.ph)
        fx = jnp.fft.fftfreq(self.pw)
        self.FY, self.FX = jnp.meshgrid(fy, fx, indexing="ij")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W = x.shape
        assert (H, W) == (
            self.H,
            self.W,
        ), f"Expected ({self.H},{self.W}), got ({H},{W})"

        w = jax.nn.softmax(self.logits_w, axis=-1)  # (npatch, M)
        mu = 0.5 * jax.nn.sigmoid(self.raw_mu)  # (npatch, M, 2)
        sigma = jax.nn.softplus(self.raw_sigma) + 1e-3  # (npatch, M, 2)

        patches = []
        idx = 0
        for i in range(self.nh):
            row = []
            for j in range(self.nw):
                sub = x[
                    :, i * self.ph : (i + 1) * self.ph, j * self.pw : (j + 1) * self.pw
                ]  # (B, ph, pw)
                # build PSD S
                S = jnp.zeros((self.ph, self.pw))
                for q in range(self.M):
                    dy = (self.FY - mu[idx, q, 0]) / sigma[idx, q, 0]
                    dx = (self.FX - mu[idx, q, 1]) / sigma[idx, q, 1]
                    ga1 = jnp.exp(-0.5 * (dy**2 + dx**2))
                    dy2 = (self.FY + mu[idx, q, 0]) / sigma[idx, q, 0]
                    dx2 = (self.FX + mu[idx, q, 1]) / sigma[idx, q, 1]
                    ga2 = jnp.exp(-0.5 * (dy2**2 + dx2**2))
                    S = S + w[idx, q] * (ga1 + ga2)
                S = S + 1e-6
                # convolution via custom JVP
                y_patch = patchwise_specmix_2d(sub, S)  # (B, ph, pw)
                # add per-patch bias map
                y_patch = y_patch + self.bias[idx]  # broadcast
                row.append(y_patch)
                idx += 1
            patches.append(jnp.concatenate(row, axis=2))  # along width
        return jnp.concatenate(patches, axis=1)  # along height
