from typing import Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import transforms

__all__ = [
    "Circulant",
    "BlockCirculant",
    "CirculantProcess",
    "BlockCirculantProcess",
    "SpectralConv1d",
    "SpectralConv2d",
    "SpectralTransposed2d"
]


@jax.custom_jvp
def fft_matmul_custom(first_row: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """
    Performs circulant matrix multiplication via FFT.
    Given the first row of a circulant matrix and an input X,
    computes:
        result = ifft( fft(first_row) * fft(X) ).real
    """
    # Compute FFT of the circulant's defining vector and of the input.
    first_row_fft = jnp.fft.fft(first_row, axis=-1)
    X_fft = jnp.fft.fft(X, axis=-1)
    # Multiply (with broadcasting) in Fourier domain.
    result_fft = first_row_fft[None, :] * X_fft
    result = jnp.fft.ifft(result_fft, axis=-1).real
    return result


@fft_matmul_custom.defjvp
def fft_matmul_custom_jvp(primals, tangents):
    first_row, X = primals
    d_first_row, dX = tangents

    # Recompute FFTs from the primal inputs (to avoid extra FFT calls in reverse mode).
    first_row_fft = jnp.fft.fft(first_row, axis=-1)
    X_fft = jnp.fft.fft(X, axis=-1)
    primal_out = jnp.fft.ifft(first_row_fft[None, :] * X_fft, axis=-1).real

    # Compute the directional derivatives.
    d_first_row_fft = (
        jnp.fft.fft(d_first_row, axis=-1) if d_first_row is not None else 0.0
    )
    dX_fft = jnp.fft.fft(dX, axis=-1) if dX is not None else 0.0
    tangent_out = jnp.fft.ifft(
        d_first_row_fft[None, :] * X_fft + first_row_fft[None, :] * dX_fft, axis=-1
    ).real
    return primal_out, tangent_out


@jax.custom_jvp
def block_circulant_matmul_custom(
    W: jnp.ndarray, x: jnp.ndarray, d_bernoulli: jnp.ndarray = None
) -> jnp.ndarray:
    """
    Performs block–circulant matrix multiplication via FFT.

    Parameters:
      W: shape (k_out, k_in, b) – each W[i,j,:] is the first row of a b×b circulant block.
      x: shape (batch, d_in) or (d_in,)
      d_bernoulli: optional diagonal (of ±1) to decorrelate projections.

    Returns:
      Output of shape (batch, k_out * b) (sliced to d_out if needed).

    The forward pass computes:
      1. Optionally scales x by the fixed Bernoulli diagonal.
      2. Pads x to length k_in*b and reshapes it to (batch, k_in, b).
      3. Computes W_fft = fft(W, axis=-1) and X_fft = fft(x_blocks, axis=-1).
      4. For each block row, sums over j:
            Y_fft[:, i, :] = sum_j X_fft[:, j, :] * conj(W_fft[i, j, :])
      5. Returns Y = ifft(Y_fft).real reshaped to (batch, k_out*b).
    """
    # Ensure x is 2D.
    if x.ndim == 1:
        x = x[None, :]
    batch_size, d_in = x.shape
    k_out, k_in, b = W.shape
    d_out = k_out * b

    # Optionally apply the Bernoulli diagonal.
    if d_bernoulli is not None:
        x = x * d_bernoulli[None, :]

    # Zero-pad x if needed.
    pad_len = k_in * b - d_in
    if pad_len > 0:
        x = jnp.pad(x, ((0, 0), (0, pad_len)))
    # Reshape x into blocks.
    x_blocks = x.reshape(batch_size, k_in, b)

    # Compute FFTs.
    W_fft = jnp.fft.fft(W, axis=-1)  # shape: (k_out, k_in, b)
    X_fft = jnp.fft.fft(x_blocks, axis=-1)  # shape: (batch, k_in, b)

    # Multiply in Fourier domain and sum over the input blocks.
    # For each output block row i:
    #   Y_fft[:, i, :] = sum_j X_fft[:, j, :] * conj(W_fft[i, j, :])
    Y_fft = jnp.sum(X_fft[:, None, :, :] * jnp.conjugate(W_fft)[None, :, :, :], axis=2)
    Y = jnp.fft.ifft(Y_fft, axis=-1).real
    out = Y.reshape(batch_size, k_out * b)
    return out


@block_circulant_matmul_custom.defjvp
def block_circulant_matmul_custom_jvp(primals, tangents):
    W, x, d_bernoulli = primals
    (
        dW,
        dx,
        dd,
    ) = tangents  # dd is the tangent for d_bernoulli (ignored here for simplicity)

    if x.ndim == 1:
        x = x[None, :]
    batch_size, d_in = x.shape
    k_out, k_in, b = W.shape

    # Forward pass (as above).
    if d_bernoulli is not None:
        x_eff = x * d_bernoulli[None, :]
    else:
        x_eff = x
    pad_len = k_in * b - d_in
    if pad_len > 0:
        x_eff = jnp.pad(x_eff, ((0, 0), (0, pad_len)))
    x_blocks = x_eff.reshape(batch_size, k_in, b)
    W_fft = jnp.fft.fft(W, axis=-1)
    X_fft = jnp.fft.fft(x_blocks, axis=-1)
    Y_fft = jnp.sum(X_fft[:, None, :, :] * jnp.conjugate(W_fft)[None, :, :, :], axis=2)
    primal_out = jnp.fft.ifft(Y_fft, axis=-1).real.reshape(batch_size, k_out * b)

    # Compute tangent for x.
    if dx is None:
        dx_eff = 0.0
    else:
        if d_bernoulli is not None:
            dx_eff = dx * d_bernoulli[None, :]
        else:
            dx_eff = dx
    if dx_eff is not None:
        if pad_len > 0:
            dx_eff = jnp.pad(dx_eff, ((0, 0), (0, pad_len)))
        x_tangent_blocks = dx_eff.reshape(batch_size, k_in, b)
        X_tangent_fft = jnp.fft.fft(x_tangent_blocks, axis=-1)
    else:
        X_tangent_fft = 0.0

    # Compute tangent for W.
    if dW is not None:
        W_tangent_fft = jnp.fft.fft(dW, axis=-1)
    else:
        W_tangent_fft = 0.0

    # The directional derivative in the Fourier domain is:
    # dY_fft = sum_j [ X_tangent_fft[:, None, j, :] * conj(W_fft)[None, :, j, :] +
    #                  X_fft[:, None, j, :] * conj(W_tangent_fft)[None, :, j, :] ]
    dY_fft = jnp.sum(
        X_tangent_fft[:, None, :, :] * jnp.conjugate(W_fft)[None, :, :, :]
        + X_fft[:, None, :, :] * jnp.conjugate(W_tangent_fft)[None, :, :, :],
        axis=2,
    )
    tangent_out = jnp.fft.ifft(dY_fft, axis=-1).real.reshape(batch_size, k_out * b)
    return primal_out, tangent_out


class Circulant:
    """
    FFT–based circulant layer that uses a custom JVP rule for faster gradients.
    The forward pass computes:
        hidden = ifft( fft(first_row) * fft(X) ).real + bias
    and the JVP uses the saved FFT computations.

    If `padded_dim` is provided, the first row and bias are sampled for that dimension,
    and the input is padded with zeros on the right to match the padded dimension.
    """

    def __init__(
        self,
        in_features: int,
        padded_dim: Optional[int] = None,
        name: str = "fft_layer",
        first_row_prior_fn=lambda shape: dist.Normal(0, 1)
        .expand(shape)
        .to_event(len(shape)),
        bias_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape).to_event(1),
    ):
        self.in_features = in_features
        # Use padded_dim if provided; otherwise, no padding (i.e. padded_dim equals in_features).
        self.padded_dim = padded_dim if padded_dim is not None else in_features
        self.name = name
        self.first_row_prior_fn = first_row_prior_fn
        self.bias_prior_fn = bias_prior_fn

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        # If padding is used, pad the input X along its last dimension.
        if self.padded_dim != self.in_features:
            pad_width = [(0, 0)] * (X.ndim - 1) + [
                (0, self.padded_dim - self.in_features)
            ]
            X = jnp.pad(X, pad_width)
        first_row = numpyro.sample(
            f"{self.name}_first_row", self.first_row_prior_fn([self.padded_dim])
        )
        bias_circulant = numpyro.sample(
            f"{self.name}_bias_circulant", self.bias_prior_fn([self.padded_dim])
        )
        hidden = fft_matmul_custom(first_row, X) + bias_circulant[None, :]
        return hidden


class BlockCirculant:
    """
    Block–circulant layer with custom JVP rules.

    This layer:
      1. Samples W of shape (k_out, k_in, block_size) (each block is circulant).
      2. Optionally samples a Bernoulli diagonal for the input.
      3. Computes the forward pass via the custom FFT–based block–circulant matmul.
      4. Uses the custom JVP for faster gradient computation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int,
        name: str = "block_circ_layer",
        W_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape).to_event(len(shape)),
        use_diag: bool = True,
        bias_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape).to_event(1),
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.name = name

        # Determine block counts along each dimension.
        self.k_in = (in_features + block_size - 1) // block_size
        self.k_out = (out_features + block_size - 1) // block_size

        self.W_prior_fn = W_prior_fn
        self.use_diag = use_diag
        self.bias_prior_fn = bias_prior_fn
        self.diag_prior = lambda shape: dist.TransformedDistribution(
            dist.Bernoulli(0.5).expand(shape).to_event(len(shape)),
            [transforms.AffineTransform(loc=-1.0, scale=2.0)],
        )

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        # Sample block–circulant weights.
        W = numpyro.sample(
            f"{self.name}_W",
            self.W_prior_fn([self.k_out, self.k_in, self.block_size]),
        )

        # Optionally sample and apply the Bernoulli diagonal.
        if self.use_diag:
            d_bernoulli = numpyro.sample(
                f"{self.name}_D",
                self.diag_prior([self.in_features]),
            )
        else:
            d_bernoulli = None

        # Compute the block–circulant multiplication via the custom JVP function.
        out = block_circulant_matmul_custom(W, X, d_bernoulli)

        # Sample and add bias.
        b = numpyro.sample(
            f"{self.name}_bias",
            self.bias_prior_fn([self.out_features]),
        )
        out = out + b[None, :]

        # If the padded output dimension is larger than out_features, slice the result.
        if self.k_out * self.block_size > self.out_features:
            out = out[:, : self.out_features]

        return out


@jax.custom_jvp
def spectral_circulant_matmul(x: jnp.ndarray, fft_full: jnp.ndarray) -> jnp.ndarray:
    padded_dim = fft_full.shape[0]
    single_example = x.ndim == 1
    if single_example:
        x = x[None, :]
    d_in = x.shape[-1]
    if d_in < padded_dim:
        pad_len = padded_dim - d_in
        x_pad = jnp.pad(x, ((0, 0), (0, pad_len)))
    elif d_in > padded_dim:
        # truncate if needed
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


class CirculantProcess:
    """
    NumPyro-based 'spectral' circulant layer with a custom JVP and an explicit bias.
    Matches JVPCirculant's parameter count: we have padded_dim for the bias,
    plus real/imag spectral params. In practice, this code uses the half-spectrum
    representation (so it's slightly fewer than 2*padded_dim parameters), but you
    can store the full real/imag arrays if you want an exact 2*padded_dim count.

    Usage:
      layer = JVPCirculantProcess(in_features=..., padded_dim=..., name="...", ...)
      y = layer(x)  # inside a numpyro model
    """

    def __init__(
        self,
        in_features: int,
        padded_dim: int = None,
        alpha: float = 1.0,
        K: int = None,
        name: str = "spectral_circ_jvp",
        prior_fn=None,  # a function: scale -> distribution
    ):
        """
        :param in_features: input dimension
        :param padded_dim: if not None, zero-pad/truncate to that dimension
        :param alpha: exponent for frequency-dependent scale
        :param K: number of active frequencies to keep
        :param name: base name for NumPyro samples
        :param prior_fn: a callable scale -> distribution (default Normal(0, scale))
        """
        self.in_features = in_features
        self.padded_dim = padded_dim if padded_dim is not None else in_features
        self.alpha = alpha
        self.name = name

        self.k_half = (self.padded_dim // 2) + 1
        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        if prior_fn is None:
            # By default, Normal(0, scale)
            self.prior_fn = lambda scale: dist.Normal(0.0, scale)
        else:
            self.prior_fn = prior_fn

        self._last_fft_full = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1) Sample real + imag parts for active frequencies
        freq_idx = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_idx**self.alpha)

        active_idx = jnp.arange(self.K)
        active_real = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(prior_std[active_idx]).expand([self.K]).to_event(1),
        )
        active_imag = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(prior_std[active_idx]).expand([self.K]).to_event(1),
        )

        # 2) Build up full real/imag up to k_half
        full_real = jnp.zeros((self.k_half,))
        full_imag = jnp.zeros((self.k_half,))
        full_real = full_real.at[active_idx].set(active_real)
        full_imag = full_imag.at[active_idx].set(active_imag)

        # Force DC component purely real
        full_imag = full_imag.at[0].set(0.0)
        # If even length, also force Nyquist purely real
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            full_imag = full_imag.at[-1].set(0.0)

        half_complex = full_real + 1j * full_imag

        # 3) Mirror to get the full FFT mask
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            nyquist = half_complex[-1].real[None]
            fft_full = jnp.concatenate(
                [half_complex[:-1], nyquist, jnp.conjugate(half_complex[1:-1])[::-1]]
            )
        else:
            fft_full = jnp.concatenate(
                [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
            )

        self._last_fft_full = jax.lax.stop_gradient(fft_full)

        # 4) Sample a bias (padded_dim) and do the spectral matmul
        bias_spectral = numpyro.sample(
            f"{self.name}_bias_spectral",
            dist.Normal(0.0, 1.0).expand([self.padded_dim]).to_event(1),
        )

        out = spectral_circulant_matmul(x, fft_full)
        if out.ndim == 2:
            return out + bias_spectral[None, :]
        else:
            return out + bias_spectral

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self._last_fft_full is None:
            raise ValueError("No Fourier coefficients available; call the layer first.")
        return self._last_fft_full


# ------------------------------------------------------------------


# ----------------------------------------------------------------
# Custom JVP function for block–circulant multiplication in the frequency domain.
@jax.custom_jvp
def block_circulant_spectral_matmul_custom(
    fft_full: jnp.ndarray, x: jnp.ndarray, d_bernoulli: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Multiply input x by a block–circulant operator whose full Fourier mask is given by fft_full.

    fft_full: shape (k_out, k_in, b), representing the full FFT for each block.
    x: shape (batch, d_in) or (d_in,)
    d_bernoulli: optional diagonal (of ±1) to decorrelate projections.

    This function works entirely in the frequency domain and is decorated with a custom JVP.
    """
    # Ensure x is 2D.
    if x.ndim == 1:
        x = x[None, :]
    bs, d_in = x.shape
    k_out, k_in, b = fft_full.shape

    if d_bernoulli is not None:
        x = x * d_bernoulli[None, :]

    # Zero-pad x to length k_in * b if needed.
    pad_len = k_in * b - d_in
    if pad_len > 0:
        x = jnp.pad(x, ((0, 0), (0, pad_len)))
    # Reshape into blocks: (bs, k_in, b)
    x_blocks = x.reshape(bs, k_in, b)

    # Compute FFT over the blocks.
    X_fft = jnp.fft.fft(x_blocks, axis=-1)
    # Multiply: for each output block row i, sum over j:
    # Y_fft[:, i, :] = sum_j [X_fft[:, j, :] * conj(fft_full[i, j, :])]
    Y_fft = jnp.sum(
        X_fft[:, None, :, :] * jnp.conjugate(fft_full)[None, :, :, :], axis=2
    )
    # Inverse FFT per block and reshape.
    Y = jnp.fft.ifft(Y_fft, axis=-1).real
    out = Y.reshape(bs, k_out * b)
    return out


@block_circulant_spectral_matmul_custom.defjvp
def block_circulant_spectral_matmul_custom_jvp(primals, tangents):
    fft_full, x, d_bernoulli = primals
    dfft, dx, dd = tangents  # dd is for d_bernoulli (we can ignore it for now)

    # Forward pass (as above).
    if x.ndim == 1:
        x = x[None, :]
    bs, d_in = x.shape
    k_out, k_in, b = fft_full.shape
    if d_bernoulli is not None:
        x_eff = x * d_bernoulli[None, :]
    else:
        x_eff = x
    pad_len = k_in * b - d_in
    if pad_len > 0:
        x_eff = jnp.pad(x_eff, ((0, 0), (0, pad_len)))
    x_blocks = x_eff.reshape(bs, k_in, b)
    FFT_x = jnp.fft.fft(x_blocks, axis=-1)
    FFT_full = fft_full  # Already in Fourier domain.
    Y_fft = jnp.sum(
        FFT_x[:, None, :, :] * jnp.conjugate(FFT_full)[None, :, :, :], axis=2
    )
    primal_out = jnp.fft.ifft(Y_fft, axis=-1).real.reshape(bs, k_out * b)

    # Compute directional derivatives.
    # For dx:
    if dx is None:
        dX_fft = 0.0
    else:
        dx_eff = dx * d_bernoulli[None, :] if d_bernoulli is not None else dx
        if pad_len > 0:
            dx_eff = jnp.pad(dx_eff, ((0, 0), (0, pad_len)))
        dx_blocks = dx_eff.reshape(bs, k_in, b)
        dX_fft = jnp.fft.fft(dx_blocks, axis=-1)

    # For dfft:
    if dfft is None:
        dFFT = 0.0
    else:
        dFFT = jnp.fft.fft(dfft, axis=-1)

    dY_fft = jnp.sum(
        dX_fft[:, None, :, :] * jnp.conjugate(FFT_full)[None, :, :, :]
        + FFT_x[:, None, :, :] * jnp.conjugate(dFFT)[None, :, :, :],
        axis=2,
    )
    tangent_out = jnp.fft.ifft(dY_fft, axis=-1).real.reshape(bs, k_out * b)
    if x.ndim == 1:
        return primal_out[0], tangent_out[0]
    return primal_out, tangent_out


# ----------------------------------------------------------------
# Frequency–domain block circulant layer with custom JVP.
class BlockCirculantProcess:
    """
    JVP version of BlockCirculantProcess: a frequency–domain block-circulant layer
    that is identical to BlockCirculantProcess except that it uses a custom JVP–decorated
    multiplication routine (block_circulant_spectral_matmul_custom) to speed up gradients.

    Each b x b block is parameterized by a half-spectrum with frequency-dependent prior scale
    and optional truncation. A custom prior on the Fourier coefficients can be specified via
    `prior_fn` (default: Gaussian). If `alpha` is None, a prior is placed on it using `alpha_prior`
    (default: Gamma(2.0, 1.0)).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int,
        alpha: float = 1.0,  # if set to None, a prior will be placed on alpha
        alpha_prior=None,  # custom prior for alpha (if alpha is None)
        K: int = None,
        name: str = "smooth_trunc_block_circ_spectral",
        prior_fn=None,  # callable to return a distribution given a scale
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.alpha = alpha
        # Use the provided alpha_prior if given, else default to Gamma(2.0, 1.0)
        self.alpha_prior = (
            alpha_prior if alpha_prior is not None else dist.Gamma(2.0, 1.0)
        )
        self.name = name

        self.k_in = (in_features + block_size - 1) // block_size
        self.k_out = (out_features + block_size - 1) // block_size
        self.b = block_size
        self.k_half = (block_size // 2) + 1
        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        self.prior_fn = (
            prior_fn
            if prior_fn is not None
            else (lambda scale: dist.Normal(0.0, scale))
        )
        self._last_block_fft = None  # will store the full block Fourier mask

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        if X.ndim == 1:
            X = X[None, :]
        bs, d_in = X.shape

        # If alpha is None, sample it using the provided alpha prior.
        alpha = (
            numpyro.sample(f"{self.name}_alpha", self.alpha_prior)
            if self.alpha is None
            else self.alpha
        )
        # Frequency-dependent scale.
        freq_idx = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_idx**alpha)
        active_indices = jnp.arange(self.K)
        n_active = self.K

        # Sample Fourier coefficients for active frequencies.
        active_scale = prior_std[active_indices]
        active_real = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(active_scale)
            .expand([self.k_out, self.k_in, n_active])
            .to_event(3),
        )
        active_imag = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(active_scale)
            .expand([self.k_out, self.k_in, n_active])
            .to_event(3),
        )

        # Build full coefficient arrays for each block.
        real_coeff = jnp.zeros((self.k_out, self.k_in, self.k_half))
        imag_coeff = jnp.zeros((self.k_out, self.k_in, self.k_half))
        real_coeff = real_coeff.at[..., active_indices].set(active_real)
        imag_coeff = imag_coeff.at[..., active_indices].set(active_imag)

        # Enforce DC component to be real.
        imag_coeff = imag_coeff.at[..., 0].set(0.0)
        if (self.b % 2 == 0) and (self.k_half > 1):
            imag_coeff = imag_coeff.at[..., -1].set(0.0)

        # Reconstruct the full block-level FFT for each (i,j) block.
        def reconstruct_fft(r_ij, i_ij):
            half_c = r_ij + 1j * i_ij
            if (self.b % 2 == 0) and (self.k_half > 1):
                nyquist = half_c[-1].real[None]
                block_fft = jnp.concatenate(
                    [half_c[:-1], nyquist, jnp.conjugate(half_c[1:-1])[::-1]]
                )
            else:
                block_fft = jnp.concatenate([half_c, jnp.conjugate(half_c[1:])[::-1]])
            return block_fft

        block_fft_full = jax.vmap(
            lambda Rrow, Irow: jax.vmap(reconstruct_fft)(Rrow, Irow),
            in_axes=(0, 0),
        )(real_coeff, imag_coeff)
        self._last_block_fft = jax.lax.stop_gradient(block_fft_full)

        # Zero-pad and reshape X into blocks is handled inside the custom JVP multiplication.
        # Use the custom JVP spectral multiplication.
        out = block_circulant_spectral_matmul_custom(self._last_block_fft, X)

        # Sample and add bias.
        b = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0, 1).expand([self.out_features]).to_event(1),
        )
        # If the padded output dimension is larger than out_features, slice it.
        if self.k_out * self.b > self.out_features:
            out = out[..., : self.out_features]
        return out + b[None, :]

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self._last_block_fft is None:
            raise ValueError(
                "No Fourier coefficients available. Call the layer on some input first."
            )
        return self._last_block_fft


# =============================================================================
# Custom JVP functions for spectral convolution operations
# =============================================================================

# -----------------------------
# 1D Spectral Convolution
# -----------------------------
@jax.custom_jvp
def spectral_conv1d_func(x: jnp.ndarray, weight_fft: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """
    x: shape (B, Cin, W)
    weight_fft: shape (Cout, Cin, rfft_length)
    bias: shape (Cout,)
    """
    B, Cin, W = x.shape
    rfft_length = weight_fft.shape[-1]
    # Determine FFT size from the half-spectrum length.
    fft_size = 2 * (rfft_length - 1) if rfft_length > 1 else 1
    if fft_size > W:
        pad_amount = fft_size - W
        x = jnp.pad(x, ((0, 0), (0, 0), (0, pad_amount)))
    X_fft = jnp.fft.rfft(x, n=fft_size, axis=-1)
    # Expand dimensions to align for multiplication.
    X_fft_exp = X_fft[:, None, :, :]         # (B, 1, Cin, rfft_length)
    W_fft_exp = weight_fft[None, :, :, :]      # (1, Cout, Cin, rfft_length)
    Y_fft = jnp.sum(X_fft_exp * W_fft_exp, axis=2)  # (B, Cout, rfft_length)
    y = jnp.fft.irfft(Y_fft, n=fft_size, axis=-1)
    y = y[..., :W]  # Crop back to original width.
    return y + bias[None, :, None]

@spectral_conv1d_func.defjvp
def spectral_conv1d_func_jvp(primals, tangents):
    x, weight_fft, bias = primals
    dx, dweight_fft, dbias = tangents
    y = spectral_conv1d_func(x, weight_fft, bias)
    dy = jnp.zeros_like(y)
    if dx is not None:
        dy += spectral_conv1d_func(dx, weight_fft, jnp.zeros_like(bias))
    if dweight_fft is not None:
        dy += spectral_conv1d_func(x, dweight_fft, jnp.zeros_like(bias))
    if dbias is not None:
        dy += dbias[None, :, None]
    return y, dy

# -----------------------------
# 2D Spectral Convolution
# -----------------------------
@jax.custom_jvp
def spectral_conv2d_func(x: jnp.ndarray, weight_fft: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """
    x: shape (B, Cin, H, W)
    weight_fft: shape (Cout, Cin, Hf, Wf_rfft)
    bias: shape (Cout,)
    """
    B, Cin, H, W = x.shape
    Cout, _, Hf, Wf_rfft = weight_fft.shape
    # For 2D rfft: determine the full width from Wf_rfft.
    W_full = 2 * (Wf_rfft - 1) if Wf_rfft > 1 else 1
    # If needed, pad x so that its spatial dims match the desired FFT size.
    pad_h = Hf - H if Hf > H else 0
    pad_w = W_full - W if W_full > W else 0
    if pad_h or pad_w:
        x = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
    X_fft = jnp.fft.rfft2(x, s=(Hf, W_full), axes=(2, 3))
    # Expand dimensions for broadcasting.
    X_fft_exp = X_fft[:, None, :, :, :]       # (B, 1, Cin, Hf, Wf_rfft)
    # Use the conjugate of weight_fft (as in your derivation).
    W_fft_exp = jnp.conjugate(weight_fft)[None, :, :, :, :]  # (1, Cout, Cin, Hf, Wf_rfft)
    Y_fft = jnp.sum(X_fft_exp * W_fft_exp, axis=2)  # (B, Cout, Hf, Wf_rfft)
    y = jnp.fft.irfft2(Y_fft, s=(Hf, W_full), axes=(2, 3))
    y = y[..., :H, :W]
    return y + bias[None, :, None, None]

@spectral_conv2d_func.defjvp
def spectral_conv2d_func_jvp(primals, tangents):
    x, weight_fft, bias = primals
    dx, dweight_fft, dbias = tangents
    y = spectral_conv2d_func(x, weight_fft, bias)
    dy = jnp.zeros_like(y)
    if dx is not None:
        dy += spectral_conv2d_func(dx, weight_fft, jnp.zeros_like(bias))
    if dweight_fft is not None:
        dy += spectral_conv2d_func(x, dweight_fft, jnp.zeros_like(bias))
    if dbias is not None:
        dy += dbias[None, :, None, None]
    return y, dy

# -----------------------------
# 2D Spectral Transposed Convolution
# -----------------------------
@jax.custom_jvp
def spectral_transposed_conv2d_func(x: jnp.ndarray, weight_fft: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """
    x: shape (B, Cout, H, W)
    weight_fft: shape (Cin, Cout, Hf, Wf_rfft)
      (Note: for the transposed layer, the roles of in/out channels are swapped relative to conv2d.)
    bias: shape (Cin,)
    """
    B, Cout, H, W = x.shape
    _, Cin, Hf, Wf_rfft = weight_fft.shape
    X_fft = jnp.fft.rfft2(x, axes=(2, 3))
    # Expand X_fft: (B, Cout, 1, Hf, Wf_rfft)
    X_fft_exp = X_fft[:, :, None, :, :]
    W_fft_exp = weight_fft[None, :, :, :, :]  # (1, Cin, Cout, Hf, Wf_rfft)
    Z_fft = jnp.sum(X_fft_exp * W_fft_exp, axis=1)  # Sum over the original Cout dimension → (B, Cin, Hf, Wf_rfft)
    W_full = 2 * (Wf_rfft - 1) if Wf_rfft > 1 else 1
    y = jnp.fft.irfft2(Z_fft, s=(Hf, W_full), axes=(2, 3))
    y = y[..., :H, :W]
    return y + (bias[None, :, None, None] if bias is not None else 0.0)

@spectral_transposed_conv2d_func.defjvp
def spectral_transposed_conv2d_func_jvp(primals, tangents):
    x, weight_fft, bias = primals
    dx, dweight_fft, dbias = tangents
    y = spectral_transposed_conv2d_func(x, weight_fft, bias)
    dy = jnp.zeros_like(y)
    if dx is not None:
        dy += spectral_transposed_conv2d_func(dx, weight_fft, jnp.zeros_like(bias))
    if dweight_fft is not None:
        dy += spectral_transposed_conv2d_func(x, dweight_fft, jnp.zeros_like(bias))
    if dbias is not None:
        dy += dbias[None, :, None, None]
    return y, dy

# =============================================================================
# NumPyro layer classes that use the above custom functions
# =============================================================================

# -----------------------------
# SpectralConv1d in NumPyro
# -----------------------------
class SpectralConv1d:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fft_size: int,
        name: str = "spectral_conv1d",
        alpha: float = 1.0,
        K: int = None,
        prior_fn=None,
    ):
        """
        Parameters:
          in_channels, out_channels: number of input/output channels.
          fft_size: full length for the FFT (determines the half-spectrum length).
          alpha: exponent for frequency–dependent decay.
          K: number of active frequencies (if None, uses full half-spectrum).
          prior_fn: callable that given a scale returns a distribution (default: Normal(0, scale)).
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fft_size = fft_size
        self.rfft_length = fft_size // 2 + 1 if fft_size > 1 else 1
        self.alpha = alpha
        self.K = K if (K is not None and K <= self.rfft_length) else self.rfft_length
        self.name = name
        self.prior_fn = prior_fn if prior_fn is not None else (lambda scale: dist.Normal(0.0, scale))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: shape (B, in_channels, W)
        freq_idx = jnp.arange(self.rfft_length, dtype=jnp.float32)
        prior_std = 1.0 / jnp.sqrt(1.0 + (freq_idx ** self.alpha))
        active_idx = jnp.arange(self.K)
        active_shape = (self.out_channels, self.in_channels, self.K)
        active_real = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(prior_std[active_idx]).expand(active_shape).to_event(3),
        )
        active_imag = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(prior_std[active_idx]).expand(active_shape).to_event(3),
        )
        full_real = jnp.zeros((self.out_channels, self.in_channels, self.rfft_length))
        full_imag = jnp.zeros((self.out_channels, self.in_channels, self.rfft_length))
        full_real = full_real.at[..., active_idx].set(active_real)
        full_imag = full_imag.at[..., active_idx].set(active_imag)
        # Enforce DC frequency to be real.
        full_imag = full_imag.at[..., 0].set(0.0)
        if self.fft_size % 2 == 0 and self.rfft_length > 1:
            full_imag = full_imag.at[..., -1].set(0.0)
        weight_fft = full_real + 1j * full_imag
        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0.0, 1.0).expand([self.out_channels]).to_event(1),
        )
        return spectral_conv1d_func(x, weight_fft, bias)

# -----------------------------
# SpectralConv2d in NumPyro
# -----------------------------
class SpectralConv2d:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fft_size: tuple,
        name: str = "spectral_conv2d",
        alpha: float = 1.0,
        K: tuple = None,
        prior_fn=None,
    ):
        """
        Parameters:
          fft_size: tuple (H_fft, W_fft) specifying the desired FFT size.
          K: tuple (K_h, K_w) for the number of active frequencies (default: full spectrum).
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fft_size = fft_size
        self.H_fft, self.W_fft = fft_size
        self.Wf_rfft = self.W_fft // 2 + 1 if self.W_fft > 1 else 1
        self.Hf = self.H_fft
        self.alpha = alpha
        if K is None:
            self.K_h, self.K_w = self.Hf, self.Wf_rfft
        else:
            self.K_h, self.K_w = K
        self.name = name
        self.prior_fn = prior_fn if prior_fn is not None else (lambda scale: dist.Normal(0.0, scale))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: shape (B, in_channels, H, W)
        freq_idx_h = jnp.arange(self.Hf, dtype=jnp.float32)
        freq_idx_w = jnp.arange(self.Wf_rfft, dtype=jnp.float32)
        decay_h = 1.0 / jnp.sqrt(1.0 + (freq_idx_h ** self.alpha))
        decay_w = 1.0 / jnp.sqrt(1.0 + (freq_idx_w ** self.alpha))
        prior_std = jnp.outer(decay_h, decay_w)  # shape (Hf, Wf_rfft)
        active_idx_h = jnp.arange(self.K_h)
        active_idx_w = jnp.arange(self.K_w)
        active_shape = (self.out_channels, self.in_channels, self.K_h, self.K_w)
        active_real = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(prior_std[active_idx_h[:, None], active_idx_w[None, :]])
            .expand(active_shape)
            .to_event(4),
        )
        active_imag = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(prior_std[active_idx_h[:, None], active_idx_w[None, :]])
            .expand(active_shape)
            .to_event(4),
        )
        full_real = jnp.zeros((self.out_channels, self.in_channels, self.Hf, self.Wf_rfft))
        full_imag = jnp.zeros((self.out_channels, self.in_channels, self.Hf, self.Wf_rfft))
        # Fill the top–left active block.
        full_real = full_real.at[:, :, active_idx_h[:, None], active_idx_w].set(active_real)
        full_imag = full_imag.at[:, :, active_idx_h[:, None], active_idx_w].set(active_imag)
        # Enforce the DC component (top–left element) to be real.
        full_imag = full_imag.at[:, :, 0, 0].set(0.0)
        if self.W_fft % 2 == 0 and self.Wf_rfft > 1:
            full_imag = full_imag.at[:, :, :, -1].set(0.0)
        weight_fft = full_real + 1j * full_imag
        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0.0, 1.0).expand([self.out_channels]).to_event(1),
        )
        return spectral_conv2d_func(x, weight_fft, bias)

# -----------------------------
# SpectralTransposed2d in NumPyro
# -----------------------------
class SpectralTransposed2d:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fft_size: tuple,
        name: str = "spectral_transposed_conv2d",
        alpha: float = 1.0,
        K: tuple = None,
        prior_fn=None,
    ):
        """
        Here the layer maps an input of shape (B, out_channels, H, W) to an output of shape (B, in_channels, H, W).
        Note: the roles of in_channels and out_channels are swapped relative to SpectralConv2d.
        """
        self.in_channels = in_channels   # output channels of the transposed layer
        self.out_channels = out_channels # input channels of the transposed layer
        self.fft_size = fft_size
        self.H_fft, self.W_fft = fft_size
        self.Wf_rfft = self.W_fft // 2 + 1 if self.W_fft > 1 else 1
        self.Hf = self.H_fft
        self.alpha = alpha
        if K is None:
            self.K_h, self.K_w = self.Hf, self.Wf_rfft
        else:
            self.K_h, self.K_w = K
        self.name = name
        self.prior_fn = prior_fn if prior_fn is not None else (lambda scale: dist.Normal(0.0, scale))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: shape (B, out_channels, H, W)
        freq_idx_h = jnp.arange(self.Hf, dtype=jnp.float32)
        freq_idx_w = jnp.arange(self.Wf_rfft, dtype=jnp.float32)
        decay_h = 1.0 / jnp.sqrt(1.0 + (freq_idx_h ** self.alpha))
        decay_w = 1.0 / jnp.sqrt(1.0 + (freq_idx_w ** self.alpha))
        prior_std = jnp.outer(decay_h, decay_w)
        active_idx_h = jnp.arange(self.K_h)
        active_idx_w = jnp.arange(self.K_w)
        active_shape = (self.in_channels, self.out_channels, self.K_h, self.K_w)
        active_real = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(prior_std[active_idx_h[:, None], active_idx_w[None, :]])
            .expand(active_shape)
            .to_event(4),
        )
        active_imag = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(prior_std[active_idx_h[:, None], active_idx_w[None, :]])
            .expand(active_shape)
            .to_event(4),
        )
        full_real = jnp.zeros((self.in_channels, self.out_channels, self.Hf, self.Wf_rfft))
        full_imag = jnp.zeros((self.in_channels, self.out_channels, self.Hf, self.Wf_rfft))
        full_real = full_real.at[:, :, active_idx_h[:, None], active_idx_w].set(active_real)
        full_imag = full_imag.at[:, :, active_idx_h[:, None], active_idx_w].set(active_imag)
        full_imag = full_imag.at[:, :, 0, 0].set(0.0)
        if self.W_fft % 2 == 0 and self.Wf_rfft > 1:
            full_imag = full_imag.at[:, :, :, -1].set(0.0)
        weight_fft = full_real + 1j * full_imag
        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0.0, 1.0).expand([self.in_channels]).to_event(1),
        )
        return spectral_transposed_conv2d_func(x, weight_fft, bias)