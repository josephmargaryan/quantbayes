from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import transforms

__all__ = [
    "Circulant",
    "BlockCirculant",
    "SpectralCirculantLayer",
    "BlockCirculantProcess",
    "SpectralCirculantLayer2d",
    "PatchWiseSpectralMixture1D",
    "PatchWiseSpectralMixture2D",
    "AdaptiveSpectralCirculantLayer",
    "AdaptiveSpectralCirculantLayer2d",
]


@jax.custom_jvp
def fft_matmul_custom(first_row: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """
    Performs circulant matrix multiplication via FFT.
    Given the first row of a circulant matrix and an input X,
    computes:
        result = ifft( fft(first_row) * fft(X) ).real
    """
    first_row_fft = jnp.fft.fft(first_row, axis=-1)
    X_fft = jnp.fft.fft(X, axis=-1)
    result_fft = first_row_fft[None, :] * X_fft
    result = jnp.fft.ifft(result_fft, axis=-1).real
    return result


@fft_matmul_custom.defjvp
def fft_matmul_custom_jvp(primals, tangents):
    first_row, X = primals
    d_first_row, dX = tangents

    first_row_fft = jnp.fft.fft(first_row, axis=-1)
    X_fft = jnp.fft.fft(X, axis=-1)
    primal_out = jnp.fft.ifft(first_row_fft[None, :] * X_fft, axis=-1).real

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
    if x.ndim == 1:
        x = x[None, :]
    batch_size, d_in = x.shape
    k_out, k_in, b = W.shape

    if d_bernoulli is not None:
        x = x * d_bernoulli[None, :]
    pad_len = k_in * b - d_in
    if pad_len > 0:
        x = jnp.pad(x, ((0, 0), (0, pad_len)))
    x_blocks = x.reshape(batch_size, k_in, b)

    W_fft = jnp.fft.fft(W, axis=-1)
    X_fft = jnp.fft.fft(x_blocks, axis=-1)

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
    ) = tangents

    if x.ndim == 1:
        x = x[None, :]
    batch_size, d_in = x.shape
    k_out, k_in, b = W.shape

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

    if dW is not None:
        W_tangent_fft = jnp.fft.fft(dW, axis=-1)
    else:
        W_tangent_fft = 0.0

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
        self.padded_dim = padded_dim if padded_dim is not None else in_features
        self.name = name
        self.first_row_prior_fn = first_row_prior_fn
        self.bias_prior_fn = bias_prior_fn

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
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
        W = numpyro.sample(
            f"{self.name}_W",
            self.W_prior_fn([self.k_out, self.k_in, self.block_size]),
        )

        if self.use_diag:
            d_bernoulli = numpyro.sample(
                f"{self.name}_D",
                self.diag_prior([self.in_features]),
            )
        else:
            d_bernoulli = None

        out = block_circulant_matmul_custom(W, X, d_bernoulli)

        b = numpyro.sample(
            f"{self.name}_bias",
            self.bias_prior_fn([self.out_features]),
        )
        out = out + b[None, :]

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


class SpectralCirculantLayer:
    def __init__(
        self,
        in_features: int,
        padded_dim: int = None,
        alpha: float = None,
        alpha_prior=dist.HalfNormal(1),
        K: int = None,
        name: str = "spectral_circ_jvp",
        prior_fn=None,
    ):
        """
        :param in_features: Input dimension.
        :param padded_dim: If provided, pad/truncate inputs to this dimension.
        :param alpha: Fixed value for the decay exponent; if None, a hyperprior is used.
        :param alpha_prior: Prior distribution for alpha if it is not fixed.
        :param K: Number of active frequencies to keep; if None, use full half-spectrum.
        :param name: Base name for NumPyro sample sites.
        :param prior_fn: Function mapping a scale to a distribution (default: Normal(0, scale)).
        """
        self.in_features = in_features
        self.padded_dim = padded_dim if padded_dim is not None else in_features
        self.alpha = alpha
        self.alpha_prior = alpha_prior
        self.name = name

        # Length of half spectrum including DC (and Nyquist when applicable)
        self.k_half = (self.padded_dim // 2) + 1
        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        self.prior_fn = (
            prior_fn
            if prior_fn is not None
            else (lambda scale: dist.Normal(0.0, scale))
        )
        self._last_fft_full = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.alpha is None:
            alpha = numpyro.sample(f"{self.name}_alpha", self.alpha_prior)
        else:
            alpha = self.alpha

        freq_idx = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_idx**alpha)

        active_idx = jnp.arange(self.K)
        active_real = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(prior_std[active_idx]).expand([self.K]).to_event(1),
        )
        active_imag = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(prior_std[active_idx]).expand([self.K]).to_event(1),
        )

        full_real = jnp.zeros((self.k_half,))
        full_imag = jnp.zeros((self.k_half,))
        full_real = full_real.at[active_idx].set(active_real)
        full_imag = full_imag.at[active_idx].set(active_imag)

        full_imag = full_imag.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            full_imag = full_imag.at[-1].set(0.0)

        half_complex = full_real + 1j * full_imag

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

        bias = numpyro.sample(
            f"{self.name}_bias_spectral",
            dist.Normal(0.0, 1.0).expand([self.padded_dim]).to_event(1),
        )
        out = spectral_circulant_matmul(x, fft_full) + bias
        return out

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self._last_fft_full is None:
            raise ValueError("No Fourier coefficients available; call the layer first.")
        return self._last_fft_full

    # compute the theoretical
    # covariance kernel from the prior PSD.
    def compute_covariance_kernel(self):
        """
        Compute the theoretical covariance kernel using the inverse FFT of the
        theoretical PSD. Here, the PSD is defined as s(f)^2, with s(f)=1/sqrt(1+f^alpha).
        This kernel represents the autocovariance function of the GP.
        """
        if self.alpha is None:
            raise ValueError("Alpha must be set to compute the covariance kernel.")
        freq_idx = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_idx**self.alpha)
        theoretical_psd = prior_std**2

        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            nyquist = theoretical_psd[-1][None]
            full_psd = jnp.concatenate(
                [
                    theoretical_psd[:-1],
                    nyquist,
                    jnp.conjugate(theoretical_psd[1:-1])[::-1],
                ]
            )
        else:
            full_psd = jnp.concatenate(
                [theoretical_psd, jnp.conjugate(theoretical_psd[1:])[::-1]]
            )

        kernel = jnp.fft.ifft(full_psd).real
        # For a GP over inputs 0,1,...,N-1, the covariance matrix is Toeplitz:
        # K[i,j] = kernel[|i-j|]
        return kernel


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
    if x.ndim == 1:
        x = x[None, :]
    bs, d_in = x.shape
    k_out, k_in, b = fft_full.shape

    if d_bernoulli is not None:
        x = x * d_bernoulli[None, :]

    pad_len = k_in * b - d_in
    if pad_len > 0:
        x = jnp.pad(x, ((0, 0), (0, pad_len)))
    x_blocks = x.reshape(bs, k_in, b)

    X_fft = jnp.fft.fft(x_blocks, axis=-1)
    Y_fft = jnp.sum(
        X_fft[:, None, :, :] * jnp.conjugate(fft_full)[None, :, :, :], axis=2
    )
    Y = jnp.fft.ifft(Y_fft, axis=-1).real
    out = Y.reshape(bs, k_out * b)
    return out


@block_circulant_spectral_matmul_custom.defjvp
def block_circulant_spectral_matmul_custom_jvp(primals, tangents):
    fft_full, x, d_bernoulli = primals
    dfft, dx, dd = tangents

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
    FFT_full = fft_full
    Y_fft = jnp.sum(
        FFT_x[:, None, :, :] * jnp.conjugate(FFT_full)[None, :, :, :], axis=2
    )
    primal_out = jnp.fft.ifft(Y_fft, axis=-1).real.reshape(bs, k_out * b)

    if dx is None:
        dX_fft = 0.0
    else:
        dx_eff = dx * d_bernoulli[None, :] if d_bernoulli is not None else dx
        if pad_len > 0:
            dx_eff = jnp.pad(dx_eff, ((0, 0), (0, pad_len)))
        dx_blocks = dx_eff.reshape(bs, k_in, b)
        dX_fft = jnp.fft.fft(dx_blocks, axis=-1)

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
        alpha: float = 1.0,
        alpha_prior=None,
        K: int = None,
        name: str = "smooth_trunc_block_circ_spectral",
        prior_fn=None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.alpha = alpha
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
        self._last_block_fft = None

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        if X.ndim == 1:
            X = X[None, :]
        alpha = (
            numpyro.sample(f"{self.name}_alpha", self.alpha_prior)
            if self.alpha is None
            else self.alpha
        )
        freq_idx = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_idx**alpha)
        active_indices = jnp.arange(self.K)
        n_active = self.K

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

        real_coeff = jnp.zeros((self.k_out, self.k_in, self.k_half))
        imag_coeff = jnp.zeros((self.k_out, self.k_in, self.k_half))
        real_coeff = real_coeff.at[..., active_indices].set(active_real)
        imag_coeff = imag_coeff.at[..., active_indices].set(active_imag)

        imag_coeff = imag_coeff.at[..., 0].set(0.0)
        if (self.b % 2 == 0) and (self.k_half > 1):
            imag_coeff = imag_coeff.at[..., -1].set(0.0)

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

        out = block_circulant_spectral_matmul_custom(self._last_block_fft, X)

        b = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0, 1).expand([self.out_features]).to_event(1),
        )
        if self.k_out * self.b > self.out_features:
            out = out[..., : self.out_features]
        return out + b[None, :]

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self._last_block_fft is None:
            raise ValueError(
                "No Fourier coefficients available. Call the layer on some input first."
            )
        return self._last_block_fft


def _enforce_hermitian(fft2d: jnp.ndarray) -> jnp.ndarray:
    """Return the closest Hermitian‑symmetric tensor so that the spatial iFFT
    is strictly real‑valued.  Also forces DC / Nyquist lines to be real.
    """
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
    """Circulant 2‑D convolution using pre‑computed Fourier kernel coefficients.

    Parameters
    ----------
    x           : (batch?, H_in, W_in) real inputs.
    fft_kernel  : (H_pad,  W_pad)     complex Fourier coefficients (Hermitian).

    Returns
    -------
    y           : Same leading batch dims, spatial dims (H_pad, W_pad).
                  If you want *valid* convolution simply slice afterwards.
    """
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
        y, dy = y[0], dy[0]
    return y, dy


class SpectralCirculantLayer2d:
    """Spectral convolution layer whose prior PSD is
            S(f) ∝ (1 + ||f||^α)^‑1/2   with a learnable or random‑draw α.
    Works well on smooth data; serves as the *baseline* in our ablations.
    """

    def __init__(
        self,
        H_in: int,
        W_in: int,
        H_pad: Optional[int] = None,
        W_pad: Optional[int] = None,
        alpha: Optional[float] = None,
        alpha_prior: dist.Distribution = dist.HalfNormal(1.0),
        name: str = "spectral_pw",
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
    ):
        self.H_in, self.W_in = H_in, W_in
        self.H_pad = H_pad or H_in
        self.W_pad = W_pad or W_in
        self.alpha = alpha
        self.alpha_prior = alpha_prior
        self.name = name
        self.prior_fn = prior_fn or (lambda scale: dist.Normal(0.0, scale))
        self._fft_kernel = None

    # ---------------------------------------------------------------------
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        alpha = (
            self.alpha
            if self.alpha is not None
            else numpyro.sample(f"{self.name}_alpha", self.alpha_prior)
        )

        u = jnp.fft.fftfreq(self.H_pad) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)

        std = 1.0 / jnp.sqrt(1.0 + R**alpha)

        real = numpyro.sample(f"{self.name}_real", self.prior_fn(std).to_event(2))
        imag = numpyro.sample(f"{self.name}_imag", self.prior_fn(std).to_event(2))

        fft2d = _enforce_hermitian(real + 1j * imag)
        self._fft_kernel = jax.lax.stop_gradient(fft2d)

        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0.0, 1.0).expand([self.H_pad, self.W_pad]).to_event(2),
        )

        return spectral_circulant_conv2d(x, fft2d) + bias

    def get_fft_kernel(self):
        if self._fft_kernel is None:
            raise RuntimeError("Layer has not been called yet.")
        return self._fft_kernel


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


class AdaptiveSpectralCirculantLayer(SpectralCirculantLayer):
    def __init__(
        self,
        in_features,
        padded_dim=None,
        alpha_global=1.0,
        alpha_prior=dist.HalfNormal(1.0),
        K=None,
        name="ns_spectral_circ",
        prior_fn=None,
    ):
        super().__init__(
            in_features,
            padded_dim,
            alpha=None,
            alpha_prior=alpha_prior,
            K=K,
            name=name,
            prior_fn=prior_fn,
        )
        self.alpha_global = alpha_global

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        delta_alpha = numpyro.sample(
            f"{self.name}_delta_alpha",
            self.alpha_prior.expand([self.k_half]).to_event(1),
        )
        alpha_vec = jnp.clip(self.alpha_global + delta_alpha, 0.1, 5.0)

        freq_idx = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_idx**alpha_vec)

        active_idx = jnp.arange(self.K)
        real_hp = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(prior_std[active_idx]).expand([self.K]).to_event(1),
        )
        imag_hp = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(prior_std[active_idx]).expand([self.K]).to_event(1),
        )

        full_real = jnp.zeros((self.k_half,))
        full_real = full_real.at[active_idx].set(real_hp)
        full_imag = jnp.zeros((self.k_half,))
        full_imag = full_imag.at[active_idx].set(imag_hp)

        full_imag = full_imag.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            full_imag = full_imag.at[-1].set(0.0)

        half_complex = full_real + 1j * full_imag
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            nyq = half_complex[-1].real[None]
            fft_full = jnp.concatenate(
                [half_complex[:-1], nyq, jnp.conj(half_complex[1:-1])[::-1]]
            )
        else:
            fft_full = jnp.concatenate([half_complex, jnp.conj(half_complex[1:])[::-1]])

        self._last_fft_full = jax.lax.stop_gradient(fft_full)
        bias = numpyro.sample(
            f"{self.name}_bias_spectral",
            dist.Normal(0.0, 1.0).expand([self.padded_dim]).to_event(1),
        )
        return spectral_circulant_matmul(x, fft_full) + bias


class AdaptiveSpectralCirculantLayer2d(SpectralCirculantLayer2d):
    def __init__(
        self,
        H_in,
        W_in,
        H_pad=None,
        W_pad=None,
        alpha_global=1.0,
        alpha_coarse_shape=(8, 8),
        alpha_prior=dist.HalfNormal(1.0),
        name="ns_spec_pw",
        prior_fn=None,
    ):
        super().__init__(
            H_in,
            W_in,
            H_pad,
            W_pad,
            alpha=None,
            alpha_prior=alpha_prior,
            name=name,
            prior_fn=prior_fn,
        )
        self.alpha_global = alpha_global
        self.alpha_coarse_shape = alpha_coarse_shape

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        delta_alpha_coarse = numpyro.sample(
            f"{self.name}_delta_alpha_map",
            self.alpha_prior.expand(self.alpha_coarse_shape).to_event(2),
        )
        delta_alpha_full = upsample_bilinear(delta_alpha_coarse, self.H_pad, self.W_pad)
        alpha_full = jnp.clip(self.alpha_global + delta_alpha_full, 0.1, 5.0)

        u = jnp.fft.fftfreq(self.H_pad) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        std = 1.0 / jnp.sqrt(1.0 + R**alpha_full)

        real = numpyro.sample(f"{self.name}_real", self.prior_fn(std).to_event(2))
        imag = numpyro.sample(f"{self.name}_imag", self.prior_fn(std).to_event(2))
        fft2d = _enforce_hermitian(real + 1j * imag)

        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0.0, 1.0).expand([self.H_pad, self.W_pad]).to_event(2),
        )
        return spectral_circulant_conv2d(x, fft2d) + bias


@jax.custom_jvp
def _specmix_patch_1d(x, S):
    """y = IFFT( FFT(x) * S ) for one patch; x:(B,L), S:(L,) real."""
    Xf = jnp.fft.fft(x, axis=-1)
    return jnp.fft.ifft(Xf * S[None, :], axis=-1).real


@_specmix_patch_1d.defjvp
def _specmix_patch_1d_jvp(primals, tangents):
    x, S = primals
    dx, dS = tangents
    Xf = jnp.fft.fft(x, axis=-1)
    dXf = jnp.fft.fft(dx, axis=-1) if dx is not None else 0.0
    Yf = Xf * S[None, :]
    dYf = dXf * S[None, :] + Xf * dS[None, :]
    return jnp.fft.ifft(Yf, axis=-1).real, jnp.fft.ifft(dYf, axis=-1).real


class PatchWiseSpectralMixture1D:
    """
    Partition length L into P patches of length l.
    Each patch uses a Q‑component spectral mixture PSD.
    """

    def __init__(self, L, patch_len, Q=3, name="psm1d", jitter=1e-6):
        assert L % patch_len == 0, "L must be divisible by patch_len"
        self.L, self.l = L, patch_len
        self.P = L // patch_len
        self.Q = Q
        self.name = name
        self.jitter = jitter
        self.f = jnp.fft.fftfreq(patch_len)

    def __call__(self, x):
        """
        x : (B, L)
        returns same shape
        """
        B, L = x.shape
        assert L == self.L
        logits = numpyro.sample(
            f"{self.name}_logits", dist.Normal(0, 1).expand([self.P, self.Q])
        )
        w = jax.nn.softmax(logits, axis=-1)  # (P,Q)
        mu = numpyro.sample(
            f"{self.name}_mu", dist.Normal(0, 0.3).expand([self.P, self.Q])
        )
        sig = numpyro.sample(
            f"{self.name}_sigma", dist.LogNormal(0, 0.3).expand([self.P, self.Q])
        )

        def make_S(p):
            w_p, mu_p, sig_p = w[p], mu[p], sig[p]
            g1 = jnp.exp(
                -0.5 * ((self.f[None, :] - mu_p[:, None]) / sig_p[:, None]) ** 2
            )
            g2 = jnp.exp(
                -0.5 * ((self.f[None, :] + mu_p[:, None]) / sig_p[:, None]) ** 2
            )
            return (w_p[:, None] * (g1 + g2)).sum(0) + self.jitter

        S = jax.vmap(make_S)(jnp.arange(self.P))
        x_split = x.reshape(B, self.P, self.l)
        y_split = jax.vmap(
            lambda xp, Sp: _specmix_patch_1d(xp, Sp), in_axes=(1, 0), out_axes=1
        )(x_split, S)
        bias = numpyro.sample(
            f"{self.name}_bias", dist.Normal(0, 1).expand([self.P, self.l]).to_event(2)
        )
        return (y_split + bias).reshape(B, L)


@jax.custom_jvp
def _specmix_patch_2d(x, S):
    """x:(B,h,w), S:(h,w) real PSD."""
    Xf = jnp.fft.fftn(x, axes=(-2, -1))
    return jnp.fft.ifftn(Xf * S[None, :, :], axes=(-2, -1)).real


@_specmix_patch_2d.defjvp
def _specmix_patch_2d_jvp(primals, tangents):
    x, S = primals
    dx, dS = tangents
    Xf = jnp.fft.fftn(x, axes=(-2, -1))
    dXf = jnp.fft.fftn(dx, axes=(-2, -1)) if dx is not None else 0.0
    Yf = Xf * S[None, :, :]
    dYf = dXf * S[None, :, :] + Xf * dS[None, :, :]
    return jnp.fft.ifftn(Yf, axes=(-2, -1)).real, jnp.fft.ifftn(dYf, axes=(-2, -1)).real


class PatchWiseSpectralMixture2D:
    """
    Non-overlapping h×w patches; Q-component spectral mixture PSD per patch.
    """

    def __init__(
        self,
        H: int,
        W: int,
        patch_h: int,
        patch_w: int,
        Q: int = 3,
        name: str = "psm2d",
        jitter: float = 1e-6,
    ):
        assert H % patch_h == 0 and W % patch_w == 0
        self.H, self.W = H, W
        self.ph, self.pw = patch_h, patch_w
        self.nh, self.nw = H // patch_h, W // patch_w
        self.P = self.nh * self.nw
        self.Q, self.name, self.jitter = Q, name, jitter

        self.fy = jnp.fft.fftfreq(self.ph)[:, None]
        self.fx = jnp.fft.fftfreq(self.pw)[None, :]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x : (B, H, W)
        returns (B, H, W)
        """
        B, H, W = x.shape
        assert (H, W) == (self.H, self.W)

        logits = numpyro.sample(
            f"{self.name}_logits", dist.Normal(0, 1).expand([self.P, self.Q])
        )
        w = jax.nn.softmax(logits, axis=-1)
        mu = numpyro.sample(
            f"{self.name}_mu", dist.Normal(0, 0.3).expand([self.P, self.Q, 2])
        )
        sig = numpyro.sample(
            f"{self.name}_sigma", dist.LogNormal(0, 0.3).expand([self.P, self.Q, 2])
        )

        def make_S(params):
            w_p, mu_p, sig_p = params

            mu_y = mu_p[:, 0][..., None, None]
            mu_x = mu_p[:, 1][..., None, None]
            sig_y = sig_p[:, 0][..., None, None]
            sig_x = sig_p[:, 1][..., None, None]

            fy = self.fy[None, :, :]
            fx = self.fx[None, :, :]

            g1 = jnp.exp(
                -0.5 * (((fy - mu_y) / sig_y) ** 2 + ((fx - mu_x) / sig_x) ** 2)
            )
            g2 = jnp.exp(
                -0.5 * (((fy + mu_y) / sig_y) ** 2 + ((fx + mu_x) / sig_x) ** 2)
            )

            return (w_p[:, None, None] * (g1 + g2)).sum(0) + self.jitter

        S = jax.vmap(make_S)((w, mu, sig))

        x_patches = x.reshape(B, self.nh, self.ph, self.nw, self.pw)
        x_patches = x_patches.transpose(1, 3, 0, 2, 4)

        out_rows = []
        p = 0
        for i in range(self.nh):
            row_patches = []
            for j in range(self.nw):
                ypj = _specmix_patch_2d(x_patches[i, j], S[p])
                bias = numpyro.sample(
                    f"{self.name}_bias_{p}",
                    dist.Normal(0, 1).expand([self.ph, self.pw]).to_event(2),
                )
                row_patches.append(ypj + bias)
                p += 1
            out_rows.append(jnp.concatenate(row_patches, axis=-1))
        return jnp.concatenate(out_rows, axis=-2)
