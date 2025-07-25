from typing import Callable, Optional, Tuple, Sequence

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import transforms

__all__ = [
    "Circulant",
    "Circulant2d",
    "BlockCirculant",
    "BlockCirculantProcess",
    "SpectralCirculantLayer",
    "SpectralCirculantLayer2d",
    "AdaptiveSpectralCirculantLayer",
    "AdaptiveSpectralCirculantLayer2d",
    "SpectralDense",
    "SpectralConv",
    "SpectralConv2d",
    "AdaptiveSpectralConv2d",
    "GraphSpectralDense",
    "AdaptiveGraphSpectralDense",
    "GraphSpectralConv2d",
    "AdaptiveGraphSpectralConv2d",
]


@jax.custom_jvp
def fft_matmul_custom(first_row: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """
    Performs circulant matrix multiplication via real‐FFT.
    Given the first row of a circulant matrix (shape [n]) and input X
    (shape [..., n]), computes
        result = irfft( rfft(first_row)[None, ...] * rfft(X), n=n )
    returning a real array of shape [..., n].
    """
    n = first_row.shape[-1]
    fr = jnp.fft.rfft(first_row, axis=-1)
    Xf = jnp.fft.rfft(X, axis=-1)
    Y = jnp.fft.irfft(fr[None, ...] * Xf, n=n, axis=-1)
    return Y


@fft_matmul_custom.defjvp
def fft_matmul_custom_jvp(primals, tangents):
    first_row, X = primals
    d_first_row, dX = tangents
    n = first_row.shape[-1]
    fr = jnp.fft.rfft(first_row, axis=-1)
    Xf = jnp.fft.rfft(X, axis=-1)
    primal_out = jnp.fft.irfft(fr[None, ...] * Xf, n=n, axis=-1)
    df_part = jnp.zeros_like(primal_out)
    if d_first_row is not None:
        dfr = jnp.fft.rfft(d_first_row, axis=-1)
        df_part = jnp.fft.irfft(dfr[None, ...] * Xf, n=n, axis=-1)

    dx_part = jnp.zeros_like(primal_out)
    if dX is not None:
        dXf = jnp.fft.rfft(dX, axis=-1)
        dx_part = jnp.fft.irfft(fr[None, ...] * dXf, n=n, axis=-1)

    return primal_out, df_part + dx_part


class Circulant:
    """
    FFT‐based probabilistic circulant layer for NumPyro.

    Samples `first_row` and `bias` from priors, pads input if needed,
    and applies `fft_matmul_custom` + bias.

    Args:
        in_features:    original input dimension
        padded_dim:     FFT dimension (defaults to in_features)
        name:           name prefix for NumPyro sites
        first_row_prior_fn: callable(shape)→dist, prior for first_row
        bias_prior_fn:      callable(shape)→dist, prior for bias
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
        bias = numpyro.sample(
            f"{self.name}_bias", self.bias_prior_fn([self.padded_dim])
        )

        hidden = fft_matmul_custom(first_row, X) + bias[None, ...]
        return hidden


@jax.custom_jvp
def fft_conv2d_real(kernel: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Circular 2-D conv with real-space kernel via rFFT.
      kernel : (C_out, C_in, H_pad, W_pad)
      x      : (..., C_in,  H_in,  W_in)
    """
    C_out, C_in, H_pad, W_pad = kernel.shape
    single = x.ndim == 3
    if single:
        x = x[None, ...]
    *lead, C_in_x, H_in, W_in = x.shape
    if C_in_x != C_in:
        raise ValueError(f"Cin mismatch (kernel={C_in}, x={C_in_x})")

    pad_h, pad_w = max(0, H_pad - H_in), max(0, W_pad - W_in)
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[..., :H_pad, :W_pad]

    Kf = jnp.fft.rfftn(kernel, axes=(-2, -1))
    Xf = jnp.fft.rfftn(x_pad, axes=(-2, -1))
    Yf = jnp.einsum("oihw,bihw->bohw", Kf, Xf)
    y = jnp.fft.irfftn(Yf, s=(H_pad, W_pad), axes=(-2, -1))
    return y[0] if single else y


@fft_conv2d_real.defjvp
def _fft_conv2d_real_jvp(primals, tangents):
    kernel, x = primals
    dk, dx = tangents
    single = x.ndim == 3
    if single:
        x = x[None, ...]
        dx = None if dx is None else dx[None, ...]
    pad_h = max(0, kernel.shape[-2] - x.shape[-2])
    pad_w = max(0, kernel.shape[-1] - x.shape[-1])
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[
        ..., : kernel.shape[-2], : kernel.shape[-1]
    ]
    dx_pad = (
        None
        if dx is None
        else jnp.pad(dx, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[
            ..., : kernel.shape[-2], : kernel.shape[-1]
        ]
    )
    Kf = jnp.fft.rfftn(kernel, axes=(-2, -1))
    Xf = jnp.fft.rfftn(x_pad, axes=(-2, -1))
    Yf = jnp.einsum("oihw,bihw->bohw", Kf, Xf)
    y = jnp.fft.irfftn(Yf, s=(kernel.shape[-2], kernel.shape[-1]), axes=(-2, -1))
    dKf = jnp.fft.rfftn(dk, axes=(-2, -1)) if dk is not None else jnp.zeros_like(Kf)
    dXf = (
        jnp.fft.rfftn(dx_pad, axes=(-2, -1))
        if dx_pad is not None
        else jnp.zeros_like(Xf)
    )

    dYf = jnp.einsum("oihw,bihw->bohw", dKf, Xf) + jnp.einsum(
        "oihw,bihw->bohw", Kf, dXf
    )
    dy = jnp.fft.irfftn(dYf, s=(kernel.shape[-2], kernel.shape[-1]), axes=(-2, -1))

    if single:
        return y[0], dy[0]
    else:
        return y, dy


class Circulant2d:
    """Real weight-space circulant convolution (Bayesian or det.)."""

    def __init__(
        self,
        C_in: int,
        C_out: int,
        H_in: int,
        W_in: int,
        *,
        H_pad: Optional[int] = None,
        W_pad: Optional[int] = None,
        name: str = "circ2d",
        kernel_prior_fn: Callable[
            [Tuple[int, ...], jnp.dtype], dist.Distribution
        ] = lambda shape, dtype: dist.Normal(0, 1)
        .expand(shape)
        .to_event(len(shape)),
        bias_prior_fn: Callable[
            [Tuple[int, ...], jnp.dtype], dist.Distribution
        ] = lambda shape, dtype: dist.Normal(0, 1)
        .expand(shape)
        .to_event(len(shape)),
        dtype=jnp.float32,
    ):
        self.C_in, self.C_out = C_in, C_out
        self.H_pad = H_pad or H_in
        self.W_pad = W_pad or W_in
        self.dtype = dtype
        self.name = name
        self._kernel_prior_fn = kernel_prior_fn
        self._bias_prior_fn = bias_prior_fn

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        kernel = numpyro.sample(
            f"{self.name}_kernel",
            self._kernel_prior_fn(
                (self.C_out, self.C_in, self.H_pad, self.W_pad), self.dtype
            ),
        )
        bias = numpyro.sample(
            f"{self.name}_bias",
            self._bias_prior_fn((self.C_out, self.H_pad, self.W_pad), self.dtype),
        )
        y = fft_conv2d_real(kernel.astype(x.dtype), x)
        return y + (bias if y.ndim == 3 else bias[None, :])

    def get_kernel(self, params) -> jnp.ndarray:
        return params[f"{self.name}_kernel"]


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
        alpha_z = numpyro.sample(f"{self.name}_alpha_z", dist.Normal(0.0, 1.0))
        α_min, α_max = 0.1, 3.0
        alpha = α_min + (α_max - α_min) * jax.nn.sigmoid(alpha_z)

        freq_idx = jnp.arange(self.k_half, dtype=x.dtype)
        freq_norm = freq_idx / jnp.maximum(self.k_half - 1, 1)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_norm**alpha)

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

        self._m = jnp.log(jnp.exp(5.0 - 0.1) - 1.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        delta_z = numpyro.sample(
            f"{self.name}_delta_alpha",
            self.alpha_prior.expand([self.k_half]).to_event(1),
        )
        delta = jax.nn.softplus(delta_z)
        z = self.alpha_global + delta
        α_min, α_max = 0.1, 3.0
        alpha_vec = α_min + (α_max - α_min) * jax.nn.sigmoid(z)

        freq_idx = jnp.arange(self.k_half, dtype=x.dtype)
        freq_norm = freq_idx / jnp.maximum(self.k_half - 1, 1)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_norm**alpha_vec)

        active_idx = jnp.arange(self.K)
        real_hp = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(prior_std[active_idx]).expand([self.K]).to_event(1),
        )
        imag_hp = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(prior_std[active_idx]).expand([self.K]).to_event(1),
        )

        full_real = jnp.zeros((self.k_half,)).at[active_idx].set(real_hp)
        full_imag = jnp.zeros((self.k_half,)).at[active_idx].set(imag_hp)

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


def _enforce_hermitian(fft2d: jnp.ndarray) -> jnp.ndarray:
    H, W = fft2d.shape[-2:]
    conj_flip = jnp.flip(jnp.conj(fft2d), axis=(-2, -1))
    herm = 0.5 * (fft2d + conj_flip)
    herm = herm.at[..., 0, 0].set(jnp.real(herm[..., 0, 0]))
    if H % 2 == 0:
        herm = herm.at[..., H // 2, :].set(jnp.real(herm[..., H // 2, :]))
    if W % 2 == 0:
        herm = herm.at[..., :, W // 2].set(jnp.real(herm[..., :, W // 2]))
    return herm


def upsample_bilinear(grid, H_pad, W_pad):
    """Bilinearly up-sample a coarse grid (no grad issues)."""
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


@jax.custom_jvp
def spectral_circulant_conv2d(x: jnp.ndarray, fft_kernel: jnp.ndarray) -> jnp.ndarray:
    """
    fft_kernel : (C_out, C_in, H_pad, W_pad) complex Hermitian
    x          : (..., C_in, H_in, W_in)         real
    """
    C_out, C_in, H_pad, W_pad = fft_kernel.shape
    single = x.ndim == 3
    if single:
        x = x[None, ...]
    *lead, C_in_x, H_in, W_in = x.shape
    if C_in_x != C_in:
        raise ValueError("Cin mismatch.")

    pad_h, pad_w = max(0, H_pad - H_in), max(0, W_pad - W_in)
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[..., :H_pad, :W_pad]

    Xf = jnp.fft.fftn(x_pad, axes=(-2, -1))
    Yf = jnp.einsum("oihw,bihw->bohw", fft_kernel, Xf)
    y = jnp.fft.ifftn(Yf, axes=(-2, -1)).real
    return y[0] if single else y


@spectral_circulant_conv2d.defjvp
def _spectral_circulant_conv2d_jvp(primals, tangents):
    x, fft_kernel = primals
    dx, dk = tangents

    C_out, C_in, H_pad, W_pad = fft_kernel.shape
    single = x.ndim == 3
    if single:
        x = x[None, ...]
        dx = None if dx is None else dx[None, ...]

    pad_h, pad_w = max(0, H_pad - x.shape[-2]), max(0, W_pad - x.shape[-1])
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[..., :H_pad, :W_pad]
    dx_pad = (
        None
        if dx is None
        else jnp.pad(dx, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[..., :H_pad, :W_pad]
    )

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


class SpectralCirculantLayer2d:
    """
    GP‐origin spectral layer with *fixed* α (shared across channels).
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
        alpha_prior: dist.Distribution = dist.HalfNormal(1.0),
        name: str = "spec2d",
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        dtype=jnp.float32,
    ):
        self.C_in, self.C_out = C_in, C_out
        self.H_pad = H_pad or H_in
        self.W_pad = W_pad or W_in
        self.alpha = alpha
        self.alpha_prior = alpha_prior
        self.name = name
        self.dtype = dtype
        self._prior_fn = prior_fn or (lambda scale: dist.Normal(0.0, scale))
        self._fft_kernel = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        alpha_z = numpyro.sample(f"{self.name}_alpha_z", dist.Normal(0.0, 1.0))
        α_min, α_max = 0.1, 3.0
        alpha = α_min + (α_max - α_min) * jax.nn.sigmoid(alpha_z)
        u = jnp.fft.fftfreq(self.H_pad, dtype=self.dtype) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad, dtype=self.dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        R_norm = R / jnp.max(R)
        std = (1.0 / jnp.sqrt(1.0 + R_norm**alpha))[None, None, :, :]

        real = numpyro.sample(
            f"{self.name}_real",
            self._prior_fn(std)
            .expand((self.C_out, self.C_in, self.H_pad, self.W_pad))
            .to_event(4),
        )
        imag = numpyro.sample(
            f"{self.name}_imag",
            self._prior_fn(std)
            .expand((self.C_out, self.C_in, self.H_pad, self.W_pad))
            .to_event(4),
        )

        fft_kernel = _enforce_hermitian(real + 1j * imag)
        self._fft_kernel = jax.lax.stop_gradient(fft_kernel)

        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0, 1).expand((self.C_out, self.H_pad, self.W_pad)).to_event(3),
        )

        y = spectral_circulant_conv2d(x, fft_kernel)
        return y + (bias if y.ndim == 3 else bias[None, :])

    def get_fft_kernel(self):
        if self._fft_kernel is None:
            raise RuntimeError("Call the layer once to build its kernel.")
        return self._fft_kernel


class AdaptiveSpectralCirculantLayer2d(SpectralCirculantLayer2d):
    """
    Adaptive per-frequency α(u,v) via a coarse HalfNormal residual grid.
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
        alpha_prior: dist.Distribution = dist.HalfNormal(1.0),
        name: str = "adap_spec2d",
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        dtype=jnp.float32,
    ):
        super().__init__(
            C_in,
            C_out,
            H_in,
            W_in,
            H_pad=H_pad,
            W_pad=W_pad,
            alpha=None,
            alpha_prior=alpha_prior,
            name=name,
            prior_fn=prior_fn,
            dtype=dtype,
        )
        self.alpha_global = alpha_global
        self.alpha_coarse_shape = alpha_coarse_shape
        self._m = jnp.log(jnp.exp(5.0 - 0.1) - 1.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        delta_z_coarse = numpyro.sample(
            f"{self.name}_delta_alpha",
            self.alpha_prior.expand(self.alpha_coarse_shape).to_event(2),
        )
        delta_full = jax.nn.softplus(delta_z_coarse)
        z_full = self.alpha_global + delta_full
        α_min, α_max = 0.1, 3.0
        alpha_map = α_min + (α_max - α_min) * jax.nn.sigmoid(z_full)

        u = jnp.fft.fftfreq(self.H_pad, dtype=self.dtype) * self.H_pad
        v = jnp.fft.fftfreq(self.W_pad, dtype=self.dtype) * self.W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        R_norm = R / jnp.max(R)
        std2d = 1.0 / jnp.sqrt(1.0 + R_norm**alpha_map)
        std = std2d[None, None, :, :].astype(self.dtype)

        real = numpyro.sample(
            f"{self.name}_real",
            self._prior_fn(std)
            .expand((self.C_out, self.C_in, self.H_pad, self.W_pad))
            .to_event(4),
        )
        imag = numpyro.sample(
            f"{self.name}_imag",
            self._prior_fn(std)
            .expand((self.C_out, self.C_in, self.H_pad, self.W_pad))
            .to_event(4),
        )

        fft_kernel = _enforce_hermitian(real + 1j * imag)
        self._fft_kernel = jax.lax.stop_gradient(fft_kernel)

        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0, 1).expand((self.C_out, self.H_pad, self.W_pad)).to_event(3),
        )
        y = spectral_circulant_conv2d(x, fft_kernel)
        return y + (bias if y.ndim == 3 else bias[None, :])


class SpectralDense:
    """
    x ↦ x Wᵀ + b,  W = U diag(s) Vᵀ with U,V fixed orthonormal.
    Per-freq std = init_scale / sqrt(1 + k**alpha).
    """

    def __init__(
        self,
        U: jnp.ndarray,
        V: jnp.ndarray,
        *,
        alpha: float = 1.0,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        name: str = "fc_spec",
    ):
        self.U = jax.lax.stop_gradient(U)
        self.V = jax.lax.stop_gradient(V)
        self.d = U.shape[0]
        self.alpha = alpha
        self.init_scale = init_scale
        self.bias_scale = bias_scale
        self.prior_fn = prior_fn or (lambda scale: dist.Normal(0.0, scale))
        self.name = name

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        k = jnp.arange(self.d, dtype=x.dtype)
        k_norm = k / jnp.maximum(self.d - 1, 1)
        alpha_z = numpyro.sample(f"{self.name}_alpha_z", dist.Normal(0.0, 1.0))
        α_min, α_max = 0.1, 3.0
        alpha = α_min + (α_max - α_min) * jax.nn.sigmoid(alpha_z)
        std = self.init_scale / jnp.sqrt(1.0 + k_norm**alpha)
        s = numpyro.sample(
            f"{self.name}_s",
            self.prior_fn(std).expand([self.d]).to_event(1),
        )
        b = numpyro.sample(
            f"{self.name}_b",
            dist.Normal(0.0, self.bias_scale).expand([self.d]).to_event(1),
        )

        W = self.U @ jnp.diag(s) @ self.V.T
        return x @ W.T + b


class AdaptiveSpectralDense(SpectralDense):
    """
    Adds per-frequency exponent deltas delta_k ~ alpha_prior,
    so alpha_k = alpha_global + delta_k.
    """

    def __init__(
        self,
        U: jnp.ndarray,
        V: jnp.ndarray,
        *,
        alpha_global: float = 1.0,
        alpha_prior: dist.Distribution = dist.HalfNormal(1.0),
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        name: str = "adap_fc_spec",
    ):
        super().__init__(
            U,
            V,
            alpha=alpha_global,
            init_scale=init_scale,
            bias_scale=bias_scale,
            prior_fn=prior_fn,
            name=name,
        )
        self.alpha_prior = alpha_prior

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        k = jnp.arange(self.d, dtype=x.dtype)
        delta_z = numpyro.sample(
            f"{self.name}_delta",
            self.alpha_prior.expand([self.d]).to_event(1),
        )
        delta = jax.nn.softplus(delta_z)
        z_k = self.alpha + delta
        α_min, α_max = 0.1, 3.0
        alpha_k = α_min + (α_max - α_min) * jax.nn.sigmoid(z_k)

        k_norm = k / jnp.maximum(self.d - 1, 1)
        std = self.init_scale / jnp.sqrt(1.0 + k_norm**alpha_k)

        s = numpyro.sample(
            f"{self.name}_s",
            self.prior_fn(std).expand([self.d]).to_event(1),
        )
        b = numpyro.sample(
            f"{self.name}_b",
            dist.Normal(0.0, self.bias_scale).expand([self.d]).to_event(1),
        )

        W = self.U @ jnp.diag(s) @ self.V.T
        return x @ W.T + b


class SpectralConv2d:
    """
    Bayesian SVD-parameterized 2D conv in NumPyro:
      - U, V fixed orthonormal factors (passed in)
      - Learn singular values s via Sobolev-inspired prior
      - Learn a bias per output channel
      - Sample a global α via Normal → sigmoid → [0.1,3.0]
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
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        prior_fn=None,
        name: str = "spec_conv2d",
    ):
        self.U = jax.lax.stop_gradient(U)
        self.V = jax.lax.stop_gradient(V)
        self.d = U.shape[1]
        self.C_out = C_out
        self.C_in = C_in
        self.H_k = H_k
        self.W_k = W_k
        self.strides = strides
        self.padding = padding
        self.init_scale = init_scale
        self.bias_scale = bias_scale
        self.name = name
        self.prior_fn = prior_fn or (lambda scale: dist.Normal(0.0, scale))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        α_z = numpyro.sample(f"{self.name}_alpha_z", dist.Normal(0.0, 1.0))
        α_min, α_max = 0.1, 3.0
        α = α_min + (α_max - α_min) * jax.nn.sigmoid(α_z)

        k = jnp.arange(self.d, dtype=x.dtype)
        k_norm = k / jnp.maximum(self.d - 1, 1)
        std = self.init_scale * (1.0 / jnp.sqrt(1.0 + k_norm**α))

        s = numpyro.sample(
            f"{self.name}_s",
            self.prior_fn(std).expand([self.d]).to_event(1),
        )
        b = numpyro.sample(
            f"{self.name}_b",
            dist.Normal(0.0, self.bias_scale).expand([self.C_out]).to_event(1),
        )

        W_mat = self.U @ jnp.diag(s) @ self.V.T
        W = W_mat.reshape(self.C_out, self.C_in, self.H_k, self.W_k)

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
    Adaptive SVD-Conv2d with per-mode δ_z:
      - Samples δ_z ~ α_prior per singular index
      - α_k = sigmoid(α_z_global + softplus(δ_z)) mapped to [0.1,3.0]
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
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        alpha_global: float = 1.0,
        alpha_prior: dist.Distribution = dist.HalfNormal(1.0),
        prior_fn=None,
        name: str = "adap_spec_conv2d",
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
            init_scale=init_scale,
            bias_scale=bias_scale,
            prior_fn=prior_fn,
            name=name,
        )
        self.alpha_global = alpha_global
        self.alpha_prior = alpha_prior

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        α_z = numpyro.sample(f"{self.name}_alpha_z", dist.Normal(0.0, 1.0))
        α_min, α_max = 0.1, 3.0
        α0 = α_min + (α_max - α_min) * jax.nn.sigmoid(α_z)

        δ_z = numpyro.sample(
            f"{self.name}_delta_z",
            self.alpha_prior.expand([self.d]).to_event(1),
        )
        δ = jax.nn.softplus(δ_z)
        z = self.alpha_global + δ
        α_k = α_min + (α_max - α_min) * jax.nn.sigmoid(z)

        k = jnp.arange(self.d, dtype=x.dtype)
        k_norm = k / jnp.maximum(self.d - 1, 1)
        std = self.init_scale * (1.0 / jnp.sqrt(1.0 + k_norm**α_k))

        s = numpyro.sample(
            f"{self.name}_s",
            self.prior_fn(std).expand([self.d]).to_event(1),
        )
        b = numpyro.sample(
            f"{self.name}_b",
            dist.Normal(0.0, self.bias_scale).expand([self.C_out]).to_event(1),
        )

        W_mat = self.U @ jnp.diag(s) @ self.V.T
        W = W_mat.reshape(self.C_out, self.C_in, self.H_k, self.W_k)
        y = jax.lax.conv_general_dilated(
            x,
            W,
            window_strides=self.strides,
            padding=self.padding,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        return y + b[None, :, None, None]


def graph_laplacian(adj: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    deg = jnp.diag(adj.sum(axis=1))
    L = deg - adj
    lam, Q = jnp.linalg.eigh(L)
    return Q, lam


class GraphSpectralDense:
    """
    x ↦ x Wᵀ + b,  W = Q diag(s) Qᵀ from adjacency Laplacian.
    """

    def __init__(
        self,
        adj: jnp.ndarray,
        *,
        alpha: float = 1.0,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        name: str = "graph_spec",
    ):
        Q, lam = graph_laplacian(adj)
        self.Q = jax.lax.stop_gradient(Q)
        self.lam = jax.lax.stop_gradient(lam)
        self.d = lam.shape[0]
        self.alpha = alpha
        self.init_scale = init_scale
        self.bias_scale = bias_scale
        self.prior_fn = prior_fn or (lambda scale: dist.Normal(0.0, scale))
        self.name = name

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., d)
        std = self.init_scale / jnp.sqrt(1.0 + self.lam**self.alpha)
        s = numpyro.sample(
            f"{self.name}_s",
            self.prior_fn(std).expand([self.d]).to_event(1),
        )
        b = numpyro.sample(
            f"{self.name}_b",
            dist.Normal(0.0, self.bias_scale).expand([self.d]).to_event(1),
        )
        W = self.Q @ jnp.diag(s) @ self.Q.T
        return x @ W.T + b


class AdaptiveGraphSpectralDense(GraphSpectralDense):
    """
    Adaptive per-frequency α for graph-spectral dense.
    """

    def __init__(
        self,
        adj: jnp.ndarray,
        *,
        alpha_global: float = 1.0,
        alpha_prior: dist.Distribution = dist.HalfNormal(1.0),
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        name: str = "adap_graph_spec",
    ):
        super().__init__(
            adj,
            alpha=alpha_global,
            init_scale=init_scale,
            bias_scale=bias_scale,
            prior_fn=prior_fn,
            name=name,
        )
        self.alpha_prior = alpha_prior

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        delta = numpyro.sample(
            f"{self.name}_delta",
            self.alpha_prior.expand([self.d]).to_event(1),
        )
        alpha_k = self.alpha + delta
        std = self.init_scale / jnp.sqrt(1.0 + self.lam**alpha_k)

        s = numpyro.sample(
            f"{self.name}_s",
            self.prior_fn(std).expand([self.d]).to_event(1),
        )
        b = numpyro.sample(
            f"{self.name}_b",
            dist.Normal(0.0, self.bias_scale).expand([self.d]).to_event(1),
        )

        W = self.Q @ jnp.diag(s) @ self.Q.T
        return x @ W.T + b


class GraphSpectralConv2d:
    """
    1×1 graph conv: x:(B, C, H, W) → y:(B, C, H, W)
    with W = Q diag(s) Qᵀ per pixel.
    """

    def __init__(
        self,
        adj: jnp.ndarray,
        *,
        alpha: float = 1.0,
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        name: str = "graph1x1",
    ):
        Q, lam = graph_laplacian(adj)
        self.Q = jax.lax.stop_gradient(Q)
        self.lam = jax.lax.stop_gradient(lam)
        self.C = lam.shape[0]
        self.alpha = alpha
        self.init_scale = init_scale
        self.bias_scale = bias_scale
        self.prior_fn = prior_fn or (lambda scale: dist.Normal(0.0, scale))
        self.name = name

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        std = self.init_scale / jnp.sqrt(1.0 + self.lam**self.alpha)
        s = numpyro.sample(
            f"{self.name}_s",
            self.prior_fn(std).expand([self.C]).to_event(1),
        )
        b = numpyro.sample(
            f"{self.name}_b",
            dist.Normal(0.0, self.bias_scale).expand([self.C]).to_event(1),
        )

        W = self.Q @ jnp.diag(s) @ self.Q.T
        y = jnp.einsum("oc,bchw->bohw", W, x)
        return y + b[None, :, None, None]


class AdaptiveGraphSpectralConv2d(GraphSpectralConv2d):
    """
    Adaptive per-frequency α for graph 1×1 conv.
    """

    def __init__(
        self,
        adj: jnp.ndarray,
        *,
        alpha_global: float = 1.0,
        alpha_prior: dist.Distribution = dist.HalfNormal(1.0),
        init_scale: float = 0.1,
        bias_scale: float = 0.1,
        prior_fn: Optional[Callable[[jnp.ndarray], dist.Distribution]] = None,
        name: str = "adap_graph1x1",
    ):
        super().__init__(
            adj,
            alpha=alpha_global,
            init_scale=init_scale,
            bias_scale=bias_scale,
            prior_fn=prior_fn,
            name=name,
        )
        self.alpha_prior = alpha_prior

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        delta = numpyro.sample(
            f"{self.name}_delta",
            self.alpha_prior.expand([self.C]).to_event(1),
        )
        alpha_k = self.alpha + delta
        std = self.init_scale / jnp.sqrt(1.0 + self.lam**alpha_k)

        s = numpyro.sample(
            f"{self.name}_s",
            self.prior_fn(std).expand([self.C]).to_event(1),
        )
        b = numpyro.sample(
            f"{self.name}_b",
            dist.Normal(0.0, self.bias_scale).expand([self.C]).to_event(1),
        )

        W = self.Q @ jnp.diag(s) @ self.Q.T
        y = jnp.einsum("oc,bchw->bohw", W, x)
        return y + b[None, :, None, None]


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


class PatchWiseSpectralMixture:
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
        w = jax.nn.softmax(logits, axis=-1)
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


class PatchWiseSpectralMixture2d:
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
