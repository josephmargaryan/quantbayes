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
