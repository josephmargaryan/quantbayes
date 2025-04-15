import jax
import jax.numpy as jnp
from numpyro.infer.autoguide import AutoGuide
import numpyro
import numpyro.distributions as dist

@jax.custom_jvp
def spectral_circulant_conv1d(x: jnp.ndarray,
                              fft_kernel: jnp.ndarray) -> jnp.ndarray:
    """
    x: shape (batch, length)  - single-channel 1D signals
    fft_kernel: shape (padded_len,) - the 1D FFT of the kernel (with Hermitian symmetry)
    Returns convolved output of shape (batch, padded_len) [or truncated as needed].
    """
    padded_len = fft_kernel.shape[0]
    single_example = (x.ndim == 1)
    if single_example:
        # Expand batch dimension
        x = x[None, :]

    length = x.shape[-1]
    # Pad or truncate x
    if length < padded_len:
        pad_len = padded_len - length
        x_padded = jnp.pad(x, ((0,0),(0,pad_len)))
    elif length > padded_len:
        x_padded = x[..., :padded_len]
    else:
        x_padded = x

    # Forward FFT
    X_fft = jnp.fft.fft(x_padded, axis=-1)  # shape (batch, padded_len)
    # Multiply in freq domain
    Y_fft = X_fft * fft_kernel[None, :]     # broadcast mul
    # iFFT
    y = jnp.fft.ifft(Y_fft, axis=-1).real

    # (Optional) you might truncate or keep full length
    if single_example:
        return y[0]
    return y

@spectral_circulant_conv1d.defjvp
def spectral_circulant_conv1d_jvp(primals, tangents):
    x, fft_kernel = primals
    dx, dfft = tangents
    padded_len = fft_kernel.shape[0]

    single_example = (x.ndim == 1)
    if single_example:
        x = x[None, :]
        dx = None if dx is None else dx[None, :]

    length = x.shape[-1]
    if length < padded_len:
        pad_len = padded_len - length
        x_padded = jnp.pad(x, ((0,0),(0,pad_len)))
        dx_padded = (None if dx is None else jnp.pad(dx, ((0,0),(0,pad_len))))
    elif length > padded_len:
        x_padded = x[..., :padded_len]
        dx_padded = (None if dx is None else dx[..., :padded_len])
    else:
        x_padded = x
        dx_padded = dx

    X_fft = jnp.fft.fft(x_padded, axis=-1)
    primal_Y_fft = X_fft * fft_kernel[None, :]
    primal_y = jnp.fft.ifft(primal_Y_fft, axis=-1).real

    # JVP
    if dx_padded is None:
        dX_fft = 0.0
    else:
        dX_fft = jnp.fft.fft(dx_padded, axis=-1)

    if dfft is None:
        dK_fft = 0.0
    else:
        dK_fft = dfft

    dY_fft = dX_fft * fft_kernel[None, :] + X_fft * dK_fft[None, :]
    dY = jnp.fft.ifft(dY_fft, axis=-1).real

    if single_example:
        return primal_y[0], dY[0]
    return primal_y, dY


class SpectralCirculantConv1d:
    def __init__(self,
                 signal_len: int,
                 padded_len: int = None,
                 alpha: float = None,
                 alpha_prior=dist.HalfNormal(1.0),
                 K: int = None,
                 name="spectral_conv1d",
                 prior_fn=None):
        """
        :param signal_len: Nominal input signal length.
        :param padded_len: If provided, pad/truncate to this dimension; else = signal_len.
        :param alpha: Optional fixed value for freq-decay exponent; else sample from alpha_prior.
        :param alpha_prior: Prior distribution for alpha if not fixed.
        :param K: Number of active frequencies to keep; if None, use full half-spectrum.
        :param name: Base name for NumPyro sample sites.
        :param prior_fn: Function mapping a scale -> distribution (default Normal(0, scale)).
        """
        self.signal_len = signal_len
        self.padded_len = padded_len or signal_len
        self.alpha = alpha
        self.alpha_prior = alpha_prior
        self.name = name

        self.k_half = (self.padded_len // 2) + 1
        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        self.prior_fn = prior_fn or (lambda scale: dist.Normal(0.0, scale))
        self._fft_kernel = None  # store for debugging

    def __call__(self, x: jnp.ndarray):
        """
        Perform the 1D 'circulant convolution' using the spectral kernel.
        x: shape (batch, signal_len) or (signal_len,)
        Returns convolved shape (batch, padded_len) by default.
        """
        # 1. Sample alpha if not fixed
        if self.alpha is None:
            alpha = numpyro.sample(f"{self.name}_alpha", self.alpha_prior)
        else:
            alpha = self.alpha

        # 2. Compute prior std in freq domain
        freq_idx = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_idx**alpha)

        # 3. Truncate to K active frequencies
        active_idx = jnp.arange(self.K)
        real_part = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(prior_std[active_idx]).expand([self.K]).to_event(1)
        )
        imag_part = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(prior_std[active_idx]).expand([self.K]).to_event(1)
        )

        # 4. Fill up the half-spectrum
        half_real = jnp.zeros(self.k_half)
        half_imag = jnp.zeros(self.k_half)
        half_real = half_real.at[active_idx].set(real_part)
        half_imag = half_imag.at[active_idx].set(imag_part)

        # enforce real DC (idx=0) and Nyquist if even
        half_imag = half_imag.at[0].set(0.0)
        if (self.padded_len % 2 == 0) and (self.k_half > 1):
            half_imag = half_imag.at[-1].set(0.0)

        half_complex = half_real + 1j * half_imag

        # 5. Construct full spectrum via Hermitian symmetry
        if (self.padded_len % 2 == 0) and (self.k_half > 1):
            nyquist = half_complex[-1].real[None]  # shape (1,)
            fft_kernel = jnp.concatenate(
                [half_complex[:-1], nyquist, jnp.conjugate(half_complex[1:-1])[::-1]]
            )
        else:
            fft_kernel = jnp.concatenate(
                [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
            )

        # (Optional) store for inspection
        self._fft_kernel = jax.lax.stop_gradient(fft_kernel)

        # 6. Optional bias
        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0.0, 1.0).expand([self.padded_len]).to_event(1)
        )

        # 7. Call the custom circulant conv
        y = spectral_circulant_conv1d(x, fft_kernel)
        return y + bias

    def get_fft_kernel(self):
        if self._fft_kernel is None:
            raise ValueError("No kernel has been sampled yet.")
        return self._fft_kernel

@jax.custom_jvp
def spectral_circulant_conv2d(x: jnp.ndarray,
                              fft_kernel_2d: jnp.ndarray) -> jnp.ndarray:
    """
    x: shape (batch, H, W)  - single-channel 2D inputs
    fft_kernel_2d: shape (H_pad, W_pad) - the 2D FFT of the kernel
    Returns shape (batch, H_pad, W_pad) by default (or you can slice).
    """
    H_pad, W_pad = fft_kernel_2d.shape
    single_example = (x.ndim == 2)
    if single_example:
        x = x[None, ...]  # shape (1, H, W)

    H_in, W_in = x.shape[-2], x.shape[-1]

    # Pad or truncate in 2D
    pad_h = max(0, H_pad - H_in)
    pad_w = max(0, W_pad - W_in)
    x_padded = jnp.pad(x, ((0,0),(0,pad_h),(0,pad_w)))

    # If H_in>H_pad or W_in>W_pad, you might also slice, e.g.: x_padded = x[...,:H_pad,:W_pad]
    # 2D FFT
    X_fft = jnp.fft.fftn(x_padded, axes=(-2, -1))

    # multiply
    Y_fft = X_fft * fft_kernel_2d[None, :, :]
    # iFFT
    y = jnp.fft.ifftn(Y_fft, axes=(-2, -1)).real

    if single_example:
        y = y[0]
    return y

@spectral_circulant_conv2d.defjvp
def spectral_circulant_conv2d_jvp(primals, tangents):
    x, fft_kernel_2d = primals
    dx, dfft = tangents

    H_pad, W_pad = fft_kernel_2d.shape
    single_example = (x.ndim == 2)

    if single_example:
        x = x[None, ...]
        dx = None if dx is None else dx[None, ...]

    H_in, W_in = x.shape[-2], x.shape[-1]

    # Pad x
    pad_h = max(0, H_pad - H_in)
    pad_w = max(0, W_pad - W_in)
    x_padded = jnp.pad(x, ((0,0),(0,pad_h),(0,pad_w)))
    dx_padded = None
    if dx is not None:
        dx_padded = jnp.pad(dx, ((0,0),(0,pad_h),(0,pad_w)))

    X_fft = jnp.fft.fftn(x_padded, axes=(-2, -1))
    primal_Y_fft = X_fft * fft_kernel_2d[None, :, :]
    primal_y = jnp.fft.ifftn(primal_Y_fft, axes=(-2, -1)).real

    if dx_padded is None:
        dX_fft = 0.0
    else:
        dX_fft = jnp.fft.fftn(dx_padded, axes=(-2, -1))

    dK_fft = 0.0 if dfft is None else dfft

    dY_fft = dX_fft * fft_kernel_2d[None, :, :] + X_fft * dK_fft[None, :, :]
    dY = jnp.fft.ifftn(dY_fft, axes=(-2, -1)).real

    if single_example:
        return primal_y[0], dY[0]
    return primal_y, dY

class SpectralCirculantConv2d:
    def __init__(self,
                 H_in: int,
                 W_in: int,
                 H_pad: int = None,
                 W_pad: int = None,
                 alpha: float = None,
                 alpha_prior=dist.HalfNormal(1.0),
                 name="spectral_conv2d",
                 prior_fn=None):
        """
        :param H_in, W_in: nominal input height and width
        :param H_pad, W_pad: padded dimension for circulant kernel
        :param alpha: optional fixed exponent for freq-decay
        :param alpha_prior: prior distribution for alpha
        :param name: base name for sampling sites
        :param prior_fn: function mapping scale -> distribution
        """
        self.H_in = H_in
        self.W_in = W_in
        self.H_pad = H_pad if H_pad is not None else H_in
        self.W_pad = W_pad if W_pad is not None else W_in
        self.alpha = alpha
        self.alpha_prior = alpha_prior
        self.name = name

        self.prior_fn = prior_fn or (lambda scale: dist.Normal(0.0, scale))
        self._fft_kernel_2d = None

    def __call__(self, x: jnp.ndarray):
        """
        x: shape (batch, H_in, W_in) or (H_in, W_in)
        returns shape (batch, H_pad, W_pad) or (H_pad, W_pad)
        """
        # 1. alpha
        if self.alpha is None:
            alpha = numpyro.sample(f"{self.name}_alpha", self.alpha_prior)
        else:
            alpha = self.alpha

        # 2. 2D freq indices
        freq_u = jnp.arange(self.H_pad)
        freq_v = jnp.arange(self.W_pad)
        # For each (u,v), define some "radius" in freq domain.
        # e.g. r = sqrt(u^2 + v^2), or we keep them separate
        # Let's do a simple radial version:
        UU, VV = jnp.meshgrid(freq_u, freq_v, indexing='ij')
        R = jnp.sqrt(UU**2 + VV**2)
        # prior_std = 1 / sqrt(1 + R^alpha)
        prior_std = 1.0 / jnp.sqrt(1.0 + (R**alpha))

        # 3. Sample real + imag from this PSD
        # This is shape (H_pad, W_pad).
        # In principle, you could do "active frequencies," but let's keep it simpler:
        real_part = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(prior_std).to_event(2)  # shape (H_pad, W_pad)
        )
        imag_part = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(prior_std).to_event(2)
        )

        # 4. Enforce Hermitian symmetry
        # We want kernel to be real in spatial domain. One way is:
        #   1) For every (u,v), set Kernel[H-u, W-v] = conj(Kernel[u,v])
        #   2) Specifically enforce real at DC, Nyquist, etc.
        # For brevity, let's do a naive approach:
        real_part = real_part.at[0,0].set(real_part[0,0])  # DC is real
        imag_part = imag_part.at[0,0].set(0.0)
        # We won't do a full indexing approach, but you might do something like:
        #   for u in range(H_pad):
        #       for v in range(W_pad):
        #           conj_u = (-u) % H_pad
        #           conj_v = (-v) % W_pad
        #           imag_part[conj_u, conj_v] = -imag_part[u,v]
        # etc.

        fft2d = real_part + 1j * imag_part

        # 5. store for debugging
        self._fft_kernel_2d = jax.lax.stop_gradient(fft2d)

        # 6. Sample bias
        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0.0, 1.0).expand([self.H_pad, self.W_pad]).to_event(2)
        )

        # 7. forward conv
        y = spectral_circulant_conv2d(x, fft2d)
        return y + bias

    def get_fft_kernel_2d(self):
        if self._fft_kernel_2d is None:
            raise ValueError("Kernel not sampled yet.")
        return self._fft_kernel_2d

class Spectral2DRealGuide(AutoGuide):
    def __init__(self, model, shape_2d):
        """
        shape_2d: (H_pad, W_pad) for the real part of the 2D spectral kernel
        """
        super().__init__(model)
        self.shape_2d = shape_2d

    def __call__(self, *args, **kwargs):
        # We'll assume the site name is "spectral_conv2d_real"
        size = self.shape_2d
        mean_real = numpyro.param("spectral_conv2d_real_mean",
                                  jnp.zeros(size))
        log_std_real = numpyro.param("spectral_conv2d_real_log_std",
                                     -3.0 * jnp.ones(size))

        sample_real = numpyro.sample(
            "spectral_conv2d_real",
            dist.Normal(mean_real, jnp.exp(log_std_real)).to_event(2)
        )
        return {"spectral_conv2d_real": sample_real}

    def sample_posterior(self, rng_key, params, sample_shape=()):
        mean_real = params["spectral_conv2d_real_mean"]
        log_std_real = params["spectral_conv2d_real_log_std"]
        dist_ = dist.Normal(mean_real, jnp.exp(log_std_real)).to_event(2)
        sample_real = dist_.sample(rng_key, sample_shape)
        return {"spectral_conv2d_real": sample_real}
