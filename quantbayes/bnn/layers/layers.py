import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import transforms

__all__ = [
    "Linear",
    "Circulant",
    "BlockCirculant",
    "CirculantProcess",
    "BlockCirculantProcess",
    "DeepKernelCirc",
    "DeepKerneBlockCirc",
    "FourierNeuralOperator1D",
    "SpectralDenseBlock",
    "ParticleLinear",
    "FFTParticleLinear",
    "Conv1d",
    "Conv2d",
    "FFTConv1d",
    "FFTConv2d",
    "TransposedConv2d",
    "FFTTransposedConv2d",
    "MaxPool2d",
    "SelfAttention",
    "MultiHeadSelfAttention",
    "PositionalEncoding",
    "TransformerEncoder",
    "LayerNorm",
    "LSTM",
    "GaussianProcessLayer",
    "VariationalLayer",
    "MixtureOfTwoLayers",
]


class LayerNorm:
    def __init__(self, num_features, name="layer_norm"):
        self.num_features = num_features
        self.name = name

    def __call__(self, X):
        mean = jnp.mean(X, axis=-1, keepdims=True)
        variance = jnp.var(X, axis=-1, keepdims=True)
        epsilon = 1e-5  # Small constant for numerical stability
        scale = numpyro.sample(
            f"{self.name}_scale", dist.Normal(1.0, 0.1).expand([self.num_features])
        )
        shift = numpyro.sample(
            f"{self.name}_shift", dist.Normal(0.0, 0.1).expand([self.num_features])
        )
        normalized = (X - mean) / jnp.sqrt(variance + epsilon)
        return scale * normalized + shift


class MaxPool2d:
    """
    A 2D max-pooling layer.
    """

    def __init__(self, kernel_size=2, stride=2, name="maxpool2d"):
        """
        :param kernel_size: int
            Size of the pooling kernel.
        :param stride: int
            Stride for the pooling.
        :param name: str
            Name of the layer (not strictly used in your Bayesian parameter naming,
            but good for clarity).
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform max pooling on the input.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, channels, height, width)`.

        :return: jnp.ndarray
            Pooled output tensor of shape `(batch_size, channels, pooled_height, pooled_width)`.
        """
        # Reduce window applies max pooling over the spatial dimensions (height, width).
        return jax.lax.reduce_window(
            X,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, 1, self.kernel_size, self.kernel_size),
            window_strides=(1, 1, self.stride, self.stride),
            padding="VALID",  # For typical UNet, "VALID" pooling is common
        )


class Linear:
    """
    A fully connected layer with weights and biases sampled from specified distributions.

    Transforms inputs via a linear operation: `output = X @ weights + biases`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        name: str = "layer",
        weight_prior_fn=lambda shape: dist.Normal(0, 1)
        .expand(shape)
        .to_event(len(shape)),
        bias_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape).to_event(1),
    ):
        """
        Initializes the Linear layer.

        :param in_features: int
            Number of input features.
        :param out_features: int
            Number of output features.
        :param name: str
            Name of the layer for parameter tracking (default: "layer").
        :param weight_prior_fn: function
            A function that takes a shape and returns a NumPyro distribution for the weights.
        :param bias_prior_fn: function
            A function that takes a shape and returns a NumPyro distribution for the biases.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.weight_prior_fn = weight_prior_fn
        self.bias_prior_fn = bias_prior_fn

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the linear transformation on the input.

        :param X: jnp.ndarray
            Input array of shape `(batch_size, in_features)`.
        :returns: jnp.ndarray
            Output array of shape `(batch_size, out_features)`.
        """
        w = numpyro.sample(
            f"{self.name}_w",
            self.weight_prior_fn([self.in_features, self.out_features]),
        )
        b = numpyro.sample(f"{self.name}_b", self.bias_prior_fn([self.out_features]))
        return jnp.dot(X, w) + b


def _fft_matmul(first_row: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """
    Perform circulant matrix multiplication using FFT.

    This computes the multiplication of a circulant matrix (defined by its first row)
    with a matrix `X` using the Fast Fourier Transform (FFT) for efficiency.

    :param first_row: jnp.ndarray
        The first row of the circulant matrix, shape `(in_features,)`.
    :param X: jnp.ndarray
        The input matrix, shape `(batch_size, in_features)`.

    :returns: jnp.ndarray
        Result of the circulant matrix multiplication, shape `(batch_size, in_features)`.
    """
    first_row_fft = jnp.fft.fft(first_row, axis=-1)
    X_fft = jnp.fft.fft(X, axis=-1)
    result_fft = first_row_fft[None, :] * X_fft
    result = jnp.fft.ifft(result_fft, axis=-1).real
    return result


class Circulant:
    """
    FFT-based linear layer for efficient circulant matrix multiplication.

    This layer uses a circulant matrix (parameterized by its first row) and
    applies FFT-based matrix multiplication for computational efficiency.
    """

    def __init__(
        self,
        in_features: int,
        name: str = "fft_layer",
        first_row_prior_fn=lambda shape: dist.Normal(0, 1)
        .expand(shape)
        .to_event(len(shape)),
        bias_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape).to_event(1),
    ):
        """
        Initialize the FFTLinear layer.

        :param in_features: int
            Number of input features.
        :param name: str
            Name of the layer, used for parameter naming (default: "fft_layer").
        :param first_row_prior_fn: function
            A function that takes a shape and returns a NumPyro distribution for the
            circulant matrix's first row.
        :param bias_prior_fn: function
            A function that takes a shape and returns a NumPyro distribution for the
            bias.
        """
        self.in_features = in_features
        self.name = name
        self.first_row_prior_fn = first_row_prior_fn
        self.bias_prior_fn = bias_prior_fn

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the FFTLinear layer.

        :param X: jnp.ndarray
            Input data, shape `(batch_size, in_features)`.
        :returns: jnp.ndarray
            Output of the FFT-based linear layer, shape `(batch_size, in_features)`.
        """
        first_row = numpyro.sample(
            f"{self.name}_first_row", self.first_row_prior_fn([self.in_features])
        )
        bias_circulant = numpyro.sample(
            f"{self.name}_bias_circulant", self.bias_prior_fn([self.in_features])
        )
        hidden = _fft_matmul(first_row, X) + bias_circulant[None, :]
        return hidden


def _block_circulant_matmul(W, x, d_bernoulli=None):
    """
    Perform block-circulant matmul using FFT.
    W: shape (k_out, k_in, b), each row W[i,j,:] is the "first row" of a b x b circulant block.
    x: shape (batch, d_in) or (d_in,)
    d_bernoulli: shape (d_in,) of ±1, if given
    Returns: shape (batch, d_out).
    """
    # If x is 1D, reshape to (1, d_in)
    if x.ndim == 1:
        x = x[None, :]
    batch_size, d_in = x.shape

    k_out, k_in, b = W.shape
    d_out = k_out * b  # noqa

    # Possibly multiply x by the Bernoulli diagonal
    if d_bernoulli is not None:
        x = x * d_bernoulli[None, :]

    # Zero-pad x to length (k_in * b)
    pad_len = k_in * b - d_in
    if pad_len > 0:
        x = jnp.pad(x, ((0, 0), (0, pad_len)))

    # Reshape into blocks: shape (batch, k_in, b)
    x_blocks = x.reshape(batch_size, k_in, b)

    # We'll accumulate output for each block-row i
    def one_block_mul(w_ij, x_j):
        # w_ij: (b,)   first row of circulant
        # x_j:  (batch, b)
        c_fft = jnp.fft.fft(w_ij)  # (b,)
        X_fft = jnp.fft.fft(x_j, axis=-1)  # (batch, b)
        block_fft = X_fft * jnp.conjugate(c_fft)[None, :]
        return jnp.fft.ifft(block_fft, axis=-1).real  # (batch, b)

    def compute_blockrow(i):
        # Sum over j=0..k_in-1 of circ(W[i,j]) x_j
        def sum_over_j(carry, j):
            w_ij = W[i, j, :]  # (b,)
            x_j = x_blocks[:, j, :]  # (batch, b)
            block_out = one_block_mul(w_ij, x_j)
            return carry + block_out, None

        init = jnp.zeros((batch_size, b))
        out_time, _ = jax.lax.scan(sum_over_j, init, jnp.arange(k_in))
        return out_time  # shape (batch, b)

    out_blocks = jax.vmap(compute_blockrow)(jnp.arange(k_out))  # (k_out, batch, b)
    out_reshaped = jnp.transpose(out_blocks, (1, 0, 2)).reshape(batch_size, k_out * b)
    return out_reshaped


class BlockCirculant:
    """
    NumPyro-style block-circulant layer.

    This layer:
      - Samples W with shape (k_out, k_in, b) using a user-specified prior (default is Normal(0,1)).
      - Optionally samples a Bernoulli diagonal for the input (controlled by use_diag).
      - Always samples a bias vector (of shape (out_features,)) using a user-specified prior
        (default is Normal(0,1)).
      - Performs block-circulant matrix multiplication using FFT.

    Parameters:
      in_features: int
          The overall input dimension.
      out_features: int
          The overall output dimension.
      block_size: int
          The size b of each circulant block.
      name: str
          Name used to tag parameters.
      W_prior_fn: callable
          A function that, given a shape, returns a NumPyro distribution for W.
      use_diag: bool
          Whether to sample and apply a Bernoulli diagonal to the input. (Default: True)
      bias_prior_fn: callable
          A function that, given a shape, returns a NumPyro distribution for the bias.
          (Default: Normal(0,1))
    """

    def __init__(
        self,
        in_features,
        out_features,
        block_size,
        name="block_circ_layer",
        W_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape).to_event(len(shape)),
        use_diag=True,
        bias_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape).to_event(1),
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.name = name

        # Calculate the number of blocks along the input and output dimensions.
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
        # 1) Sample W with shape (k_out, k_in, block_size)
        W = numpyro.sample(
            f"{self.name}_W",
            self.W_prior_fn([self.k_out, self.k_in, self.block_size]),
        )

        # 2) Sample the Bernoulli diagonal if enabled; otherwise, set to None.
        if self.use_diag:
            d_bernoulli = numpyro.sample(
                f"{self.name}_D",
                self.diag_prior([self.in_features]),
            )
        else:
            d_bernoulli = None

        # 3) Perform the block-circulant multiplication.
        out = _block_circulant_matmul(W, X, d_bernoulli)

        # 4) Sample and add the bias.
        b = numpyro.sample(
            f"{self.name}_bias",
            self.bias_prior_fn([self.out_features]),
        )
        out = out + b[None, :]

        # 5) If the padded output dimension is larger than out_features, slice it.
        k_out_b = self.k_out * self.block_size
        if k_out_b > self.out_features:
            out = out[:, : self.out_features]

        return out


class CirculantProcess:
    """
    NumPyro-based circulant layer that places a frequency-dependent prior on the Fourier coefficients
    and truncates high frequencies (freq >= K).

    The prior distribution for each Fourier coefficient is defined via a callable `prior_fn` that takes
    a scale (or array of scales) and returns a distribution (default: Gaussian).

    If `alpha` is set to None, a prior is placed on it. You can define a custom prior on alpha via
    the `alpha_prior` argument (default: Gamma(2.0, 1.0)).
    """

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,  # if set to None, a prior will be placed on alpha
        alpha_prior=None,  # custom prior for alpha (if alpha is None)
        K: int = None,
        name: str = "smooth_trunc_circ",
        prior_fn=None,  # callable to return a distribution given a scale
    ):
        self.in_features = in_features
        self.alpha = alpha
        # Use a custom prior for alpha if provided; otherwise default to Gamma(2.0, 1.0)
        self.alpha_prior = (
            alpha_prior if alpha_prior is not None else dist.Gamma(2.0, 1.0)
        )
        self.name = name
        self.k_half = in_features // 2 + 1  # number of independent coefficients

        if K is None or K > self.k_half:
            K = self.k_half
        self.K = K

        self.prior_fn = (
            prior_fn
            if prior_fn is not None
            else (lambda scale: dist.Normal(0.0, scale))
        )
        self._last_fft_full = None  # will store the full FFT after forward pass

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        # If alpha is None, sample it from the provided alpha prior.
        alpha = (
            numpyro.sample(f"{self.name}_alpha", self.alpha_prior)
            if self.alpha is None
            else self.alpha
        )
        # Compute frequency indices and frequency-dependent standard deviations.
        freq_indices = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_indices**alpha)

        # Determine active indices: the first K indices are active.
        active_indices = jnp.arange(self.K)  # shape (K,)
        n_active = self.K

        # Sample Fourier coefficients for active frequencies using the provided prior function.
        active_scale = prior_std[active_indices]
        active_real = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(active_scale).expand([n_active]).to_event(1),
        )
        active_imag = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(active_scale).expand([n_active]).to_event(1),
        )

        # Build full coefficient arrays.
        real_full = jnp.zeros((self.k_half,))
        imag_full = jnp.zeros((self.k_half,))
        real_full = real_full.at[active_indices].set(active_real)
        imag_full = imag_full.at[active_indices].set(active_imag)

        # Enforce that the DC component is real.
        imag_full = imag_full.at[0].set(0.0)
        if (self.in_features % 2 == 0) and (self.k_half > 1):
            imag_full = imag_full.at[-1].set(0.0)

        half_complex = real_full + 1j * imag_full

        # Reconstruct the full FFT coefficients via conjugate symmetry.
        if (self.in_features % 2 == 0) and (self.k_half > 1):
            nyquist = half_complex[-1].real[None]
            fft_full = jnp.concatenate(
                [half_complex[:-1], nyquist, jnp.conjugate(half_complex[1:-1])[::-1]]
            )
        else:
            fft_full = jnp.concatenate(
                [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
            )

        self._last_fft_full = jax.lax.stop_gradient(fft_full)

        # Compute output via FFT, multiplication, and inverse FFT.
        if X.ndim == 2:
            X_fft = jnp.fft.fft(X, axis=-1)
            out_fft = X_fft * fft_full[None, :]
            out_time = jnp.fft.ifft(out_fft, axis=-1).real
        else:
            X_fft = jnp.fft.fft(X)
            out_fft = X_fft * fft_full
            out_time = jnp.fft.ifft(out_fft).real

        return out_time

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self._last_fft_full is None:
            raise ValueError(
                "No Fourier coefficients available. Call the layer on some input first."
            )
        return self._last_fft_full


class BlockCirculantProcess:
    """
    NumPyro-based block-circulant layer. Each b x b block is parameterized by a half-spectrum
    with frequency-dependent prior scale and optional truncation. Vectorized for faster sampling.

    A custom prior on the Fourier coefficients can be specified via `prior_fn` (default: Gaussian).
    If `alpha` is None, a prior is placed on it. You can define a custom prior on alpha via the
    `alpha_prior` argument (default: Gamma(2.0, 1.0)).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int,
        alpha: float = 1.0,  # if set to None, a prior will be placed on alpha
        alpha_prior=None,  # custom prior for alpha (if alpha is None)
        K: int = None,
        name: str = "smooth_trunc_block_circ",
        prior_fn=None,  # callable to return a distribution given a scale
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
        self._last_block_fft = None  # will store the full block FFT

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

        # Zero-pad and reshape X into blocks.
        pad_len = self.k_in * self.b - d_in
        if pad_len > 0:
            X = jnp.pad(X, ((0, 0), (0, pad_len)))
        X_blocks = X.reshape(bs, self.k_in, self.b)

        # Multiply in the time domain over the blocks.
        def multiply_blockrow(i):
            def scan_j(carry, j):
                w_ij = block_fft_full[i, j]  # shape (b,)
                x_j = X_blocks[:, j, :]  # shape (bs, b)
                X_fft = jnp.fft.fft(x_j, axis=-1)
                out_fft = X_fft * jnp.conjugate(w_ij)[None, :]
                out_time = jnp.fft.ifft(out_fft, axis=-1).real
                return carry + out_time, None

            init = jnp.zeros((bs, self.b))
            out_time, _ = jax.lax.scan(scan_j, init, jnp.arange(self.k_in))
            return out_time

        out_blocks = jax.vmap(multiply_blockrow)(jnp.arange(self.k_out))
        out_reshaped = jnp.transpose(out_blocks, (1, 0, 2)).reshape(
            bs, self.k_out * self.b
        )
        if self.k_out * self.b > self.out_features:
            out_reshaped = out_reshaped[:, : self.out_features]
        if X.shape[0] == 1 and bs == 1:
            out_reshaped = out_reshaped[0]
        return out_reshaped

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self._last_block_fft is None:
            raise ValueError("No Fourier coefficients yet. Call the layer first.")
        return self._last_block_fft


class MixtureOfTwoLayers:
    """
    Combine outputs from two sub-layers via gating:
      out = gate * outA + (1 - gate) * outB
    gate can be:
      (a) a param in [0,1],
      (b) a sample from Beta distribution,
      (c) an MLP output for data-dependent gating,
      (d) a per-feature gating vector, etc.
    """

    def __init__(self, layerA, layerB, name="mixture_of_experts", gating_mode="param"):
        """
        :param layerA, layerB: callables, each takes X -> Y
        :param gating_mode: "param", "beta", or "mlp" for demonstration
        """
        self.layerA = layerA
        self.layerB = layerB
        self.name = name
        self.gating_mode = gating_mode

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        outA = self.layerA(X)
        outB = self.layerB(X)
        # Suppose outA, outB have shape (batch_size, d_out).
        if self.gating_mode == "param":
            # learn a single global gate in [0,1]
            logit_gate = numpyro.param(f"{self.name}_logit_gate", 0.0)  # scalar
            gate = jax.nn.sigmoid(logit_gate)
            # broadcast
            return gate * outA + (1.0 - gate) * outB
        elif self.gating_mode == "beta":
            # sample a random gate from Beta
            alpha0 = numpyro.param(
                f"{self.name}_alpha0", 2.0, constraint=dist.constraints.positive
            )
            beta0 = numpyro.param(
                f"{self.name}_beta0", 2.0, constraint=dist.constraints.positive
            )
            gate_sample = numpyro.sample(f"{self.name}_gate", dist.Beta(alpha0, beta0))
            return gate_sample * outA + (1.0 - gate_sample) * outB
        elif self.gating_mode == "mlp":
            # data-dependent gate (per example). We'll do a single-layer net from X to a scalar gate:
            d_in = X.shape[-1]
            w = numpyro.sample(
                f"{self.name}_gate_w", dist.Normal(0, 1).expand([d_in, 1])
            )
            b = numpyro.sample(f"{self.name}_gate_b", dist.Normal(0, 1).expand([1]))
            # shape => (batch_size, 1)
            logit = jnp.dot(X, w) + b
            gate = jax.nn.sigmoid(logit)  # shape (batch_size, 1)

            return gate * outA + (1.0 - gate) * outB
        else:
            raise ValueError(f"Unrecognized gating_mode={self.gating_mode}")


class DeepKernelCirc:
    """
    NumPyro-based deep kernel layer that first embeds the input via a learnable transform
    (e.g. an MLP) and then applies a circulant spectral transformation. The resulting kernel is
    k(ϕ(x), ϕ(x')), where the circulant part defines a stationary kernel in the index dimension.

    The spectral part places a frequency-dependent prior on the Fourier coefficients and truncates
    high frequencies (freq >= K). The prior for each Fourier coefficient is defined via a callable
    `prior_fn` that takes a scale (or array of scales) and returns a distribution (default: Gaussian).

    If `alpha` is set to None, a prior is placed on it (default prior: Gamma(2.0, 1.0)).

    The embedding is provided via the callable `phi_fn` (e.g. an MLP). If not provided, it defaults
    to the identity function.
    """

    def __init__(
        self,
        in_features: int,
        phi_fn=None,  # callable embedding function (e.g. an MLP). Defaults to identity.
        alpha: float = 1.0,  # if set to None, a prior will be placed on alpha
        alpha_prior=None,  # custom prior for alpha (if alpha is None)
        K: int = None,
        name: str = "deep_kernel_smooth_trunc_circ",
        prior_fn=None,  # callable to return a distribution given a scale (default: Normal)
    ):
        self.in_features = in_features
        # Embedding function (ϕ). If None, we use identity (i.e. no embedding).
        self.phi_fn = phi_fn if phi_fn is not None else (lambda x: x)
        self.alpha = alpha
        self.alpha_prior = (
            alpha_prior if alpha_prior is not None else dist.Gamma(2.0, 1.0)
        )
        self.name = name

        # The Fourier coefficients are defined for a half spectrum.
        self.k_half = in_features // 2 + 1
        if K is None or K > self.k_half:
            K = self.k_half
        self.K = K

        self.prior_fn = (
            prior_fn
            if prior_fn is not None
            else (lambda scale: dist.Normal(0.0, scale))
        )
        self._last_fft_full = None  # store the full FFT coefficients after forward pass

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the embedding transform ϕ to the input X, then performs the spectral (FFT-based)
        transformation with learned Fourier coefficients.
        """
        # First, transform the input using the provided phi_fn (e.g. an MLP).
        embedded_X = self.phi_fn(X)

        # Sample or use fixed alpha.
        alpha = (
            numpyro.sample(f"{self.name}_alpha", self.alpha_prior)
            if self.alpha is None
            else self.alpha
        )
        # Compute frequency indices and frequency-dependent standard deviations.
        freq_indices = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_indices**alpha)

        # Only use the first K frequencies.
        active_indices = jnp.arange(self.K)
        n_active = self.K

        # Sample Fourier coefficients for active frequencies.
        active_scale = prior_std[active_indices]
        active_real = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(active_scale).expand([n_active]).to_event(1),
        )
        active_imag = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(active_scale).expand([n_active]).to_event(1),
        )

        # Build full coefficient arrays.
        real_full = jnp.zeros((self.k_half,))
        imag_full = jnp.zeros((self.k_half,))
        real_full = real_full.at[active_indices].set(active_real)
        imag_full = imag_full.at[active_indices].set(active_imag)

        # Enforce that the DC component is real.
        imag_full = imag_full.at[0].set(0.0)
        if (self.in_features % 2 == 0) and (self.k_half > 1):
            imag_full = imag_full.at[-1].set(0.0)

        half_complex = real_full + 1j * imag_full

        # Reconstruct the full FFT coefficients using conjugate symmetry.
        if (self.in_features % 2 == 0) and (self.k_half > 1):
            nyquist = half_complex[-1].real[None]
            fft_full = jnp.concatenate(
                [half_complex[:-1], nyquist, jnp.conjugate(half_complex[1:-1])[::-1]]
            )
        else:
            fft_full = jnp.concatenate(
                [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
            )

        self._last_fft_full = jax.lax.stop_gradient(fft_full)

        # Compute output by applying FFT to the embedded input, modulating by the spectral coefficients,
        # and then taking the inverse FFT.
        if embedded_X.ndim == 2:
            X_fft = jnp.fft.fft(embedded_X, axis=-1)
            out_fft = X_fft * fft_full[None, :]
            out_time = jnp.fft.ifft(out_fft, axis=-1).real
        else:
            X_fft = jnp.fft.fft(embedded_X)
            out_fft = X_fft * fft_full
            out_time = jnp.fft.ifft(out_fft).real

        return out_time

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self._last_fft_full is None:
            raise ValueError(
                "No Fourier coefficients available. Call the layer on some input first."
            )
        return self._last_fft_full


class DeepKerneBlockCirc:
    """
    NumPyro-based deep kernel block-circulant layer. This layer first embeds the input using
    a learnable transform ϕ (e.g. an MLP) and then applies a block circulant spectral transformation.
    The effective kernel is given by k(ϕ(x), ϕ(x')), where the block circulant part defines a
    stationary kernel in the index (block) dimension.

    The block circulant part places a frequency-dependent prior on the Fourier coefficients
    for each b x b block and truncates high frequencies (freq >= K). The Fourier coefficient prior is
    defined via a callable `prior_fn` that takes a scale and returns a distribution (default: Gaussian).

    If `alpha` is set to None, a prior is placed on it (default prior: Gamma(2.0, 1.0)).

    Parameters:
      - in_features: int, number of input features.
      - out_features: int, number of output features.
      - block_size: int, size of each block (b).
      - phi_fn: callable, embedding function (e.g. an MLP) applied to the input. Defaults to identity.
      - alpha: float or None, if None a prior is placed on alpha.
      - alpha_prior: distribution, prior for alpha if learning it.
      - K: int, number of active frequencies (truncation); if None, defaults to maximum (b//2 + 1).
      - name: str, name used to label NumPyro sample sites.
      - prior_fn: callable, returns a distribution given a scale (default: Normal).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int,
        phi_fn=None,
        alpha: float = 1.0,  # if set to None, a prior will be placed on alpha
        alpha_prior=None,  # custom prior for alpha (if alpha is None)
        K: int = None,
        name: str = "deep_kernel_smooth_trunc_block_circ",
        prior_fn=None,  # callable to return a distribution given a scale
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        # Embedding function ϕ: default to identity if not provided.
        self.phi_fn = phi_fn if phi_fn is not None else (lambda x: x)
        self.alpha = alpha
        self.alpha_prior = (
            alpha_prior if alpha_prior is not None else dist.Gamma(2.0, 1.0)
        )
        self.name = name

        # Compute the number of blocks in the input and output dimensions.
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
        self._last_block_fft = (
            None  # will store the full block FFT after a forward pass
        )

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the embedding function ϕ to the input, then performs the block circulant spectral
        transformation with learned Fourier coefficients.
        """
        # First, embed the input via the provided phi_fn.
        embedded_X = self.phi_fn(X)
        if embedded_X.ndim == 1:
            embedded_X = embedded_X[None, :]
        bs, d_in = embedded_X.shape

        # If alpha is None, sample it using the provided prior.
        alpha = (
            numpyro.sample(f"{self.name}_alpha", self.alpha_prior)
            if self.alpha is None
            else self.alpha
        )
        # Frequency-dependent scale for the Fourier coefficients.
        freq_idx = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_idx**alpha)
        active_indices = jnp.arange(self.K)
        n_active = self.K

        # Sample Fourier coefficients for active frequencies for each block.
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

        # Build full coefficient arrays for each (i,j) block.
        real_coeff = jnp.zeros((self.k_out, self.k_in, self.k_half))
        imag_coeff = jnp.zeros((self.k_out, self.k_in, self.k_half))
        real_coeff = real_coeff.at[..., active_indices].set(active_real)
        imag_coeff = imag_coeff.at[..., active_indices].set(active_imag)

        # Enforce the DC component to be real.
        imag_coeff = imag_coeff.at[..., 0].set(0.0)
        if (self.b % 2 == 0) and (self.k_half > 1):
            imag_coeff = imag_coeff.at[..., -1].set(0.0)

        # Function to reconstruct the full FFT for one block.
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

        # Reconstruct the full block-level FFT for each block using vectorized mapping.
        block_fft_full = jax.vmap(
            lambda Rrow, Irow: jax.vmap(reconstruct_fft)(Rrow, Irow),
            in_axes=(0, 0),
        )(real_coeff, imag_coeff)
        self._last_block_fft = jax.lax.stop_gradient(block_fft_full)

        # Zero-pad and reshape embedded_X into blocks.
        pad_len = self.k_in * self.b - d_in
        if pad_len > 0:
            embedded_X = jnp.pad(embedded_X, ((0, 0), (0, pad_len)))
        X_blocks = embedded_X.reshape(bs, self.k_in, self.b)

        # Multiply in the time domain over the blocks.
        def multiply_blockrow(i):
            def scan_j(carry, j):
                # Get block-level FFT for block (i, j)
                w_ij = block_fft_full[i, j]  # shape (b,)
                x_j = X_blocks[:, j, :]  # shape (bs, b)
                X_fft = jnp.fft.fft(x_j, axis=-1)
                # Multiply (you may choose conjugation or not depending on desired symmetry)
                out_fft = X_fft * jnp.conjugate(w_ij)[None, :]
                out_time = jnp.fft.ifft(out_fft, axis=-1).real
                return carry + out_time, None

            init = jnp.zeros((bs, self.b))
            out_time, _ = jax.lax.scan(scan_j, init, jnp.arange(self.k_in))
            return out_time

        out_blocks = jax.vmap(multiply_blockrow)(jnp.arange(self.k_out))
        out_reshaped = jnp.transpose(out_blocks, (1, 0, 2)).reshape(
            bs, self.k_out * self.b
        )
        # Trim to out_features if necessary.
        if self.k_out * self.b > self.out_features:
            out_reshaped = out_reshaped[:, : self.out_features]
        # If originally a single sample was provided, remove the batch dimension.
        if X.shape[0] == 1 and bs == 1:
            out_reshaped = out_reshaped[0]
        return out_reshaped

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self._last_block_fft is None:
            raise ValueError("No Fourier coefficients available. Call the layer first.")
        return self._last_block_fft


class FourierNeuralOperator1D:
    """
    A toy 1D Fourier Neural Operator with L "Fourier layers."
    Each layer:
      - transforms the input to the frequency domain via FFT,
      - truncates high frequencies by keeping only the first n_modes (and the symmetric tail),
      - multiplies by a trainable complex mask (implemented via a single real-valued weight vector),
      - performs the inverse FFT,
      - then applies a small pointwise MLP with a residual connection.

    If out_features != in_features, a final linear layer is applied.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        n_modes: int = None,
        L: int = 2,
        hidden_dim: int = 16,
        name: str = "fourier_operator",
    ):
        """
        Parameters:
          in_features: int, the dimension of the input function.
          out_features: int, the output dimension (default: same as input).
          n_modes: int, how many low-frequency modes to keep (default: in_features//2).
          L: int, number of Fourier layers.
          hidden_dim: int, hidden dimension for the pointwise MLP.
          name: str, base name for parameter naming.
        """
        self.in_features = in_features
        self.out_features = out_features if out_features is not None else in_features
        self.n_modes = n_modes if n_modes is not None else in_features // 2
        self.L = L
        self.hidden_dim = hidden_dim
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        if X.ndim == 1:
            X = X[None, :]  # Ensure batched input.
        batch_size, d_in = X.shape
        out = X
        for ell in range(self.L):
            out = self._fourier_layer(out, ell)
        if self.out_features != d_in:
            # Final linear mapping.
            w = numpyro.sample(
                f"{self.name}_final_w",
                dist.Normal(0, 1).expand([d_in, self.out_features]).to_event(2),
            )
            b = numpyro.sample(
                f"{self.name}_final_b",
                dist.Normal(0, 1).expand([self.out_features]).to_event(1),
            )
            out = jnp.dot(out, w) + b
        return out

    def _fourier_layer(self, x: jnp.ndarray, layer_idx: int) -> jnp.ndarray:
        # 1) FFT along last axis.
        X_fft = jnp.fft.fft(x, axis=-1)
        n = x.shape[-1]
        # 2) Sample spectral weight for the full frequency range.
        w_complex = numpyro.sample(
            f"{self.name}_layer{layer_idx}_spectral_weight",
            dist.Normal(0, 1).expand([n]).to_event(1),
        )

        # 3) Build a mask that keeps the first and last n_modes frequencies.
        def make_mask(n, n_modes):
            mask = jnp.zeros((n,))
            mask = mask.at[:n_modes].set(1.0)
            mask = mask.at[-n_modes:].set(1.0)
            return mask

        mask = make_mask(n, self.n_modes)
        spectral_scale = (mask * w_complex)[None, :]  # Broadcast to (1, n)
        X_fft_mod = X_fft * (1.0 + spectral_scale)
        # 4) Inverse FFT.
        x_ifft = jnp.fft.ifft(X_fft_mod, axis=-1).real
        # 5) Apply a small pointwise MLP with a residual connection.
        hidden_w = numpyro.sample(
            f"{self.name}_layer{layer_idx}_pw_w",
            dist.Normal(0, 1).expand([self.in_features, self.hidden_dim]).to_event(2),
        )
        hidden_b = numpyro.sample(
            f"{self.name}_layer{layer_idx}_pw_b",
            dist.Normal(0, 1).expand([self.hidden_dim]).to_event(1),
        )
        out_w = numpyro.sample(
            f"{self.name}_layer{layer_idx}_pw_out_w",
            dist.Normal(0, 1).expand([self.hidden_dim, self.in_features]).to_event(2),
        )
        out_b = numpyro.sample(
            f"{self.name}_layer{layer_idx}_pw_out_b",
            dist.Normal(0, 1).expand([self.in_features]).to_event(1),
        )
        h = jax.nn.relu(jnp.dot(x_ifft, hidden_w) + hidden_b)
        x_mlp = jnp.dot(h, out_w) + out_b
        # Residual connection.
        return x_ifft + x_mlp


class SpectralDenseBlock:
    """
    A block that performs:
      1) FFT on the input,
      2) Multiplication by a trainable complex mask,
      3) Inverse FFT,
      4) A pointwise MLP that maps from in_features to out_features,
      5) And adds a residual connection (with projection if needed).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        out_features: int = None,
        name: str = "spectral_dense_block",
    ):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        # Default output dimension is the same as input if not provided.
        self.out_features = out_features if out_features is not None else in_features
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        # Ensure X is at least 2D.
        if X.ndim == 1:
            X = X[None, :]
        batch_size, d_in = X.shape

        # 1) FFT of the input.
        X_fft = jnp.fft.fft(X, axis=-1)

        # 2) Sample and construct a trainable Fourier mask.
        w_real = numpyro.sample(
            f"{self.name}_fft_w_real", dist.Normal(0, 1).expand([d_in]).to_event(1)
        )
        w_imag = numpyro.sample(
            f"{self.name}_fft_w_imag", dist.Normal(0, 1).expand([d_in]).to_event(1)
        )
        mask_complex = w_real + 1j * w_imag
        out_fft = X_fft * mask_complex[None, :]

        # 3) Inverse FFT to go back to the time domain.
        x_time = jnp.fft.ifft(out_fft, axis=-1).real

        # 4) Apply a pointwise MLP.
        # First linear transformation: in_features -> hidden_dim.
        w1 = numpyro.sample(
            f"{self.name}_w1",
            dist.Normal(0, 1).expand([d_in, self.hidden_dim]).to_event(2),
        )
        b1 = numpyro.sample(
            f"{self.name}_b1", dist.Normal(0, 1).expand([self.hidden_dim]).to_event(1)
        )
        h = jax.nn.relu(jnp.dot(x_time, w1) + b1)
        # Second linear transformation: hidden_dim -> out_features.
        w2 = numpyro.sample(
            f"{self.name}_w2",
            dist.Normal(0, 1).expand([self.hidden_dim, self.out_features]).to_event(2),
        )
        b2 = numpyro.sample(
            f"{self.name}_b2", dist.Normal(0, 1).expand([self.out_features]).to_event(1)
        )
        x_dense = jnp.dot(h, w2) + b2

        # 5) Residual connection.
        # If the dimensions differ, project x_time to match out_features.
        if d_in != self.out_features:
            w_proj = numpyro.sample(
                f"{self.name}_w_proj",
                dist.Normal(0, 1).expand([d_in, self.out_features]).to_event(2),
            )
            b_proj = numpyro.sample(
                f"{self.name}_b_proj",
                dist.Normal(0, 1).expand([self.out_features]).to_event(1),
            )
            shortcut = jnp.dot(x_time, w_proj) + b_proj
        else:
            shortcut = x_time

        return shortcut + x_dense


class ParticleLinear:
    """
    A particle-aware fully connected layer.

    Applies linear transformations to inputs for each particle, then aggregates
    the outputs to yield a (batch_size, out_features) output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        name: str = "particle_layer",
        aggregation: str = "mean",
        prior=lambda shape: dist.Normal(0, 1).expand(shape).to_event(len(shape)),
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.aggregation = aggregation
        self.prior = prior

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the particle-aware transformation.

        Expects X of shape (particles, batch_size, in_features).
        If the particle dimension is missing (i.e. X has shape (batch_size, in_features)),
        a singleton particle dimension is added.
        """
        if X.ndim == 2:
            X = X[jnp.newaxis, ...]  # Shape: (1, batch_size, in_features)

        particles = X.shape[0]
        # Sample weights and biases with full event treatment using the provided prior.
        w = numpyro.sample(
            f"{self.name}_w",
            self.prior([particles, self.in_features, self.out_features]),
        )
        b = numpyro.sample(
            f"{self.name}_b",
            self.prior([particles, self.out_features]),
        )

        # Compute output for each particle.
        particle_outputs = jnp.einsum("pbi,pij->pbj", X, w) + b

        # Aggregate across particles.
        if self.aggregation == "mean":
            aggregated_output = jnp.mean(particle_outputs, axis=0)
        elif self.aggregation == "sum":
            aggregated_output = jnp.sum(particle_outputs, axis=0)
        elif self.aggregation == "max":
            aggregated_output = jnp.max(particle_outputs, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

        return aggregated_output


class FFTParticleLinear:
    """
    FFT-based particle-aware linear layer.

    For each particle, samples a circulant matrix parameterized by its first row
    and a bias vector. Applies FFT-based multiplication and then aggregates
    across particles.
    """

    def __init__(
        self,
        in_features: int,
        name: str = "fft_particle_layer",
        aggregation: str = "mean",
        prior=lambda shape: dist.Normal(0, 1).expand(shape).to_event(len(shape)),
    ):
        self.in_features = in_features
        self.name = name
        self.aggregation = aggregation
        self.prior = prior

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the FFT-based particle-aware transformation.

        Expects X of shape (particles, batch_size, in_features). If missing the
        particle dimension, it is added.
        Returns an output of shape (batch_size, in_features).
        """
        if X.ndim == 2:
            X = X[jnp.newaxis, ...]  # Shape: (1, batch_size, in_features)

        particles = X.shape[0]
        # Sample first rows and biases for each particle using the provided prior.
        first_rows = numpyro.sample(
            f"{self.name}_first_rows",
            self.prior([particles, self.in_features]),
        )
        biases = numpyro.sample(
            f"{self.name}_biases",
            self.prior([particles, self.in_features]),
        )

        def fft_particle_transform(p_idx):
            first_row = first_rows[p_idx]  # (in_features,)
            bias = biases[p_idx]  # (in_features,)
            transformed = _fft_matmul(first_row, X[p_idx])  # (batch_size, in_features)
            return transformed + bias

        particle_outputs = jax.vmap(fft_particle_transform)(jnp.arange(particles))

        # Aggregate across particles.
        if self.aggregation == "mean":
            aggregated_output = jnp.mean(particle_outputs, axis=0)
        elif self.aggregation == "sum":
            aggregated_output = jnp.sum(particle_outputs, axis=0)
        elif self.aggregation == "max":
            aggregated_output = jnp.max(particle_outputs, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

        return aggregated_output


class SelfAttention:
    """
    Implements a single-head self-attention mechanism with learnable weights.

    Self-attention computes attention scores for a sequence, enabling the model to
    focus on different parts of the sequence for each position.
    """

    def __init__(self, embed_dim: int, name: str = "self_attention"):
        """
        Initialize the SelfAttention layer.

        :param embed_dim: int
            Dimensionality of the embedding space.
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "self_attention").
        """
        self.embed_dim = embed_dim
        self.name = name

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Perform the forward pass of the self-attention layer.

        :param query: jnp.ndarray
            Query tensor of shape `(batch_size, seq_len, embed_dim)`.
        :param key: jnp.ndarray
            Key tensor of shape `(batch_size, seq_len, embed_dim)`.
        :param value: jnp.ndarray
            Value tensor of shape `(batch_size, seq_len, embed_dim)`.
        :param mask: jnp.ndarray, optional
            Attention mask of shape `(batch_size, seq_len, seq_len)`, where 1 indicates
            valid positions and 0 indicates masked positions (default: None).

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, seq_len, embed_dim)`.
        """
        d_k = self.embed_dim

        w_q = numpyro.sample(
            f"{self.name}_w_q",
            dist.Normal(0, 1).expand([self.embed_dim, self.embed_dim]),
        )
        w_k = numpyro.sample(
            f"{self.name}_w_k",
            dist.Normal(0, 1).expand([self.embed_dim, self.embed_dim]),
        )
        w_v = numpyro.sample(
            f"{self.name}_w_v",
            dist.Normal(0, 1).expand([self.embed_dim, self.embed_dim]),
        )

        q = jnp.dot(query, w_q)
        k = jnp.dot(key, w_k)
        v = jnp.dot(value, w_v)

        scores = jnp.matmul(q, k.transpose(0, 2, 1)) / jnp.sqrt(d_k)
        if mask is not None:
            scores = scores - 1e9 * (1 - mask)

        attention_weights = jax.nn.softmax(scores, axis=-1)
        output = jnp.matmul(attention_weights, v)

        return output


class MultiHeadSelfAttention:
    """
    Implements a multi-head self-attention mechanism with learnable weights.

    Multi-head self-attention splits the embedding space into multiple attention heads
    to capture diverse relationships within the input sequence.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, name: str = "multihead_self_attention"
    ):
        """
        Initialize the MultiHeadSelfAttention layer.

        :param embed_dim: int
            Dimensionality of the embedding space.
        :param num_heads: int
            Number of attention heads. Must divide `embed_dim` evenly.
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "multihead_self_attention").
        """
        assert (
            embed_dim % num_heads == 0
        ), "Embed dim must be divisible by the number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.name = name

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Perform the forward pass of the multi-head self-attention layer.

        :param query: jnp.ndarray
            Query tensor of shape `(batch_size, seq_len, embed_dim)`.
        :param key: jnp.ndarray
            Key tensor of shape `(batch_size, seq_len, embed_dim)`.
        :param value: jnp.ndarray
            Value tensor of shape `(batch_size, seq_len, embed_dim)`.
        :param mask: jnp.ndarray, optional
            Attention mask of shape `(batch_size, seq_len, seq_len)`, where 1 indicates
            valid positions and 0 indicates masked positions (default: None).

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, seq_len, embed_dim)`.
        """
        batch_size, seq_len, _ = query.shape

        def project(input_tensor, suffix):
            w = numpyro.sample(
                f"{self.name}_w_{suffix}",
                dist.Normal(0, 1).expand([self.embed_dim, self.embed_dim]),
            )
            return jnp.dot(input_tensor, w)

        q = project(query, "q").reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = project(key, "k").reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v = project(value, "v").reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        q, k, v = map(lambda x: x.transpose(0, 2, 1, 3), (q, k, v))

        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        if mask is not None:
            scores = scores - 1e9 * (1 - mask[:, None, :, :])

        attention_weights = jax.nn.softmax(scores, axis=-1)
        attention_output = jnp.matmul(attention_weights, v)

        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.embed_dim
        )

        w_o = numpyro.sample(
            f"{self.name}_w_o",
            dist.Normal(0, 1).expand([self.embed_dim, self.embed_dim]),
        )
        return jnp.dot(attention_output, w_o)


class PositionalEncoding:
    """
    Implements learnable positional encodings for input sequences.

    Positional encodings are added to the input embeddings to inject information
    about the relative and absolute positions of tokens in a sequence.
    """

    def __init__(self, seq_len: int, embed_dim: int, name: str = "positional_encoding"):
        """
        Initialize the PositionalEncoding layer.

        :param seq_len: int
            The length of the input sequences.
        :param embed_dim: int
            The dimensionality of the embeddings.
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "positional_encoding").
        """
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.name = name

        # Learnable positional embeddings
        self.positional_embeddings = numpyro.param(
            f"{self.name}_positional_embeddings",
            jax.random.normal(jax.random.PRNGKey(0), (seq_len, embed_dim)),
        )

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Add positional encodings to the input tensor.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, seq_len, embed_dim)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, seq_len, embed_dim)` with positional encodings added.

        :raises AssertionError:
            If the sequence length of the input tensor does not match the initialized `seq_len`.
        """
        assert (
            X.shape[1] == self.seq_len
        ), f"Input sequence length {X.shape[1]} does not match initialized sequence length {self.seq_len}."
        return X + self.positional_embeddings


class TransformerEncoder:
    """
    Implements a single transformer encoder block.

    A transformer encoder block consists of a multi-head self-attention mechanism,
    followed by a feedforward network with residual connections and layer normalization.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        name: str = "transformer_encoder",
    ):
        """
        Initialize the TransformerEncoder block.

        :param embed_dim: int
            Dimensionality of the input embeddings.
        :param num_heads: int
            Number of attention heads for the self-attention mechanism.
        :param hidden_dim: int
            Dimensionality of the hidden layer in the feedforward network.
        :param name: str, optional
            Name of the block, used for parameter naming (default: "transformer_encoder").
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.name = name

        self.self_attention = MultiHeadSelfAttention(
            embed_dim, num_heads, name=f"{name}_self_attention"
        )
        self.layer_norm1 = LayerNorm(embed_dim, name=f"{name}_layer_norm1")
        self.feedforward = Linear(embed_dim, hidden_dim, name=f"{name}_feedforward")
        self.out = Linear(hidden_dim, embed_dim, name=f"{name}_out")
        self.layer_norm2 = LayerNorm(embed_dim, name=f"{name}_layer_norm2")

    def __call__(self, X: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
        """
        Perform the forward pass of the TransformerEncoder block.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, seq_len, embed_dim)`.
        :param mask: jnp.ndarray, optional
            Attention mask of shape `(batch_size, seq_len, seq_len)`, where 1 indicates
            valid positions and 0 indicates masked positions (default: None).

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, seq_len, embed_dim)`.
        """
        attention_output = self.self_attention(X, X, X, mask)
        X = self.layer_norm1(attention_output + X)

        ff_output = jax.nn.relu(self.feedforward(X))
        ff_output = self.out(ff_output)

        X = self.layer_norm2(ff_output + X)
        return X


class Conv1d:
    """
    Implements a 1D convolutional layer with learnable weights and biases.

    This layer performs a convolution operation along the temporal (or spatial) dimension
    of the input, with support for custom stride, padding, and kernel size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = "valid",
        name: str = "conv1d",
    ):
        """
        Initialize the Conv1d layer.

        :param in_channels: int
            Number of input channels.
        :param out_channels: int
            Number of output channels.
        :param kernel_size: int
            Size of the convolutional kernel.
        :param stride: int, optional
            Stride of the convolution (default: 1).
        :param padding: str, optional
            Padding type, either "valid" (no padding) or "same" (zero-padding, maintains size) (default: "valid").
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "conv1d").
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding.upper()  # Convert to uppercase for JAX compatibility
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the Conv1d layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, in_channels, input_length)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, out_channels, output_length)`, where `output_length`
            depends on the padding and stride configuration.
        """

        weight = numpyro.sample(
            f"{self.name}_weight",
            dist.Normal(0, 1).expand(
                [self.out_channels, self.in_channels, self.kernel_size]
            ),
        )
        bias = numpyro.sample(
            f"{self.name}_bias", dist.Normal(0, 1).expand([self.out_channels])
        )

        convolved = jax.lax.conv_general_dilated(
            X,
            weight,
            window_strides=(self.stride,),
            padding=self.padding,
            dimension_numbers=("NCH", "OIH", "NCH"),
        )

        convolved += bias[:, None]

        return convolved


class Conv2d:
    """
    Implements a 2D convolutional layer with learnable weights and biases.

    This layer performs a 2D convolution operation over spatial dimensions
    (height and width) of the input tensor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: str = "valid",
        name: str = "conv2d",
    ):
        """
        Initialize the Conv2d layer.

        :param in_channels: int
            Number of input channels.
        :param out_channels: int
            Number of output channels.
        :param kernel_size: int or tuple
            Size of the convolutional kernel, either as an integer or a tuple (kernel_h, kernel_w).
        :param stride: int or tuple, optional
            Stride of the convolution, either as an integer or a tuple (stride_h, stride_w) (default: 1).
        :param padding: str, optional
            Padding mode, either "valid" (no padding) or "same" (maintains spatial dimensions) (default: "valid").
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "conv2d").
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the Conv2d layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, in_channels, height, width)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, out_channels, new_height, new_width)`, where:
            - `new_height` and `new_width` depend on the padding, kernel size, and stride.

        :raises ValueError:
            If an unsupported padding mode is provided.
        """
        batch_size, in_channels, input_h, input_w = X.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        # Determine padding
        if self.padding == "same":
            padding = "SAME"
        elif self.padding == "valid":
            padding = "VALID"
        else:
            raise ValueError(f"Unsupported padding mode: {self.padding}")

        weight = numpyro.sample(
            f"{self.name}_weight",
            dist.Normal(0, 1).expand(
                [self.out_channels, self.in_channels, kernel_h, kernel_w]
            ),
        )
        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0, 1).expand([self.out_channels]),
        )

        convolved = jax.lax.conv_general_dilated(
            X,
            weight,
            window_strides=(stride_h, stride_w),
            padding=padding,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

        bias = bias[:, None, None]
        convolved += bias
        return convolved


class FFTConv1d:
    """
    Implements a 1D convolutional layer using FFT-based computation.

    This layer performs a 1D convolution in the frequency domain for efficiency,
    particularly with large kernel sizes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        name: str = "fft_conv1d",
    ):
        """
        Initialize the FFTConv1d layer.

        :param in_channels: int
            Number of input channels.
        :param out_channels: int
            Number of output channels.
        :param kernel_size: int
            Size of the convolutional kernel.
        :param stride: int, optional
            Stride of the convolution (default: 1).
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "fft_conv1d").
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the FFTConv1d layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, in_channels, width)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, out_channels, new_width)`, where:
            - `new_width = floor((width + kernel_size - 1) / stride)`.
        """
        batch_size, in_channels, width = X.shape

        weight = numpyro.sample(
            f"{self.name}_weight",
            dist.Normal(0, 1).expand(
                [self.out_channels, self.in_channels, self.kernel_size]
            ),
        )
        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0, 1).expand([self.out_channels]),
        )

        output = []
        for out_c in range(self.out_channels):
            channel_out = 0
            for in_c in range(self.in_channels):
                X_fft = jnp.fft.fft(X[:, in_c], n=width + self.kernel_size - 1, axis=-1)
                weight_fft = jnp.fft.fft(
                    weight[out_c, in_c], n=width + self.kernel_size - 1, axis=-1
                )

                conv_fft = X_fft * weight_fft

                channel_out += jnp.fft.ifft(conv_fft, axis=-1).real

            channel_out += bias[out_c]
            output.append(channel_out)

        output = jnp.stack(output, axis=1)

        if self.stride > 1:
            output = output[:, :, :: self.stride]

        return output


class FFTConv2d:
    """
    Implements a 2D convolutional layer using FFT-based computation.

    This layer performs a 2D convolution in the frequency domain, which is
    efficient for large kernels and input sizes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        padding: str = "same",  # Added padding parameter
        name: str = "fft_conv2d",
    ):
        """
        Initialize the FFTConv2d layer.

        :param in_channels: int
            Number of input channels.
        :param out_channels: int
            Number of output channels.
        :param kernel_size: int or tuple
            Size of the convolutional kernel, either as an integer or a tuple (kernel_h, kernel_w).
        :param padding: str, optional
            Padding mode, either "same" or "valid" (default: "same").
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "fft_conv2d").
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        if padding not in ("same", "valid"):
            raise ValueError(f"Unsupported padding mode: {padding}")
        self.padding = padding
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the FFTConv2d layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, in_channels, height, width)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, out_channels, new_height, new_width)`, where:
            - For "same" padding: `new_height = height`, `new_width = width`
            - For "valid" padding: `new_height = height - kernel_h + 1`, `new_width = width - kernel_w + 1`
        """
        batch_size, in_channels, height, width = X.shape
        kernel_h, kernel_w = self.kernel_size

        # Determine padding sizes
        if self.padding == "same":
            pad_h = kernel_h // 2
            pad_w = kernel_w // 2
            # Apply padding to height and width
            X_padded = jnp.pad(
                X,
                pad_width=((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
                constant_values=0,
            )
        elif self.padding == "valid":
            X_padded = X
        else:
            raise ValueError(f"Unsupported padding mode: {self.padding}")

        # Update dimensions after padding
        _, _, H_padded, W_padded = X_padded.shape

        # Prepare weight and bias
        weight = numpyro.sample(
            f"{self.name}_weight",
            dist.Normal(0, 1).expand(
                [self.out_channels, self.in_channels, kernel_h, kernel_w]
            ),
        )
        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0, 1).expand([self.out_channels]),
        )

        # Determine FFT size
        out_h = H_padded + kernel_h - 1
        out_w = W_padded + kernel_w - 1

        output = []
        for out_c in range(self.out_channels):
            channel_out = 0
            for in_c in range(self.in_channels):
                # FFT of input
                X_fft = jnp.fft.fft2(X_padded[:, in_c], s=(out_h, out_w))
                # FFT of weight
                weight_fft = jnp.fft.fft2(weight[out_c, in_c], s=(out_h, out_w))

                # Element-wise multiplication in frequency domain
                conv_fft = X_fft * weight_fft

                # Inverse FFT to get the convolved output
                conv = jnp.fft.ifft2(conv_fft).real

                # Crop to desired size
                if self.padding == "same":
                    # To get the same size as input, center crop
                    start_h = (conv.shape[1] - height) // 2
                    start_w = (conv.shape[2] - width) // 2
                    conv_cropped = conv[
                        :, start_h : start_h + height, start_w : start_w + width
                    ]
                elif self.padding == "valid":
                    conv_cropped = conv[:, kernel_h - 1 : height, kernel_w - 1 : width]
                else:
                    raise ValueError(f"Unsupported padding mode: {self.padding}")

                channel_out += conv_cropped

            # Add bias
            channel_out += bias[out_c].reshape(1, 1, 1)

            output.append(channel_out)

        # Stack all output channels
        output = jnp.stack(output, axis=1)  # Shape: (batch_size, out_channels, H, W)

        return output


class TransposedConv2d:
    """
    Implements a 2D transposed convolutional layer with learnable weights and biases.
    This layer is commonly used for upsampling in neural networks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: str = "valid",
        name: str = "transposed_conv2d",
    ):
        """
        Initialize the TransposedConv2d layer.

        :param in_channels: int
            Number of input channels.
        :param out_channels: int
            Number of output channels.
        :param kernel_size: int or tuple
            Size of the transposed convolutional kernel.
        :param stride: int or tuple, optional
            Stride of the transposed convolution (default: 1).
        :param padding: str, optional
            Padding type, either "valid" or "same" (default: "valid").
        :param name: str, optional
            Name of the layer (default: "transposed_conv2d").
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the TransposedConv2d layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, in_channels, height, width)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, out_channels, new_height, new_width)`.
        """
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        # Determine padding
        if self.padding == "same":
            padding = "SAME"
        elif self.padding == "valid":
            padding = "VALID"
        else:
            raise ValueError(f"Unsupported padding mode: {self.padding}")

        weight = numpyro.sample(
            f"{self.name}_weight",
            dist.Normal(0, 1).expand(
                [self.in_channels, self.out_channels, kernel_h, kernel_w]
            ),
        )
        bias = numpyro.sample(
            f"{self.name}_bias", dist.Normal(0, 1).expand([self.out_channels])
        )

        convolved = jax.lax.conv_transpose(
            lhs=X,
            rhs=weight,
            strides=(stride_h, stride_w),
            padding=padding,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

        convolved += bias[None, :, None, None]  # Add bias per channel
        return convolved


class FFTTransposedConv2d:
    """
    Implements a 2D transposed convolutional layer using FFT-based computation.

    This layer performs a 2D transposed convolution in the frequency domain
    for efficiency with large kernel sizes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,  # Add stride for compatibility
        padding: str = "same",  # Added padding parameter
        name: str = "fft_transposed_conv2d",
    ):
        """
        Initialize the FFTTransposedConv2d layer.

        :param in_channels: int
            Number of input channels.
        :param out_channels: int
            Number of output channels.
        :param kernel_size: int or tuple
            Size of the transposed convolutional kernel.
        :param stride: int or tuple, optional
            Stride of the transposed convolution (default: 1).
        :param padding: str, optional
            Padding mode, either "same" or "valid" (default: "same").
        :param name: str, optional
            Name of the layer (default: "fft_transposed_conv2d").
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        if padding not in ("same", "valid"):
            raise ValueError(f"Unsupported padding mode: {padding}")
        self.padding = padding
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the FFTTransposedConv2d layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, in_channels, height, width)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, out_channels, new_height, new_width)`.
        """
        batch_size, in_channels, height, width = X.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        # Prepare weight and bias
        weight = numpyro.sample(
            f"{self.name}_weight",
            dist.Normal(0, 1).expand(
                [self.in_channels, self.out_channels, kernel_h, kernel_w]
            ),
        )
        bias = numpyro.sample(
            f"{self.name}_bias", dist.Normal(0, 1).expand([self.out_channels])
        )

        # Calculate output spatial dimensions
        out_h = height * stride_h
        out_w = width * stride_w

        # Apply stride-based upsampling if stride > 1
        if self.stride != (1, 1):
            # Initialize a zero array with upsampled spatial dimensions
            upsampled = jnp.zeros((batch_size, in_channels, out_h, out_w))
            # Assign input values to the upsampled array with the given stride
            upsampled = upsampled.at[:, :, ::stride_h, ::stride_w].set(X)
        else:
            upsampled = X

        # Determine padding sizes for transposed convolution
        if self.padding == "same":
            pad_h = kernel_h // 2
            pad_w = kernel_w // 2
            # Apply padding to height and width
            upsampled_padded = jnp.pad(
                upsampled,
                pad_width=((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
                constant_values=0,
            )
        elif self.padding == "valid":
            upsampled_padded = upsampled
        else:
            raise ValueError(f"Unsupported padding mode: {self.padding}")

        # Update dimensions after padding
        _, _, H_padded, W_padded = upsampled_padded.shape

        # Determine FFT size
        conv_h = H_padded + kernel_h - 1
        conv_w = W_padded + kernel_w - 1

        output = []
        for out_c in range(self.out_channels):
            channel_out = 0
            for in_c in range(self.in_channels):
                # FFT of upsampled input
                X_fft = jnp.fft.fft2(upsampled_padded[:, in_c], s=(conv_h, conv_w))
                # FFT of weight (note the transpose for transposed convolution)
                weight_fft = jnp.fft.fft2(weight[in_c, out_c], s=(conv_h, conv_w))

                # Element-wise multiplication in frequency domain
                conv_fft = X_fft * weight_fft

                # Inverse FFT to get the convolved output
                conv = jnp.fft.ifft2(conv_fft).real

                # Crop to desired size
                if self.padding == "same":
                    # To maintain the upsampled size
                    start_h = (conv.shape[1] - out_h) // 2
                    start_w = (conv.shape[2] - out_w) // 2
                    conv_cropped = conv[
                        :, start_h : start_h + out_h, start_w : start_w + out_w
                    ]
                elif self.padding == "valid":
                    # Calculate the amount to crop based on kernel size
                    conv_cropped = conv[:, kernel_h - 1 : out_h, kernel_w - 1 : out_w]
                else:
                    raise ValueError(f"Unsupported padding mode: {self.padding}")

                channel_out += conv_cropped

            # Add bias
            channel_out += bias[out_c].reshape(1, 1, 1)

            output.append(channel_out)

        # Stack all output channels
        output = jnp.stack(
            output, axis=1
        )  # Shape: (batch_size, out_channels, out_h, out_w)

        return output


class LSTM:
    """
    Implements a Long Short-Term Memory (LSTM) layer.

    The LSTM processes sequential data and maintains hidden and cell states
    across time steps for tasks like sequence modeling and time series forecasting.
    """

    def __init__(self, input_dim: int, hidden_dim: int, name: str = "lstm"):
        """
        Initialize the LSTM layer.

        :param input_dim: int
            Dimensionality of the input features.
        :param hidden_dim: int
            Dimensionality of the hidden state and cell state.
        :param name: str, optional
            Name of the LSTM layer, used for parameter naming (default: "lstm").
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.name = name

    def __call__(
        self, X: jnp.ndarray, init_state: tuple | None = None
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Perform the forward pass of the LSTM layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, seq_len, input_dim)`.
        :param init_state: tuple or None, optional
            Initial hidden and cell states as `(h_0, c_0)` of shape `(batch_size, hidden_dim)`.
            If None, states are initialized to zeros (default: None).

        :returns: tuple
            - `outputs`: jnp.ndarray
                Output tensor of shape `(batch_size, seq_len, hidden_dim)` containing
                the hidden states across all time steps.
            - `final_state`: tuple[jnp.ndarray, jnp.ndarray]
                Final hidden state (`h_t`) and cell state (`c_t`), both of shape `(batch_size, hidden_dim)`.
        """
        batch_size, seq_len, _ = X.shape

        # Initialize weights
        Wf = numpyro.sample(
            f"{self.name}_Wf",
            dist.Normal(0, 1).expand(
                [self.input_dim + self.hidden_dim, self.hidden_dim]
            ),
        )
        Wi = numpyro.sample(
            f"{self.name}_Wi",
            dist.Normal(0, 1).expand(
                [self.input_dim + self.hidden_dim, self.hidden_dim]
            ),
        )
        Wc = numpyro.sample(
            f"{self.name}_Wc",
            dist.Normal(0, 1).expand(
                [self.input_dim + self.hidden_dim, self.hidden_dim]
            ),
        )
        Wo = numpyro.sample(
            f"{self.name}_Wo",
            dist.Normal(0, 1).expand(
                [self.input_dim + self.hidden_dim, self.hidden_dim]
            ),
        )
        bf = numpyro.sample(
            f"{self.name}_bf", dist.Normal(0, 1).expand([self.hidden_dim])
        )
        bi = numpyro.sample(
            f"{self.name}_bi", dist.Normal(0, 1).expand([self.hidden_dim])
        )
        bc = numpyro.sample(
            f"{self.name}_bc", dist.Normal(0, 1).expand([self.hidden_dim])
        )
        bo = numpyro.sample(
            f"{self.name}_bo", dist.Normal(0, 1).expand([self.hidden_dim])
        )

        # Initialize hidden and cell states
        if init_state is None:
            h_t = jnp.zeros((batch_size, self.hidden_dim))
            c_t = jnp.zeros((batch_size, self.hidden_dim))
        else:
            h_t, c_t = init_state

        outputs = []

        for t in range(seq_len):
            x_t = X[:, t, :]
            combined = jnp.concatenate([x_t, h_t], axis=-1)
            f_t = jax.nn.sigmoid(jnp.dot(combined, Wf) + bf)
            i_t = jax.nn.sigmoid(jnp.dot(combined, Wi) + bi)
            o_t = jax.nn.sigmoid(jnp.dot(combined, Wo) + bo)
            c_t_candidate = jnp.tanh(jnp.dot(combined, Wc) + bc)
            c_t = f_t * c_t + i_t * c_t_candidate
            h_t = o_t * jnp.tanh(c_t)
            outputs.append(h_t)

        return jnp.stack(outputs, axis=1), (h_t, c_t)


# --- Unified Gaussian Process Layer ---
class GaussianProcessLayer:
    def __init__(
        self,
        input_dim: int,
        kernel_type: str = "rbf",
        name: str = "gp_layer",
        **kernel_kwargs,
    ):
        """
        A unified GP layer that supports multiple kernels.

        Parameters:
          input_dim: int - dimensionality of input features.
          kernel_type: str - one of: "rbf", "spectralmixture", "matern32", "matern52",
                            "periodic", "rq" (rational quadratic), "linear", "poly"
          name: str - parameter name prefix.
          kernel_kwargs: extra parameters passed to the kernel (e.g., Q for spectral mixture, degree for poly, etc.)
        """
        self.input_dim = input_dim
        self.name = name
        self.kernel_type = kernel_type.lower()
        # Choose kernel function
        if self.kernel_type == "rbf":
            self.kernel_fn = self.rbf_kernel
        elif self.kernel_type == "spectralmixture":
            self.kernel_fn = SpectralMixtureKernel(input_dim, **kernel_kwargs)
        elif self.kernel_type == "matern32":
            self.kernel_fn = Matern32Kernel(input_dim, **kernel_kwargs)
        elif self.kernel_type == "matern52":
            self.kernel_fn = Matern52Kernel(input_dim, **kernel_kwargs)
        elif self.kernel_type == "periodic":
            self.kernel_fn = PeriodicKernel(input_dim, **kernel_kwargs)
        elif self.kernel_type == "rq":
            self.kernel_fn = RationalQuadraticKernel(input_dim, **kernel_kwargs)
        elif self.kernel_type == "linear":
            self.kernel_fn = LinearKernel(input_dim, **kernel_kwargs)
        elif self.kernel_type == "poly":
            self.kernel_fn = PolynomialKernel(input_dim, **kernel_kwargs)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        # Compute the kernel matrix using the chosen kernel function.
        K = self.kernel_fn(X, X2)
        # For full covariance (i.e. when X2 is X), add noise.
        if X is X2:
            # Store the noise parameter in the instance.
            self.noise = numpyro.param(
                f"{self.name}_noise",
                jnp.array(1.0),
                constraint=dist.constraints.positive,
            )
            K = K + self.noise * jnp.eye(X.shape[0]) + 1e-6 * jnp.eye(X.shape[0])
        return K

    # --- RBF Kernel Implementation ---
    def rbf_kernel(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        # Retrieve kernel parameters
        length_scale = numpyro.param(
            f"{self.name}_length_scale",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        variance = numpyro.param(
            f"{self.name}_variance",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        # Compute squared Euclidean distances.
        X_sq = jnp.sum(X**2, axis=-1, keepdims=True)
        X2_sq = jnp.sum(X2**2, axis=-1, keepdims=True)
        pairwise_sq_dists = X_sq - 2 * jnp.dot(X, X2.T) + X2_sq.T
        return variance * jnp.exp(-0.5 * pairwise_sq_dists / (length_scale**2))


# --- Kernel Variants from Before ---
# Spectral Mixture Kernel:
class SpectralMixtureKernel:
    def __init__(self, input_dim: int, Q: int = 1, name: str = "sm_kernel"):
        self.input_dim = input_dim
        self.Q = Q
        self.name = name

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        diff = jnp.linalg.norm(X[:, None, :] - X2[None, :, :], axis=-1)
        weights = numpyro.param(
            f"{self.name}_weights",
            jnp.ones(self.Q) / self.Q,
            constraint=dist.constraints.simplex,
        )
        means = numpyro.param(
            f"{self.name}_means", jnp.ones(self.Q), constraint=dist.constraints.positive
        )
        variances = numpyro.param(
            f"{self.name}_variances",
            jnp.ones(self.Q),
            constraint=dist.constraints.positive,
        )
        kernel = 0.0
        for q in range(self.Q):
            wq = weights[q]
            vq = variances[q]
            muq = means[q]
            kernel += (
                wq
                * jnp.exp(-2 * (jnp.pi**2) * (diff**2) * vq)
                * jnp.cos(2 * jnp.pi * diff * muq)
            )
        return kernel


# Matern 3/2 Kernel:
class Matern32Kernel:
    def __init__(self, input_dim: int, name: str = "matern32"):
        self.input_dim = input_dim
        self.name = name

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        diff = jnp.linalg.norm(X[:, None, :] - X2[None, :, :], axis=-1)
        length_scale = numpyro.param(
            f"{self.name}_length_scale",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        variance = numpyro.param(
            f"{self.name}_variance",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        sqrt3 = jnp.sqrt(3.0)
        return (
            variance
            * (1 + sqrt3 * diff / length_scale)
            * jnp.exp(-sqrt3 * diff / length_scale)
        )


# Matern 5/2 Kernel:
class Matern52Kernel:
    def __init__(self, input_dim: int, name: str = "matern52"):
        self.input_dim = input_dim
        self.name = name

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        diff = jnp.linalg.norm(X[:, None, :] - X2[None, :, :], axis=-1)
        length_scale = numpyro.param(
            f"{self.name}_length_scale",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        variance = numpyro.param(
            f"{self.name}_variance",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        sqrt5 = jnp.sqrt(5.0)
        return (
            variance
            * (1 + sqrt5 * diff / length_scale + (5 * diff**2) / (3 * length_scale**2))
            * jnp.exp(-sqrt5 * diff / length_scale)
        )


# Periodic Kernel:
class PeriodicKernel:
    def __init__(self, input_dim: int, name: str = "periodic"):
        self.input_dim = input_dim
        self.name = name

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        diff = jnp.linalg.norm(X[:, None, :] - X2[None, :, :], axis=-1)
        length_scale = numpyro.param(
            f"{self.name}_length_scale",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        variance = numpyro.param(
            f"{self.name}_variance",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        period = numpyro.param(
            f"{self.name}_period", jnp.array(1.0), constraint=dist.constraints.positive
        )
        return variance * jnp.exp(
            -2 * (jnp.sin(jnp.pi * diff / period) ** 2) / (length_scale**2)
        )


# Rational Quadratic Kernel:
class RationalQuadraticKernel:
    def __init__(self, input_dim: int, name: str = "rq"):
        self.input_dim = input_dim
        self.name = name

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        diff = jnp.linalg.norm(X[:, None, :] - X2[None, :, :], axis=-1)
        length_scale = numpyro.param(
            f"{self.name}_length_scale",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        variance = numpyro.param(
            f"{self.name}_variance",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        alpha = numpyro.param(
            f"{self.name}_alpha", jnp.array(1.0), constraint=dist.constraints.positive
        )
        return variance * (1 + (diff**2) / (2 * alpha * length_scale**2)) ** (-alpha)


# Linear Kernel:
class LinearKernel:
    def __init__(self, input_dim: int, name: str = "linear"):
        self.input_dim = input_dim
        self.name = name

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        bias = numpyro.param(f"{self.name}_bias", jnp.array(0.0))
        return (X @ X2.T) + bias


# Polynomial Kernel:
class PolynomialKernel:
    def __init__(self, input_dim: int, degree: int = 2, name: str = "poly"):
        self.input_dim = input_dim
        self.degree = degree
        self.name = name

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        gamma = numpyro.param(
            f"{self.name}_gamma", jnp.array(1.0), constraint=dist.constraints.positive
        )
        coef0 = numpyro.param(f"{self.name}_coef0", jnp.array(1.0))
        return (gamma * (X @ X2.T) + coef0) ** self.degree


class VariationalLayer:
    """
    Implements a variational layer for Bayesian neural networks, with learnable
    weight distributions (mean and variance).

    The layer uses variational inference to approximate the posterior distributions
    of weights during training.
    """

    def __init__(
        self, input_dim: int, output_dim: int, name: str = "variational_layer"
    ):
        """
        Initialize the variational layer.

        :param input_dim: int
            Dimensionality of the input features.
        :param output_dim: int
            Dimensionality of the output features.
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "variational_layer").
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the variational layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, input_dim)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, output_dim)`, computed as the
            dot product of the input and variationally sampled weights.
        """
        # Sample mean and variance for weights
        w_mu = numpyro.sample(
            f"{self.name}_w_mu",
            dist.Normal(0, 1).expand([self.input_dim, self.output_dim]),
        )
        w_sigma = numpyro.sample(
            f"{self.name}_w_sigma",
            dist.LogNormal(0.0, 0.1).expand([self.input_dim, self.output_dim]),
        )

        # Sample weights using the mean and variance
        weights = numpyro.sample(f"{self.name}_weights", dist.Normal(w_mu, w_sigma))

        # Compute the output
        return jnp.dot(X, weights)
