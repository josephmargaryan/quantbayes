from typing import Optional
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import jax.random

__all__ = [
    "Circulant",
    "BlockCirculant",
    "CirculantProcess",
    "BlockCirculantProcess",
    "SpectralDenseBlock",
    "FourierNeuralOperator1D",
    "MixtureOfTwoLayers",
]


class Circulant(eqx.Module):
    """
    A custom Equinox layer that implements a linear layer with a circulant matrix.

    The layer stores only the first row `c` (a vector of shape (n,)) and a bias vector
    (of shape (n,)). We define the circulant matrix C so that its first row is c and its i-th row is:
      C[i, :] = jnp.roll(c, i)

    The first column is computed by flipping `c` and then rolling by 1.
    The circulant multiplication is then computed via FFT:
      y = real( ifft( fft(x) * fft(first_col) ) ) + bias
    """

    first_row: jnp.ndarray  # shape (n,)
    bias: jnp.ndarray  # shape (n,)
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()

    def __init__(self, in_features: int, *, key, init_scale: float = 1.0):
        self.in_features = in_features
        self.out_features = in_features
        # Split key for first_row and bias initialization.
        key1, key2 = jax.random.split(key)
        self.first_row = jax.random.normal(key1, (in_features,)) * init_scale
        self.bias = jax.random.normal(key2, (in_features,)) * init_scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Multiply input x (shape (..., n)) by the circulant matrix and add bias.
        """
        # Compute the "first column" of the circulant matrix:
        # first_row = [c0, c1, ..., c_{n-1}]
        # flip(first_row) = [c_{n-1}, c_{n-2}, ..., c0]
        # Rolling that by +1 gives: [c0, c_{n-1}, c_{n-2}, ..., c1]
        first_col = jnp.roll(jnp.flip(self.first_row), shift=1)
        fft_w = jnp.fft.fft(first_col)
        fft_x = jnp.fft.fft(x, axis=-1)
        y = jnp.fft.ifft(fft_x * fft_w, axis=-1)
        return jnp.real(y) + self.bias


class BlockCirculant(eqx.Module):
    """
    Equinox module implementing a block-circulant weight matrix with bias.

    - W is of shape (k_out, k_in, b), where each slice W[i, j] is the "first row"
      of a circulant block.
    - D_bernoulli is an optional diagonal (of shape (in_features,)) with ±1 entries.
    - A bias vector of shape (out_features,) is added to the final output.
    - in_features and out_features are the overall input and output dimensions,
      possibly padded up to multiples of b.
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
        """
        Initialize the BlockCirculant module.

        Parameters:
            in_features: Overall input dimension.
            out_features: Overall output dimension.
            block_size: The circulant block size, b.
            key: PRNG key for initialization.
            init_scale: Scaling factor for normal initialization.
            use_bernoulli_diag: If True, include the Bernoulli diagonal.
            use_bias: If True, include a bias parameter of shape (out_features,).
        """
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # Determine k_in and k_out, the number of blocks along the input and output.
        k_in = (in_features + block_size - 1) // block_size
        k_out = (out_features + block_size - 1) // block_size
        object.__setattr__(self, "k_in", k_in)
        object.__setattr__(self, "k_out", k_out)

        # Initialize W with shape (k_out, k_in, block_size)
        k1, k2, k3 = jr.split(key, 3)
        W_init = jr.normal(k1, (k_out, k_in, block_size)) * init_scale

        # Bernoulli diagonal for input, shape (in_features,)
        if use_bernoulli_diag:
            diag_signs = jnp.where(
                jr.bernoulli(k2, p=0.5, shape=(in_features,)), 1.0, -1.0
            )
        else:
            diag_signs = jnp.ones((in_features,))

        # Initialize bias if requested, shape (out_features,)
        if use_bias:
            bias_init = jr.normal(k3, (out_features,)) * init_scale
        else:
            bias_init = jnp.zeros((out_features,))

        object.__setattr__(self, "W", W_init)
        object.__setattr__(self, "D_bernoulli", diag_signs)
        object.__setattr__(self, "bias", bias_init)

    def __call__(
        self, x: jnp.ndarray, *, key=None, state=None, **kwargs
    ) -> jnp.ndarray:
        """
        Forward pass for block-circulant multiplication with bias.

        1. If x is a single sample, add a batch dimension.
        2. Multiply x elementwise by the Bernoulli diagonal.
        3. Zero-pad x (if needed) so that its length equals k_in * block_size.
        4. Reshape x into blocks of shape (batch, k_in, block_size).
        5. For each block-row i, sum over the circulant multiplications across the k_in blocks.
        6. Reshape the result to (batch, k_out * block_size) and slice to (batch, out_features).
        7. Add the bias.
        """
        single_example = x.ndim == 1
        if single_example:
            x = x[None, :]  # add batch dimension

        batch_size = x.shape[0]
        d_in = self.in_features
        d_out = self.out_features
        b = self.block_size
        k_in = self.k_in
        k_out = self.k_out

        # (2) Apply the Bernoulli diagonal.
        x_d = x * self.D_bernoulli[None, :]

        # (3) Zero-pad x_d if necessary.
        pad_len = k_in * b - d_in
        if pad_len > 0:
            x_d = jnp.pad(
                x_d, ((0, 0), (0, pad_len)), mode="constant", constant_values=0.0
            )

        # (4) Reshape into blocks.
        x_blocks = x_d.reshape(batch_size, k_in, b)

        # (5) Perform block-circulant multiplication via FFT.
        def one_block_mul(w_ij, x_j):
            # w_ij: (b,) - first row of a circulant block.
            # x_j: (batch, b) - corresponding block from x.
            c_fft = jnp.fft.fft(w_ij)  # shape (b,)
            X_fft = jnp.fft.fft(x_j, axis=-1)  # shape (batch, b)
            block_fft = X_fft * jnp.conjugate(c_fft)[None, :]
            return jnp.fft.ifft(block_fft, axis=-1).real  # shape (batch, b)

        def compute_blockrow(i):
            # For block-row i, sum over j=0,...,k_in-1.
            def sum_over_j(carry, j):
                w_ij = self.W[i, j, :]  # (b,)
                x_j = x_blocks[:, j, :]  # (batch, b)
                block_out = one_block_mul(w_ij, x_j)
                return carry + block_out, None

            init = jnp.zeros((batch_size, b))
            out_time, _ = jax.lax.scan(sum_over_j, init, jnp.arange(k_in))
            return out_time  # shape (batch, b)

        out_blocks = jax.vmap(compute_blockrow)(
            jnp.arange(k_out)
        )  # shape (k_out, batch, b)
        out_reshaped = jnp.transpose(out_blocks, (1, 0, 2)).reshape(
            batch_size, k_out * b
        )

        # (6) Slice if needed.
        if k_out * b > d_out:
            out_reshaped = out_reshaped[:, :d_out]

        # (7) Add bias.
        out_final = out_reshaped + self.bias[None, :]

        if single_example:
            out_final = out_final[0]
        return out_final


class CirculantProcess(eqx.Module):
    in_features: int = eqx.static_field()
    alpha: float = eqx.static_field()
    K: int = eqx.static_field()
    k_half: int = eqx.static_field()

    # Instead of making prior_std a static field that changes,
    # we define it as a property computed from in_features, alpha, and K.
    # This keeps our design functional.
    fourier_coeffs_real: jnp.ndarray  # shape (k_half,)
    fourier_coeffs_imag: jnp.ndarray  # shape (k_half,)

    # We'll store the final full FFT in a mutable field for retrieval.
    _last_fft_full: jnp.ndarray = eqx.field(default=None, repr=False)

    def __init__(self, in_features, alpha=1.0, K=None, *, key, init_scale=0.1):
        self.in_features = in_features
        self.alpha = alpha
        k_half = in_features // 2 + 1
        if (K is None) or (K > k_half):
            K = k_half
        self.K = K
        self.k_half = k_half

        key_r, key_i = jr.split(key, 2)
        r_init = jr.normal(key_r, (k_half,)) * init_scale
        i_init = jr.normal(key_i, (k_half,)) * init_scale
        # Ensure the DC component is purely real
        i_init = i_init.at[0].set(0.0)
        if (in_features % 2 == 0) and (k_half > 1):
            i_init = i_init.at[-1].set(0.0)

        self.fourier_coeffs_real = r_init
        self.fourier_coeffs_imag = i_init
        self._last_fft_full = None

    @property
    def prior_std(self) -> jnp.ndarray:
        """
        Compute and return the frequency-dependent standard deviations.
        This is based on the frequency indices and the parameter alpha.
        """
        freq_idx = jnp.arange(self.k_half)
        # Compute prior standard deviation for each frequency
        return 1.0 / jnp.sqrt(1.0 + freq_idx**self.alpha)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch = x.ndim == 2
        n = self.in_features

        freq_idx = jnp.arange(self.k_half)
        # Use the precomputed property for prior_std and also build a mask for truncation.
        # (Note: here we are not using prior_std to scale the parameters directly,
        # but it can be used in a bayesianize step to set the prior on each Fourier coeff.)
        mask = freq_idx < self.K

        # "Truncation" at forward time: zeroing out frequencies with index >= K.
        r = self.fourier_coeffs_real * mask
        i = self.fourier_coeffs_imag * mask

        # Build the half-complex array.
        half_complex = r + 1j * i
        if (n % 2 == 0) and (self.k_half > 1):
            nyquist = half_complex[-1].real[None]
            fft_full = jnp.concatenate(
                [half_complex[:-1], nyquist, jnp.conjugate(half_complex[1:-1])[::-1]]
            )
        else:
            fft_full = jnp.concatenate(
                [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
            )

        object.__setattr__(self, "_last_fft_full", fft_full)

        X_fft = jnp.fft.fft(x, axis=-1) if batch else jnp.fft.fft(x)
        if batch:
            out_fft = X_fft * fft_full[None, :]
            out_time = jnp.fft.ifft(out_fft, axis=-1).real
        else:
            out_fft = X_fft * fft_full
            out_time = jnp.fft.ifft(out_fft).real
        return out_time

    def get_fourier_coeffs(self) -> jnp.ndarray:
        """Return the last computed full FFT array (complex, length in_features)."""
        if self._last_fft_full is None:
            raise ValueError(
                "No Fourier coefficients available for layer. "
                "Call the layer on some input first."
            )
        return self._last_fft_full


class BlockCirculantProcess(eqx.Module):
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()
    block_size: int = eqx.static_field()
    alpha: float = eqx.static_field()
    K: int = eqx.static_field()  # truncation index: frequencies >= K are zeroed.
    k_in: int = eqx.static_field()
    k_out: int = eqx.static_field()
    k_half: int = eqx.static_field()

    # Fourier coefficients for each block.
    W_real: jnp.ndarray  # shape (k_out, k_in, k_half)
    W_imag: jnp.ndarray  # shape (k_out, k_in, k_half)

    _last_fourier_coeffs: jnp.ndarray = eqx.field(default=None, repr=False)
    # This will store the reconstructed full Fourier coefficients,
    # with shape (k_out, k_in, block_size)

    def __init__(
        self,
        in_features,
        out_features,
        block_size,
        alpha=1.0,
        K=None,
        *,
        key,
        init_scale=0.1,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.alpha = alpha

        # Compute the number of blocks for input and output.
        k_in = (in_features + block_size - 1) // block_size
        k_out = (out_features + block_size - 1) // block_size
        self.k_in = k_in
        self.k_out = k_out

        b = block_size
        k_half = (b // 2) + 1
        if (K is None) or (K > k_half):
            K = k_half
        self.K = K
        self.k_half = k_half

        key_r, key_i = jr.split(key, 2)
        shape = (k_out, k_in, k_half)
        Wr = jr.normal(key_r, shape) * init_scale
        Wi = jr.normal(key_i, shape) * init_scale
        # Ensure that the DC frequency is purely real.
        Wi = Wi.at[..., 0].set(0.0)
        if (b % 2 == 0) and (k_half > 1):
            Wi = Wi.at[..., -1].set(0.0)

        self.W_real = Wr
        self.W_imag = Wi
        self._last_fourier_coeffs = None

    @property
    def prior_std(self) -> jnp.ndarray:
        """
        Compute the frequency-dependent standard deviations for the block Fourier coefficients.
        This property returns an array of shape (k_half,) based on the frequency indices.
        """
        freq_idx = jnp.arange(self.k_half)
        return 1.0 / jnp.sqrt(1.0 + freq_idx**self.alpha)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch = x.ndim == 2
        if not batch:
            x = x[None, :]
        bs, d_in = x.shape

        # Zero-pad x if needed so that its length fits an integer number of blocks.
        pad_len = self.k_in * self.block_size - d_in
        if pad_len > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_len)))
        x_blocks = x.reshape(bs, self.k_in, self.block_size)

        # Use the property to get frequency-dependent std.
        freq_idx = jnp.arange(self.k_half)
        # In this deterministic forward pass, we don't multiply by prior_std,
        # but you can use it later when bayesianizing the parameters.
        freq_mask = (freq_idx < self.K).astype(jnp.float32)

        # Function to reconstruct a full Fourier block from the half-spectrum.
        def reconstruct_block(r_ij, i_ij):
            # Apply the frequency mask.
            r_ij = r_ij * freq_mask
            i_ij = i_ij * freq_mask
            half_complex = r_ij + 1j * i_ij
            b = self.block_size
            if (b % 2 == 0) and (self.k_half > 1):
                nyquist = half_complex[-1].real[None]
                block_fft = jnp.concatenate(
                    [
                        half_complex[:-1],
                        nyquist,
                        jnp.conjugate(half_complex[1:-1])[::-1],
                    ]
                )
            else:
                block_fft = jnp.concatenate(
                    [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
                )
            return block_fft

        # Vectorized reconstruction over (k_out, k_in).
        block_fft_full = jax.vmap(
            lambda Rrow, Irow: jax.vmap(reconstruct_block)(Rrow, Irow),
            in_axes=(0, 0),
        )(self.W_real, self.W_imag)

        # Store the reconstructed block FFT for later retrieval.
        object.__setattr__(self, "_last_fourier_coeffs", block_fft_full)

        # Multiply in the time domain.
        def multiply_blockrow(i):
            # Sum over the input blocks j.
            def sum_over_j(carry, j):
                fft_block = block_fft_full[i, j]  # shape (block_size,)
                x_j = x_blocks[:, j, :]  # shape (bs, block_size)
                X_fft = jnp.fft.fft(x_j, axis=-1)
                # Note the conjugation on the weight block.
                out_fft = X_fft * jnp.conjugate(fft_block)[None, :]
                out_time = jnp.fft.ifft(out_fft, axis=-1).real
                return carry + out_time, None

            init = jnp.zeros((bs, self.block_size))
            out_time, _ = jax.lax.scan(sum_over_j, init, jnp.arange(self.k_in))
            return out_time

        out_blocks = jax.vmap(multiply_blockrow)(
            jnp.arange(self.k_out)
        )  # shape (k_out, bs, block_size)
        out_reshaped = jnp.transpose(out_blocks, (1, 0, 2)).reshape(
            bs, self.k_out * self.block_size
        )

        if self.k_out * self.block_size > self.out_features:
            out_reshaped = out_reshaped[:, : self.out_features]

        if not batch:
            out_reshaped = out_reshaped[0]
        return out_reshaped

    def get_fourier_coeffs(self) -> jnp.ndarray:
        """
        Return the last-computed full Fourier coefficients array of shape (k_out, k_in, block_size).
        """
        if self._last_fourier_coeffs is None:
            raise ValueError(
                "No Fourier coefficients available. "
                "Call the layer on some input first."
            )
        return self._last_fourier_coeffs


class SpectralDenseBlock(eqx.Module):
    """
    A custom spectral dense block that:
      1. Applies FFT on the input,
      2. Multiplies by a trainable complex mask (w_real and w_imag),
      3. Applies inverse FFT (taking the real part),
      4. Applies a pointwise MLP (Linear -> ReLU -> Linear) mapping to out_features,
      5. And adds a residual connection (with a projection if in_features != out_features).

    This layer is defined for a single example with shape (in_features,).
    """

    in_features: int
    out_features: int
    hidden_dim: int
    w_real: jnp.ndarray  # shape: (in_features,)
    w_imag: jnp.ndarray  # shape: (in_features,)
    linear1: eqx.nn.Linear  # maps (in_features,) -> (hidden_dim,)
    linear2: eqx.nn.Linear  # maps (hidden_dim,) -> (out_features,)
    proj: Optional[
        eqx.nn.Linear
    ]  # optional projection from in_features to out_features

    def __init__(self, in_features: int, out_features: int, hidden_dim: int, *, key):
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        # Split keys for initialization.
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.w_real = jax.random.normal(k1, (in_features,)) * 0.1
        self.w_imag = jax.random.normal(k2, (in_features,)) * 0.1
        self.linear1 = eqx.nn.Linear(in_features, hidden_dim, key=k3)
        self.linear2 = eqx.nn.Linear(hidden_dim, out_features, key=k4)
        # If in_features != out_features, create a projection for the residual path.
        self.proj = (
            eqx.nn.Linear(in_features, out_features, key=k5)
            if in_features != out_features
            else None
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is expected to be of shape (in_features,)
        # 1. FFT of input.
        X_fft = jnp.fft.fft(x)
        # 2. Construct trainable complex mask.
        mask_complex = self.w_real + 1j * self.w_imag
        out_fft = X_fft * mask_complex
        # 3. Inverse FFT to return to time domain.
        x_time = jnp.fft.ifft(out_fft).real
        # 4. Pointwise MLP.
        h = self.linear1(x_time)
        h = jax.nn.relu(h)
        x_dense = self.linear2(h)
        # 5. Residual connection: project x_time if needed.
        shortcut = self.proj(x_time) if self.proj is not None else x_time
        return shortcut + x_dense


def make_mask(n: int, n_modes: int):
    """Create a mask of length n with ones in the first and last n_modes positions."""
    mask = jnp.zeros((n,))
    mask = mask.at[:n_modes].set(1.0)
    mask = mask.at[-n_modes:].set(1.0)
    return mask


class FourierNeuralOperator1D(eqx.Module):
    """
    A single Fourier layer that:
      1. Computes the FFT of the input,
      2. Multiplies by a trainable spectral weight (with a mask),
      3. Computes the inverse FFT,
      4. Applies a pointwise MLP (Linear -> ReLU -> Linear),
      5. And adds a residual connection.

    This layer processes a single sample of shape (in_features,).
    """

    in_features: int
    hidden_dim: int
    n_modes: int
    spectral_weight: jnp.ndarray  # shape: (in_features,)
    linear1: eqx.nn.Linear  # maps (in_features,) -> (hidden_dim,)
    linear2: eqx.nn.Linear  # maps (hidden_dim,) -> (in_features,)

    def __init__(self, in_features: int, hidden_dim: int, n_modes: int, *, key):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.n_modes = n_modes
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.spectral_weight = jax.random.normal(k1, (in_features,)) * 0.1
        self.linear1 = eqx.nn.Linear(in_features, hidden_dim, key=k2)
        self.linear2 = eqx.nn.Linear(hidden_dim, in_features, key=k3)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is expected to be shape (in_features,)
        X_fft = jnp.fft.fft(x)
        mask = make_mask(self.in_features, self.n_modes)
        scale = 1.0 + mask * self.spectral_weight
        X_fft_mod = X_fft * scale
        x_ifft = jnp.fft.ifft(X_fft_mod).real
        h = self.linear1(x_ifft)
        h = jax.nn.relu(h)
        x_mlp = self.linear2(h)
        return x_ifft + x_mlp


class MixtureOfTwoLayers(eqx.Module):
    # Layers for combining
    layerA: eqx.Module
    layerB: eqx.Module
    # gating mechanism selection
    gating_mode: str = "param"
    # parameters for "param" gating:
    logit_gate: jnp.ndarray = None
    # parameters for "beta" gating (we use mean of Beta(alpha, beta)):
    alpha0: jnp.ndarray = None
    beta0: jnp.ndarray = None
    # parameters for "mlp" gating:
    gate_w: jnp.ndarray = None
    gate_b: jnp.ndarray = None

    # key and d_in are only needed for certain gating modes
    def __init__(
        self,
        layerA: eqx.Module,
        layerB: eqx.Module,
        gating_mode: str = "param",
        key: jax.random.PRNGKey = None,
        d_in: int = None,
    ):
        self.layerA = layerA
        self.layerB = layerB
        self.gating_mode = gating_mode

        if gating_mode == "param":
            # A single learnable scalar (logit) gate.
            self.logit_gate = jnp.array(0.0)
        elif gating_mode == "beta":
            # Instead of sampling from a Beta, we’ll use a deterministic function, e.g. its mean.
            self.alpha0 = jnp.array(2.0)
            self.beta0 = jnp.array(2.0)
        elif gating_mode == "mlp":
            # For data-dependent gating we need to map from input (d_in) to a scalar gate.
            if key is None or d_in is None:
                raise ValueError("For mlp gating, key and d_in must be provided")
            # Initialize weights and bias. (For simplicity we use a single linear layer.)
            self.gate_w = jax.random.normal(key, (d_in, 1))
            self.gate_b = jax.random.normal(key, (1,))
        else:
            raise ValueError(f"Unrecognized gating_mode={gating_mode}")

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        outA = self.layerA(X)
        outB = self.layerB(X)
        if self.gating_mode == "param":
            gate = jax.nn.sigmoid(self.logit_gate)  # scalar in (0,1)
        elif self.gating_mode == "beta":
            # Deterministically use the mean of the Beta distribution:
            gate = self.alpha0 / (self.alpha0 + self.beta0)
        elif self.gating_mode == "mlp":
            # Compute per-example gate values.
            logit = jnp.dot(X, self.gate_w) + self.gate_b  # shape (batch_size, 1)
            gate = jax.nn.sigmoid(logit)
        else:
            raise ValueError(f"Unrecognized gating_mode={self.gating_mode}")
        return gate * outA + (1.0 - gate) * outB
