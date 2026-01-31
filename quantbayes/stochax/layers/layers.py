import math
from typing import Optional, Tuple
from jaxtyping import Array, PRNGKeyArray
from typing import Optional
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random
import jax.random as jr

__all__ = [
    "SpectralGRUCell",
    "SpectralLSTMCell",
    "SpectralDenseBlock",
    "FourierNeuralOperator",
    "MixtureOfTwoLayers",
    "SpectralMultiheadAttention",
]


class SpectralGRUCell(eqx.Module):
    """
    A GRU cell that applies FFT-based spectral modulation to its input
    before computing standard GRU gate operations.
    """

    weight_ih: Array  # (3 * hidden_size, input_size)
    weight_hh: Array  # (3 * hidden_size, hidden_size)
    bias: Optional[Array]  # (3 * hidden_size,)
    bias_n: Optional[Array]  # (hidden_size,)
    base_filter: Array  # (freq_bins,)
    base_bias: Array  # (freq_bins,)

    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        dtype = jnp.float32
        lim = math.sqrt(1 / hidden_size)
        ihkey, hhkey, bkey, sfkey, sbkey, bnkey = jr.split(key, 6)

        # Standard GRU weights
        ihshape = (3 * hidden_size, input_size)
        self.weight_ih = jr.uniform(
            ihkey, shape=ihshape, minval=-lim, maxval=lim, dtype=dtype
        )
        hhshape = (3 * hidden_size, hidden_size)
        self.weight_hh = jr.uniform(
            hhkey, shape=hhshape, minval=-lim, maxval=lim, dtype=dtype
        )
        if use_bias:
            self.bias = jr.uniform(
                bkey, shape=(3 * hidden_size,), minval=-lim, maxval=lim, dtype=dtype
            )
            self.bias_n = jr.uniform(
                bnkey, shape=(hidden_size,), minval=-lim, maxval=lim, dtype=dtype
            )
        else:
            self.bias = None
            self.bias_n = None

        # Parameters for spectral modulation on the input.
        self.base_filter = jr.uniform(
            sfkey, shape=(input_size // 2 + 1,), minval=0.9, maxval=1.1, dtype=dtype
        )
        self.base_bias = jr.uniform(
            sbkey, shape=(input_size // 2 + 1,), minval=-0.1, maxval=0.1, dtype=dtype
        )

    def spectral_transform(self, x: Array) -> Array:
        """
        Applies FFT-based modulation to the input vector.
        Args:
            x: input of shape (input_size,)
        Returns:
            x_mod: spectrally modulated vector, shape (input_size,)
        """
        x_fft = jnp.fft.rfft(x, norm="ortho")  # shape: (freq_bins,)
        x_fft_mod = x_fft * self.base_filter + self.base_bias
        x_mod = jnp.fft.irfft(x_fft_mod, n=self.input_size, norm="ortho")
        return x_mod

    def __call__(
        self, input: Array, hidden: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        """
        One time step of the spectral GRU cell.
        Args:
            input: shape (input_size,)
            hidden: shape (hidden_size,)
        Returns:
            new hidden state: shape (hidden_size,)
        """
        # Apply spectral modulation to the input.
        input_mod = self.spectral_transform(input)
        # Compute gates from the spectrally-modulated input.
        if self.use_bias:
            igates = jnp.split(self.weight_ih @ input_mod + self.bias, 3)
        else:
            igates = jnp.split(self.weight_ih @ input_mod, 3)
        hgates = jnp.split(self.weight_hh @ hidden, 3)

        reset = jax.nn.sigmoid(igates[0] + hgates[0])
        inp = jax.nn.sigmoid(igates[1] + hgates[1])
        new = jnp.tanh(
            igates[2] + reset * (hgates[2] + (self.bias_n if self.use_bias else 0))
        )
        new_hidden = new + inp * (hidden - new)
        return new_hidden


class SpectralLSTMCell(eqx.Module):
    """
    A LSTM cell that applies FFT-based spectral modulation to its input
    before computing standard LSTM gate operations.
    """

    weight_ih: Array  # (4 * hidden_size, input_size)
    weight_hh: Array  # (4 * hidden_size, hidden_size)
    bias: Optional[Array]  # (4 * hidden_size,)
    base_filter: Array  # (freq_bins,) where freq_bins = input_size // 2 + 1
    base_bias: Array  # (freq_bins,)

    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

        dtype = jnp.float32
        lim = math.sqrt(1 / hidden_size)
        # Split key into parts for weight initialization and spectral parameters.
        ihkey, hhkey, bkey, sfkey, sbkey = jr.split(key, 5)

        # Standard LSTM weights
        ihshape = (4 * hidden_size, input_size)
        self.weight_ih = jr.uniform(
            ihkey, shape=ihshape, minval=-lim, maxval=lim, dtype=dtype
        )
        hhshape = (4 * hidden_size, hidden_size)
        self.weight_hh = jr.uniform(
            hhkey, shape=hhshape, minval=-lim, maxval=lim, dtype=dtype
        )
        if use_bias:
            bshape = (4 * hidden_size,)
            self.bias = jr.uniform(
                bkey, shape=bshape, minval=-lim, maxval=lim, dtype=dtype
            )
        else:
            self.bias = None

        # Spectral modulation parameters.
        # Number of frequency bins: input_size//2 + 1 (for real FFT)
        freq_bins = input_size // 2 + 1
        self.base_filter = jr.uniform(
            sfkey, shape=(freq_bins,), minval=0.9, maxval=1.1, dtype=dtype
        )
        self.base_bias = jr.uniform(
            sbkey, shape=(freq_bins,), minval=-0.1, maxval=0.1, dtype=dtype
        )

    def spectral_transform(self, x: Array) -> Array:
        """
        Applies FFT-based modulation to the input vector.
        Args:
            x: Input vector of shape (input_size,)
        Returns:
            x_mod: Spectrally modulated vector, shape (input_size,)
        """
        # Compute real FFT.
        x_fft = jnp.fft.rfft(x, norm="ortho")  # shape: (freq_bins,)
        # Apply learned spectral modulation.
        x_fft_mod = x_fft * self.base_filter + self.base_bias
        # Inverse FFT to return to time-domain.
        x_mod = jnp.fft.irfft(x_fft_mod, n=self.input_size, norm="ortho")
        return x_mod

    def __call__(
        self,
        input: Array,
        hidden: Tuple[Array, Array],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Tuple[Array, Array]:
        """
        One time step of the spectral LSTM cell.

        Args:
            input: JAX array of shape (input_size,).
            hidden: Tuple (h, c), each of shape (hidden_size,).
            key: Ignored; provided for compatibility with Equinox API.

        Returns:
            new_hidden: Tuple (new_h, new_c) representing the updated hidden state.
        """
        h, c = hidden
        # Apply spectral modulation to the input.
        input_mod = self.spectral_transform(input)
        # Compute combined linear transformation.
        lin = self.weight_ih @ input_mod + self.weight_hh @ h
        if self.use_bias:
            lin = lin + self.bias
        # Split into four components.
        i, f, g, o = jnp.split(lin, 4)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        new_c = f * c + i * g
        new_h = o * jnp.tanh(new_c)
        return new_h, new_c


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


class FourierNeuralOperator(eqx.Module):
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


class SpectralMultiheadAttention(eqx.Module):
    """
    A multihead self-attention layer that applies FFT-based spectral modulation
    to the input before performing standard multihead attention.
    """

    in_features: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    W_q: Array
    W_k: Array
    W_v: Array
    W_o: Array
    base_filter: Array  # shape: (in_features//2 + 1,)
    base_bias: Array  # shape: (in_features//2 + 1,)

    def __init__(self, in_features: int, num_heads: int, *, key: PRNGKeyArray):
        self.in_features = in_features
        self.num_heads = num_heads
        if in_features % num_heads != 0:
            raise ValueError("in_features must be divisible by num_heads")
        self.head_dim = in_features // num_heads

        dtype = jnp.float32
        lim = math.sqrt(1 / in_features)
        keys = jr.split(key, 5)
        self.W_q = jr.uniform(
            keys[0], (in_features, in_features), minval=-lim, maxval=lim, dtype=dtype
        )
        self.W_k = jr.uniform(
            keys[1], (in_features, in_features), minval=-lim, maxval=lim, dtype=dtype
        )
        self.W_v = jr.uniform(
            keys[2], (in_features, in_features), minval=-lim, maxval=lim, dtype=dtype
        )
        self.W_o = jr.uniform(
            keys[3], (in_features, in_features), minval=-lim, maxval=lim, dtype=dtype
        )
        freq_bins = in_features // 2 + 1
        # For simplicity, we use the same key for both spectral parameters.
        self.base_filter = jr.uniform(
            keys[4], (freq_bins,), minval=0.9, maxval=1.1, dtype=dtype
        )
        self.base_bias = jr.uniform(
            keys[4], (freq_bins,), minval=-0.1, maxval=0.1, dtype=dtype
        )

    def spectral_transform(self, x: Array) -> Array:
        """
        Applies FFT-based modulation to the input tensor along its last dimension.
        Args:
            x: Input array of shape (..., in_features)
        Returns:
            x_mod: Spectrally modulated array, shape (..., in_features)
        """
        # Compute real FFT along the last axis.
        x_fft = jnp.fft.rfft(x, norm="ortho")
        x_fft_mod = x_fft * self.base_filter + self.base_bias
        # Inverse FFT to return to the original space.
        x_mod = jnp.fft.irfft(x_fft_mod, n=self.in_features, norm="ortho")
        return x_mod

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """
        Args:
            x: Input tensor of shape (seq_len, in_features)
            key: Ignored; provided for API compatibility.
        Returns:
            output: Tensor of shape (seq_len, in_features) after attention.
        """
        # Apply spectral modulation to the input.
        x_mod = self.spectral_transform(x)

        # Compute query, key, and value matrices.
        Q = x_mod @ self.W_q  # shape: (seq_len, in_features)
        K = x_mod @ self.W_k
        V = x_mod @ self.W_v

        # Split each into heads.
        def split_heads(t: Array) -> Array:
            seq_len = t.shape[0]
            return t.reshape(seq_len, self.num_heads, self.head_dim)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # Transpose to shape (num_heads, seq_len, head_dim)
        Q = jnp.transpose(Q, (1, 0, 2))
        K = jnp.transpose(K, (1, 0, 2))
        V = jnp.transpose(V, (1, 0, 2))

        # Compute scaled dot-product attention.
        scale = math.sqrt(self.head_dim)
        attn_scores = (
            jnp.einsum("hqd,hkd->hqk", Q, K) / scale
        )  # shape: (num_heads, seq_len, seq_len)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        attn_output = jnp.einsum(
            "hqk,hkd->hqd", attn_weights, V
        )  # shape: (num_heads, seq_len, head_dim)

        # Transpose and combine heads back.
        attn_output = jnp.transpose(
            attn_output, (1, 0, 2)
        )  # shape: (seq_len, num_heads, head_dim)
        combined = attn_output.reshape(
            x.shape[0], self.in_features
        )  # shape: (seq_len, in_features)

        # Final linear projection.
        output = combined @ self.W_o
        return output
