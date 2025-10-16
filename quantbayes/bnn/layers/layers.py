import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

__all__ = [
    "Linear",
    "FourierNeuralOperator1D",
    "SpectralDenseBlock",
    "ParticleLinear",
    "FFTParticleLinear",
    "Conv1d",
    "Conv2d",
    "TransposedConv2d",
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
    "GibbsKernelLayer",
    "InputWarpingLayer",
    "GibbsKernel2DLayer",
    "InputWarping2DLayer",
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


class GibbsKernelLayer:
    """
    Implements a Gibbs (non‑stationary) kernel convolution:
      k(x,x') = sqrt(2 l(x) l(x') / (l(x)^2 + l(x')^2))
                * exp(-(x-x')^2 / (l(x)^2 + l(x')^2))
    where l(x) is a location–dependent lengthscale from a small net.
    Adds 1e-6 to the denominator for numerical stability.
    """

    def __init__(self, in_features, lengthscale_net, name="gibbs"):
        self.N = in_features
        self.lengthscale_net = lengthscale_net
        self.name = name

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, N = x.shape
        assert N == self.N

        # positions 0..1
        pos = jnp.linspace(0.0, 1.0, N)
        log_l = self.lengthscale_net(pos)  # (N,)
        l = jnp.exp(log_l)  # (N,)

        l_i = l[:, None]  # (N,1)
        l_j = l[None, :]  # (1,N)
        denom = l_i**2 + l_j**2 + 1e-6  # (N,N)

        sq = jnp.sqrt(2 * (l_i * l_j) / denom)
        d = (jnp.arange(N)[:, None] - jnp.arange(N)[None, :]) ** 2
        exp = jnp.exp(-d / denom)

        K = sq * exp  # (N,N)
        return x @ K.T  # (B,N)


class InputWarpingLayer:
    def __init__(self, in_features, warp_net, base_psd, name="warp"):
        self.N = in_features
        self.warp_net = warp_net  # function pos→offset
        self.base_psd = base_psd  # function freqs→S(freq)
        self.name = name

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x : (batch,N)
        batch, N = x.shape
        assert N == self.N

        # positions in [0,1]
        pos = jnp.linspace(0.0, 1.0, N)
        # learn offsets Δu(pos)
        delta_u = self.warp_net(pos)  # (N,)
        u = pos + delta_u  # (N,)

        # reinterpolate each sequence onto uniform grid
        def interp_one(seq):
            # jnp.interp is supported in JAX
            return jnp.interp(pos, u, seq)

        x_warp = jax.vmap(interp_one)(x)  # (batch,N)

        # stationary conv in warped space
        freqs = jnp.fft.fftfreq(N)  # (N,)
        S = self.base_psd(freqs)  # (N,)
        Xf = jnp.fft.fft(x_warp, axis=-1)  # (batch,N)
        y = jnp.fft.ifft(Xf * S[None, :], axis=-1).real  # (batch,N)
        return y


class GibbsKernel2DLayer:
    """
    Separable 2D Gibbs kernel: k((i,j),(i',j')) = k_x(i,i') * k_y(j,j')
    where k_x, k_y are 1D Gibbs kernels from lengthscale nets.
    """

    def __init__(self, H, W, lengthscale_net_x, lengthscale_net_y, name="gibbs2d"):
        self.H, self.W = H, W
        self.net_x = lengthscale_net_x  # maps pos_x -> log l_x
        self.net_y = lengthscale_net_y  # maps pos_y -> log l_y
        self.name = name

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, H, W)
        B, H, W = x.shape
        assert (H, W) == (self.H, self.W)
        # positions
        pos_x = jnp.linspace(0.0, 1.0, W)
        pos_y = jnp.linspace(0.0, 1.0, H)
        # lengthscales
        lx = jnp.exp(self.net_x(pos_x))  # (W,)
        ly = jnp.exp(self.net_y(pos_y))  # (H,)

        # 1D Gibbs kernels
        def gibbs1d(pos, ell):
            d = pos[:, None] - pos[None, :]
            l_i = ell[:, None]
            l_j = ell[None, :]
            denom = l_i**2 + l_j**2
            sq = jnp.sqrt(2 * l_i * l_j / denom)
            return sq * jnp.exp(-(d**2) / denom)

        Kx = gibbs1d(pos_x, lx)  # (W,W)
        Ky = gibbs1d(pos_y, ly)  # (H,H)
        # separable convolution: first along width (x-axis), then height (y-axis)
        # apply per-row 1D conv along width
        x1 = jnp.einsum("bhw,wd->bhd", x, Kx)  # (B,H,W)
        # apply per-column conv along height
        y = jnp.einsum("bhd,hk->bkd", x1, Ky)  # (B,H,W)
        return y  # shape (B,H,W)


class InputWarping2DLayer:
    """
    Separable 2D warp: warp rows then warp columns with learned monotonic mapping.
    """

    def __init__(self, H, W, warp_net_x, warp_net_y, base_psd, name="warp2d"):
        self.H, self.W = H, W
        self.warp_x, self.warp_y = warp_net_x, warp_net_y
        self.base_psd = base_psd
        self.name = name

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch,H,W)
        B, H, W = x.shape
        pos_x = jnp.linspace(0.0, 1.0, W)
        pos_y = jnp.linspace(0.0, 1.0, H)

        # 1D warps
        def make_u(net, pos):
            raw = net(pos)
            inc = jax.nn.softplus(raw)
            cs = jnp.cumsum(inc)
            return (cs - cs[0]) / (cs[-1] - cs[0])

        ux = make_u(self.warp_x, pos_x)  # (W,)
        uy = make_u(self.warp_y, pos_y)  # (H,)

        # warp X: rows -> warp_x
        def warp_rows(img):
            return jnp.stack([jnp.interp(pos_x, ux, row) for row in img], axis=0)

        x1 = jax.vmap(warp_rows)(x)

        # warp Y: columns -> warp_y
        def warp_cols(img):
            return jnp.stack([jnp.interp(pos_y, uy, col) for col in img.T], axis=1)

        x2 = jax.vmap(warp_cols)(x1)
        # circulant 2D FFT conv
        fy = jnp.fft.fftfreq(H)
        fx = jnp.fft.fftfreq(W)
        FY, FX = jnp.meshgrid(fy, fx, indexing="ij")
        S = self.base_psd(FY, FX)  # (H,W)
        Sf = S[None, :, :]
        Xf = jnp.fft.fftn(x2, axes=(1, 2))
        y = jnp.fft.ifftn(Xf * Sf, axes=(1, 2)).real
        return y
