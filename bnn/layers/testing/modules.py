import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


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


class Linear:
    """
    A fully connected layer with weights and biases sampled from Normal distributions.

    Transforms inputs via a linear operation: `output = X @ weights + biases`.
    """

    def __init__(self, in_features: int, out_features: int, name: str = "layer"):
        """
        Initializes the Linear layer.

        :param in_features: int
            Number of input features (columns in the input matrix).
        :param out_features: int
            Number of output features (columns in the output matrix).
        :param name: str
            Name of the layer for parameter tracking (default: "layer").
        """
        self.in_features = in_features
        self.out_features = out_features
        self.name = name

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
            dist.Normal(0, 1).expand([self.in_features, self.out_features]),
        )
        b = numpyro.sample(
            f"{self.name}_b", dist.Normal(0, 1).expand([self.out_features])
        )
        return jnp.dot(X, w) + b


def fft_matmul(first_row: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
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


class FFTLinear:
    """
    FFT-based linear layer for efficient circulant matrix multiplication.

    This layer uses a circulant matrix (parameterized by its first row) and
    applies FFT-based matrix multiplication for computational efficiency.
    """

    def __init__(self, in_features: int, name: str = "fft_layer"):
        """
        Initialize the FFTLinear layer.

        :param in_features: int
            Number of input features.
        :param name: str
            Name of the layer, used for parameter naming (default: "fft_layer").
        """
        self.in_features = in_features
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the FFTLinear layer.

        :param X: jnp.ndarray
            Input data, shape `(batch_size, in_features)`.

        :returns: jnp.ndarray
            Output of the FFT-based linear layer, shape `(batch_size, in_features)`.
        """

        first_row = numpyro.sample(
            f"{self.name}_first_row", dist.Normal(0, 1).expand([self.in_features])
        )
        bias_circulant = numpyro.sample(
            f"{self.name}_bias_circulant", dist.Normal(0, 1).expand([self.in_features])
        )
        hidden = fft_matmul(first_row, X) + bias_circulant[None, :]

        return hidden


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
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the FFTConv2d layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, in_channels, height, width)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, out_channels, new_height, new_width)`, where:
            - `new_height = height + kernel_h - 1`
            - `new_width = width + kernel_w - 1`.
        """
        batch_size, in_channels, height, width = X.shape
        kernel_h, kernel_w = self.kernel_size

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

        output = []
        for out_c in range(self.out_channels):
            channel_out = 0
            for in_c in range(self.in_channels):
                X_fft = jnp.fft.fft2(
                    X[:, in_c], s=(height + kernel_h - 1, width + kernel_w - 1)
                )
                weight_fft = jnp.fft.fft2(
                    weight[out_c, in_c], s=(height + kernel_h - 1, width + kernel_w - 1)
                )

                conv_fft = X_fft * weight_fft

                channel_out += jnp.fft.ifft2(conv_fft).real

            channel_out += bias[out_c].reshape(1, 1)

            output.append(channel_out)

        output = jnp.stack(output, axis=1)
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


class GaussianProcessLayer:
    """
    Implements a Gaussian Process (GP) layer with a Radial Basis Function (RBF) kernel.

    This layer computes the covariance matrix (kernel) based on the input data and
    learnable parameters: length scale, variance, and noise.
    """

    def __init__(self, input_dim: int, name: str = "gp_layer"):
        """
        Initialize the Gaussian Process layer.

        :param input_dim: int
            Dimensionality of the input features.
        :param name: str, optional
            Name of the GP layer, used for parameter naming (default: "gp_layer").
        """
        self.input_dim = input_dim
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass to compute the GP kernel matrix.

        :param X: jnp.ndarray
            Input data of shape `(num_points, input_dim)`, where `num_points` is
            the number of data points, and `input_dim` is the feature dimension.

        :returns: jnp.ndarray
            Covariance matrix (kernel) of shape `(num_points, num_points)`, representing
            pairwise relationships between input data points.
        """
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
        noise = numpyro.param(
            f"{self.name}_noise", jnp.array(1.0), constraint=dist.constraints.positive
        )

        pairwise_sq_dists = (
            jnp.sum(X**2, axis=-1, keepdims=True)
            - 2 * jnp.dot(X, X.T)
            + jnp.sum(X**2, axis=-1)
        )
        kernel = variance * jnp.exp(-0.5 * pairwise_sq_dists / length_scale**2)

        return kernel + noise * jnp.eye(X.shape[0])


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
