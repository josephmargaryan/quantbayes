from quantbayes.bnn.layers import FFTConv1d, Linear
import jax.numpy as jnp

class FFTConv1DModel:
    """
    Simple 1D model that uses FFT-based 1D convolution layer.
    """

    def __init__(self, in_channels=1, out_channels=8, kernel_size=16, name="fft_conv1d_model"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.name = name

        self.fft_conv1d = FFTConv1d(in_channels, out_channels, kernel_size, name=f"{name}_fftconv")
        self.linear_out = Linear(out_channels, 1, name=f"{name}_linear_out")

    def __call__(self, X):
        """
        X: (batch_size, in_channels, seq_len)
        returns: shape (batch_size,)
        """
        conv_out = self.fft_conv1d(X)  # shape (batch_size, out_channels, new_width)
        # average pool or flatten
        pooled = jnp.mean(conv_out, axis=-1)
        out = self.linear_out(pooled)  # shape (batch_size, 1)
        return out.squeeze(-1)
