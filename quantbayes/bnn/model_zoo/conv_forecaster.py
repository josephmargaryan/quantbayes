from quantbayes.bnn.layers import Conv1d, Linear
import jax.numpy as jnp


class Conv1DTimeSeriesForecaster:
    """
    Simple 1D CNN for forecasting the next step of a time series.
    """

    def __init__(
        self, in_channels=1, out_channels=8, kernel_size=3, name="conv1d_forecaster"
    ):
        """
        :param in_channels: int, number of input channels (e.g. 1 for univariate)
        :param out_channels: int, number of channels after convolution
        :param kernel_size: int, convolution kernel
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.name = name

        self.conv1 = Conv1d(
            in_channels, out_channels, kernel_size, name=f"{name}_conv1"
        )
        # We'll do a final linear that takes out_channels -> 1
        self.linear_out = Linear(out_channels, 1, name=f"{name}_linear_out")

    def __call__(self, X):
        """
        X: shape (batch_size, in_channels, seq_len)
        returns: forecast, shape (batch_size,)
        """
        # 1) Convolution
        conv_out = self.conv1(X)
        # shape = (batch_size, out_channels, new_len)

        # 2) Global average pooling across time dimension
        pooled = jnp.mean(conv_out, axis=-1)
        # shape = (batch_size, out_channels)

        # 3) Final linear -> produce single scalar
        out = self.linear_out(pooled)  # shape = (batch_size, 1)
        return out.squeeze(-1)
