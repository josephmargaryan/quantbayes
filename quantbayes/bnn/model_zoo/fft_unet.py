from quantbayes.bnn import *
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import jax


class FFTUnet(Module):
    """
    A simple two-level FFT-based U-Net for binary image segmentation.
    """

    def __init__(
        self, in_channels: int, out_channels: int, method="nuts", task_type="binary"
    ):
        """
        Initialize the FFTUnet.

        :param in_channels: int
            Number of input channels.
        :param out_channels: int
            Number of output channels.
        :param method: str
            Inference method ('nuts', 'svi', or 'steinvi').
        :param task_type: str
            Task type ('binary').
        """
        super().__init__(method=method, task_type=task_type)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.encoder_conv1 = FFTConv2d(
            in_channels, 16, kernel_size=3, padding="same", name="enc_conv1"
        )
        self.encoder_conv2 = FFTConv2d(
            16, 32, kernel_size=3, padding="same", name="enc_conv2"
        )
        self.pool1 = MaxPool2d(kernel_size=2, stride=2, name="pool1")

        # Bottleneck
        self.bottleneck_conv = FFTConv2d(
            32, 64, kernel_size=3, padding="same", name="bottleneck_conv"
        )

        # Decoder
        self.upconv1 = FFTTransposedConv2d(
            64, 32, kernel_size=2, stride=2, padding="same", name="upconv1"
        )
        self.decoder_conv1 = FFTConv2d(
            64, 32, kernel_size=3, padding="same", name="dec_conv1"
        )
        self.decoder_conv2 = FFTConv2d(
            32, 16, kernel_size=3, padding="same", name="dec_conv2"
        )

        # Output Layer
        self.output_conv = FFTConv2d(
            16, out_channels, kernel_size=1, padding="same", name="output_conv"
        )

    def __call__(self, X: jnp.ndarray, y=None) -> jnp.ndarray:
        """
        Forward pass of the FFTUnet.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, in_channels, height, width)`.
        :param y: jnp.ndarray or None
            Ground truth labels for the binary task. Required for training.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, out_channels, height, width)`.
        """
        # Encoder
        enc1 = self.encoder_conv1(X)  # Shape: (batch, 16, H, W)
        enc1 = jax.nn.relu(enc1)
        enc2 = self.encoder_conv2(enc1)  # Shape: (batch, 32, H, W)
        enc2 = jax.nn.relu(enc2)
        pool1 = self.pool1(enc2)  # Shape: (batch, 32, H/2, W/2)

        # Bottleneck
        bottleneck = self.bottleneck_conv(pool1)  # Shape: (batch, 64, H/2, W/2)
        bottleneck = jax.nn.relu(bottleneck)

        # Decoder
        up1 = self.upconv1(bottleneck)  # Shape: (batch, 32, H, W)
        # Concatenate with encoder feature map
        concat1 = jnp.concatenate([up1, enc2], axis=1)  # Shape: (batch, 64, H, W)
        dec1 = self.decoder_conv1(concat1)  # Shape: (batch, 32, H, W)
        dec1 = jax.nn.relu(dec1)
        dec2 = self.decoder_conv2(dec1)  # Shape: (batch, 16, H, W)
        dec2 = jax.nn.relu(dec2)

        # Output Layer
        out = self.output_conv(dec2)  # Shape: (batch, out_channels, H, W)

        # Probabilistic Modeling for Binary Tasks
        numpyro.deterministic("logits", out)
        numpyro.sample("obs", dist.Bernoulli(logits=out), obs=y)

        return out
