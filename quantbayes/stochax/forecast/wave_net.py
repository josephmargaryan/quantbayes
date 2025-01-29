import jax
import jax.numpy as jnp
from flax import linen as nn

class WaveNetResidualBlock(nn.Module):
    residual_channels: int
    skip_channels: int
    dilation: int
    kernel_size: int = 2

    @nn.compact
    def __call__(self, x):
        # x: (batch, seq_len, channels)
        # Causal padding: pad left side only
        padding = (self.kernel_size - 1) * self.dilation
        x_padded = jnp.pad(x, ((0, 0), (padding, 0), (0, 0)), mode="constant")

        # Dilated convolution for filter
        conv_filter = nn.Conv(
            features=self.residual_channels,
            kernel_size=self.kernel_size,
            kernel_dilation=self.dilation,
            padding='VALID',
            name="conv_filter",
        )(x_padded)

        # Dilated convolution for gate
        conv_gate = nn.Conv(
            features=self.residual_channels,
            kernel_size=self.kernel_size,
            kernel_dilation=self.dilation,
            padding='VALID',
            name="conv_gate",
        )(x_padded)

        # Gated activation
        gated = jnp.tanh(conv_filter) * nn.sigmoid(conv_gate)

        # Skip connection
        skip = nn.Conv(
            features=self.skip_channels,
            kernel_size=1,
            padding='VALID',
            name="conv_skip",
        )(gated)

        # Residual connection
        residual = nn.Conv(
            features=x.shape[-1],  # Correctly using channel dimension
            kernel_size=1,
            padding='VALID',
            name="conv_residual",
        )(gated)

        # Add residual to input
        return x + residual, skip


# Define the WaveNet model
class WaveNet(nn.Module):
    input_dim: int = 1
    residual_channels: int = 32
    skip_channels: int = 32
    dilation_depth: int = 6
    kernel_size: int = 2

    @nn.compact
    def __call__(self, x):
        # x: (batch, seq_len, input_dim)
        # No transpose needed
        current = nn.Conv(features=self.residual_channels, kernel_size=1, padding='VALID')(x)

        skip_total = 0.0
        for i in range(self.dilation_depth):
            dilation = 2 ** i
            block = WaveNetResidualBlock(
                residual_channels=self.residual_channels,
                skip_channels=self.skip_channels,
                dilation=dilation,
                kernel_size=self.kernel_size
            )
            current, skip = block(current)
            skip_total += skip

        out = nn.relu(skip_total)
        out = nn.Conv(features=self.skip_channels, kernel_size=1, padding='VALID')(out)
        out = nn.relu(out)
        out = nn.Conv(features=self.skip_channels, kernel_size=1, padding='VALID')(out)
        out = nn.relu(out)

        # Take the last time step
        out_last = out[:, -1, :]  # (batch, channels)
        pred = nn.Dense(features=1)(out_last)

        return pred


# Simple test
def test_wavenet():
    rng = jax.random.PRNGKey(0)
    model = WaveNet(input_dim=2, residual_channels=32, skip_channels=32, dilation_depth=4, kernel_size=2)

    # Create a random input tensor with shape (batch_size, seq_len, input_dim)
    batch_size = 8
    seq_len = 30
    input_dim = 2
    x = jax.random.normal(rng, (batch_size, seq_len, input_dim))

    # Initialize the model and perform a forward pass
    variables = model.init(rng, x)
    preds = model.apply(variables, x)

    print("Output shape:", preds.shape)  # Should be (batch_size, 1)


if __name__ == "__main__":
    test_wavenet()
