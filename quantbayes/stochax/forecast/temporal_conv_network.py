import jax
import jax.numpy as jnp
from dataclasses import field
from flax import linen as nn
from typing import List

class TemporalConvBlock(nn.Module):
    """
    A single TCN residual block:
      - 2 x (Conv -> ReLU -> Dropout)
      - Residual/skip connection
      - Causal convolutions with dilation
    """
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    dilation: int = 1
    dropout: float = 0.2

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        padding = (self.kernel_size - 1) * self.dilation

        # First convolution
        out = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding=((padding, 0),),
            kernel_dilation=(self.dilation,),
            use_bias=False
        )(x)
        out = out[:, :, :x.shape[2]]  # Keep length consistent
        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)
        out = nn.Dropout(self.dropout)(out, deterministic=not train)

        # Second convolution
        out = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding=((padding, 0),),
            kernel_dilation=(self.dilation,),
            use_bias=False
        )(out)
        out = out[:, :, :x.shape[2]]  # Keep length consistent
        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)
        out = nn.Dropout(self.dropout)(out, deterministic=not train)

        # Residual connection
        res = x
        if self.in_channels != self.out_channels:
            res = nn.Conv(features=self.out_channels, kernel_size=(1,))(res)
            res = res[:, :, :x.shape[2]]  # Keep length consistent

        return nn.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    A multi-layer TCN with increasing dilation.
    """
    in_channels: int
    num_channels_list: list
    kernel_size: int = 3
    dropout: float = 0.2

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        for i, out_channels in enumerate(self.num_channels_list):
            x = TemporalConvBlock(
                in_channels=x.shape[1] if i == 0 else self.num_channels_list[i - 1],
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                dilation=2**i,
                dropout=self.dropout,
            )(x, train=train)
        return x



class TCNForecaster(nn.Module):
    """
    A TCN for time-series forecasting.
    """
    input_dim: int = 1
    tcn_channels: List[int] = field(default_factory=lambda: [32, 32, 64])
    kernel_size: int = 3
    dropout: float = 0.2

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Transpose input for Conv1D: (B, L, input_dim) -> (B, input_dim, L)
        x = jnp.swapaxes(x, 1, 2)

        # Temporal Convolutional Network
        tcn_out = TemporalConvNet(
            in_channels=self.input_dim,
            num_channels_list=self.tcn_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )(x, train=train)

        # Take the last time step and project to output dimension
        last_step = tcn_out[:, :, -1]
        out = nn.Dense(1)(last_step)  # Output dimension is 1
        return out


# Test Case
def test_tcn_forecaster():
    key = jax.random.PRNGKey(0)
    batch_size = 8
    seq_len = 20
    input_dim = 3

    model = TCNForecaster(
        input_dim=input_dim,
        tcn_channels=[16, 16, 32],
        kernel_size=3,
        dropout=0.1
    )

    # Generate random input
    x = jax.random.normal(key, (batch_size, seq_len, input_dim))  # Shape (B, L, input_dim)

    # Initialize and run the model
    variables = model.init(jax.random.PRNGKey(1), x, train=True)
    y = model.apply(variables, x, train=False)

    print("Output shape:", y.shape)  # Expected: (B, 1)

test_tcn_forecaster()
