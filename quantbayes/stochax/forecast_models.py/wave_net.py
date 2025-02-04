import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from functools import partial

# -------------------------------------------------------------------
# Helper: VConv1d
#
# Wraps an eqx.nn.Conv1d so that it can be applied to a batched input.
# Expects input shape [N, in_channels, L] and applies the conv over each sample.
# -------------------------------------------------------------------
class VConv1d(eqx.Module):
    conv: eqx.nn.Conv1d

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [N, in_channels, L]
        return jax.vmap(self.conv)(x)

# -------------------------------------------------------------------
# Helper: CausalConv1d
#
# A 1D convolution with manual left-only (causal) padding.
# Expects input of shape [N, in_channels, L] and returns output [N, out_channels, L].
# -------------------------------------------------------------------
class CausalConv1d(eqx.Module):
    conv: eqx.nn.Conv1d
    kernel_size: int = eqx.field(static=True)
    dilation: int = eqx.field(static=True)

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, *, key):
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = eqx.nn.Conv1d(
            in_channels, out_channels,
            kernel_size,
            stride=1,
            padding="VALID",  # we do manual padding below
            dilation=dilation,
            key=key
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [N, in_channels, L]
        pad = self.dilation * (self.kernel_size - 1)
        # Pad only on the left along the time dimension.
        x_padded = jnp.pad(x, ((0, 0), (0, 0), (pad, 0)))
        # Since eqx.nn.Conv1d expects unbatched input (shape (in_channels, L)),
        # we vmap it over the batch dimension.
        return jax.vmap(self.conv)(x_padded)

# -------------------------------------------------------------------
# WaveNet Residual Block
#
# Each residual block applies two dilated causal convolutions (one for the filter,
# one for the gate) to compute a gated activation unit.
# A 1×1 convolution then produces both a residual connection (added to the input)
# and a skip connection (to be summed later).
# -------------------------------------------------------------------
class WaveNetResidualBlock(eqx.Module):
    # Two dilated causal convs: one for filter and one for gate.
    conv_filter: CausalConv1d
    conv_gate: CausalConv1d
    # 1x1 convs for residual and skip connections.
    conv_res: VConv1d
    conv_skip: VConv1d

    def __init__(self, residual_channels: int, kernel_size: int,
                 dilation: int, skip_channels: int, *, key):
        # Split keys for each submodule.
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.conv_filter = CausalConv1d(residual_channels, residual_channels, kernel_size,
                                        dilation=dilation, key=k1)
        self.conv_gate = CausalConv1d(residual_channels, residual_channels, kernel_size,
                                      dilation=dilation, key=k2)
        # 1x1 conv for producing the new residual.
        conv_res_layer = eqx.nn.Conv1d(residual_channels, residual_channels, kernel_size=1,
                                       padding="SAME", key=k3)
        self.conv_res = VConv1d(conv_res_layer)
        # 1x1 conv for producing the skip connection.
        conv_skip_layer = eqx.nn.Conv1d(residual_channels, skip_channels, kernel_size=1,
                                        padding="SAME", key=k4)
        self.conv_skip = VConv1d(conv_skip_layer)

    def __call__(self, x: jnp.ndarray, *, key=None) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Tensor of shape [N, residual_channels, L]
        Returns:
          residual: Tensor of shape [N, residual_channels, L]
          skip: Tensor of shape [N, skip_channels, L]
        """
        # Compute the dilated convolutions.
        f = self.conv_filter(x)  # shape [N, residual_channels, L]
        g = self.conv_gate(x)    # shape [N, residual_channels, L]
        # Gated activation unit.
        activated = jnp.tanh(f) * jnn.sigmoid(g)
        # Compute skip connection.
        skip = self.conv_skip(activated)  # [N, skip_channels, L]
        # Compute residual (1×1 conv) and add input.
        res = self.conv_res(activated)    # [N, residual_channels, L]
        residual = res + x
        return residual, skip

# -------------------------------------------------------------------
# WaveNet
#
# The full WaveNet consists of:
#   1. A pre‑processing 1×1 convolution to map the input channels to residual_channels.
#   2. A stack of residual blocks with increasing dilation.
#   3. Post‑processing layers that sum skip connections, apply ReLU and 1×1 convs.
#   4. Extraction of the last time step, which is mapped to a scalar prediction.
# -------------------------------------------------------------------
class WaveNet(eqx.Module):
    pre_conv: VConv1d
    residual_blocks: list[WaveNetResidualBlock]
    post_conv1: VConv1d
    post_conv2: VConv1d

    def __init__(self, input_channels: int, residual_channels: int,
                 skip_channels: int, kernel_size: int, num_blocks: int,
                 dropout: float, *, key):
        # For simplicity, we ignore dropout in this simplified version.
        keys = jax.random.split(key, num_blocks + 4)
        # Preprocessing 1×1 conv to map input channels to residual_channels.
        pre_conv_layer = eqx.nn.Conv1d(input_channels, residual_channels, kernel_size=1,
                                       padding="SAME", key=keys[0])
        self.pre_conv = VConv1d(pre_conv_layer)
        # Create residual blocks. We use a doubling dilation schedule.
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** i
            block = WaveNetResidualBlock(residual_channels, kernel_size,
                                         dilation, skip_channels, key=keys[i+1])
            blocks.append(block)
        self.residual_blocks = blocks
        # Post-processing layers (1×1 convolutions).
        post_conv1_layer = eqx.nn.Conv1d(skip_channels, skip_channels, kernel_size=1,
                                         padding="SAME", key=keys[num_blocks+1])
        self.post_conv1 = VConv1d(post_conv1_layer)
        post_conv2_layer = eqx.nn.Conv1d(skip_channels, skip_channels, kernel_size=1,
                                         padding="SAME", key=keys[num_blocks+2])
        self.post_conv2 = VConv1d(post_conv2_layer)

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Args:
          x: Input tensor of shape [N, L, input_channels] (note: time is the second dimension)
             In our wrapper we will transpose the input so that channels come first.
        Returns:
          Output tensor of shape [N, skip_channels, L] after post-processing.
        """
        # x is expected to have shape [N, residual_channels, L] for the conv layers.
        # Apply pre‑conv.
        x = self.pre_conv(x)
        skip_sum = 0
        # Pass through each residual block.
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        # Post-processing.
        out = jnn.relu(skip_sum)
        out = self.post_conv1(out)
        out = jnn.relu(out)
        out = self.post_conv2(out)
        return out

# -------------------------------------------------------------------
# WaveNet Forecast Model
#
# A wrapper that:
#   1. Accepts input of shape [N, seq_len, D].
#   2. Transposes to channels-first ([N, D, seq_len]) for the WaveNet.
#   3. Applies the WaveNet.
#   4. Extracts the features from the final time step.
#   5. Maps them to a scalar prediction via a linear layer.
# -------------------------------------------------------------------
class WaveNetForecast(eqx.Module):
    wavenet: WaveNet
    final_linear: eqx.nn.Linear

    def __init__(self, input_channels: int, residual_channels: int, skip_channels: int,
                 kernel_size: int, num_blocks: int, *, key):
        keys = jax.random.split(key, 3)
        self.wavenet = WaveNet(input_channels, residual_channels, skip_channels,
                               kernel_size, num_blocks, dropout=0.0, key=keys[0])
        self.final_linear = eqx.nn.Linear(skip_channels, 1, key=keys[1])

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Args:
          x: Input tensor of shape [N, seq_len, D]
          key: Optional PRNG key.
        Returns:
          Predictions tensor of shape [N, 1] (prediction from the last time step).
        """
        # Transpose input to channels-first: [N, D, seq_len]
        x = jnp.transpose(x, (0, 2, 1))
        # Apply WaveNet. Output shape: [N, skip_channels, L]
        out = self.wavenet(x, key=key)
        # Extract the final time step along the temporal dimension.
        last = out[:, :, -1]  # shape [N, skip_channels]
        # Apply final linear layer to each sample.
        return jax.vmap(self.final_linear)(last)

# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == '__main__':
    # For example: a batch of 8 sequences, each of length 64, with 32 input channels.
    N, seq_len, D = 8, 64, 32
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (N, seq_len, D))
    
    # Define hyperparameters.
    residual_channels = 32
    skip_channels = 32
    kernel_size = 2       # small kernel size is common in WaveNet
    num_blocks = 6        # number of residual blocks (dilation doubles each block)
    
    model_key, run_key = jax.random.split(key)
    model = WaveNetForecast(input_channels=D, residual_channels=residual_channels,
                            skip_channels=skip_channels, kernel_size=kernel_size,
                            num_blocks=num_blocks, key=model_key)
    
    preds = model(x, key=run_key)
    print("Predictions shape:", preds.shape)  # Expected: (8, 1)
