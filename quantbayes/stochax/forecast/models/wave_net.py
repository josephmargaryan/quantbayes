import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr


# -------------------------------------------------------------------
# Helper: VConv1d
#
# Wraps an eqx.nn.Conv1d so that it can be applied to a batched input.
# Expects input shape [N, in_channels, L].
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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        *,
        key,
    ):
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = eqx.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding="VALID",  # We do manual padding below.
            dilation=dilation,
            key=key,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [N, in_channels, L]
        pad = self.dilation * (self.kernel_size - 1)
        # Pad only on the left along the time dimension.
        x_padded = jnp.pad(x, ((0, 0), (0, 0), (pad, 0)))
        return jax.vmap(self.conv)(x_padded)


# -------------------------------------------------------------------
# WaveNet Residual Block
#
# Applies two dilated causal convolutions (filter and gate) to compute a gated activation.
# A 1×1 convolution produces both a residual and a skip connection.
# -------------------------------------------------------------------
class WaveNetResidualBlock(eqx.Module):
    conv_filter: CausalConv1d
    conv_gate: CausalConv1d
    conv_res: VConv1d
    conv_skip: VConv1d

    def __init__(
        self,
        residual_channels: int,
        kernel_size: int,
        dilation: int,
        skip_channels: int,
        *,
        key,
    ):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.conv_filter = CausalConv1d(
            residual_channels, residual_channels, kernel_size, dilation=dilation, key=k1
        )
        self.conv_gate = CausalConv1d(
            residual_channels, residual_channels, kernel_size, dilation=dilation, key=k2
        )
        conv_res_layer = eqx.nn.Conv1d(
            residual_channels, residual_channels, kernel_size=1, padding="SAME", key=k3
        )
        self.conv_res = VConv1d(conv_res_layer)
        conv_skip_layer = eqx.nn.Conv1d(
            residual_channels, skip_channels, kernel_size=1, padding="SAME", key=k4
        )
        self.conv_skip = VConv1d(conv_skip_layer)

    def __call__(self, x: jnp.ndarray, *, key=None) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x: [N, residual_channels, L]
        f = self.conv_filter(x)
        g = self.conv_gate(x)
        activated = jnp.tanh(f) * jnn.sigmoid(g)
        skip = self.conv_skip(activated)
        res = self.conv_res(activated)
        residual = res + x
        return residual, skip


# -------------------------------------------------------------------
# WaveNet Module
#
# Consists of a pre‑processing conv, a stack of residual blocks with increasing dilation,
# and post‑processing 1×1 conv layers.
# -------------------------------------------------------------------
class WaveNet(eqx.Module):
    pre_conv: VConv1d
    residual_blocks: list[WaveNetResidualBlock]
    post_conv1: VConv1d
    post_conv2: VConv1d

    def __init__(
        self,
        input_channels: int,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        num_blocks: int,
        *,
        key,
    ):
        keys = jr.split(key, num_blocks + 4)
        pre_conv_layer = eqx.nn.Conv1d(
            input_channels,
            residual_channels,
            kernel_size=1,
            padding="SAME",
            key=keys[0],
        )
        self.pre_conv = VConv1d(pre_conv_layer)
        blocks = []
        for i in range(num_blocks):
            dilation = 2**i
            block = WaveNetResidualBlock(
                residual_channels, kernel_size, dilation, skip_channels, key=keys[i + 1]
            )
            blocks.append(block)
        self.residual_blocks = blocks
        post_conv1_layer = eqx.nn.Conv1d(
            skip_channels,
            skip_channels,
            kernel_size=1,
            padding="SAME",
            key=keys[num_blocks + 1],
        )
        self.post_conv1 = VConv1d(post_conv1_layer)
        post_conv2_layer = eqx.nn.Conv1d(
            skip_channels,
            skip_channels,
            kernel_size=1,
            padding="SAME",
            key=keys[num_blocks + 2],
        )
        self.post_conv2 = VConv1d(post_conv2_layer)

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # x: [N, residual_channels, L]
        x = self.pre_conv(x)
        skip_sum = 0
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        out = jnn.relu(skip_sum)
        out = self.post_conv1(out)
        out = jnn.relu(out)
        out = self.post_conv2(out)
        return out


# -------------------------------------------------------------------
# WaveNet Forecast Model
#
# Wraps the WaveNet model to accept raw input of shape [N, seq_len, D],
# transposes it to channels-first, applies WaveNet, extracts the last time step,
# and maps it to a scalar prediction.
# -------------------------------------------------------------------
class WaveNetForecast(eqx.Module):
    wavenet: WaveNet
    final_linear: eqx.nn.Linear

    def __init__(
        self,
        input_channels: int,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        num_blocks: int,
        *,
        key,
    ):
        keys = jr.split(key, 3)
        self.wavenet = WaveNet(
            input_channels,
            residual_channels,
            skip_channels,
            kernel_size,
            num_blocks,
            key=keys[0],
        )
        self.final_linear = eqx.nn.Linear(skip_channels, 1, key=keys[1])

    def __call__(self, x: jnp.ndarray, key, state) -> tuple[jnp.ndarray, any]:
        """
        Args:
          x: Single-sample input tensor of shape [seq_len, D]
          key: Optional PRNG key.
        Returns:
          A tuple (prediction, state) where prediction is a scalar (shape [1]).
        """
        # Add batch dimension.
        x = x[None, ...]  # now shape: [1, seq_len, D]
        # Transpose to channels-first: [1, D, seq_len]
        x = jnp.transpose(x, (0, 2, 1))
        out = self.wavenet(x, key=key)  # shape: [1, skip_channels, L]
        last = out[:, :, -1]  # shape: [1, skip_channels]
        pred = self.final_linear(last[0])  # shape: [1]
        return pred, state


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    import jax.random as jr

    from quantbayes.fake_data import create_synthetic_time_series
    from quantbayes.stochax.forecast import ForecastingModel

    # Create synthetic data.
    X_train, X_val, y_train, y_val = create_synthetic_time_series()
    # Raw input shape: [N, seq_len, input_dim] with input_dim == 1.
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_val = y_val.reshape(y_val.shape[0], -1)
    print(f"X train shape: {X_train.shape}")
    print(f"y train shape: {y_train.shape}")

    key = jr.PRNGKey(0)
    model, state = eqx.nn.make_with_state(WaveNetForecast)(
        input_channels=1,  # raw input channels
        residual_channels=32,  # model capacity for residual path
        skip_channels=4,  # skip connection channels
        kernel_size=3,
        num_blocks=4,
        key=key,
    )
    trainer = ForecastingModel(lr=1e-3)
    model, state = trainer.fit(
        model,
        state,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=500,
        patience=100,
        key=jr.PRNGKey(42),
    )
    preds = trainer.predict(model, state, X_val, key=jr.PRNGKey(123))
    print(f"preds shape {preds.shape}")
    trainer.visualize(y_val, preds, title="Forecast vs. Ground Truth")
