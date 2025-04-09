import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr


# -------------------------------------------------------------------
# Helper Module: CausalConv1d
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
            padding="VALID",  # We do manual padding.
            dilation=dilation,
            key=key,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: shape (N, in_channels, L)
        pad = self.dilation * (self.kernel_size - 1)
        x_padded = jnp.pad(x, ((0, 0), (0, 0), (pad, 0)))
        # vmap over the batch dimension so that each sample (of shape (in_channels, L)) is processed.
        return jax.vmap(self.conv)(x_padded)


# -------------------------------------------------------------------
# TCN Block (Residual Block)
# -------------------------------------------------------------------
class TCNBlock(eqx.Module):
    conv1: CausalConv1d
    conv2: CausalConv1d
    dropout: eqx.nn.Dropout
    activation: callable = eqx.field(static=True)
    resample: eqx.Module = eqx.field(static=True, default=None)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout_p: float,
        *,
        key,
    ):
        keys = jax.random.split(key, 4)
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, key=keys[0]
        )
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation=dilation, key=keys[1]
        )
        self.dropout = eqx.nn.Dropout(p=dropout_p, inference=False)
        self.activation = jnn.relu
        if in_channels != out_channels:
            self.resample = eqx.nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding="SAME", key=keys[2]
            )
        else:
            self.resample = None

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # x: shape (N, C_in, L)
        if key is not None:
            key1, key2 = jax.random.split(key)
        else:
            key1 = key2 = None
        y = self.conv1(x)
        y = self.activation(y)
        y = self.dropout(y, key=key1)
        y = self.conv2(y)
        y = self.activation(y)
        y = self.dropout(y, key=key2)
        if self.resample is None:
            res = x
        else:
            res = jax.vmap(self.resample)(x)
        return y + res


# -------------------------------------------------------------------
# Full TCN Module
# -------------------------------------------------------------------
class TCN(eqx.Module):
    blocks: list[TCNBlock]

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        num_levels: int,
        kernel_size: int,
        dropout_p: float,
        *,
        key,
    ):
        keys = jax.random.split(key, num_levels)
        blocks = []
        for i in range(num_levels):
            dilation = 2**i
            block_in = in_channels if i == 0 else num_filters
            block = TCNBlock(
                block_in, num_filters, kernel_size, dilation, dropout_p, key=keys[i]
            )
            blocks.append(block)
        self.blocks = blocks

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # x: shape (N, in_channels, L)
        if key is not None:
            block_keys = jax.random.split(key, len(self.blocks))
        else:
            block_keys = [None] * len(self.blocks)
        for block, k in zip(self.blocks, block_keys):
            x = block(x, key=k)
        return x


# -------------------------------------------------------------------
# TCN Forecast Model
# -------------------------------------------------------------------
class TCNForecast(eqx.Module):
    tcn: TCN
    final_linear: eqx.nn.Linear
    in_channels: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        num_levels: int,
        kernel_size: int,
        dropout_p: float,
        *,
        key,
    ):
        keys = jr.split(key, 2)
        self.tcn = TCN(
            in_channels, num_filters, num_levels, kernel_size, dropout_p, key=keys[0]
        )
        self.final_linear = eqx.nn.Linear(num_filters, 1, key=keys[1])
        self.in_channels = in_channels

    def __call__(self, x: jnp.ndarray, key, state) -> tuple[jnp.ndarray, any]:
        """
        Args:
          x: A single sample of shape [seq_len, D], where D == in_channels.
          key: Optional PRNG key.
        Returns:
          A prediction of shape [1] and the state.
        """
        # Add a batch dimension: [1, seq_len, D]
        x = x[None, ...]
        # Rearrange from [1, seq_len, D] to [1, D, seq_len] for Conv1d.
        x = jnp.transpose(x, (0, 2, 1))
        y = self.tcn(x, key=key)  # y: shape [1, num_filters, seq_len]
        # Select the last time step along the temporal dimension.
        last = y[:, :, -1]  # shape: [1, num_filters]
        # Remove the batch dimension by taking the first (and only) sample.
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
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_val = y_val.reshape(y_val.shape[0], -1)
    print(f"X train shape: {X_train.shape}")
    print(f"y train shape: {y_train.shape}")

    key = jr.PRNGKey(0)
    model, state = eqx.nn.make_with_state(TCNForecast)(
        in_channels=1,
        num_filters=64,
        num_levels=4,
        kernel_size=4,
        dropout_p=0.1,
        key=key,
    )
    trainer = ForecastingModel(lr=1e-4)
    trainer.fit(
        model, state, X_train, y_train, X_val, y_val, num_epochs=500, patience=100
    )
    preds = trainer.predict(model, state, X_val, key=jr.PRNGKey(123))
    print(f"preds shape {preds.shape}")
    trainer.visualize(y_val, preds, title="Forecast vs. Ground Truth")
