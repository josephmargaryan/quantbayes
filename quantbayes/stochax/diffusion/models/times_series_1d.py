import math
from typing import List, Union
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# Helper: key splitting function.
def key_split_allowing_none(key):
    if key is None:
        return key, None
    else:
        return jr.split(key)


########################################
# A 1D time-embedding (sinusoidal)
########################################
class SinusoidalTimeEmb(eqx.Module):
    emb: jax.Array

    def __init__(self, dim: int):
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: scalar (or shape ())
        emb = x * self.emb
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


########################################
# 1D Residual Block for Time-Series with Dropout
########################################
class TimeResBlock1d(eqx.Module):
    dim_in: int
    dim_out: int
    time_emb_dim: int
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    time_proj: eqx.nn.Linear
    skip_conv: eqx.nn.Conv1d
    dropout: Union[eqx.nn.Dropout, eqx.nn.Identity]
    up: bool
    down: bool

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        time_emb_dim: int,
        *,
        up: bool = False,
        down: bool = False,
        dropout: float = 0.0,
        key: jr.PRNGKey,
    ):
        """
        1D ResNet-style block that operates on tensors of shape [C, L] with time
        injection. If dropout>0, dropout will be applied.
        """
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.time_emb_dim = time_emb_dim
        self.up = up
        self.down = down

        # Split keys for the submodules.
        k1, k2, k3, k4, k5 = jr.split(key, 5)
        self.norm1 = eqx.nn.GroupNorm(1, dim_in)
        self.conv1 = eqx.nn.Conv1d(dim_in, dim_out, kernel_size=3, padding=1, key=k1)
        self.norm2 = eqx.nn.GroupNorm(1, dim_out)
        self.conv2 = eqx.nn.Conv1d(dim_out, dim_out, kernel_size=3, padding=1, key=k2)
        self.time_proj = eqx.nn.Linear(time_emb_dim, dim_out, key=k3)
        self.skip_conv = eqx.nn.Conv1d(dim_in, dim_out, kernel_size=1, key=k4)
        # Setup dropout (if dropout > 0, otherwise use Identity)
        self.dropout = (
            eqx.nn.Dropout(p=dropout, inference=False)
            if dropout > 0
            else eqx.nn.Identity()
        )

    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        x: shape [C, L]
        t_emb: shape [time_emb_dim]
        key: PRNG key for dropout
        """
        h = self.norm1(x)
        h = jax.nn.silu(h)
        h = self.conv1(h)
        # Inject time embedding (projected and broadcast along L)
        h = h + self.time_proj(t_emb)[..., None]
        h = self.norm2(h)
        h = jax.nn.silu(h)
        # Apply dropout (requires a key if dropout is active)
        if isinstance(self.dropout, eqx.nn.Dropout):
            if key is None:
                raise RuntimeError("Dropout requires a valid key when dropout>0.")
            h = self.dropout(h, key=key)
        else:
            h = self.dropout(h)
        h = self.conv2(h)

        # Skip connection: if dimensions differ or if up/down is used, apply a 1x1 conv.
        if (self.dim_in != self.dim_out) or self.up or self.down:
            x = self.skip_conv(x)
        return (x + h) / jnp.sqrt(2)


########################################
# 1D Convolutional UNet for Time-Series with Dropout support
########################################
class ConvTimeUNet(eqx.Module):
    """
    A 1D UNet-style model for time-series diffusion.
    The model expects inputs of shape (L,) or (batch, L) and returns outputs of the same shape.
    """

    time_emb: SinusoidalTimeEmb
    time_mlp: eqx.nn.MLP
    first_conv: eqx.nn.Conv1d
    down_blocks: List[List[TimeResBlock1d]]
    mid_block1: TimeResBlock1d
    mid_block2: TimeResBlock1d
    up_blocks: List[List[TimeResBlock1d]]
    final_norm: eqx.nn.GroupNorm
    final_conv: eqx.nn.Conv1d

    def __init__(
        self,
        seq_length: int,
        in_channels: int,
        hidden_dim: int,
        dim_mults: List[int],
        num_res_blocks: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        *,
        key: jr.PRNGKey,
    ):
        """
        seq_length: length of each time-series
        in_channels: number of channels (for single-channel, use 1)
        hidden_dim: base hidden dimension
        dim_mults: e.g. [2, 2] expands channels as hidden_dim -> 2*hidden_dim -> 2*hidden_dim
        num_res_blocks: number of residual blocks per level
        time_emb_dim: dimension of the time embedding
        dropout: dropout probability (if > 0, dropout is applied in blocks)
        """
        keys = jr.split(key, 6)

        # Time embedding modules
        self.time_emb = SinusoidalTimeEmb(time_emb_dim)
        self.time_mlp = eqx.nn.MLP(
            in_size=time_emb_dim,
            out_size=time_emb_dim,
            width_size=4 * time_emb_dim,
            depth=2,
            key=keys[0],
        )

        # First convolution: input shape will be (in_channels, L)
        self.first_conv = eqx.nn.Conv1d(
            in_channels, hidden_dim, kernel_size=3, padding=1, key=keys[1]
        )

        # Downsampling blocks
        dims = [hidden_dim] + [hidden_dim * m for m in dim_mults]
        self.down_blocks = []
        dkey = keys[2]
        dkeys = jr.split(dkey, len(dim_mults) * num_res_blocks)
        idx = 0
        current_dim = hidden_dim
        for m in dim_mults:
            out_dim = hidden_dim * m
            level_blocks = []
            for _ in range(num_res_blocks):
                level_blocks.append(
                    TimeResBlock1d(
                        dim_in=current_dim,
                        dim_out=out_dim,
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                        key=dkeys[idx],
                    )
                )
                idx += 1
                current_dim = out_dim
            self.down_blocks.append(level_blocks)

        # Middle (bottleneck) blocks
        self.mid_block1 = TimeResBlock1d(
            current_dim, current_dim, time_emb_dim, dropout=dropout, key=keys[3]
        )
        self.mid_block2 = TimeResBlock1d(
            current_dim, current_dim, time_emb_dim, dropout=dropout, key=keys[4]
        )

        # Upsampling blocks (for simplicity, we mirror the down blocks but without skip concatenation)
        self.up_blocks = []
        ukey = keys[5]
        ukeys = jr.split(ukey, len(dim_mults) * num_res_blocks)
        idx = 0
        # Here we simply reduce channels back to hidden_dim per level.
        for m in reversed(dim_mults):
            level_blocks = []
            for _ in range(num_res_blocks):
                # For simplicity, we use the same dimension for input and output.
                level_blocks.append(
                    TimeResBlock1d(
                        dim_in=current_dim,
                        dim_out=hidden_dim,  # reducing to hidden_dim
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                        key=ukeys[idx],
                    )
                )
                idx += 1
                current_dim = hidden_dim
            self.up_blocks.append(level_blocks)

        # Final normalization and conv to map back to original in_channels.
        self.final_norm = eqx.nn.GroupNorm(1, hidden_dim)
        self.final_conv = eqx.nn.Conv1d(
            hidden_dim, in_channels, kernel_size=1, key=jr.PRNGKey(999)
        )

    def _forward(self, t: jnp.ndarray, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        t: scalar diffusion time
        x: a single time series tensor of shape (C, L)
        Returns a tensor of shape (C, L)
        """
        # 1) Time embedding
        emb = self.time_emb(t)
        emb = self.time_mlp(emb)

        # 2) First convolution
        h = self.first_conv(x)
        skips = [h]

        # 3) Down path
        for level in self.down_blocks:
            for block in level:
                key, subkey = key_split_allowing_none(key)
                h = block(h, emb, key=subkey)
                skips.append(h)

        # 4) Middle
        key, subkey = key_split_allowing_none(key)
        h = self.mid_block1(h, emb, key=subkey)
        key, subkey = key_split_allowing_none(key)
        h = self.mid_block2(h, emb, key=subkey)

        # 5) Up path (for simplicity, we do not merge skip connections here)
        for level in self.up_blocks:
            for block in level:
                key, subkey = key_split_allowing_none(key)
                h = block(h, emb, key=subkey)

        # 6) Final normalization and output convolution
        h = self.final_norm(h)
        h = jax.nn.silu(h)
        return self.final_conv(h)

    def __call__(self, t: jnp.ndarray, y: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Accepts time series data in various shapes and returns output with the same shape.

        If y.ndim == 1: shape (L,) is treated as a single sample (converted to (1, L)).
        If y.ndim == 2: interpreted as either (batch, L) or (C, L).
        If y.ndim == 3: interpreted as (batch, C, L).
        """

        # Ensure that single-sample inputs have a channel dimension.
        def single_sample_forward(sample, k):
            if sample.ndim == 1:
                sample = sample[None, :]  # shape (1, L)
            return self._forward(t, sample, key=k)

        if y.ndim == 1:
            # (L,) -> (1, L)
            x = y[None, :]
            return self._forward(t, x, key=key)
        elif y.ndim == 2:
            # (B, L) assumed to be a batch of single-channel time series,
            # so vmap over batch dimension.
            if key is not None:
                keys = jr.split(key, y.shape[0])
                return jax.vmap(lambda sample, k: single_sample_forward(sample, k))(
                    y, keys
                )
            else:
                return jax.vmap(lambda sample: single_sample_forward(sample, None))(y)
        elif y.ndim == 3:
            # (B, C, L)
            if key is not None:
                keys = jr.split(key, y.shape[0])
                return jax.vmap(lambda sample, k: self._forward(t, sample, key=k))(
                    y, keys
                )
            else:
                return jax.vmap(lambda sample: self._forward(t, sample, key=None))(y)
        else:
            raise ValueError(f"Unsupported shape for y: {y.shape}")
