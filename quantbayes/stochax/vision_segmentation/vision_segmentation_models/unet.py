from typing import Tuple
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx


class ConvBlock(eqx.Module):
    """
    Basic 2x(Conv2D + BatchNorm + ReLU) block, using Equinox's recommended
    approach of storing batchnorm parameters in the module but the running
    stats in the separate 'state' object.
    """

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm

    def __init__(self, in_channels: int, out_channels: int, *, key: jnp.ndarray):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, key=k1
        )
        # axis_name="batch" will be used when we vmap this over a batch dimension.
        self.bn1 = eqx.nn.BatchNorm(out_channels, axis_name="batch", inference=False)
        self.conv2 = eqx.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, key=k3
        )
        self.bn2 = eqx.nn.BatchNorm(out_channels, axis_name="batch", inference=False)

    def __call__(
        self, x: jnp.ndarray, state: eqx.nn.State, *, key: jnp.ndarray
    ) -> Tuple[jnp.ndarray, eqx.nn.State]:
        # We'll split the provided `key` to feed each conv separately if we want
        k1, k2 = jr.split(key, 2)

        x = self.conv1(x, key=k1)
        x, state = self.bn1(x, state=state)
        x = jax.nn.relu(x)

        x = self.conv2(x, key=k2)
        x, state = self.bn2(x, state=state)
        x = jax.nn.relu(x)
        return x, state


class UpConvBlock(eqx.Module):
    """
    Upsample with ConvTranspose2D, concatenate skip connection,
    then apply a ConvBlock.
    """

    upconv: eqx.nn.ConvTranspose2d
    conv: ConvBlock

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, key):
        k1, k2 = jr.split(key, 2)
        self.upconv = eqx.nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, key=k1
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, key=k2)

    def __call__(
        self,
        x: jnp.ndarray,
        skip: jnp.ndarray,
        state: eqx.nn.State,
        *,
        key: jnp.ndarray
    ) -> Tuple[jnp.ndarray, eqx.nn.State]:
        k1, k2 = jr.split(key, 2)
        x = self.upconv(x, key=k1)
        # Concatenate along the channel dimension (axis 0 => (C, H, W)).
        # If your data is (B, C, H, W) then typically channel dimension is 1.
        # Here we'll assume channel-first. We just have to be consistent.
        x = jnp.concatenate([skip, x], axis=0)

        x, state = self.conv(x, state=state, key=k2)
        return x, state


class UNet(eqx.Module):
    """
    Standard UNet for segmentation: 4 levels down, bottleneck, then 4 up.
    """

    # Encoder blocks
    enc1: ConvBlock
    enc2: ConvBlock
    enc3: ConvBlock
    enc4: ConvBlock
    # Bottleneck
    bottleneck: ConvBlock
    # Decoder blocks
    dec1: UpConvBlock
    dec2: UpConvBlock
    dec3: UpConvBlock
    dec4: UpConvBlock
    # Final 1x1 conv
    final_conv: eqx.nn.Conv2d
    # MaxPool
    pool: eqx.nn.MaxPool2d

    def __init__(
        self, in_channels: int, out_channels: int, base_channels: int = 64, *, key
    ):
        keys = jr.split(key, 10)
        self.enc1 = ConvBlock(in_channels, base_channels, key=keys[0])
        self.enc2 = ConvBlock(base_channels, base_channels * 2, key=keys[1])
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, key=keys[2])
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, key=keys[3])

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16, key=keys[4])

        self.dec1 = UpConvBlock(
            base_channels * 16, base_channels * 8, base_channels * 8, key=keys[5]
        )
        self.dec2 = UpConvBlock(
            base_channels * 8, base_channels * 4, base_channels * 4, key=keys[6]
        )
        self.dec3 = UpConvBlock(
            base_channels * 4, base_channels * 2, base_channels * 2, key=keys[7]
        )
        self.dec4 = UpConvBlock(
            base_channels * 2, base_channels, base_channels, key=keys[8]
        )

        self.final_conv = eqx.nn.Conv2d(
            base_channels, out_channels, kernel_size=1, key=keys[9]
        )
        self.pool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(
        self, x: jnp.ndarray, state: eqx.nn.State, *, key: jnp.ndarray
    ) -> Tuple[jnp.ndarray, eqx.nn.State]:
        """
        x: shape (C, H, W) if single sample,
           or use vmap for a batch: (batch, C, H, W).
        """
        # We'll split the key for each stage.
        k1, k2, k3, k4, k5, k6, k7, k8, k9 = jr.split(key, 9)

        e1, state = self.enc1(x, state=state, key=k1)
        p1 = self.pool(e1)

        e2, state = self.enc2(p1, state=state, key=k2)
        p2 = self.pool(e2)

        e3, state = self.enc3(p2, state=state, key=k3)
        p3 = self.pool(e3)

        e4, state = self.enc4(p3, state=state, key=k4)
        p4 = self.pool(e4)

        b, state = self.bottleneck(p4, state=state, key=k5)

        d1, state = self.dec1(b, e4, state=state, key=k6)
        d2, state = self.dec2(d1, e3, state=state, key=k7)
        d3, state = self.dec3(d2, e2, state=state, key=k8)
        d4, state = self.dec4(d3, e1, state=state, key=k9)

        out = self.final_conv(d4)
        return out, state
