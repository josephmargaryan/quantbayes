#!/usr/bin/env python
"""
Plain U‑Net in Equinox.
Uses:
  - eqx.nn.Conv2d and eqx.nn.ConvTranspose2d for convolutions,
  - eqx.nn.BatchNorm for normalization (in inference mode),
  - eqx.nn.MaxPool2d for pooling.
Since BatchNorm now requires a state (initialized with the module),
we create the state via eqx.nn.State(self.bnX).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from equinox import nn


# -------------------------------------------------------
# Define a basic convolutional block:
# Two times (Conv2d → BatchNorm → ReLU)
# -------------------------------------------------------
class ConvBlock(eqx.Module):
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    bn1: nn.BatchNorm
    bn2: nn.BatchNorm
    bn1_state: eqx.nn.State
    bn2_state: eqx.nn.State

    def __init__(self, in_channels: int, out_channels: int, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        # 3×3 convolution with padding=1 to preserve spatial dimensions.
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, key=k1
        )
        self.bn1 = nn.BatchNorm(out_channels, axis_name="batch", inference=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, key=k3
        )
        self.bn2 = nn.BatchNorm(out_channels, axis_name="batch", inference=True)
        # Create states using the respective BatchNorm modules.
        self.bn1_state = eqx.nn.State(self.bn1)
        self.bn2_state = eqx.nn.State(self.bn2)

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        x = self.conv1(x, key=key)
        x, _ = self.bn1(x, state=self.bn1_state)
        x = jax.nn.relu(x)
        x = self.conv2(x, key=key)
        x, _ = self.bn2(x, state=self.bn2_state)
        x = jax.nn.relu(x)
        return x


# -------------------------------------------------------
# Define an up‐sampling block:
# Upsample using ConvTranspose2d, concatenate the skip connection,
# then apply a ConvBlock.
# -------------------------------------------------------
class UpConvBlock(eqx.Module):
    upconv: nn.ConvTranspose2d
    conv: ConvBlock

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, key):
        k1, k2 = jax.random.split(key, 2)
        # Use a 2×2 kernel and stride 2 to upsample.
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, key=k1
        )
        # After upsampling, the skip connection (with skip_channels) is concatenated.
        self.conv = ConvBlock(
            in_channels=out_channels + skip_channels, out_channels=out_channels, key=k2
        )

    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray, *, key=None) -> jnp.ndarray:
        x = self.upconv(x, key=key)
        # Concatenate along the channel dimension (axis 0 for channel‐first tensors)
        x = jnp.concatenate([skip, x], axis=0)
        x = self.conv(x, key=key)
        return x


# -------------------------------------------------------
# Define the full U‑Net model.
# -------------------------------------------------------
class UNet(eqx.Module):
    # Encoder blocks
    enc1: ConvBlock
    enc2: ConvBlock
    enc3: ConvBlock
    enc4: ConvBlock
    # Bottleneck block
    bottleneck: ConvBlock
    # Decoder blocks
    dec1: UpConvBlock
    dec2: UpConvBlock
    dec3: UpConvBlock
    dec4: UpConvBlock
    # Final 1×1 convolution to produce the desired output channels.
    final_conv: nn.Conv2d
    # Built‐in MaxPool2d (with kernel size 2 and stride 2)
    pool: nn.MaxPool2d

    def __init__(
        self, in_channels: int, out_channels: int, base_channels: int = 64, *, key
    ):
        keys = jax.random.split(key, 10)
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
        self.final_conv = nn.Conv2d(
            base_channels, out_channels, kernel_size=1, key=keys[9]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # Encoder path
        e1 = self.enc1(x, key=key)
        p1 = self.pool(e1)
        e2 = self.enc2(p1, key=key)
        p2 = self.pool(e2)
        e3 = self.enc3(p2, key=key)
        p3 = self.pool(e3)
        e4 = self.enc4(p3, key=key)
        p4 = self.pool(e4)
        # Bottleneck
        b = self.bottleneck(p4, key=key)
        # Decoder path with skip‐connections
        d1 = self.dec1(b, e4, key=key)
        d2 = self.dec2(d1, e3, key=key)
        d3 = self.dec3(d2, e2, key=key)
        d4 = self.dec4(d3, e1, key=key)
        out = self.final_conv(d4, key=key)
        return out


# -------------------------------------------------------
# Test the UNet model.
# -------------------------------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    # Create a UNet: for example, 3‐channel input and 1‐channel output.
    model = UNet(in_channels=3, out_channels=1, base_channels=64, key=key)
    # Create a dummy input image with shape (channels, height, width).
    x = jax.random.normal(key, (3, 256, 256))
    y = model(x, key=key)
    print("UNet output shape:", y.shape)
