#!/usr/bin/env python
"""
Attention U‑Net in Equinox.
This variant adds attention gates on the skip‐connections.
Uses:
  - nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm, and nn.MaxPool2d.
As before, we create the BatchNorm state using eqx.nn.State(module).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from equinox import nn

# -------------------------------------------------------
# Define a basic ConvBlock (as in the plain U‑Net).
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, key=k1)
        self.bn1   = nn.BatchNorm(out_channels, axis_name="batch", inference=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, key=k3)
        self.bn2   = nn.BatchNorm(out_channels, axis_name="batch", inference=True)
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
# Define the AttentionGate.
# It computes an attention coefficient via 1×1 convolutions.
# -------------------------------------------------------
class AttentionGate(eqx.Module):
    conv_g: nn.Conv2d  # for the gating signal
    conv_x: nn.Conv2d  # for the skip connection features
    conv_psi: nn.Conv2d  # to produce attention coefficients

    def __init__(self, F_g: int, F_l: int, F_int: int, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv_g = nn.Conv2d(F_g, F_int, kernel_size=1, key=k1)
        self.conv_x = nn.Conv2d(F_l, F_int, kernel_size=1, key=k2)
        self.conv_psi = nn.Conv2d(F_int, 1, kernel_size=1, key=k3)

    def __call__(self, x: jnp.ndarray, g: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # x: skip connection features (shape: (F_l, H, W))
        # g: gating signal (shape: (F_g, H, W))
        g1 = self.conv_g(g, key=key)
        x1 = self.conv_x(x, key=key)
        psi = jax.nn.relu(g1 + x1)
        psi = jax.nn.sigmoid(self.conv_psi(psi, key=key))
        return x * psi

# -------------------------------------------------------
# Define an up‐sampling block that uses an AttentionGate.
# -------------------------------------------------------
class UpConvAttentionBlock(eqx.Module):
    upconv: nn.ConvTranspose2d
    att_gate: AttentionGate
    conv: ConvBlock

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, key=k1)
        # F_int is typically set to skip_channels // 2.
        self.att_gate = AttentionGate(F_g=out_channels, F_l=skip_channels, F_int=skip_channels // 2, key=k2)
        self.conv = ConvBlock(in_channels=out_channels + skip_channels, out_channels=out_channels, key=k3)

    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray, *, key=None) -> jnp.ndarray:
        x = self.upconv(x, key=key)
        skip_att = self.att_gate(skip, x, key=key)
        x = jnp.concatenate([skip_att, x], axis=0)
        x = self.conv(x, key=key)
        return x

# -------------------------------------------------------
# Define the full Attention U‑Net model.
# -------------------------------------------------------
class AttentionUNet(eqx.Module):
    # Encoder blocks
    enc1: ConvBlock
    enc2: ConvBlock
    enc3: ConvBlock
    enc4: ConvBlock
    # Bottleneck
    bottleneck: ConvBlock
    # Decoder blocks (with attention gates)
    dec1: UpConvAttentionBlock
    dec2: UpConvAttentionBlock
    dec3: UpConvAttentionBlock
    dec4: UpConvAttentionBlock
    # Final 1×1 convolution
    final_conv: nn.Conv2d
    pool: nn.MaxPool2d

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64, *, key):
        keys = jax.random.split(key, 10)
        self.enc1 = ConvBlock(in_channels, base_channels, key=keys[0])
        self.enc2 = ConvBlock(base_channels, base_channels * 2, key=keys[1])
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, key=keys[2])
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, key=keys[3])
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16, key=keys[4])
        self.dec1 = UpConvAttentionBlock(base_channels * 16, base_channels * 8, base_channels * 8, key=keys[5])
        self.dec2 = UpConvAttentionBlock(base_channels * 8, base_channels * 4, base_channels * 4, key=keys[6])
        self.dec3 = UpConvAttentionBlock(base_channels * 4, base_channels * 2, base_channels * 2, key=keys[7])
        self.dec4 = UpConvAttentionBlock(base_channels * 2, base_channels, base_channels, key=keys[8])
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1, key=keys[9])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # Encoder
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
        # Decoder with attention gates and skip connections
        d1 = self.dec1(b, e4, key=key)
        d2 = self.dec2(d1, e3, key=key)
        d3 = self.dec3(d2, e2, key=key)
        d4 = self.dec4(d3, e1, key=key)
        out = self.final_conv(d4, key=key)
        return out

# -------------------------------------------------------
# Test the Attention U‑Net model.
# -------------------------------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    # Create an AttentionUNet: for example, 3 input channels and 1 output channel.
    model = AttentionUNet(in_channels=3, out_channels=1, base_channels=64, key=key)
    # Create a dummy input image with shape (channels, height, width).
    x = jax.random.normal(key, (3, 256, 256))
    y = model(x, key=key)
    print("Attention UNet output shape:", y.shape)
