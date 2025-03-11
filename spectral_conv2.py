import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import math

from quantbayes.stochax.layers import (
    SpectralTransposed2d,
    SpectralConv1d,
    SpectralConv2d
)

# ---------------------------------------------------------------
# SpectralLeNet_Direct (using the direct spectral layers)
# ---------------------------------------------------------------
class SpectralLeNet_Direct(eqx.Module):
    conv1: SpectralConv2d
    conv2: SpectralConv2d
    fc: eqx.nn.Linear

    def __init__(self, key: jax.random.PRNGKey):
        k1, k2, k3 = jr.split(key, 3)
        in_channels = 1
        out_channels1 = 8
        out_channels2 = 16
        fft_size = (28, 28)
        self.conv1 = SpectralConv2d(
            in_channels=in_channels,
            out_channels=out_channels1,
            fft_size=fft_size,
            key=k1,
            init_scale=0.1,
            alpha=1.0,
        )
        self.conv2 = SpectralConv2d(
            in_channels=out_channels1,
            out_channels=out_channels2,
            fft_size=fft_size,
            key=k2,
            init_scale=0.1,
            alpha=1.0,
        )
        self.fc = eqx.nn.Linear(in_features=out_channels2 * 28 * 28, out_features=10, key=k3)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        return jax.vmap(self.fc)(x)


# ---------------------------------------------------------------
# SpectralUNet_Direct (using the direct spectral layers)
# ---------------------------------------------------------------
def pool(x):
    window = (1, 1, 2, 2)
    strides = (1, 1, 2, 2)
    pooled = jax.lax.reduce_window(x, 0.0, jax.lax.add, window, strides, padding="SAME")
    return pooled / 4.0

def upsample(x, factor=2):
    return jnp.repeat(jnp.repeat(x, factor, axis=-2), factor, axis=-1)

class SpectralUNet_Direct(eqx.Module):
    conv1: SpectralConv2d
    conv2: SpectralConv2d
    bottleneck: SpectralConv2d
    upconv1: SpectralTransposed2d
    conv3: SpectralConv2d
    upconv2: SpectralTransposed2d
    conv4: SpectralConv2d
    out_channels: int = eqx.static_field()

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16, key: jax.random.PRNGKey = jr.PRNGKey(0)):
        self.out_channels = out_channels
        keys = jr.split(key, 7)
        self.conv1 = SpectralConv2d(in_channels, base_channels, fft_size=(128, 128), key=keys[0], init_scale=0.1, alpha=1.0)
        self.conv2 = SpectralConv2d(base_channels, base_channels*2, fft_size=(64, 64), key=keys[1], init_scale=0.1, alpha=1.0)
        self.bottleneck = SpectralConv2d(base_channels*2, base_channels*4, fft_size=(32, 32), key=keys[2], init_scale=0.1, alpha=1.0)
        self.upconv1 = SpectralTransposed2d(out_channels=base_channels*4, in_channels=base_channels*2, fft_size=(64, 64), key=keys[3], init_scale=0.1)
        self.conv3 = SpectralConv2d(base_channels*4, base_channels*2, fft_size=(64, 64), key=keys[4], init_scale=0.1, alpha=1.0)
        self.upconv2 = SpectralTransposed2d(out_channels=base_channels*2, in_channels=base_channels, fft_size=(128, 128), key=keys[5], init_scale=0.1)
        self.conv4 = SpectralConv2d(base_channels*2, out_channels, fft_size=(128, 128), key=keys[6], init_scale=0.1, alpha=1.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x1 = self.conv1(x)
        x1_act = jax.nn.relu(x1)
        x1_pool = pool(x1_act)
        x2 = self.conv2(x1_pool)
        x2_act = jax.nn.relu(x2)
        x2_pool = pool(x2_act)
        x3 = self.bottleneck(x2_pool)
        x3_act = jax.nn.relu(x3)
        x3_up = upsample(x3_act, factor=2)
        x4 = self.upconv1(x3_up)
        x4_cat = jnp.concatenate([x4, x2_act], axis=1)
        x5 = self.conv3(x4_cat)
        x5_act = jax.nn.relu(x5)
        x5_up = upsample(x5_act, factor=2)
        x6 = self.upconv2(x5_up)
        x6_cat = jnp.concatenate([x6, x1_act], axis=1)
        x7 = self.conv4(x6_cat)
        return x7


# ---------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------
def test_spectral_conv1d_direct():
    print("Testing SpectralConv1d_Direct...")
    key = jr.PRNGKey(0)
    conv1d = SpectralConv1d(in_channels=3, out_channels=5, fft_size=16, key=key, init_scale=0.1, alpha=1.0)
    x = jr.normal(key, (4, 3, 20))
    y = conv1d(x)
    print("  Input shape :", x.shape)
    print("  Output shape:", y.shape)

def test_spectral_transposed2d_direct():
    print("Testing SpectralTransposed2d_Direct...")
    key = jr.PRNGKey(1)
    trans_conv = SpectralTransposed2d(out_channels=8, in_channels=3, fft_size=(28, 28), key=key, init_scale=0.1)
    x = jr.normal(key, (4, 8, 28, 28))
    y = trans_conv(x)
    print("  Input shape :", x.shape)
    print("  Output shape:", y.shape)

def test_spectral_le_net_direct():
    print("Testing SpectralLeNet_Direct...")
    key = jr.PRNGKey(42)
    model = SpectralLeNet_Direct(key)
    x = jr.normal(key, (2, 1, 28, 28))
    logits = model(x)
    print("  Input shape :", x.shape)
    print("  Output shape:", logits.shape)

def test_spectral_unet_direct():
    print("Testing SpectralUNet_Direct...")
    key = jr.PRNGKey(123)
    model = SpectralUNet_Direct(in_channels=1, out_channels=1, base_channels=16, key=key)
    x = jr.normal(key, (2, 1, 128, 128))
    y = model(x)
    print("  Input shape :", x.shape)
    print("  Output shape:", y.shape)

if __name__ == "__main__":
    test_spectral_conv1d_direct()
    print()
    test_spectral_transposed2d_direct()
    print()
    test_spectral_le_net_direct()
    print()
    test_spectral_unet_direct()
