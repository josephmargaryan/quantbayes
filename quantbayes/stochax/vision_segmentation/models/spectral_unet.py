from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# --- A simple spectral 2D convolution using FFT ---
@jax.custom_jvp
def spectral_conv2d(x: jnp.ndarray, fft_mask: jnp.ndarray) -> jnp.ndarray:
    """
    Applies spectral modulation by performing FFT on spatial dimensions,
    multiplying by a Fourier mask, then applying the inverse FFT.
    x: (H, W)
    fft_mask: (H, W) learnable Fourier mask.
    """
    X_fft = jnp.fft.fftn(x, axes=(0, 1))
    Y_fft = X_fft * fft_mask
    y = jnp.fft.ifftn(Y_fft, axes=(0, 1)).real
    return y


@spectral_conv2d.defjvp
def spectral_conv2d_jvp(primals, tangents):
    x, fft_mask = primals
    dx, dmask = tangents
    X_fft = jnp.fft.fftn(x, axes=(0, 1))
    Y_fft = X_fft * fft_mask
    y = jnp.fft.ifftn(Y_fft, axes=(0, 1)).real

    dX_fft = jnp.fft.fftn(dx, axes=(0, 1)) if dx is not None else 0.0
    term1 = dX_fft * fft_mask
    term2 = X_fft * dmask if dmask is not None else 0.0
    dy = jnp.fft.ifftn(term1 + term2, axes=(0, 1)).real
    return y, dy


# --- Spectral Convolution Block ---
class SpectralConvBlock(eqx.Module):
    conv: eqx.nn.Conv2d  # standard conv to map channels
    fft_mask: jnp.ndarray  # learnable Fourier mask per output channel
    use_spectral: bool = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_shape: Tuple[int, int],
        *,
        key
    ):
        k1, k2 = jr.split(key)
        # First, use a standard conv to map to out_channels.
        self.conv = eqx.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, key=k1
        )
        # Create a Fourier mask for each output channel.
        H, W = spatial_shape
        self.fft_mask = jr.normal(k2, (out_channels, H, W))
        self.use_spectral = True

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is assumed to have shape (C, H, W)
        x = self.conv(x)  # now shape becomes (out_channels, H, W)
        if self.use_spectral:
            # Map over the channel axis: now both x and fft_mask have matching channel dimension.
            return jax.vmap(lambda xi, mask: spectral_conv2d(xi, mask))(
                x, self.fft_mask
            )
        else:
            return x


# --- A minimal Spectral UNet ---
class SpectralUNet(eqx.Module):
    encoder: SpectralConvBlock
    decoder: SpectralConvBlock
    final_conv: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_shape: Tuple[int, int],
        *,
        key
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.encoder = SpectralConvBlock(in_channels, 64, spatial_shape, key=k1)
        self.decoder = SpectralConvBlock(64, 64, spatial_shape, key=k2)
        self.final_conv = eqx.nn.Conv2d(64, out_channels, kernel_size=1, key=k3)

    def __call__(self, x: jnp.ndarray, state=None, *, key=None) -> jnp.ndarray:
        # x: (C, H, W)
        enc = self.encoder(x)
        dec = self.decoder(enc)
        out = self.final_conv(dec)
        return out, state


# --- Example usage ---
if __name__ == "__main__":
    key = jr.PRNGKey(0)
    # Suppose an input image has 3 channels and spatial size 128x128.
    x = jr.normal(key, (3, 128, 128))
    model_key, run_key = jr.split(key)
    model = SpectralUNet(
        in_channels=3, out_channels=1, spatial_shape=(128, 128), key=model_key
    )
    y, state = model(x)
    print("Spectral UNet output shape:", y.shape)
