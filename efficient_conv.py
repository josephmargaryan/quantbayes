import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import math


# ----------------------------------------------------------------------
# 1D SPECTRAL CONV
# ----------------------------------------------------------------------
@jax.custom_jvp
def spectral_conv1d(x: jnp.ndarray, weight_fft: jnp.ndarray, bias: jnp.ndarray):
    """
    1D spectral convolution.

    x: (batch, in_channels, width)
    weight_fft: (out_channels, in_channels, rfft_length) where rfft_length = fft_size//2+1
    bias: (out_channels,)
    Returns:
      out: (batch, out_channels, width)
    """
    B, Cin, W = x.shape
    # Recover the original FFT spatial size from the weight FFT.
    # If weight_fft.shape[-1] = rfft_length = fft_size//2+1, then:
    fft_size = 2 * (weight_fft.shape[-1] - 1) if weight_fft.shape[-1] > 1 else 1

    # Pad x if needed so that its length matches fft_size.
    if fft_size > W:
        pad_amount = fft_size - W
        x = jnp.pad(x, ((0, 0), (0, 0), (0, pad_amount)))

    # Compute rFFT on x with the fixed fft_size.
    X_fft = jnp.fft.rfft(x, n=fft_size, axis=-1)  # shape: (B, Cin, fft_size//2+1)
    # Multiply in Fourier domain:
    # Expand dimensions so that:
    #   X_fft_expanded: (B, 1, Cin, fft_size//2+1)
    #   W_fft_expanded: (1, Cout, Cin, fft_size//2+1)
    X_fft_expanded = X_fft[:, None, :, :]
    W_fft_expanded = weight_fft[None, :, :, :]
    Y_fft = jnp.sum(X_fft_expanded * W_fft_expanded, axis=2)  # (B, Cout, fft_size//2+1)

    # Inverse rFFT to get back to spatial domain, using fft_size.
    y = jnp.fft.irfft(Y_fft, n=fft_size, axis=-1)
    # Slice back to the original input width
    y = y[..., :W]
    # Add bias (broadcast over batch and width)
    y = y + bias[None, :, None]
    return y


@spectral_conv1d.defjvp
def spectral_conv1d_jvp(primals, tangents):
    x, weight_fft, bias = primals
    dx, dweight_fft, dbias = tangents

    y = spectral_conv1d(x, weight_fft, bias)
    dy = jnp.zeros_like(y)
    if dx is not None:
        dy += spectral_conv1d(dx, weight_fft, jnp.zeros_like(bias))
    if dweight_fft is not None:
        dy += spectral_conv1d(x, dweight_fft, jnp.zeros_like(bias))
    if dbias is not None:
        dy += dbias[None, :, None]
    return y, dy


class SpectralConv1d(eqx.Module):
    """A 1D convolution layer using FFT-based multiplication in the frequency domain."""

    weight_fft: jnp.ndarray  # shape: (out_channels, in_channels, rfft_length)
    bias: jnp.ndarray  # shape: (out_channels,)
    in_channels: int = eqx.static_field()
    out_channels: int = eqx.static_field()
    kernel_size: int = eqx.static_field()
    fft_size: int = eqx.static_field()

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        fft_size: int,
        key,
        init_scale: float = 0.1,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.fft_size = fft_size

        k1, k2 = jr.split(key, 2)
        # Create spatial-domain filters: shape (out_channels, in_channels, kernel_size)
        w_spatial = jr.normal(k1, (out_channels, in_channels, kernel_size)) * init_scale

        # Pad spatial filters to fft_size if needed.
        if kernel_size < fft_size:
            pad_amount = fft_size - kernel_size
            w_spatial = jnp.pad(w_spatial, ((0, 0), (0, 0), (0, pad_amount)))

        # Compute rFFT on the last dimension.
        w_fft = jnp.fft.rfft(
            w_spatial, axis=-1
        )  # shape: (out_channels, in_channels, fft_size//2+1)
        b = jr.normal(k2, (out_channels,)) * init_scale

        object.__setattr__(self, "weight_fft", w_fft)
        object.__setattr__(self, "bias", b)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, in_channels, width)
        Returns: (batch, out_channels, width)
        """
        return spectral_conv1d(x, self.weight_fft, self.bias)


# ----------------------------------------------------------------------
# 2D SPECTRAL CONV
# ----------------------------------------------------------------------
@jax.custom_jvp
def spectral_conv2d(x: jnp.ndarray, weight_fft: jnp.ndarray, bias: jnp.ndarray):
    """
    2D spectral convolution.

    x: (batch, in_channels, H, W)
    weight_fft: (out_channels, in_channels, Hf, Wf') -- the rfft2 of the conv filters
    bias: (out_channels,)
    Returns:
      (batch, out_channels, H, W)
    """
    B, Cin, H, W = x.shape
    Cout, _, Hf, Wf_rfft = weight_fft.shape

    # Compute rFFT2 on x.
    X_fft = jnp.fft.rfft2(x, axes=(2, 3))  # shape: (B, Cin, H', W')
    # Expand dimensions for broadcasting.
    X_fft_expanded = X_fft[:, None, :, :, :]  # (B, 1, Cin, H', W')
    W_fft_expanded = jnp.conjugate(weight_fft)[
        None, :, :, :, :
    ]  # (1, Cout, Cin, Hf, Wf')
    Y_fft = jnp.sum(X_fft_expanded * W_fft_expanded, axis=2)  # (B, Cout, H', W')

    # Determine the real width from the rFFT output.
    if Wf_rfft > 1:
        real_Wf = 2 * (Wf_rfft - 1)
    else:
        real_Wf = 1

    # Inverse rFFT2 using the padded shape (Hf, real_Wf), then slice.
    y = jnp.fft.irfft2(Y_fft, s=(Hf, real_Wf), axes=(2, 3))
    y = y[..., :H, :W]
    y = y + bias[None, :, None, None]
    return y


@spectral_conv2d.defjvp
def spectral_conv2d_jvp(primals, tangents):
    x, weight_fft, bias = primals
    dx, dweight_fft, dbias = tangents

    y = spectral_conv2d(x, weight_fft, bias)
    dy = jnp.zeros_like(y)
    if dx is not None:
        dy += spectral_conv2d(dx, weight_fft, jnp.zeros_like(bias))
    if dweight_fft is not None:
        dy += spectral_conv2d(x, dweight_fft, jnp.zeros_like(bias))
    if dbias is not None:
        dy += dbias[None, :, None, None]
    return y, dy


class SpectralConv2d(eqx.Module):
    """
    2D FFT-based convolution with precomputed kernel in the frequency domain.
    """

    weight_fft: jnp.ndarray  # (out_channels, in_channels, Hf, Wf//2+1)
    bias: jnp.ndarray  # (out_channels,)
    in_channels: int = eqx.static_field()
    out_channels: int = eqx.static_field()
    kernel_size: tuple = eqx.static_field()
    fft_size: tuple = eqx.static_field()

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        fft_size: tuple,
        key,
        init_scale: float = 0.1,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.fft_size = fft_size

        k1, k2 = jr.split(key, 2)
        Kh, Kw = kernel_size
        Hf, Wf = fft_size

        # Create a spatial kernel: (out_channels, in_channels, Kh, Kw)
        w_spatial = jr.normal(k1, (out_channels, in_channels, Kh, Kw)) * init_scale
        # Pad the kernel to (Hf, Wf)
        pad_h = Hf - Kh
        pad_w = Wf - Kw
        w_spatial = jnp.pad(w_spatial, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
        w_fft = jnp.fft.rfft2(w_spatial, axes=(2, 3))
        b = jr.normal(k2, (out_channels,)) * init_scale

        object.__setattr__(self, "weight_fft", w_fft)
        object.__setattr__(self, "bias", b)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, in_channels, H, W)
        Returns: (batch, out_channels, H, W)
        """
        return spectral_conv2d(x, self.weight_fft, self.bias)


# ----------------------------------------------------------------------
# 2D SPECTRAL TRANSPOSED CONV
# ----------------------------------------------------------------------
@jax.custom_jvp
def spectral_transposed_conv2d(
    x: jnp.ndarray, weight_fft: jnp.ndarray, bias: jnp.ndarray
):
    """
    Spectral transposed (deconvolution) 2D convolution.

    x: (batch, out_channels, H, W) -- input feature map (output of a forward conv)
    weight_fft: (out_channels, in_channels, Hf, Wf') -- FFT of the forward conv kernel.
    bias: (in_channels,) -- bias for the transposed conv.
    Returns:
      (batch, in_channels, H, W)
    """
    B, Cout, H, W = x.shape
    _, Cin, Hf, Wf_rfft = weight_fft.shape

    # Compute rFFT2 on x.
    X_fft = jnp.fft.rfft2(x, axes=(2, 3))  # shape: (B, Cout, H', W')
    # Expand dimensions for broadcasting.
    X_fft_expanded = X_fft[:, :, None, :, :]  # (B, Cout, 1, H', W')
    W_fft_expanded = weight_fft[None, :, :, :, :]  # (1, Cout, Cin, Hf, Wf')
    # Sum over Cout (the forward conv’s output channels) to produce input channels.
    Z_fft = jnp.sum(X_fft_expanded * W_fft_expanded, axis=1)  # (B, Cin, H', W')

    if Wf_rfft > 1:
        real_Wf = 2 * (Wf_rfft - 1)
    else:
        real_Wf = 1

    # Inverse rFFT2 using the padded shape and slice to (H, W).
    y = jnp.fft.irfft2(Z_fft, s=(Hf, real_Wf), axes=(2, 3))
    y = y[..., :H, :W]

    if bias is not None:
        y = y + bias[None, :, None, None]
    return y


@spectral_transposed_conv2d.defjvp
def spectral_transposed_conv2d_jvp(primals, tangents):
    x, weight_fft, bias = primals
    dx, dweight_fft, dbias = tangents

    y = spectral_transposed_conv2d(x, weight_fft, bias)
    dy = jnp.zeros_like(y)
    if dx is not None:
        dy += spectral_transposed_conv2d(dx, weight_fft, jnp.zeros_like(bias))
    if dweight_fft is not None:
        dy += spectral_transposed_conv2d(x, dweight_fft, jnp.zeros_like(bias))
    if dbias is not None:
        dy += dbias[None, :, None, None]
    return y, dy


class SpectralTransposed2d(eqx.Module):
    """
    2D spectral transposed convolution.

    For a forward conv with kernel weight_fft of shape
      (out_channels, in_channels, Hf, Wf//2+1),
    this layer takes an input of shape (batch, out_channels, H, W)
    and produces an output of shape (batch, in_channels, H, W).
    """

    weight_fft: jnp.ndarray  # (out_channels, in_channels, Hf, Wf//2+1)
    bias: jnp.ndarray  # (in_channels,)
    out_channels: int = eqx.static_field()  # from forward conv (now input channels)
    in_channels: int = eqx.static_field()  # from forward conv (now output channels)
    kernel_size: tuple = eqx.static_field()
    fft_size: tuple = eqx.static_field()

    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        kernel_size: tuple,
        fft_size: tuple,
        key,
        init_scale: float = 0.1,
    ):
        """
        Initialize the spectral transposed convolution.

        :param out_channels: Number of channels from the forward conv (input channels for transposed conv).
        :param in_channels: Number of channels of the forward conv input (output channels for transposed conv).
        :param kernel_size: (Kh, Kw) filter size.
        :param fft_size: (Hf, Wf) target size for padding before FFT.
        """
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.fft_size = fft_size

        k1, k2 = jr.split(key, 2)
        Kh, Kw = kernel_size
        Hf, Wf = fft_size

        # Create spatial-domain filters: shape (out_channels, in_channels, Kh, Kw)
        w_spatial = jr.normal(k1, (out_channels, in_channels, Kh, Kw)) * init_scale
        # Pad the kernel to (Hf, Wf)
        pad_h = Hf - Kh
        pad_w = Wf - Kw
        w_spatial = jnp.pad(w_spatial, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
        w_fft = jnp.fft.rfft2(w_spatial, axes=(2, 3))
        b = jr.normal(k2, (in_channels,)) * init_scale

        object.__setattr__(self, "weight_fft", w_fft)
        object.__setattr__(self, "bias", b)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, out_channels, H, W)
        Returns: (batch, in_channels, H, W)
        """
        return spectral_transposed_conv2d(x, self.weight_fft, self.bias)


# ----------------------------------------------------------------------
# SpectralLeNet (for completeness)
# ----------------------------------------------------------------------
class SpectralLeNet(eqx.Module):
    conv1: SpectralConv2d
    conv2: SpectralConv2d
    fc: eqx.nn.Linear

    def __init__(self, key: jax.random.PRNGKey):
        k1, k2, k3 = jr.split(key, 3)

        in_channels = 1
        out_channels1 = 8
        out_channels2 = 16
        kernel_size = (5, 5)

        self.conv1 = SpectralConv2d(
            in_channels=in_channels,
            out_channels=out_channels1,
            kernel_size=kernel_size,
            fft_size=(28, 28),
            key=k1,
            init_scale=0.1,
        )
        self.conv2 = SpectralConv2d(
            in_channels=out_channels1,
            out_channels=out_channels2,
            kernel_size=kernel_size,
            fft_size=(28, 28),
            key=k2,
            init_scale=0.1,
        )

        self.fc = eqx.nn.Linear(
            in_features=out_channels2 * 28 * 28,
            out_features=10,
            key=k3,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, 1, 28, 28)
        Returns: (batch, 10)
        """
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        logits = jax.vmap(self.fc)(x)
        return logits


# ----------------------------------------------------------------------
# SPECTRAL UNET
# ----------------------------------------------------------------------
def pool(x):
    """Average pool with a 2x2 window and stride 2."""
    # For input shape (B, C, H, W)
    window = (1, 1, 2, 2)
    strides = (1, 1, 2, 2)
    pooled = jax.lax.reduce_window(x, 0.0, jax.lax.add, window, strides, padding="SAME")
    return pooled / 4.0


def upsample(x, factor=2):
    """Nearest neighbor upsampling by repeating along spatial dims."""
    return jnp.repeat(jnp.repeat(x, factor, axis=-2), factor, axis=-1)


class SpectralUNet(eqx.Module):
    # Encoder
    conv1: SpectralConv2d  # Input conv (e.g. 1 -> 16 channels)
    conv2: SpectralConv2d  # Second encoder conv (16 -> 32 channels)
    bottleneck: SpectralConv2d  # Bottleneck (32 -> 64 channels)
    # Decoder
    upconv1: SpectralTransposed2d  # Upsample: (64 -> 32 channels)
    conv3: SpectralConv2d  # After skip concatenation (64 -> 32 channels)
    upconv2: SpectralTransposed2d  # Upsample: (32 -> 16 channels)
    conv4: SpectralConv2d  # Final conv after skip concatenation (32 -> out_channels)
    # Configuration
    out_channels: int = eqx.static_field()

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        key: jax.random.PRNGKey = jr.PRNGKey(0),
    ):
        self.out_channels = out_channels
        # Assume input image resolution is 128x128.
        keys = jr.split(key, 7)
        # Encoder:
        # conv1: 1 -> base_channels, fft_size = (128, 128)
        self.conv1 = SpectralConv2d(
            in_channels,
            base_channels,
            kernel_size=(3, 3),
            fft_size=(128, 128),
            key=keys[0],
        )
        # After pooling, resolution becomes 64x64.
        # conv2: base_channels -> base_channels*2, fft_size = (64, 64)
        self.conv2 = SpectralConv2d(
            base_channels,
            base_channels * 2,
            kernel_size=(3, 3),
            fft_size=(64, 64),
            key=keys[1],
        )
        # After another pooling, resolution becomes 32x32.
        # Bottleneck: base_channels*2 -> base_channels*4, fft_size = (32, 32)
        self.bottleneck = SpectralConv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=(3, 3),
            fft_size=(32, 32),
            key=keys[2],
        )
        # Decoder:
        # upconv1: transforms from bottleneck channels (base_channels*4) to base_channels*2.
        # For SpectralTransposed2d, we specify forward conv parameters:
        #   forward conv had: out_channels = base_channels*4, in_channels = base_channels*2,
        # so here: out_channels=base_channels*4, in_channels=base_channels*2, fft_size = (64,64)
        self.upconv1 = SpectralTransposed2d(
            out_channels=base_channels * 4,
            in_channels=base_channels * 2,
            kernel_size=(3, 3),
            fft_size=(64, 64),
            key=keys[3],
        )
        # After concatenating with conv2 output (which has base_channels*2 channels), we get 2*base_channels*2 channels.
        # conv3: (base_channels*2 + base_channels*2) = base_channels*4 -> base_channels*2, fft_size = (64,64)
        self.conv3 = SpectralConv2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=(3, 3),
            fft_size=(64, 64),
            key=keys[4],
        )
        # upconv2: transforms from base_channels*2 to base_channels.
        # For forward conv: out_channels = base_channels*2, in_channels = base_channels,
        # so here: out_channels=base_channels*2, in_channels=base_channels, fft_size = (128,128)
        self.upconv2 = SpectralTransposed2d(
            out_channels=base_channels * 2,
            in_channels=base_channels,
            kernel_size=(3, 3),
            fft_size=(128, 128),
            key=keys[5],
        )
        # After concatenating with conv1 output (base_channels channels), total channels = base_channels + base_channels = base_channels*2.
        # conv4: (base_channels*2) -> out_channels, fft_size = (128,128)
        self.conv4 = SpectralConv2d(
            base_channels * 2,
            out_channels,
            kernel_size=(3, 3),
            fft_size=(128, 128),
            key=keys[6],
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, in_channels, H, W)  e.g. (batch, 1, 128, 128)
        Returns: (batch, out_channels, H, W)
        """
        # Encoder
        x1 = self.conv1(x)  # (B, base_channels, 128, 128)
        x1_act = jax.nn.relu(x1)
        x1_pool = pool(x1_act)  # (B, base_channels, 64, 64)

        x2 = self.conv2(x1_pool)  # (B, base_channels*2, 64, 64)
        x2_act = jax.nn.relu(x2)
        x2_pool = pool(x2_act)  # (B, base_channels*2, 32, 32)

        # Bottleneck
        x3 = self.bottleneck(x2_pool)  # (B, base_channels*4, 32, 32)
        x3_act = jax.nn.relu(x3)

        # Decoder
        x3_up = upsample(x3_act, factor=2)  # (B, base_channels*4, 64, 64)
        x4 = self.upconv1(x3_up)  # (B, base_channels*2, 64, 64)
        # Skip connection from x2_act (shape: (B, base_channels*2, 64, 64))
        x4_cat = jnp.concatenate([x4, x2_act], axis=1)  # (B, base_channels*4, 64, 64)
        x5 = self.conv3(x4_cat)  # (B, base_channels*2, 64, 64)
        x5_act = jax.nn.relu(x5)

        x5_up = upsample(x5_act, factor=2)  # (B, base_channels*2, 128, 128)
        x6 = self.upconv2(x5_up)  # (B, base_channels, 128, 128)
        # Skip connection from x1_act (shape: (B, base_channels, 128, 128))
        x6_cat = jnp.concatenate([x6, x1_act], axis=1)  # (B, base_channels*2, 128, 128)
        x7 = self.conv4(x6_cat)  # (B, out_channels, 128, 128)
        return x7


# ----------------------------------------------------------------------
# Test functions
# ----------------------------------------------------------------------
def test_spectral_unet():
    print("Testing SpectralUNet...")
    key = jr.PRNGKey(123)
    # Create a UNet with input 1 channel and output 1 channel.
    model = SpectralUNet(in_channels=1, out_channels=1, base_channels=16, key=key)
    # Create a batch of 2 images of shape (1, 128, 128)
    x = jr.normal(key, (2, 1, 128, 128))
    y = model(x)
    print("  Input shape :", x.shape)
    print("  Output shape:", y.shape)
    # Expected output shape: (2, 1, 128, 128)


def test_spectral_conv1d():
    print("Testing SpectralConv1d...")
    key = jr.PRNGKey(0)
    # Create a 1D conv: in_channels=3, out_channels=5, kernel_size=7, fft_size=16.
    conv1d = SpectralConv1d(
        in_channels=3, out_channels=5, kernel_size=7, fft_size=16, key=key
    )
    # Create random input: (batch, in_channels, width)
    x = jr.normal(key, (4, 3, 20))
    y = conv1d(x)
    print("  Input shape :", x.shape)
    print("  Output shape:", y.shape)
    # Expected output shape: (4, 5, 20)


def test_spectral_transposed2d():
    print("Testing SpectralTransposed2d...")
    key = jr.PRNGKey(1)
    # For transposed conv, assume the forward conv had:
    #   out_channels=8 and in_channels=3.
    # Thus the transposed conv takes input of shape (batch, 8, H, W)
    # and produces output of shape (batch, 3, H, W).
    trans_conv = SpectralTransposed2d(
        out_channels=8, in_channels=3, kernel_size=(3, 3), fft_size=(28, 28), key=key
    )
    x = jr.normal(key, (4, 8, 28, 28))
    y = trans_conv(x)
    print("  Input shape :", x.shape)
    print("  Output shape:", y.shape)
    # Expected output shape: (4, 3, 28, 28)


def test_spectral_lenet():
    print("Testing SpectralLeNet...")
    key = jr.PRNGKey(42)
    model = SpectralLeNet(key)
    x = jr.normal(key, (2, 1, 28, 28))
    logits = model(x)
    print("  Input shape :", x.shape)
    print("  Output shape:", logits.shape)
    # Expected output shape: (2, 10)


if __name__ == "__main__":
    test_spectral_unet()
    test_spectral_conv1d()
    print()
    test_spectral_transposed2d()
    print()
    test_spectral_lenet()
