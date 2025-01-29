import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from quantbayes.bnn import Module
from typing import Optional


class Unet(Module):
    """
    A simplified 2-level UNet for segmentation (binary by default).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        method: str = "nuts",
        task_type: str = "image_segmentation",
    ):
        super().__init__(method=method, task_type=task_type)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def double_conv(self, x: jnp.ndarray, in_ch: int, out_ch: int, name_prefix: str):
        """
        Two consecutive Conv2d(3x3) -> ReLU
        """
        from quantbayes.bnn import Conv2d

        conv1 = Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding="same",
            name=f"{name_prefix}_conv1",
        )(x)
        relu1 = jax.nn.relu(conv1)

        conv2 = Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding="same",
            name=f"{name_prefix}_conv2",
        )(relu1)
        relu2 = jax.nn.relu(conv2)
        return relu2

    def up(self, x: jnp.ndarray, out_ch: int, name_prefix: str):
        """
        Upsampling via transposed conv.
        1x1 conv adjusts channels before the transposed convolution.
        """
        from quantbayes.bnn import Conv2d, TransposedConv2d

        # Adjust channels first
        x = Conv2d(
            in_channels=x.shape[1],
            out_channels=out_ch,
            kernel_size=1,
            padding="valid",
            name=f"{name_prefix}_adjust_channels",
        )(x)

        # Transposed conv to upsample
        x = TransposedConv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=2,
            stride=2,
            padding="same",
            name=f"{name_prefix}_transposed",
        )(x)
        return x

    def __call__(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None):
        """
        Forward pass:
          - Two "down" levels
          - Bottleneck
          - Two "up" levels
          - 1x1 output conv
        """
        from quantbayes.bnn import MaxPool2d, Conv2d

        # Down 1
        d1 = self.double_conv(X, self.in_channels, 64, "down1")
        p1 = MaxPool2d(kernel_size=2, stride=2, name="pool1")(d1)

        # Down 2
        d2 = self.double_conv(p1, 64, 128, "down2")
        p2 = MaxPool2d(kernel_size=2, stride=2, name="pool2")(d2)

        # Bottleneck
        bottleneck = self.double_conv(p2, 128, 256, "bottleneck")

        # Up 2
        u2 = self.up(bottleneck, 128, "up2")
        merge2 = jnp.concatenate([d2, u2], axis=1)
        uc2 = self.double_conv(merge2, 128 + 128, 128, "upconv2")

        # Up 1
        u1 = self.up(uc2, 64, "up1")
        merge1 = jnp.concatenate([d1, u1], axis=1)
        uc1 = self.double_conv(merge1, 64 + 64, 64, "upconv1")

        # Final 1x1 conv
        logits = Conv2d(
            in_channels=64,
            out_channels=self.out_channels,
            kernel_size=1,
            padding="valid",
            name="final_conv",
        )(uc1)

        numpyro.deterministic("logits", logits)
        # For binary segmentation:
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)
        return logits
