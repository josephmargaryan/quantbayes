from quantbayes.bnn import * 
import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp

class UNet(Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, method="nuts", task_type="binary"):
        super().__init__(method=method, task_type=task_type)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def double_conv(self, x: jnp.ndarray, in_ch: int, out_ch: int, name_prefix: str):
        conv1 = Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding="same", name=f"{name_prefix}_conv1")(x)
        relu1 = jax.nn.relu(conv1)

        conv2 = Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding="same", name=f"{name_prefix}_conv2")(relu1)
        relu2 = jax.nn.relu(conv2)

        return relu2

    def up(self, x: jnp.ndarray, out_ch: int, name_prefix: str, stride=2):
        adjust_channels = Conv2d(
            in_channels=x.shape[1],
            out_channels=out_ch,
            kernel_size=1,
            padding="valid",
            name=f"{name_prefix}_adjust_channels",
        )(x)

        upsampled = TransposedConv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=(2, 2),
            stride=(stride, stride),
            padding="same",
            name=f"{name_prefix}_transposed_conv",
        )(adjust_channels)

        return upsampled

    def __call__(self, X: jnp.ndarray, y: jnp.ndarray = None):
        d1 = self.double_conv(X, self.in_channels, 64, "down1")
        p1 = MaxPool2d(kernel_size=2, stride=2, name="pool1")(d1)

        d2 = self.double_conv(p1, 64, 128, "down2")
        p2 = MaxPool2d(kernel_size=2, stride=2, name="pool2")(d2)

        d3 = self.double_conv(p2, 128, 256, "down3")
        p3 = MaxPool2d(kernel_size=2, stride=2, name="pool3")(d3)

        d4 = self.double_conv(p3, 256, 512, "down4")
        p4 = MaxPool2d(kernel_size=2, stride=2, name="pool4")(d4)

        bottleneck = self.double_conv(p4, 512, 1024, "bottleneck")

        u4 = self.up(bottleneck, 512, "up4")
        merge4 = jnp.concatenate([d4, u4], axis=1)
        uc4 = self.double_conv(merge4, 512 + 512, 512, "upconv4")

        u3 = self.up(uc4, 256, "up3")
        merge3 = jnp.concatenate([d3, u3], axis=1)
        uc3 = self.double_conv(merge3, 256 + 256, 256, "upconv3")

        u2 = self.up(uc3, 128, "up2")
        merge2 = jnp.concatenate([d2, u2], axis=1)
        uc2 = self.double_conv(merge2, 128 + 128, 128, "upconv2")

        u1 = self.up(uc2, 64, "up1")
        merge1 = jnp.concatenate([d1, u1], axis=1)
        uc1 = self.double_conv(merge1, 64 + 64, 64, "upconv1")

        logits = Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=1, padding="valid", name="final_conv")(uc1)
        numpyro.deterministic("logits", logits)
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)
        return logits