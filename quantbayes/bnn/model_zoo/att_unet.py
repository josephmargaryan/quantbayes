import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from quantbayes.bnn import Module
from typing import Optional

class AttentionUNet(Module):
    """
    A simplified 2-level UNet with attention gates on skip connections.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        method: str = "nuts",
        task_type: str = "binary",
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

    def attention_gate(
        self, skip: jnp.ndarray, gating: jnp.ndarray, out_ch: int, name_prefix: str
    ):
        """
        Attention gate that takes skip connection and gating signal,
        returns skip * attention_coefficients.
        """
        from quantbayes.bnn import Conv2d

        # 1x1 conv to reduce channel dimensions
        theta_x = Conv2d(
            in_channels=skip.shape[1],
            out_channels=out_ch,
            kernel_size=1,
            padding="valid",
            name=f"{name_prefix}_theta_x",
        )(skip)

        phi_g = Conv2d(
            in_channels=gating.shape[1],
            out_channels=out_ch,
            kernel_size=1,
            padding="valid",
            name=f"{name_prefix}_phi_g",
        )(gating)

        # Reshape if spatial dims differ (basic up/down if needed).
        # Here, we assume gating is smaller; we can do simple interpolation:
        if theta_x.shape[2] != phi_g.shape[2] or theta_x.shape[3] != phi_g.shape[3]:
            # Basic approach: nearest neighbor upsampling to match skip's spatial size
            scale_h = theta_x.shape[2] // phi_g.shape[2]
            scale_w = theta_x.shape[3] // phi_g.shape[3]
            phi_g = jax.image.resize(
                phi_g,
                (phi_g.shape[0], phi_g.shape[1], theta_x.shape[2], theta_x.shape[3]),
                method="nearest",
            )

        concat = jax.nn.relu(theta_x + phi_g)

        psi = Conv2d(
            in_channels=out_ch,
            out_channels=1,
            kernel_size=1,
            padding="valid",
            name=f"{name_prefix}_psi",
        )(concat)
        # Attention coefficients
        alpha = jax.nn.sigmoid(psi)

        # Broadcast alpha over channel dimension
        alpha = jnp.broadcast_to(alpha, skip.shape)

        # Multiply skip with attention coefficients
        attended_skip = skip * alpha
        return attended_skip

    def up(self, x: jnp.ndarray, out_ch: int, name_prefix: str):
        """
        Upsampling via transposed conv (like normal UNet).
        """
        from quantbayes.bnn import Conv2d, TransposedConv2d

        # Adjust channels
        x = Conv2d(
            in_channels=x.shape[1],
            out_channels=out_ch,
            kernel_size=1,
            name=f"{name_prefix}_adj_channels",
        )(x)

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
        2-level UNet with attention gates:
          - down1 -> pool1
          - down2 -> pool2
          - bottleneck
          - up2 (attend skip2)
          - up1 (attend skip1)
          - final conv
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
        # Attention gate for skip d2
        attn_d2 = self.attention_gate(d2, u2, out_ch=128, name_prefix="attn2")
        merge2 = jnp.concatenate([attn_d2, u2], axis=1)
        uc2 = self.double_conv(merge2, 128 + 128, 128, "upconv2")

        # Up 1
        u1 = self.up(uc2, 64, "up1")
        # Attention gate for skip d1
        attn_d1 = self.attention_gate(d1, u1, out_ch=64, name_prefix="attn1")
        merge1 = jnp.concatenate([attn_d1, u1], axis=1)
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
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)
        return logits
