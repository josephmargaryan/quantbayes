import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Attention Gate
# ------------------------------------------------------------------------------
class AttentionGate(nn.Module):
    """Attention gate that upsamples the skip connection to match the decoder's gating."""
    features: int
    gating_features: int
    inter_features: int

    @nn.compact
    def __call__(self, skip, gating):
        """
        skip: feature map from the encoder (smaller spatial size)
        gating: upsampled feature map from the decoder (bigger spatial size)
        """
        # 1) 1x1 Conv on skip
        #    skip is typically smaller (e.g. 16x16), so we'll upsample it below.
        theta_x = nn.Conv(features=self.inter_features, kernel_size=(1, 1))(skip)

        # 2) Upsample skip to match gating's spatial dimensions (2x2, stride=2)
        #    Now theta_x has the same height/width as gating.
        theta_x = nn.ConvTranspose(
            features=self.inter_features,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="SAME"
        )(theta_x)

        # 3) 1x1 Conv on gating (already bigger, e.g. 32x32)
        phi_g = nn.Conv(features=self.inter_features, kernel_size=(1, 1))(gating)

        # 4) Combine
        add_xg = theta_x + phi_g
        act_xg = nn.relu(add_xg)

        # 5) Another conv + sigmoid
        psi = nn.Conv(features=1, kernel_size=(1, 1))(act_xg)
        sigmoid_xg = nn.sigmoid(psi)

        # 6) Multiply attention coefficients with the upsampled skip
        #    Expand across the channel dimension of the upsampled skip.
        #    (theta_x.shape[-1] == self.inter_features, but the original skip had more channels.)
        #    We'll simply match however many channels skip had originally. If you want to
        #    keep the "masked" skip with fewer channels, adjust accordingly.
        #    For typical Attention UNet logic, you'd multiply skip itself, but since
        #    we've upsampled skip to `theta_x`, we multiply `theta_x`:
        #    "Do we want to multiply the 'raw skip' or 'upsampled skip'?" -- Implementation choice.
        #    We'll do the upsampled skip here:
        channels_to_restore = theta_x.shape[-1]  # e.g. 256
        mask = jnp.repeat(sigmoid_xg, channels_to_restore, axis=-1)

        return theta_x * mask


# ------------------------------------------------------------------------------
# Double Convolution Block
# ------------------------------------------------------------------------------
class DoubleConv(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        return x


# ------------------------------------------------------------------------------
# Downsampling Block
# ------------------------------------------------------------------------------
class Down(nn.Module):
    """DoubleConv => MaxPool, halving spatial dimension."""
    features: int

    @nn.compact
    def __call__(self, x):
        x = DoubleConv(features=self.features)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


# ------------------------------------------------------------------------------
# Upsampling Block (with Attention)
# ------------------------------------------------------------------------------
class Up(nn.Module):
    """
    1) ConvTranspose to double the spatial dimension of `x` (decoder).
    2) Apply AttentionGate(skip, x) -- skip is smaller, x is bigger.
    3) Concatenate
    4) DoubleConv
    """
    features: int
    attention_features: int
    gating_features: int

    @nn.compact
    def __call__(self, x, skip):
        """
        x: decoder feature map (e.g. shape [batch, 16, 16, channels])
        skip: encoder feature map (smaller shape [batch, 8, 8, channels]) => we attention-gate it
        """
        # 1) Upsample x from e.g. (16x16) -> (32x32)
        x = nn.ConvTranspose(
            features=self.features,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="SAME",
        )(x)

        # 2) Apply attention gate: skip is smaller => it gets upsampled internally.
        #    gating is the bigger x from the decoder.
        #    The return is an upsampled/masked skip with shape matching x.
        att_skip = AttentionGate(
            features=skip.shape[-1],
            gating_features=self.gating_features,
            inter_features=self.attention_features,
        )(skip, x)

        # 3) Concatenate upsampled x with the attention-masked skip
        x = jnp.concatenate([x, att_skip], axis=-1)

        # 4) DoubleConv
        x = DoubleConv(features=self.features)(x)
        return x


# ------------------------------------------------------------------------------
# Attention U-Net (with optional capture of intermediates)
# ------------------------------------------------------------------------------
class AttentionUNet(nn.Module):
    """
    Attention U-Net that:
      - Downsamples 4 times (Down blocks)
      - Bottleneck
      - Upsamples 4 times (Up blocks)
      - Output => `num_classes`
      - Optionally captures intermediate feature maps for visualization
    """
    num_classes: int
    capture_intermediates: bool = True

    @nn.compact
    def __call__(self, x):
        # ----------------
        # Encoder
        # ----------------
        down1 = Down(features=64)(x)    # => [B, H/2,  W/2, 64]
        if self.capture_intermediates:
            self.sow("intermediates", "encoder1", down1)

        down2 = Down(features=128)(down1)  # => [B, H/4,  W/4, 128]
        if self.capture_intermediates:
            self.sow("intermediates", "encoder2", down2)

        down3 = Down(features=256)(down2)  # => [B, H/8,  W/8, 256]
        if self.capture_intermediates:
            self.sow("intermediates", "encoder3", down3)

        down4 = Down(features=512)(down3)  # => [B, H/16, W/16, 512]
        if self.capture_intermediates:
            self.sow("intermediates", "encoder4", down4)

        # ----------------
        # Bottleneck
        # ----------------
        bottleneck = DoubleConv(features=1024)(down4)  # => [B, H/16, W/16, 1024]
        if self.capture_intermediates:
            self.sow("intermediates", "bottleneck", bottleneck)

        # ----------------
        # Decoder
        # ----------------
        up1 = Up(features=512, attention_features=256, gating_features=512)(
            bottleneck, down4
        )  # => [B, H/8, W/8, 512]
        if self.capture_intermediates:
            self.sow("intermediates", "decoder1", up1)

        up2 = Up(features=256, attention_features=128, gating_features=256)(
            up1, down3
        )  # => [B, H/4, W/4, 256]
        if self.capture_intermediates:
            self.sow("intermediates", "decoder2", up2)

        up3 = Up(features=128, attention_features=64, gating_features=128)(
            up2, down2
        )  # => [B, H/2, W/2, 128]
        if self.capture_intermediates:
            self.sow("intermediates", "decoder3", up3)

        up4 = Up(features=64, attention_features=32, gating_features=64)(
            up3, down1
        )  # => [B, H, W, 64]
        if self.capture_intermediates:
            self.sow("intermediates", "decoder4", up4)

        # ----------------
        # Output
        # ----------------
        output = nn.Conv(features=self.num_classes, kernel_size=(1, 1))(up4)
        if self.capture_intermediates:
            self.sow("intermediates", "output", output)

        return output


# ------------------------------------------------------------------------------
# Utilities for Feature Extraction & Visualization
# ------------------------------------------------------------------------------
def get_intermediate_outputs(model, params, input_image):
    """Returns the final output and intermediate features in a dict."""
    final_out, mutable_dict = model.apply(
        {"params": params},
        input_image,
        capture_intermediates=True,
        mutable=["intermediates"],
    )
    return final_out, mutable_dict["intermediates"]


def visualize_feature_maps_jax(feature_maps, layer_name, num_maps=6):
    """Visualize feature maps from JAX/Flax model."""
    # Convert JAX array to numpy
    # feature_maps[0] => the actual array (since 'intermediates' are stored in lists/tuples).
    fmaps = np.array(feature_maps[0])  # remove the outer list/tuple

    # If there's a batch dimension, remove it
    if len(fmaps.shape) == 4:
        fmaps = fmaps[0]  # [H, W, C]

    num_channels = fmaps.shape[-1]
    num_maps = min(num_maps, num_channels)

    fig, axes = plt.subplots(1, num_maps, figsize=(15, 15))
    if num_maps == 1:
        axes = [axes]

    for i in range(num_maps):
        ax = axes[i]
        channel_map = fmaps[..., i]
        ax.imshow(channel_map, cmap="viridis")
        ax.axis("off")
        ax.set_title(f"{layer_name} Channel {i+1}")

    plt.show()


# ------------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------------
def test_feature_extraction():
    # 1) Create model
    model = AttentionUNet(num_classes=2, capture_intermediates=True)
    rng = jax.random.PRNGKey(0)

    # 2) Dummy input of shape [batch=1, 256, 256, 3]
    dummy_input = jnp.ones((1, 256, 256, 3))

    # 3) Initialize
    variables = model.init(rng, dummy_input)

    # 4) Get final output + intermediate maps
    final_out, intermediates = get_intermediate_outputs(model, variables["params"], dummy_input)
    print("Final output shape:", final_out.shape)

    # 5) Visualize some layers
    for layer in ["encoder1", "encoder2", "bottleneck", "decoder1", "decoder2"]:
        if layer in intermediates:
            arr = intermediates[layer]
            print(f"Layer={layer}, shape={arr[0].shape}")
            visualize_feature_maps_jax(arr, layer, num_maps=4)


if __name__ == "__main__":
    test_feature_extraction()
