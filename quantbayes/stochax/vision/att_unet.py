import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import matplotlib.pyplot as plt


# Define an Attention Gate
class AttentionGate(nn.Module):
    features: int
    gating_features: int
    inter_features: int

    @nn.compact
    def __call__(self, x, gating):
        """
        x: Input feature map from the encoder (skip connection)
        gating: Gating signal from the decoder
        """
        # Apply 1x1 conv to x
        theta_x = nn.Conv(features=self.inter_features, kernel_size=(1, 1))(x)

        # Apply 1x1 conv to gating
        phi_g = nn.Conv(features=self.inter_features, kernel_size=(1, 1))(gating)

        # Resize gating to match theta_x's spatial dimensions
        phi_g = jax.image.resize(phi_g, theta_x.shape, method="bilinear")

        # Combine
        add_xg = theta_x + phi_g
        act_xg = nn.relu(add_xg)

        # Apply another conv and sigmoid
        psi = nn.Conv(features=1, kernel_size=(1, 1))(act_xg)
        sigmoid_xg = nn.sigmoid(psi)

        # Multiply attention coefficients with the input feature map
        # Expand sigmoid_xg to match the number of channels in x
        sigmoid_xg = jnp.repeat(sigmoid_xg, x.shape[-1], axis=-1)
        return x * sigmoid_xg


# Define a Double Convolution Block
class DoubleConv(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        return x

# Define the Downsampling Block
class Down(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = DoubleConv(features=self.features)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x

# Define the Upsampling Block with Attention
class Up(nn.Module):
    features: int
    attention_features: int
    gating_features: int

    @nn.compact
    def __call__(self, x, skip):
        # Upsample
        x = nn.ConvTranspose(features=self.features, kernel_size=(2, 2), strides=(2, 2))(x)

        # Apply attention gate
        attention = AttentionGate(features=skip.shape[-1],
                                  gating_features=self.gating_features,
                                  inter_features=self.attention_features)(skip, x)

        # Resize attention to match x
        attention = jax.image.resize(attention, x.shape, method="bilinear")

        # Concatenate
        x = jnp.concatenate([x, attention], axis=-1)

        # Apply double convolution
        x = DoubleConv(features=self.features)(x)
        return x


# Modified Attention U-Net with intermediate capture
class AttentionUNet(nn.Module):
    num_classes: int
    capture_intermediates: bool = True  # New flag to control capture

    @nn.compact
    def __call__(self, x):
        intermediates = {}

        # Encoder
        down1 = Down(features=64)(x)
        if self.capture_intermediates:
            self.sow('intermediates', 'encoder1', down1)

        down2 = Down(features=128)(down1)
        if self.capture_intermediates:
            self.sow('intermediates', 'encoder2', down2)

        down3 = Down(features=256)(down2)
        if self.capture_intermediates:
            self.sow('intermediates', 'encoder3', down3)

        down4 = Down(features=512)(down3)
        if self.capture_intermediates:
            self.sow('intermediates', 'encoder4', down4)

        # Bottleneck
        bottleneck = DoubleConv(features=1024)(down4)
        if self.capture_intermediates:
            self.sow('intermediates', 'bottleneck', bottleneck)

        # Decoder
        up1 = Up(features=512, attention_features=256, gating_features=512)(bottleneck, down4)
        if self.capture_intermediates:
            self.sow('intermediates', 'decoder1', up1)

        up2 = Up(features=256, attention_features=128, gating_features=256)(up1, down3)
        if self.capture_intermediates:
            self.sow('intermediates', 'decoder2', up2)

        up3 = Up(features=128, attention_features=64, gating_features=128)(up2, down2)
        if self.capture_intermediates:
            self.sow('intermediates', 'decoder3', up3)

        up4 = Up(features=64, attention_features=32, gating_features=64)(up3, down1)
        if self.capture_intermediates:
            self.sow('intermediates', 'decoder4', up4)

        # Output
        output = nn.Conv(features=self.num_classes, kernel_size=(1, 1))(up4)
        if self.capture_intermediates:
            self.sow('intermediates', 'output', output)

        return output

# Utility functions for feature extraction and visualization
def get_intermediate_outputs(model, params, input_image):
    """Returns both final output and intermediate features"""
    output, intermediates = model.apply(
        {'params': params},
        input_image,
        capture_intermediates=True,
        mutable=['intermediates']
    )
    return intermediates['intermediates']

def visualize_feature_maps_jax(feature_maps, layer_name, num_maps=6):
    """Visualize feature maps from JAX/Flax model."""
    # Convert JAX array to numpy
    fmaps = np.array(feature_maps[0])  # First (and only) batch element

    # Remove the batch dimension if it exists
    if len(fmaps.shape) == 4:
        fmaps = fmaps[0]  # Remove batch dimension

    num_channels = fmaps.shape[-1]
    num_maps = min(num_maps, num_channels)

    fig, axes = plt.subplots(1, num_maps, figsize=(15, 15))
    if num_maps == 1:
        axes = [axes]
    for i in range(num_maps):
        ax = axes[i]
        channel_map = fmaps[..., i]
        ax.imshow(channel_map, cmap='viridis')
        ax.axis('off')
        ax.set_title(f'{layer_name} Channel {i+1}')
    plt.show()

# Example usage
def test_feature_extraction():
    # Initialize model and parameters
    model = AttentionUNet(num_classes=2)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 256, 256, 3))
    variables = model.init(rng, dummy_input)

    # Get intermediate features
    intermediates = get_intermediate_outputs(model, variables['params'], dummy_input)

    # Visualize feature maps
    for layer in ['encoder1', 'encoder2', 'decoder1', 'decoder2', 'bottleneck']:
        if layer in intermediates:
            print(f"Visualizing {layer}")
            visualize_feature_maps_jax(intermediates[layer], layer)

if __name__ == "__main__":
  test_feature_extraction()