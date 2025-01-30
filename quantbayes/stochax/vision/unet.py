import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


def visualize_feature_maps_jax(feature_maps, layer_name, num_maps=6):
    """
    Visualize feature maps from a JAX/Flax model.
    - feature_maps: a tuple/list of feature maps (we usually use [0] if there's a batch dimension)
    - layer_name: string for labeling the plots
    - num_maps: how many channels to visualize
    """
    # Convert from JAX array to numpy
    fmaps = np.array(feature_maps[0])  # [0] if there's a batch dimension

    # If the shape is (batch, H, W, C), remove the batch dim
    if len(fmaps.shape) == 4:
        fmaps = fmaps[0]  # Now fmaps shape is (H, W, C)

    num_channels = fmaps.shape[-1]
    num_maps = min(num_maps, num_channels)

    # Plot some of the channels
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


class UNet(nn.Module):
    """
    A U-Net implementation with three levels in Flax Linen, but now
    capturing intermediate feature maps for visualization.
    """
    num_classes: int = 1
    capture_intermediates: bool = False  # Toggle to store intermediate outputs

    @nn.compact
    def __call__(self, x):
        # ----------------
        # Downsample path
        # ----------------
        conv1 = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        conv1 = nn.relu(conv1)
        conv1 = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(conv1)
        conv1 = nn.relu(conv1)

        if self.capture_intermediates:
            self.sow("intermediates", "conv1", conv1)

        pool1 = nn.max_pool(conv1, window_shape=(2, 2), strides=(2, 2))

        conv2 = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME")(pool1)
        conv2 = nn.relu(conv2)
        conv2 = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME")(conv2)
        conv2 = nn.relu(conv2)

        if self.capture_intermediates:
            self.sow("intermediates", "conv2", conv2)

        pool2 = nn.max_pool(conv2, window_shape=(2, 2), strides=(2, 2))

        conv3 = nn.Conv(features=256, kernel_size=(3, 3), padding="SAME")(pool2)
        conv3 = nn.relu(conv3)
        conv3 = nn.Conv(features=256, kernel_size=(3, 3), padding="SAME")(conv3)
        conv3 = nn.relu(conv3)

        if self.capture_intermediates:
            self.sow("intermediates", "conv3", conv3)

        pool3 = nn.max_pool(conv3, window_shape=(2, 2), strides=(2, 2))

        # -----------
        # Bottleneck
        # -----------
        bottleneck = nn.Conv(features=512, kernel_size=(3, 3), padding="SAME")(pool3)
        bottleneck = nn.relu(bottleneck)
        bottleneck = nn.Conv(features=512, kernel_size=(3, 3), padding="SAME")(bottleneck)
        bottleneck = nn.relu(bottleneck)

        if self.capture_intermediates:
            self.sow("intermediates", "bottleneck", bottleneck)

        # -----------
        # Upsampling
        # -----------
        up3 = nn.ConvTranspose(features=256, kernel_size=(2, 2), strides=(2, 2))(bottleneck)
        concat3 = jnp.concatenate([up3, conv3], axis=-1)
        conv_up3 = nn.Conv(features=256, kernel_size=(3, 3), padding="SAME")(concat3)
        conv_up3 = nn.relu(conv_up3)
        conv_up3 = nn.Conv(features=256, kernel_size=(3, 3), padding="SAME")(conv_up3)
        conv_up3 = nn.relu(conv_up3)

        if self.capture_intermediates:
            self.sow("intermediates", "decoder3", conv_up3)

        up2 = nn.ConvTranspose(features=128, kernel_size=(2, 2), strides=(2, 2))(conv_up3)
        concat2 = jnp.concatenate([up2, conv2], axis=-1)
        conv_up2 = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME")(concat2)
        conv_up2 = nn.relu(conv_up2)
        conv_up2 = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME")(conv_up2)
        conv_up2 = nn.relu(conv_up2)

        if self.capture_intermediates:
            self.sow("intermediates", "decoder2", conv_up2)

        up1 = nn.ConvTranspose(features=64, kernel_size=(2, 2), strides=(2, 2))(conv_up2)
        concat1 = jnp.concatenate([up1, conv1], axis=-1)
        conv_up1 = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(concat1)
        conv_up1 = nn.relu(conv_up1)
        conv_up1 = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(conv_up1)
        conv_up1 = nn.relu(conv_up1)

        if self.capture_intermediates:
            self.sow("intermediates", "decoder1", conv_up1)

        # ----------
        # Output
        # ----------
        output = nn.Conv(features=self.num_classes, kernel_size=(1, 1), padding="SAME")(conv_up1)
        if self.capture_intermediates:
            self.sow("intermediates", "output", output)

        return output


def get_intermediate_outputs(model, params, x):
    """
    Return a dictionary of intermediate feature maps
    captured during the forward pass.
    """
    # We pass capture_intermediates=True at call time,
    # plus mutable=["intermediates"] so we can retrieve them.
    output, intermediates = model.apply(
        {"params": params},
        x,
        capture_intermediates=True,  # Tells the forward pass to sow
        mutable=["intermediates"],
    )
    return intermediates["intermediates"]


def test_feature_extraction():
    """
    Example test function that:
    1) Creates a dummy input
    2) Initializes the UNet with capture_intermediates=True
    3) Captures and visualizes some of the feature maps
    """
    # 1. Construct dummy input
    dummy_input = jnp.ones((1, 128, 128, 3))  # (batch_size, H, W, channels)

    # 2. Create model with capture_intermediates
    model = UNet(num_classes=1, capture_intermediates=True)

    # 3. Init model
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, dummy_input)  # This gives us {"params": ..., ...}

    # 4. Retrieve intermediate outputs
    intermediates = get_intermediate_outputs(model, variables["params"], dummy_input)
    print("Captured intermediate keys:", list(intermediates.keys()))

    # 5. Visualize a few layers
    for layer_name in ["conv1", "conv2", "conv3", "bottleneck", "decoder3"]:
        if layer_name in intermediates:
            print(f"\nVisualizing {layer_name}, shape: {intermediates[layer_name][0].shape}")
            visualize_feature_maps_jax(intermediates[layer_name], layer_name, num_maps=4)


def test_simple_unet():
    """
    Original simpler test to confirm output shape, not capturing intermediates.
    """
    # Create a dummy input
    dummy_input = jnp.ones((1, 128, 128, 3))
    model = UNet(num_classes=1, capture_intermediates=False)

    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, dummy_input)
    output = model.apply(variables, dummy_input)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    # Run the feature extraction test (with visualization)
    test_feature_extraction()

    # Alternatively, run the simpler shape test
    # test_simple_unet()
