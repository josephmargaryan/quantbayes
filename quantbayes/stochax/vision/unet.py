import flax.linen as nn 
import jax 
import jax.numpy as jnp

class UNet(nn.Module):
    """
    A U-Net implementation with three levels in Flax Linen.
    """
    num_classes: int = 1

    @nn.compact
    def __call__(self, x):
        # Downsample path
        conv1 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        conv1 = nn.relu(conv1)
        conv1 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(conv1)
        conv1 = nn.relu(conv1)
        pool1 = nn.max_pool(conv1, window_shape=(2, 2), strides=(2, 2))

        conv2 = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(pool1)
        conv2 = nn.relu(conv2)
        conv2 = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(conv2)
        conv2 = nn.relu(conv2)
        pool2 = nn.max_pool(conv2, window_shape=(2, 2), strides=(2, 2))

        conv3 = nn.Conv(features=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(pool2)
        conv3 = nn.relu(conv3)
        conv3 = nn.Conv(features=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(conv3)
        conv3 = nn.relu(conv3)
        pool3 = nn.max_pool(conv3, window_shape=(2, 2), strides=(2, 2))

        # Bottleneck
        bottleneck = nn.Conv(features=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(pool3)
        bottleneck = nn.relu(bottleneck)
        bottleneck = nn.Conv(features=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(bottleneck)
        bottleneck = nn.relu(bottleneck)

        # Upsample path
        up3 = nn.ConvTranspose(features=256, kernel_size=(2, 2), strides=(2, 2))(bottleneck)
        concat3 = jnp.concatenate([up3, conv3], axis=-1)
        conv_up3 = nn.Conv(features=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(concat3)
        conv_up3 = nn.relu(conv_up3)
        conv_up3 = nn.Conv(features=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(conv_up3)
        conv_up3 = nn.relu(conv_up3)

        up2 = nn.ConvTranspose(features=128, kernel_size=(2, 2), strides=(2, 2))(conv_up3)
        concat2 = jnp.concatenate([up2, conv2], axis=-1)
        conv_up2 = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(concat2)
        conv_up2 = nn.relu(conv_up2)
        conv_up2 = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(conv_up2)
        conv_up2 = nn.relu(conv_up2)

        up1 = nn.ConvTranspose(features=64, kernel_size=(2, 2), strides=(2, 2))(conv_up2)
        concat1 = jnp.concatenate([up1, conv1], axis=-1)
        conv_up1 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(concat1)
        conv_up1 = nn.relu(conv_up1)
        conv_up1 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(conv_up1)
        conv_up1 = nn.relu(conv_up1)

        # Output layer
        output = nn.Conv(features=self.num_classes, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(conv_up1)
        return output

def test_simple_unet():
    # Create a dummy input tensor with shape (batch, height, width, channel)
    dummy_input = jnp.ones((1, 128, 128, 3))  # Batch size = 1, Height = 128, Width = 128, Channels = 3

    # Initialize the model
    model = UNet(num_classes=1)

    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, dummy_input)  # Initialize model parameters

    # Perform a forward pass
    output = model.apply(variables, dummy_input)

    # Print the output shape
    print("Output shape:", output.shape)

if __name__ == "__main__":
    test_simple_unet()
