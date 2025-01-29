from quantbayes.bnn.layers import Conv2d, Linear


class Conv2DClassifier:
    """
    Minimal 2D CNN for image classification, e.g. MNIST (batch_size, 1, 28, 28).
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=8,
        kernel_size=3,
        stride=1,
        name="conv2d_classifier",
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.name = name

        # Convolution layer
        self.conv2d = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding="same",
            name=f"{name}_conv2d",
        )
        # After convolution, let's flatten and do a final linear
        # We'll not know the exact output shape until we fix image size,
        # so let's do e.g. out_channels * 28 * 28 => 10 classes.
        self.linear = Linear(out_channels * 28 * 28, 10, name=f"{name}_linear")

    def __call__(self, X):
        """
        X: (batch_size, in_channels, height, width)
        returns: logits shape (batch_size, 10) if we define 10 classes
        """
        conv_out = self.conv2d(X)  # shape (batch_size, out_channels, h, w)
        batch_size, c, h, w = conv_out.shape
        flattened = conv_out.reshape((batch_size, c * h * w))
        logits = self.linear(flattened)  # (batch_size, 10)
        return logits
