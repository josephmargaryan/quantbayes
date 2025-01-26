import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from quantbayes.bnn.utils.generalization_bound import BayesianAnalysis
from quantbayes import bnn
from quantbayes.fake_data import *


class TransposedConvTest(bnn.Module):
    def __init__(self):
        super().__init__(method="nuts", task_type="binary")

    def __call__(self, X, y=None):
        # Define a transposed convolution layer
        conv = bnn.TransposedConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding="same",
            name="transposed_conv",
        )

        # Apply the transposed convolution
        X_conv = conv(X)  # Output shape: (batch_size, out_channels, height, width)

        # Flatten the output for classification
        X_flat = X_conv.reshape((X_conv.shape[0], -1))  # (batch_size, flattened_features)

        # Use the predefined Linear layer
        dense = bnn.Linear(in_features=X_flat.shape[-1], out_features=1, name="dense")
        logits = dense(X_flat)

        # Define binary classification likelihood
        probs = jax.nn.sigmoid(logits)
        numpyro.sample("obs", dist.Bernoulli(probs=probs), obs=y)

class FFTTransposedConvTest(bnn.Module):
    def __init__(self):
        super().__init__(method="nuts", task_type="binary")

    def __call__(self, X, y=None):
        # Define an FFT-based transposed convolution layer
        conv = bnn.FFTTransposedConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 3),
            name="fft_transposed_conv",
        )

        # Apply the FFT-based transposed convolution
        X_conv = conv(X)  # Output shape: (batch_size, out_channels, height, width)

        # Flatten the output for classification
        X_flat = X_conv.reshape((X_conv.shape[0], -1))  # (batch_size, flattened_features)

        # Use FFTLinear for initial transformation
        fft_dense = bnn.FFTLinear(in_features=X_flat.shape[-1], name="fft_dense")
        hidden = fft_dense(X_flat)  # Output shape: (batch_size, in_features)

        # Use Linear for final transformation to output dimension
        dense = bnn.Linear(in_features=hidden.shape[-1], out_features=1, name="dense")
        logits = dense(hidden)  # Output shape: (batch_size, 1)

        # Define binary classification likelihood
        probs = jax.nn.sigmoid(logits.squeeze(-1))  # Squeeze to match (batch_size,)
        numpyro.sample("obs", dist.Bernoulli(probs=probs), obs=y)



import jax

# Generate dummy data
rng = jax.random.PRNGKey(42)
X = jax.random.normal(rng, shape=(32, 1, 28, 28))  # Batch of grayscale images
y = jax.random.bernoulli(rng, p=0.5, shape=(32,))  # Binary labels

# Test TransposedConv2d
print("Testing TransposedConv2d with Linear...")
transposed_conv_test = TransposedConvTest()
transposed_conv_test.compile(num_samples=100, num_warmup=50, num_chains=1)
transposed_conv_test.fit(X, y, rng)

# Test FFTTransposedConv2d
print("Testing FFTTransposedConv2d with FFTLinear...")
fft_transposed_conv_test = FFTTransposedConvTest()
fft_transposed_conv_test.compile(num_samples=100, num_warmup=50, num_chains=1)
fft_transposed_conv_test.fit(X, y, rng)
