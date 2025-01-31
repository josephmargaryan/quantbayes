import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module
from jax import random

from quantbayes.bnn import Module

# Import AttentionUNet
from quantbayes.stochax.vision.att_unet import AttentionUNet


class MyOwnModel(Module):
    def __init__(self, task_type: str, method: str):
        super().__init__(task_type=task_type, method=method)

    def __call__(self, X, y=None):
        """
        Defines the NumPyro probabilistic model using `random_flax_module`
        """
        # Bayesian Neural Network with Flax
        net = random_flax_module(
            "AttUNet",
            nn_module=AttentionUNet(num_classes=1, capture_intermediates=False),
            prior=dist.Normal(0, 1),
            input_shape=(1, 128, 128, 1),  # Correct input shape
        )

        # Forward pass
        logits = net(X)
        logits = jnp.clip(logits, a_min=-10, a_max=10)  # Prevent numerical instability
        numpyro.deterministic("logits", logits)

        # Likelihood function (binary segmentation)
        with numpyro.plate("batch", len(X), dim=-4):
            with numpyro.plate("height", X.shape[1], dim=-3):
                with numpyro.plate("width", X.shape[2], dim=-2):
                    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)


# ------------------------------------------------------------------------------
# 2️⃣ Create & Train Model
# ------------------------------------------------------------------------------
# Instantiate model
model = MyOwnModel(task_type="image_segmentation", method="svi")  # Define inference method

# Compile (chooses between NUTS, SVI, or SteinVI)
model.compile(num_samples=10, learning_rate=0.01)

# Define Training Data
rng_key = random.PRNGKey(42)
X_train = jnp.ones((10, 128, 128, 1))  # Example training images (batch=64)
y_train = jnp.ones((10, 128, 128, 1))  # Example training masks

# Train the Model
model.fit(X_train, y_train, rng_key, num_steps=10)
model.visualize(X_train[:3], y_train[:3])