import jax
import jax.numpy as jnp
from numpyro import sample, deterministic
from numpyro import deterministic, plate, sample, set_platform, subsample
from numpyro.distributions import Normal, Gamma, Categorical, Bernoulli
import numpyro.distributions as dist
import numpyro
from fft_matrix import circulant_matrix_multiply
from jax import nn


def regression_model(X, y=None):
    """
    Bayesian Regression Model with Circulant Matrix Layer, inspired by SVGD.
    """
    input_size = X.shape[1]
    num_particles = 10  # Number of particles for SVGD approximation

    # Priors for circulant matrix and bias
    first_row = numpyro.sample(
        "first_row", dist.Normal(0, 1).expand([num_particles, input_size])
    )
    bias_circulant = numpyro.sample(
        "bias_circulant", dist.Normal(0, 1).expand([num_particles])
    )

    # Circulant layer for each particle
    hidden_list = []
    for i in range(num_particles):
        hidden = circulant_matrix_multiply(first_row[i], X) + bias_circulant[i]
        hidden = jax.nn.relu(hidden)
        hidden_list.append(hidden)

    hidden = jnp.stack(
        hidden_list, axis=0
    )  # Shape: (num_particles, batch_size, hidden_dim)

    # Output layer
    weights_out = numpyro.sample(
        "weights_out", dist.Normal(0, 1).expand([num_particles, input_size, 1])
    )
    bias_out = numpyro.sample("bias_out", dist.Normal(0, 1).expand([num_particles]))

    # Predictions for each particle
    predictions = (
        jnp.einsum("pbi,pio->pbo", hidden, weights_out).squeeze(-1) + bias_out[:, None]
    )

    # Combine particle predictions (mean across particles for simplicity)
    mean_predictions = jnp.mean(predictions, axis=0)

    # Likelihood
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("obs", dist.Normal(mean_predictions, sigma), obs=y)


def multiclass_model(X, y=None, num_classes=3):
    """
    Bayesian Multiclass Classification Model with Circulant Matrix Layer.
    """
    input_size = X.shape[1]
    num_particles = 10  # Number of particles for SVGD approximation

    # Priors for circulant matrix and bias
    first_row = numpyro.sample(
        "first_row", dist.Normal(0, 1).expand([num_particles, input_size])
    )
    bias_circulant = numpyro.sample(
        "bias_circulant", dist.Normal(0, 1).expand([num_particles])
    )

    # Circulant layer for each particle
    hidden_list = []
    for i in range(num_particles):
        hidden = circulant_matrix_multiply(first_row[i], X) + bias_circulant[i]
        hidden = jax.nn.relu(hidden)
        hidden_list.append(hidden)

    hidden = jnp.stack(
        hidden_list, axis=0
    )  # Shape: (num_particles, batch_size, hidden_dim)

    # Output layer
    weights_out = numpyro.sample(
        "weights_out",
        dist.Normal(0, 1).expand([num_particles, input_size, num_classes]),
    )
    bias_out = numpyro.sample(
        "bias_out", dist.Normal(0, 1).expand([num_particles, num_classes])
    )

    # Predictions for each particle
    logits = jnp.einsum("pbi,pio->pbo", hidden, weights_out) + bias_out[:, None, :]

    # Combine particle logits (mean across particles for simplicity)
    mean_logits = jnp.mean(logits, axis=0)

    # Softmax activation for multiclass classification
    probs = jax.nn.softmax(mean_logits, axis=-1)

    # Likelihood
    numpyro.deterministic("probs", probs)
    numpyro.sample("obs", dist.Categorical(probs=probs), obs=y)


def binary_model(X, y=None):
    """
    Bayesian Binary Classification Model with Circulant Matrix Layer.
    """
    input_size = X.shape[1]
    num_particles = 10  # Number of particles for SVGD approximation

    # Priors for circulant matrix and bias
    first_row = numpyro.sample(
        "first_row", dist.Normal(0, 1).expand([num_particles, input_size])
    )
    bias_circulant = numpyro.sample(
        "bias_circulant", dist.Normal(0, 1).expand([num_particles])
    )

    # Circulant layer for each particle
    hidden_list = []
    for i in range(num_particles):
        hidden = circulant_matrix_multiply(first_row[i], X) + bias_circulant[i]
        hidden = jax.nn.relu(hidden)
        hidden_list.append(hidden)

    hidden = jnp.stack(hidden_list, axis=0)

    weights_out = numpyro.sample(
        "weights_out", dist.Normal(0, 1).expand([num_particles, input_size, 1])
    )
    bias_out = numpyro.sample("bias_out", dist.Normal(0, 1).expand([num_particles]))

    logits = (
        jnp.einsum("pbi,pio->pbo", hidden, weights_out).squeeze(-1) + bias_out[:, None]
    )

    mean_logits = jnp.mean(logits, axis=0)

    probs = jax.nn.sigmoid(mean_logits)

    numpyro.deterministic("probs", probs)
    numpyro.sample("obs", dist.Bernoulli(probs=probs), obs=y)
