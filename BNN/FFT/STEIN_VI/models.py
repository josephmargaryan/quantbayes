import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro
from BNN.FFT.STEIN_VI.fft_matrix import circulant_matrix_multiply


def regression_model(X, y=None):
    """
    Bayesian Regression Model with Circulant Matrix Layer, inspired by SVGD.
    """
    input_size = X.shape[1]
    num_particles = 10

    first_row = numpyro.sample(
        "first_row", dist.Normal(0, 1).expand([num_particles, input_size])
    )
    bias_circulant = numpyro.sample(
        "bias_circulant", dist.Normal(0, 1).expand([num_particles, input_size])
    )

    hidden_list = []
    for i in range(num_particles):
        hidden = circulant_matrix_multiply(first_row[i], X) + bias_circulant[i]
        hidden = jax.nn.relu(hidden)
        hidden_list.append(hidden)

    hidden = jnp.stack(hidden_list, axis=0)

    weights_out = numpyro.sample(
        "weights_out", dist.Normal(0, 1).expand([num_particles, hidden.shape[2], 1])
    )
    bias_out = numpyro.sample("bias_out", dist.Normal(0, 1).expand([num_particles]))

    predictions = (
        jnp.einsum("pbi,pio->pbo", hidden, weights_out).squeeze(-1) + bias_out[:, None]
    )

    mean_predictions = jnp.mean(predictions, axis=0)
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("obs", dist.Normal(mean_predictions, sigma), obs=y)


def multiclass_model(X, y=None, num_classes=3):
    """
    Bayesian Multiclass Classification Model with Circulant Matrix Layer.
    """
    input_size = X.shape[1]
    num_particles = 10

    first_row = numpyro.sample(
        "first_row", dist.Normal(0, 1).expand([num_particles, input_size])
    )
    bias_circulant = numpyro.sample(
        "bias_circulant", dist.Normal(0, 1).expand([num_particles, input_size])
    )

    hidden_list = []
    for i in range(num_particles):
        hidden = circulant_matrix_multiply(first_row[i], X) + bias_circulant[i]
        hidden = jax.nn.relu(hidden)
        hidden_list.append(hidden)

    hidden = jnp.stack(hidden_list, axis=0)

    weights_out = numpyro.sample(
        "weights_out",
        dist.Normal(0, 1).expand([num_particles, hidden.shape[2], num_classes]),
    )
    bias_out = numpyro.sample(
        "bias_out", dist.Normal(0, 1).expand([num_particles, num_classes])
    )

    logits = jnp.einsum("pbi,pio->pbo", hidden, weights_out) + bias_out[:, None, :]
    logits = jnp.mean(logits, axis=0)
    logits = jnp.clip(logits, a_min=-10, a_max=10)
    numpyro.deterministic("logits", logits)
    numpyro.sample("obs", dist.Categorical(logits=logits), obs=y)


def binary_model(X, y=None):
    """
    Bayesian Binary Classification Model with Circulant Matrix Layer.
    """
    input_size = X.shape[1]
    num_particles = 10

    first_row = numpyro.sample(
        "first_row", dist.Normal(0, 1).expand([num_particles, input_size])
    )
    bias_circulant = numpyro.sample(
        "bias_circulant", dist.Normal(0, 1).expand([num_particles, input_size])
    )

    hidden_list = []
    for i in range(num_particles):
        hidden = circulant_matrix_multiply(first_row[i], X) + bias_circulant[i]
        hidden = jax.nn.relu(hidden)
        hidden_list.append(hidden)

    hidden = jnp.stack(hidden_list, axis=0)

    weights_out = numpyro.sample(
        "weights_out", dist.Normal(0, 1).expand([num_particles, hidden.shape[2], 1])
    )
    bias_out = numpyro.sample("bias_out", dist.Normal(0, 1).expand([num_particles]))

    logits = (
        jnp.einsum("pbi,pio->pbo", hidden, weights_out).squeeze(-1) + bias_out[:, None]
    )  # Logits for each particle
    logits = jnp.mean(logits, axis=0)

    logits = jnp.clip(logits, a_min=-10, a_max=10)
    numpyro.deterministic("logits", logits)
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)
