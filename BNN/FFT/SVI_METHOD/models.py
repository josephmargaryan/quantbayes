import numpyro
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from fft_matrix import circulant_matrix_multiply


def regression_model(X, y=None):
    """
    Bayesian Neural Network with Circulant Matrix Layer.
    """
    input_size = X.shape[1]

    first_row = numpyro.sample("first_row", dist.Normal(0, 1).expand([input_size]))
    bias_circulant = numpyro.sample("bias_circulant", dist.Normal(0, 1))

    hidden = circulant_matrix_multiply(first_row, X) + bias_circulant
    hidden = jax.nn.relu(hidden)

    weights_out = numpyro.sample(
        "weights_out", dist.Normal(0, 1).expand([input_size, 1])
    )
    bias_out = numpyro.sample("bias_out", dist.Normal(0, 1))

    predictions = jnp.matmul(hidden, weights_out).squeeze() + bias_out

    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("obs", dist.Normal(predictions, sigma), obs=y)


def binary_model(X, y=None):
    """
    Bayesian Neural Network for Binary Classification.
    """
    input_size = X.shape[1]

    first_row = numpyro.sample("first_row", dist.Normal(0, 1).expand([input_size]))
    bias_circulant = numpyro.sample("bias_circulant", dist.Normal(0, 1))

    hidden = circulant_matrix_multiply(first_row, X) + bias_circulant
    hidden = jax.nn.relu(hidden)

    weights_out = numpyro.sample(
        "weights_out", dist.Normal(0, 1).expand([hidden.shape[1], 1])
    )
    bias_out = numpyro.sample("bias_out", dist.Normal(0, 1))
    logits = jnp.matmul(hidden, weights_out).squeeze() + bias_out
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)


def multiclass_model(X, y=None, n_classes=3):
    """
    Bayesian Neural Network with Circulant Matrix Layer for Multiclass Classification.
    """
    input_size = X.shape[1]

    first_row = numpyro.sample("first_row", dist.Normal(0, 1).expand([input_size]))
    bias_circulant = numpyro.sample("bias_circulant", dist.Normal(0, 1))

    hidden = circulant_matrix_multiply(first_row, X) + bias_circulant
    hidden = jax.nn.relu(hidden)

    weights_out = numpyro.sample(
        "weights_out", dist.Normal(0, 1).expand([hidden.shape[1], n_classes])
    )
    bias_out = numpyro.sample("bias_out", dist.Normal(0, 1).expand([n_classes]))

    logits = jnp.matmul(hidden, weights_out) + bias_out
    numpyro.deterministic("logits", logits)
    numpyro.sample("obs", dist.Categorical(logits=logits), obs=y)
