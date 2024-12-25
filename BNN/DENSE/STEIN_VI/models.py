import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp


def regression_model(X, y=None):
    """
    Bayesian Regression Model
    """
    input_size = X.shape[1]
    hidden_size = 10  # Hidden layer size
    num_particles = 10  # Number of particles for SVGD approximation

    # Priors for dense weights and bias
    weights_dense = numpyro.sample(
        "weights_dense",
        dist.Normal(0, 1).expand([num_particles, input_size, hidden_size]),
    )
    bias_dense = numpyro.sample(
        "bias_dense", dist.Normal(0, 1).expand([num_particles, hidden_size])
    )

    # Dense layer for each particle
    hidden_list = []
    for i in range(num_particles):
        hidden = jnp.matmul(X, weights_dense[i]) + bias_dense[i]
        hidden = jax.nn.relu(hidden)
        hidden_list.append(hidden)

    hidden = jnp.stack(
        hidden_list, axis=0
    )  # Shape: (num_particles, batch_size, hidden_dim)

    # Output layer
    weights_out = numpyro.sample(
        "weights_out", dist.Normal(0, 1).expand([num_particles, hidden_size, 1])
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


def binary_model(X, y=None):
    """
    Bayesian Binary Classification Model with a Dense Layer.
    """
    input_size = X.shape[1]
    hidden_size = 10  # Hidden layer size
    num_particles = 10  # Number of particles for SVGD approximation

    # Priors for dense weights and bias
    weights_dense = numpyro.sample(
        "weights_dense",
        dist.Normal(0, 1).expand([num_particles, input_size, hidden_size]),
    )
    bias_dense = numpyro.sample(
        "bias_dense", dist.Normal(0, 1).expand([num_particles, hidden_size])
    )

    # Dense layer for each particle
    hidden_list = []
    for i in range(num_particles):
        hidden = jnp.matmul(X, weights_dense[i]) + bias_dense[i]
        hidden = jax.nn.relu(hidden)
        hidden_list.append(hidden)

    hidden = jnp.stack(
        hidden_list, axis=0
    )  # Shape: (num_particles, batch_size, hidden_dim)

    # Output layer
    weights_out = numpyro.sample(
        "weights_out", dist.Normal(0, 1).expand([num_particles, hidden_size, 1])
    )
    bias_out = numpyro.sample("bias_out", dist.Normal(0, 1).expand([num_particles]))

    # Predictions for each particle
    logits = (
        jnp.einsum("pbi,pio->pbo", hidden, weights_out).squeeze(-1) + bias_out[:, None]
    )

    # Combine particle logits (mean across particles for simplicity)
    mean_logits = jnp.mean(logits, axis=0)

    # Sigmoid activation for binary classification
    probs = jax.nn.sigmoid(mean_logits)

    # Likelihood
    numpyro.deterministic("probs", probs)
    numpyro.sample("obs", dist.Bernoulli(probs=probs), obs=y)


def multiclass_model(X, y=None, num_classes=3):
    """
    Bayesian Multiclass Classification Model with a Dense Layer.
    """
    input_size = X.shape[1]
    num_particles = 10  # Number of particles for SVGD approximation

    # Priors for dense weights and bias
    weights_dense = numpyro.sample(
        "weights_dense",
        dist.Normal(0, 1).expand([num_particles, input_size, input_size]),
    )
    bias_dense = numpyro.sample(
        "bias_dense", dist.Normal(0, 1).expand([num_particles, input_size])
    )

    # Dense layer for each particle
    hidden_list = []
    for i in range(num_particles):
        # Perform dense layer computation
        hidden = jnp.matmul(X, weights_dense[i]) + bias_dense[i]
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
