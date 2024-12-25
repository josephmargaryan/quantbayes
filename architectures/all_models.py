import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist

################################ Regression ################################


def regression_model1(X, y=None):
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


def regression_model2(X, y=None, hidden_dim=10):
    input_dim = X.shape[1]

    # Priors for weights and biases
    w_hidden = numpyro.sample(
        "w_hidden", dist.Normal(0, 1).expand([input_dim, hidden_dim])
    )
    b_hidden = numpyro.sample("b_hidden", dist.Normal(0, 1).expand([hidden_dim]))

    w_out = numpyro.sample("w_out", dist.Normal(0, 1).expand([hidden_dim, 1]))
    b_out = numpyro.sample("b_out", dist.Normal(0, 1).expand([1]))

    # Hidden layer
    hidden = jax.nn.relu(jnp.dot(X, w_hidden) + b_hidden)

    # Outputs (regression predictions)
    mean = numpyro.deterministic("mean", jnp.dot(hidden, w_out).squeeze() + b_out)

    # Likelihood
    sigma = numpyro.sample(
        "sigma", dist.Exponential(1.0)
    )  # Standard deviation of the noise
    numpyro.sample("y", dist.Normal(mean, sigma), obs=y)


def regression_model3(x, y=None, hidden_dim=10):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    w1 = numpyro.sample(
        "w1", dist.Normal(0, 1 / jnp.sqrt(prec)).expand([1, hidden_dim])
    )
    b1 = numpyro.sample("b1", dist.Normal(0, 1 / jnp.sqrt(prec)).expand([hidden_dim]))
    w2 = numpyro.sample(
        "w2", dist.Normal(0, 1 / jnp.sqrt(prec)).expand([hidden_dim, 1])
    )
    b2 = numpyro.sample("b2", dist.Normal(0, 1 / jnp.sqrt(prec)).expand([1]))
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(1.0, 0.1))

    hidden = jnp.maximum(jnp.dot(x[:, None], w1) + b1, 0)
    mean = numpyro.deterministic("mean", jnp.dot(hidden, w2) + b2)

    numpyro.sample("y", dist.Normal(mean.squeeze(), 1 / jnp.sqrt(prec_obs)), obs=y)


################################### Binary ######################################
def binary_model1(X, y=None):
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


def binary_model2(X, y=None, hidden_dim=10):
    input_dim = X.shape[1]

    # Priors for weights and biases
    w_hidden = numpyro.sample(
        "w_hidden", dist.Normal(0, 1).expand([input_dim, hidden_dim])
    )
    b_hidden = numpyro.sample("b_hidden", dist.Normal(0, 1).expand([hidden_dim]))

    w_out = numpyro.sample("w_out", dist.Normal(0, 1).expand([hidden_dim, 1]))
    b_out = numpyro.sample("b_out", dist.Normal(0, 1).expand([1]))

    # Hidden layer
    hidden = jax.nn.relu(jnp.dot(X, w_hidden) + b_hidden)

    # Logits (deterministic)
    logits = numpyro.deterministic("logits", jnp.dot(hidden, w_out).squeeze() + b_out)

    # Observations
    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)


def bnn_model3(X, y=None, hidden_dim=16):
    prec = numpyro.sample("precision", dist.Gamma(1.0, 1.0))
    n_features = X.shape[1]

    w1 = numpyro.sample(
        "w1", dist.Normal(0, 1 / jnp.sqrt(prec)).expand([n_features, hidden_dim])
    )
    b1 = numpyro.sample("b1", dist.Normal(0, 1 / jnp.sqrt(prec)).expand([hidden_dim]))

    w2 = numpyro.sample(
        "w2", dist.Normal(0, 1 / jnp.sqrt(prec)).expand([hidden_dim, 1])
    )
    b2 = numpyro.sample("b2", dist.Normal(0, 1 / jnp.sqrt(prec)))

    hidden = jax.nn.relu(jnp.dot(X, w1) + b1)
    logits = numpyro.deterministic("logits", jnp.dot(hidden, w2) + b2)

    numpyro.sample("y", dist.Bernoulli(logits=logits.squeeze()), obs=y)


############################# Multiclass ##########################
def multiclass_model1(X, y=None, num_classes=3):
    """
    Bayesian Multiclass Classification Model with a Dense Layer.
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
        "weights_out",
        dist.Normal(0, 1).expand([num_particles, hidden_size, num_classes]),
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


def multiclass_model2(X, y=None, hidden_dim=10, num_classes=3):
    input_dim = X.shape[1]

    # Priors for weights and biases
    w_hidden = numpyro.sample(
        "w_hidden", dist.Normal(0, 1).expand([input_dim, hidden_dim])
    )
    b_hidden = numpyro.sample("b_hidden", dist.Normal(0, 1).expand([hidden_dim]))

    w_out = numpyro.sample("w_out", dist.Normal(0, 1).expand([hidden_dim, num_classes]))
    b_out = numpyro.sample("b_out", dist.Normal(0, 1).expand([num_classes]))

    # Hidden layer
    hidden = jax.nn.relu(jnp.dot(X, w_hidden) + b_hidden)

    # Logits (unnormalized scores for each class)
    logits = numpyro.deterministic("logits", jnp.dot(hidden, w_out) + b_out)

    # Likelihood
    numpyro.sample("y", dist.Categorical(logits=logits), obs=y)


def multiclass_model3(X, y=None, hidden_dim=16, n_classes=3):
    prec = numpyro.sample("precision", dist.Gamma(1.0, 1.0))
    n_features = X.shape[1]

    w1 = numpyro.sample(
        "w1", dist.Normal(0, 1 / jnp.sqrt(prec)).expand([n_features, hidden_dim])
    )
    b1 = numpyro.sample("b1", dist.Normal(0, 1 / jnp.sqrt(prec)).expand([hidden_dim]))

    w2 = numpyro.sample(
        "w2", dist.Normal(0, 1 / jnp.sqrt(prec)).expand([hidden_dim, n_classes])
    )
    b2 = numpyro.sample("b2", dist.Normal(0, 1 / jnp.sqrt(prec)).expand([n_classes]))

    hidden = jax.nn.relu(jnp.dot(X, w1) + b1)
    logits = numpyro.deterministic("logits", jnp.dot(hidden, w2) + b2)

    numpyro.sample("y", dist.Categorical(logits=logits), obs=y)
