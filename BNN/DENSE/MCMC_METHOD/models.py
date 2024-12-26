import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp


def regression_model(X, y=None, hidden_dim=10):
    input_dim = X.shape[1]

    w_hidden = numpyro.sample(
        "w_hidden", dist.Normal(0, 1).expand([input_dim, hidden_dim])
    )
    b_hidden = numpyro.sample("b_hidden", dist.Normal(0, 1).expand([hidden_dim]))

    w_out = numpyro.sample("w_out", dist.Normal(0, 1).expand([hidden_dim, 1]))
    b_out = numpyro.sample("b_out", dist.Normal(0, 1).expand([1]))

    hidden = jax.nn.relu(jnp.dot(X, w_hidden) + b_hidden)

    mean = numpyro.deterministic("mean", jnp.dot(hidden, w_out).squeeze() + b_out)

    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("y", dist.Normal(mean, sigma), obs=y)


def binary_model(X, y=None, hidden_dim=10):
    input_dim = X.shape[1]

    w_hidden = numpyro.sample(
        "w_hidden", dist.Normal(0, 1).expand([input_dim, hidden_dim])
    )
    b_hidden = numpyro.sample("b_hidden", dist.Normal(0, 1).expand([hidden_dim]))

    w_out = numpyro.sample("w_out", dist.Normal(0, 1).expand([hidden_dim, 1]))
    b_out = numpyro.sample("b_out", dist.Normal(0, 1).expand([1]))

    hidden = jax.nn.relu(jnp.dot(X, w_hidden) + b_hidden)

    logits = numpyro.deterministic("logits", jnp.dot(hidden, w_out).squeeze() + b_out)

    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)


def multiclass_model(X, y=None, hidden_dim=10, num_classes=3):
    input_dim = X.shape[1]

    w_hidden = numpyro.sample(
        "w_hidden", dist.Normal(0, 1).expand([input_dim, hidden_dim])
    )
    b_hidden = numpyro.sample("b_hidden", dist.Normal(0, 1).expand([hidden_dim]))

    w_out = numpyro.sample("w_out", dist.Normal(0, 1).expand([hidden_dim, num_classes]))
    b_out = numpyro.sample("b_out", dist.Normal(0, 1).expand([num_classes]))

    hidden = jax.nn.relu(jnp.dot(X, w_hidden) + b_hidden)

    logits = numpyro.deterministic("logits", jnp.dot(hidden, w_out) + b_out)

    numpyro.sample("y", dist.Categorical(logits=logits), obs=y)
