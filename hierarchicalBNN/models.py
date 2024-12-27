import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist


def BNNRegressor(x, y=None, hidden_dim=10):
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


def BNNBinary(X, y=None, hidden_dim=16):
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
    logits = jnp.clip(logits, a_min=-10, a_max=10)

    numpyro.sample("y", dist.Bernoulli(logits=logits.squeeze()), obs=y)


def BNNMultiClass(X, y=None, hidden_dim=16, n_classes=3):
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
    logits = jnp.clip(logits, a_min=-10, a_max=10)

    numpyro.sample("y", dist.Categorical(logits=logits), obs=y)
