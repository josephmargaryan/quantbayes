import matplotlib.pyplot as plt
import numpy as np

from jax import vmap, random
import jax.numpy as jnp

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


# Non-linearity function
def nonlin(x):
    return jnp.tanh(x)


# Bayesian neural network model
def model(X, Y, D_H, D_Y=1):
    N, D_X = X.shape

    # Sample weights for each layer
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H))))
    z1 = nonlin(jnp.matmul(X, w1))

    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    z2 = nonlin(jnp.matmul(z1, w2))

    w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y))))
    z3 = jnp.matmul(z2, w3)

    # Observation noise
    sigma_obs = 1.0 / jnp.sqrt(numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0)))

    # Observations
    numpyro.sample("Y", dist.Normal(z3, sigma_obs).to_event(1), obs=Y)


# Helper function to run inference
def run_inference(model, rng_key, X, Y, D_H, num_samples=1000, num_warmup=500):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(rng_key, X, Y, D_H)
    mcmc.print_summary()
    return mcmc.get_samples()


# Helper function to predict
def predict(model, rng_key, samples, X, D_H):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=D_H)
    return trace["Y"]["value"]


# Generate synthetic data
def get_data(N=50, D_X=3, sigma_obs=0.05, N_test=500):
    D_Y = 1
    np.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    X = jnp.power(X[:, np.newaxis], jnp.arange(D_X))
    W = 0.5 * np.random.randn(D_X)
    Y = jnp.dot(X, W) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])
    Y += sigma_obs * np.random.randn(N)
    Y = Y[:, np.newaxis]
    Y = (Y - jnp.mean(Y)) / jnp.std(Y)

    X_test = jnp.linspace(-1.3, 1.3, N_test)
    X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))

    return X, Y, X_test


# Main script
rng_key = random.PRNGKey(0)
X, Y, X_test = get_data(N=100, D_X=3)
D_H = 5  # Number of hidden units

# Run inference
samples = run_inference(model, rng_key, X, Y, D_H, num_samples=1000, num_warmup=500)

# Predict on test data
rng_keys = random.split(rng_key, 1000)
predictions = vmap(
    lambda samples, rng_key: predict(model, rng_key, samples, X_test, D_H)
)(samples, rng_keys)
predictions = predictions[..., 0]

# Calculate mean prediction and confidence intervals
mean_prediction = jnp.mean(predictions, axis=0)
percentiles = np.percentile(predictions, [5, 95], axis=0)

# Visualization
plt.figure(figsize=(8, 6))
plt.plot(X[:, 1], Y[:, 0], "kx", label="Training data")
plt.fill_between(
    X_test[:, 1],
    percentiles[0, :],
    percentiles[1, :],
    color="lightblue",
    label="90% CI",
)
plt.plot(X_test[:, 1], mean_prediction, "blue", label="Mean prediction")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Bayesian Neural Network Predictions")
plt.show()
