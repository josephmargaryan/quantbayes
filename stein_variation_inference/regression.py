import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpyro
from numpyro import sample, deterministic
from numpyro.contrib.einstein import SteinVI, RBFKernel, MixtureGuidePredictive
from numpyro.distributions import Normal, Gamma
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoNormal


# Step 1: Generate synthetic data
def generate_synthetic_data(n_samples=100):
    np.random.seed(42)
    x = np.linspace(-3, 3, n_samples)
    y = 0.5 * x + np.sin(x) + np.random.normal(0, 0.3, size=n_samples)
    return x, y


x_train, y_train = generate_synthetic_data()
x_train, y_train = jnp.array(x_train), jnp.array(y_train)


# Step 2: Define the Bayesian Neural Network
def bnn_model(x, y=None, hidden_dim=10):
    # Priors for weights and biases
    prec = sample("prec", Gamma(1.0, 0.1))
    w1 = sample("w1", Normal(0, 1 / jnp.sqrt(prec)).expand([1, hidden_dim]))
    b1 = sample("b1", Normal(0, 1 / jnp.sqrt(prec)).expand([hidden_dim]))
    w2 = sample("w2", Normal(0, 1 / jnp.sqrt(prec)).expand([hidden_dim, 1]))
    b2 = sample("b2", Normal(0, 1 / jnp.sqrt(prec)).expand([1]))
    prec_obs = sample("prec_obs", Gamma(1.0, 0.1))

    # Forward pass
    hidden = jnp.maximum(jnp.dot(x[:, None], w1) + b1, 0)  # ReLU activation
    mean = deterministic("mean", jnp.dot(hidden, w2) + b2)

    # Likelihood
    sample("y", Normal(mean.squeeze(), 1 / jnp.sqrt(prec_obs)), obs=y)


# Step 3: Train the model using SteinVI
rng_key = random.PRNGKey(0)
guide = AutoNormal(bnn_model)
optimizer = Adam(step_size=0.01)

stein = SteinVI(
    model=bnn_model,
    guide=guide,
    optim=optimizer,
    kernel_fn=RBFKernel(),
    num_stein_particles=10,
    num_elbo_particles=10,
)

stein_result = stein.run(rng_key, num_steps=1000, x=x_train, y=y_train)

# Step 4: Make predictions
predictive = MixtureGuidePredictive(
    bnn_model,
    guide=stein.guide,
    params=stein.get_params(stein_result.state),
    guide_sites=stein.guide_sites,
    num_samples=100,
)

x_test = jnp.linspace(-3, 3, 100)
predictions = predictive(rng_key, x=x_test)["mean"]

# Assuming `predictions` is a (num_samples, num_test_points) array from MixtureGuidePredictive
mean_prediction = predictions.mean(axis=0)
lower = jnp.percentile(predictions, 5, axis=0)
upper = jnp.percentile(predictions, 95, axis=0)

# Ensure all are 1-dimensional
x_test = x_test.squeeze()
mean_prediction = mean_prediction.squeeze()
lower = lower.squeeze()
upper = upper.squeeze()

# Plotting with corrected shapes
plt.figure(figsize=(8, 6))
plt.plot(x_test, mean_prediction, "r-", label="Mean Prediction")
plt.fill_between(x_test, lower, upper, color="gray", alpha=0.3, label="90% CI")
plt.scatter(x_train, y_train, color="blue", alpha=0.5, label="Training Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Bayesian Neural Network Regression with SteinVI")
plt.legend()
plt.show()
