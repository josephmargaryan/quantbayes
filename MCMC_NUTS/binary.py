import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import jax

import numpyro
from numpyro import sample, deterministic
from numpyro.infer import MCMC, NUTS
from numpyro.distributions import Normal, Bernoulli
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


# Generate synthetic binary classification data
def generate_data(n_samples=500, noise=0.2):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return jnp.array(X_train), jnp.array(y_train), jnp.array(X_test), jnp.array(y_test)


# Define the Bayesian neural network model for binary classification
def bnn_model(X, y=None, hidden_dim=16):
    w1 = sample("w1", Normal(0, 1).expand([X.shape[1], hidden_dim]))
    b1 = sample("b1", Normal(0, 1).expand([hidden_dim]))
    w2 = sample("w2", Normal(0, 1).expand([hidden_dim, 1]))
    b2 = sample("b2", Normal(0, 1))

    hidden = jax.nn.relu(jnp.dot(X, w1) + b1)
    logits = deterministic("logits", jnp.dot(hidden, w2) + b2)
    sample("y", Bernoulli(logits=logits.squeeze()), obs=y)


# Train the BNN using NUTS
def train_bnn(X_train, y_train, hidden_dim=16, num_samples=1000, num_warmup=500):
    rng_key = random.PRNGKey(0)
    kernel = NUTS(bnn_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, X=X_train, y=y_train, hidden_dim=hidden_dim)
    return mcmc


# Evaluate the BNN
def evaluate_bnn(mcmc, X_test, hidden_dim=16):
    samples = mcmc.get_samples()

    # Predictive samples for logits
    logits_samples = []
    for i in range(samples["w1"].shape[0]):  # Iterate over posterior samples
        w1 = samples["w1"][i]
        b1 = samples["b1"][i]
        w2 = samples["w2"][i]
        b2 = samples["b2"][i]

        hidden = jax.nn.relu(jnp.dot(X_test, w1) + b1)
        logits = jnp.dot(hidden, w2) + b2
        logits_samples.append(logits)

    logits_samples = jnp.stack(logits_samples)
    probs = jax.nn.sigmoid(logits_samples)  # Convert logits to probabilities
    mean_probs = probs.mean(axis=0)  # Average probabilities across posterior samples

    return mean_probs


# Plot the decision boundary
def plot_decision_boundary(mcmc, X_train, y_train, X_test, y_test, hidden_dim=16):
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = jnp.c_[xx.ravel(), yy.ravel()]

    # Predict probabilities for the grid
    mean_probs = evaluate_bnn(mcmc, grid, hidden_dim=hidden_dim)
    zz = mean_probs.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, zz, levels=50, cmap="RdBu", alpha=0.8)

    # Add training and test points
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        edgecolor="k",
        cmap="RdBu",
        label="Train",
    )
    plt.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k", cmap="RdBu", label="Test"
    )

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label("Predicted Probability")

    # Add labels and title
    plt.title("Binary Classification Decision Boundary (NUTS MCMC)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


# Main script
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = generate_data()
    print("Training Bayesian Neural Network (Binary Classification)...")
    mcmc = train_bnn(X_train, y_train)

    plot_decision_boundary(mcmc, X_train, y_train, X_test, y_test)
