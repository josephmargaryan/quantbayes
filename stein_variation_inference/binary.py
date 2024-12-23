import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

import numpyro
from numpyro import sample, deterministic
from numpyro.contrib.einstein import SteinVI, RBFKernel, MixtureGuidePredictive
from numpyro.distributions import Normal, Gamma, Bernoulli
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoNormal
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


# Step 1: Generate synthetic binary classification data
def generate_data(n_samples=500, noise=0.2):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return (
        jnp.array(X_train),
        jnp.array(y_train),
        jnp.array(X_test),
        jnp.array(y_test),
    )


# Step 2: Define the Bayesian neural network model
def bnn_model(X, y=None, hidden_dim=16):
    # Priors for weights and biases
    prec = sample("precision", Gamma(1.0, 1.0))
    n_features = X.shape[1]  # Number of input features

    # Hidden layer
    w1 = sample("w1", Normal(0, 1 / jnp.sqrt(prec)).expand([n_features, hidden_dim]))
    b1 = sample("b1", Normal(0, 1 / jnp.sqrt(prec)).expand([hidden_dim]))

    # Output layer
    w2 = sample("w2", Normal(0, 1 / jnp.sqrt(prec)).expand([hidden_dim, 1]))
    b2 = sample("b2", Normal(0, 1 / jnp.sqrt(prec)))

    # Forward pass through the network
    hidden = jax.nn.relu(jnp.dot(X, w1) + b1)  # Hidden layer activation
    logits = deterministic("logits", jnp.dot(hidden, w2) + b2)  # Logits for Bernoulli

    # Likelihood for binary classification
    numpyro.sample("y", Bernoulli(logits=logits.squeeze()), obs=y)


# Step 3: Train the model using SteinVI
def train_bnn(X_train, y_train, hidden_dim=16, num_steps=1000):
    guide = AutoNormal(bnn_model)

    # Set up SteinVI with an RBF kernel
    stein = SteinVI(
        bnn_model,
        guide,
        Adam(0.01),
        RBFKernel(),
        num_stein_particles=10,
        num_elbo_particles=10,
    )

    rng_key = random.PRNGKey(0)
    stein_result = stein.run(
        rng_key,
        num_steps,
        X_train,
        y_train,
        hidden_dim=hidden_dim,
        progress_bar=True,
    )
    return stein, stein_result


# Step 4: Evaluate the model
def evaluate_bnn(stein, stein_result, X_test, hidden_dim=16):
    predictive = MixtureGuidePredictive(
        bnn_model,
        stein.guide,
        params=stein.get_params(stein_result.state),
        num_samples=100,
        guide_sites=stein.guide_sites,
    )

    # Obtain predictions
    rng_key = random.PRNGKey(1)
    pred_samples = predictive(rng_key, X_test, hidden_dim=hidden_dim)["y"]
    mean_pred = pred_samples.mean(axis=0)

    return mean_pred


# Step 5: Visualize decision boundary
def plot_decision_boundary(X, y, X_test, y_test, y_pred, title="BNN Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = jnp.c_[xx.ravel(), yy.ravel()]

    # Predict on grid
    pred_grid = evaluate_bnn(*y_pred, grid).reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, pred_grid, levels=50, cmap="RdBu", alpha=0.8)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors="k", cmap="RdBu", s=40)
    plt.colorbar(label="Predicted Probability")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# Main script
if __name__ == "__main__":
    # Generate synthetic data
    X_train, y_train, X_test, y_test = generate_data()

    # Train the model
    print("Training Bayesian Neural Network...")
    stein, stein_result = train_bnn(X_train, y_train)

    # Evaluate the model
    mean_pred = evaluate_bnn(stein, stein_result, X_test)

    # Plot decision boundary
    plot_decision_boundary(X_train, y_train, X_test, y_test, (stein, stein_result))
