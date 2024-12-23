import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

import numpyro
from numpyro import sample, deterministic
from numpyro.contrib.einstein import SteinVI, RBFKernel, MixtureGuidePredictive
from numpyro.distributions import Normal, Gamma, Categorical
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoNormal
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Step 1: Generate synthetic multiclass data
def generate_multiclass_data(
    n_samples=1000, n_classes=3, n_features=2, cluster_std=1.0
):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return (
        jnp.array(X_train),
        jnp.array(y_train),
        jnp.array(X_test),
        jnp.array(y_test),
    )


# Step 2: Define the Bayesian neural network for multiclass classification
def bnn_model(X, y=None, hidden_dim=16, n_classes=3):
    # Priors for weights and biases
    prec = sample("precision", Gamma(1.0, 1.0))
    n_features = X.shape[1]  # Number of input features

    # Hidden layer
    w1 = sample("w1", Normal(0, 1 / jnp.sqrt(prec)).expand([n_features, hidden_dim]))
    b1 = sample("b1", Normal(0, 1 / jnp.sqrt(prec)).expand([hidden_dim]))

    # Output layer
    w2 = sample("w2", Normal(0, 1 / jnp.sqrt(prec)).expand([hidden_dim, n_classes]))
    b2 = sample("b2", Normal(0, 1 / jnp.sqrt(prec)).expand([n_classes]))

    # Forward pass through the network
    hidden = jax.nn.relu(jnp.dot(X, w1) + b1)  # Hidden layer activation
    logits = deterministic("logits", jnp.dot(hidden, w2) + b2)  # Logits for Categorical

    # Likelihood for multiclass classification
    numpyro.sample("y", Categorical(logits=logits), obs=y)


# Step 3: Train the model using SteinVI
def train_bnn(X_train, y_train, hidden_dim=16, n_classes=3, num_steps=1000):
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
        n_classes=n_classes,
        progress_bar=True,
    )
    return stein, stein_result


# Step 4: Evaluate the model
def evaluate_bnn(stein, stein_result, X_test, hidden_dim=16, n_classes=3):
    predictive = MixtureGuidePredictive(
        bnn_model,
        stein.guide,
        params=stein.get_params(stein_result.state),
        num_samples=100,
        guide_sites=stein.guide_sites,
    )

    # Obtain predictions
    rng_key = random.PRNGKey(1)
    pred_samples = predictive(
        rng_key, X_test, hidden_dim=hidden_dim, n_classes=n_classes
    )["y"]
    mean_pred = pred_samples.mean(axis=0)
    return mean_pred


# Step 5: Visualize decision boundary
def plot_decision_boundary_multiclass(
    X, y, X_test, y_test, y_pred, n_classes=3, title="BNN Multiclass Decision Boundary"
):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = jnp.c_[xx.ravel(), yy.ravel()]

    # Predict on grid
    pred_grid = evaluate_bnn(*y_pred, grid, n_classes=n_classes).reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(
        xx,
        yy,
        pred_grid,
        levels=np.arange(-0.5, n_classes + 0.5, 1),
        cmap="viridis",
        alpha=0.8,
    )
    scatter = plt.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, edgecolors="k", cmap="viridis", s=40
    )
    plt.colorbar(scatter, ticks=np.arange(0, n_classes))
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# Main script
if __name__ == "__main__":
    # Generate synthetic data
    X_train, y_train, X_test, y_test = generate_multiclass_data()

    # Train the model
    print("Training Bayesian Neural Network for Multiclass Classification...")
    stein, stein_result = train_bnn(X_train, y_train, n_classes=3)

    # Evaluate the model
    mean_pred = evaluate_bnn(stein, stein_result, X_test, n_classes=3)

    # Plot decision boundary
    plot_decision_boundary_multiclass(
        X_train, y_train, X_test, y_test, (stein, stein_result), n_classes=3
    )
