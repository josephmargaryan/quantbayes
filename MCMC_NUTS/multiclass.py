import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import jax

import numpyro
from numpyro import sample, deterministic
from numpyro.infer import MCMC, NUTS
from numpyro.distributions import Normal, Categorical
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


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
    return jnp.array(X_train), jnp.array(y_train), jnp.array(X_test), jnp.array(y_test)


# Step 2: Define the Bayesian neural network for multiclass classification
def bnn_model(X, y=None, hidden_dim=16, n_classes=3):
    w1 = sample("w1", Normal(0, 1).expand([X.shape[1], hidden_dim]))
    b1 = sample("b1", Normal(0, 1).expand([hidden_dim]))
    w2 = sample("w2", Normal(0, 1).expand([hidden_dim, n_classes]))
    b2 = sample("b2", Normal(0, 1).expand([n_classes]))

    hidden = jax.nn.relu(jnp.dot(X, w1) + b1)
    logits = deterministic("logits", jnp.dot(hidden, w2) + b2)
    sample("y", Categorical(logits=logits), obs=y)


# Step 3: Train the BNN using NUTS
def train_bnn(
    X_train, y_train, hidden_dim=16, n_classes=3, num_samples=1000, num_warmup=500
):
    rng_key = random.PRNGKey(0)
    kernel = NUTS(bnn_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, X=X_train, y=y_train, hidden_dim=hidden_dim, n_classes=n_classes)
    return mcmc


# Step 4: Evaluate the model
def evaluate_bnn(mcmc, X_test, hidden_dim=16, n_classes=3):
    samples = mcmc.get_samples()
    logits_samples = []

    for i in range(samples["w1"].shape[0]):  # Iterate over posterior samples
        w1 = samples["w1"][i]
        b1 = samples["b1"][i]
        w2 = samples["w2"][i]
        b2 = samples["b2"][i]

        hidden = jax.nn.relu(jnp.dot(X_test, w1) + b1)
        logits = jnp.dot(hidden, w2) + b2
        logits_samples.append(logits)

    logits_samples = jnp.stack(logits_samples)  # Stack logits from posterior samples
    probs = jax.nn.softmax(logits_samples, axis=2)  # Convert logits to probabilities
    mean_probs = probs.mean(axis=0)  # Average probabilities across posterior samples

    return mean_probs


# Step 5: Visualize decision boundary
def plot_decision_boundary_multiclass(
    mcmc, X_train, y_train, X_test, y_test, hidden_dim=16, n_classes=3
):
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = jnp.c_[xx.ravel(), yy.ravel()]

    # Predict probabilities for the grid
    mean_probs = evaluate_bnn(mcmc, grid, hidden_dim=hidden_dim, n_classes=n_classes)
    zz = jnp.argmax(mean_probs, axis=1).reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(
        xx,
        yy,
        zz,
        alpha=0.8,
        cmap="viridis",
        levels=np.arange(-0.5, n_classes + 0.5, 1),
    )

    # Add training and test points
    scatter_train = plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        edgecolor="k",
        cmap="viridis",
        label="Train",
    )
    scatter_test = plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        edgecolor="k",
        cmap="viridis",
        marker="x",
        label="Test",
    )

    # Add colorbar
    cbar = plt.colorbar(scatter_train)
    cbar.set_label("Class")

    # Add labels and title
    plt.title("Multiclass Classification Decision Boundary (NUTS MCMC)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


# Main script
if __name__ == "__main__":
    # Generate synthetic data
    X_train, y_train, X_test, y_test = generate_multiclass_data()

    # Train the model
    print("Training Bayesian Neural Network for Multiclass Classification...")
    mcmc = train_bnn(X_train, y_train, n_classes=3)

    # Plot decision boundary
    plot_decision_boundary_multiclass(
        mcmc, X_train, y_train, X_test, y_test, n_classes=3
    )
