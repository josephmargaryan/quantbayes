import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

import numpyro
from numpyro import sample, deterministic
from numpyro.distributions import Normal, Bernoulli
from numpyro.optim import Adam
from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Step 1: Generate Synthetic Binary Classification Data
def generate_binary_data(n_samples=500, noise=0.2):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return jnp.array(X), jnp.array(y)


# Step 2: Define the BNN Model
def bnn_model(x, y=None, hidden_dim=10):
    input_dim = x.shape[1]

    # Hidden layer weights and biases
    w_hidden = sample("w_hidden", Normal(0, 1).expand([input_dim, hidden_dim]))
    b_hidden = sample("b_hidden", Normal(0, 1).expand([hidden_dim]))

    # Output layer weights and biases
    w_out = sample("w_out", Normal(0, 1).expand([hidden_dim, 1]))
    b_out = sample("b_out", Normal(0, 1).expand([1]))

    # Forward pass
    hidden = jax.nn.relu(jnp.dot(x, w_hidden) + b_hidden)  # Hidden layer
    logits = deterministic(
        "logits", jnp.dot(hidden, w_out).squeeze() + b_out
    )  # Output layer

    # Likelihood
    sample("y", Bernoulli(logits=logits), obs=y)


# Step 3: Train the Model Using SVI
def train_bnn(x_train, y_train, hidden_dim=10, num_steps=1000):
    guide = autoguide.AutoNormal(bnn_model)
    optimizer = Adam(0.01)

    svi = SVI(
        bnn_model,
        guide,
        optimizer,
        loss=Trace_ELBO(),
    )

    rng_key = random.PRNGKey(0)
    svi_state = svi.init(rng_key, x_train, y_train, hidden_dim=hidden_dim)

    # Train the model
    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state, x_train, y_train, hidden_dim=hidden_dim)
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    params = svi.get_params(svi_state)
    return svi, params


# Step 4: Make Predictions
def make_predictions(svi, params, x_test, hidden_dim=10):
    """
    Generate predictions using the trained SVI model.
    """
    predictive = Predictive(svi.model, guide=svi.guide, params=params, num_samples=100)
    rng_key = random.PRNGKey(1)
    pred_samples = predictive(rng_key, x=x_test, hidden_dim=hidden_dim)

    logits = pred_samples["logits"].mean(axis=0)
    probs = jax.nn.sigmoid(logits)
    preds = (probs > 0.5).astype(int)
    return preds, probs


# Step 5: Plot Decision Boundary
def plot_decision_boundary(x, y, x_test, y_test, svi, params, hidden_dim=10):
    """
    Visualize the decision boundary of the trained BNN.
    """
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = jnp.c_[xx.ravel(), yy.ravel()]

    preds, _ = make_predictions(svi, params, grid, hidden_dim=hidden_dim)
    zz = preds.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, zz, alpha=0.8, cmap="RdBu")
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, edgecolor="k", cmap="RdBu", s=40)
    plt.title("Binary Classification Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# Step 6: Main Script
if __name__ == "__main__":
    x, y = generate_binary_data(n_samples=500, noise=0.2)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    print("Training Bayesian Neural Network for Binary Classification...")
    svi, params = train_bnn(x_train, y_train, hidden_dim=10, num_steps=1000)

    print("Making Predictions...")
    preds, probs = make_predictions(svi, params, x_test, hidden_dim=10)

    accuracy = accuracy_score(y_test, preds)
    print(f"Accuracy: {accuracy:.4f}")

    plot_decision_boundary(x_train, y_train, x_test, y_test, svi, params, hidden_dim=10)
