import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

import numpyro
from numpyro import sample, deterministic
from numpyro.distributions import Normal
from numpyro.optim import Adam
from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Step 1: Generate Synthetic Regression Data
def generate_regression_data(n_samples=500, noise=0.1):
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = np.sin(X) + noise * np.random.randn(n_samples, 1)
    return jnp.array(X), jnp.array(y).squeeze()


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
    mean = deterministic(
        "mean", jnp.dot(hidden, w_out).squeeze() + b_out
    )  # Output layer

    # Likelihood
    sample("y", Normal(mean, 0.1), obs=y)


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

    means = pred_samples["mean"]
    preds = means.mean(axis=0)
    uncertainties = means.std(axis=0)
    return preds, uncertainties


# Step 5: Plot Results
def plot_results(x_train, y_train, x_test, y_test, preds, uncertainties):
    """
    Plot the true data, predictions, and uncertainty intervals.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color="blue", label="Train Data", alpha=0.5)
    plt.scatter(x_test, y_test, color="red", label="Test Data", alpha=0.5)
    plt.plot(x_test, preds, color="black", label="Predictions", linewidth=2)
    plt.fill_between(
        x_test.squeeze(),
        preds - 2 * uncertainties,
        preds + 2 * uncertainties,
        color="gray",
        alpha=0.3,
        label="Uncertainty (±2 SD)",
    )
    plt.title("Bayesian Neural Network Regression with SVI")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()


# Step 6: Main Script
if __name__ == "__main__":
    x, y = generate_regression_data(n_samples=500, noise=0.1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    print("Training Bayesian Neural Network for Regression...")
    svi, params = train_bnn(x_train, y_train, hidden_dim=10, num_steps=1000)

    print("Making Predictions...")
    preds, uncertainties = make_predictions(svi, params, x_test, hidden_dim=10)

    mse = mean_squared_error(y_test, preds)
    print(f"Mean Squared Error: {mse:.4f}")

    plot_results(x_train, y_train, x_test, y_test, preds, uncertainties)
