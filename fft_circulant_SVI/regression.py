import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

from numpyro import sample, deterministic
from numpyro.distributions import Normal, Gamma
from numpyro.optim import Adam
from numpyro.infer import SVI, Trace_ELBO, autoguide
from numpyro.infer import Predictive
from sklearn.metrics import mean_squared_error


# Step 1: Generate Synthetic Data for Regression
def generate_data(n_samples=500, noise=0.2):
    X = np.random.uniform(-3, 3, size=(n_samples, 1))
    y = 0.5 * X[:, 0] + np.sin(X[:, 0]) + np.random.normal(0, noise, size=n_samples)
    return jnp.array(X), jnp.array(y)


# Step 2: FFT Circulant Matrix Multiplication
def circulant_matrix_multiply(first_row, X):
    """
    Perform circulant matrix multiplication using FFT.
    """
    first_row_fft = jnp.fft.fft(first_row, axis=-1)
    X_fft = jnp.fft.fft(X, axis=-1)
    result_fft = first_row_fft[None, :] * X_fft
    result = jnp.fft.ifft(result_fft, axis=-1).real
    return result


# Step 3: Define BNN with FFT-based Circulant Layers
def bnn_model(x, y=None, hidden_dim=10):
    input_dim = x.shape[1]

    # Priors for input projection layer
    first_row_proj = sample("first_row_proj", Normal(0, 1).expand([hidden_dim]))
    bias_proj = sample("bias_proj", Normal(0, 1).expand([hidden_dim]))

    # Project input to hidden_dim
    x_proj = circulant_matrix_multiply(first_row_proj, x) + bias_proj

    # Priors for circulant layers
    first_row_hidden = sample("first_row_hidden", Normal(0, 1).expand([hidden_dim]))
    bias_hidden = sample("bias_hidden", Normal(0, 1).expand([hidden_dim]))

    # Hidden layer
    hidden = circulant_matrix_multiply(first_row_hidden, x_proj) + bias_hidden
    hidden = jnp.maximum(hidden, 0)  # ReLU activation

    # Output layer for regression
    prec_obs = sample("prec_obs", Gamma(1.0, 1.0))
    w_out = sample("w_out", Normal(0, 1).expand([hidden_dim, 1]))
    b_out = sample("b_out", Normal(0, 1))
    mean = deterministic("mean", jnp.dot(hidden, w_out) + b_out)
    sample("y", Normal(mean.squeeze(), 1 / jnp.sqrt(prec_obs)), obs=y)


# Step 4: Train the Model Using SVI
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


# Step 5: Make Predictions
def make_predictions(svi, params, x_test, hidden_dim=10):
    """
    Generate predictions using the trained SVI model.
    """
    rng_key = random.PRNGKey(1)

    # Get posterior samples using the guide
    guide_predictive = Predictive(svi.guide, params=params, num_samples=100)
    posterior_samples = guide_predictive(rng_key)

    # Define the model for prediction
    def model_with_posterior_samples(x):
        return svi.model(x=x, hidden_dim=hidden_dim)

    # Use posterior samples to evaluate the model
    predictive = Predictive(
        model_with_posterior_samples, posterior_samples=posterior_samples
    )
    pred_samples = predictive(rng_key, x_test)

    # Extract mean predictions
    means = pred_samples["mean"]
    return means.mean(axis=0), means


# Step 6: Plot Results
def plot_results(x_train, y_train, x_test, y_test, preds, uncertainties):
    """
    Visualize the training data, predictions, and uncertainty.
    """
    lower = jnp.percentile(uncertainties, 5, axis=0).squeeze()  # 5th percentile
    upper = jnp.percentile(uncertainties, 95, axis=0).squeeze()  # 95th percentile
    preds = preds.squeeze()

    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color="blue", label="Training data")
    plt.scatter(x_test, y_test, color="green", label="Test data", alpha=0.7)
    plt.plot(x_test.squeeze(), preds, color="red", label="Predictions", linewidth=2)
    plt.fill_between(
        x_test.squeeze(), lower, upper, color="gray", alpha=0.3, label="90% CI"
    )
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("BNN Regression with FFT Circulant Matrices (SVI)")
    plt.legend()
    plt.show()


# Step 7: Main Script
if __name__ == "__main__":
    x_train, y_train = generate_data(n_samples=500, noise=0.2)
    x_test, y_test = generate_data(n_samples=200, noise=0.2)

    print("Training Bayesian Neural Network for Regression...")
    svi, params = train_bnn(x_train, y_train, hidden_dim=10, num_steps=1000)

    print("Making Predictions...")
    preds, uncertainties = make_predictions(svi, params, x_test, hidden_dim=10)

    mse = mean_squared_error(y_test, preds)
    print(f"Mean Squared Error: {mse:.4f}")

    plot_results(x_train, y_train, x_test, y_test, preds, uncertainties)
