import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpyro
from numpyro import sample, deterministic
from numpyro.contrib.einstein import SteinVI, RBFKernel, MixtureGuidePredictive
from numpyro.distributions import Normal, Gamma
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoNormal
from sklearn.metrics import mean_squared_error


# Step 1: Generate Synthetic Data
def generate_synthetic_data(n_samples=100):
    """
    Generate synthetic data for regression.
    """
    np.random.seed(42)
    x = np.linspace(-3, 3, n_samples)
    y = 0.5 * x + np.sin(x) + np.random.normal(0, 0.3, size=n_samples)
    return x, y


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


# Step 3: Bayesian Neural Network Model
def bnn_model_fft(x, y=None, hidden_dim=10):
    """
    Bayesian Neural Network with FFT-based circulant layers.
    """
    # Priors for circulant matrix parameters and biases
    first_row_1 = sample("first_row_1", Normal(0, 1).expand([hidden_dim]))
    bias_circulant_1 = sample("bias_circulant_1", Normal(0, 1).expand([hidden_dim]))

    first_row_2 = sample("first_row_2", Normal(0, 1).expand([hidden_dim]))
    bias_circulant_2 = sample("bias_circulant_2", Normal(0, 1))

    prec_obs = sample("prec_obs", Gamma(1.0, 0.1))

    # Circulant Layer 1
    hidden = circulant_matrix_multiply(first_row_1, x[:, None]) + bias_circulant_1
    hidden = jnp.maximum(hidden, 0)  # ReLU activation

    # Circulant Layer 2 with scalar output
    mean = (
        circulant_matrix_multiply(first_row_2, hidden).sum(axis=-1) + bias_circulant_2
    )

    # Likelihood
    sample("y", Normal(mean, 1 / jnp.sqrt(prec_obs)), obs=y)


# Step 4: Train the Model Using SteinVI
def train_steinvi(x_train, y_train, hidden_dim=10):
    """
    Train the Bayesian Neural Network using SteinVI.
    """
    rng_key = random.PRNGKey(0)

    # Define guide and optimizer
    guide = AutoNormal(bnn_model_fft)
    optimizer = Adam(step_size=0.01)

    # SteinVI configuration
    stein = SteinVI(
        model=bnn_model_fft,
        guide=guide,
        optim=optimizer,
        kernel_fn=RBFKernel(),
        num_stein_particles=10,
        num_elbo_particles=10,
    )

    # Run SteinVI optimization
    stein_result = stein.run(
        rng_key, num_steps=5000, x=x_train, y=y_train, hidden_dim=hidden_dim
    )
    return stein, stein_result


# Step 5: Make Predictions
def make_predictions(stein, stein_result, x_test, hidden_dim=10):
    """
    Generate predictions using the trained SteinVI model.
    """
    predictive = MixtureGuidePredictive(
        bnn_model_fft,
        guide=stein.guide,
        params=stein.get_params(stein_result.state),
        guide_sites=stein.guide_sites,
        num_samples=100,
    )

    predictions = predictive(random.PRNGKey(1), x=x_test, hidden_dim=hidden_dim)["y"]
    return predictions


# Step 6: Plot Results
def plot_results(x_train, y_train, x_test, predictions):
    """
    Visualize the training data, predictions, and uncertainty.
    """
    # Compute mean and confidence intervals
    mean_prediction = predictions.mean(axis=0)
    lower = jnp.percentile(predictions, 5, axis=0)
    upper = jnp.percentile(predictions, 95, axis=0)

    # Ensure shapes are correct
    x_test = x_test.squeeze()
    mean_prediction = mean_prediction.squeeze()
    lower = lower.squeeze()
    upper = upper.squeeze()

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(x_test, mean_prediction, "r-", label="Mean Prediction")
    plt.fill_between(x_test, lower, upper, color="gray", alpha=0.3, label="90% CI")
    plt.scatter(x_train, y_train, color="blue", alpha=0.5, label="Training Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Bayesian Neural Network Regression with SteinVI")
    plt.legend()
    plt.show()


# Compute and print MSE
def compute_mse(y_true, predictions):
    """
    Compute Mean Squared Error (MSE) for predictions.
    """
    mean_prediction = predictions.mean(axis=0)  # Use the mean of the predictions
    mse = mean_squared_error(y_true, mean_prediction)
    return mse


if __name__ == "__main__":
    # Generate synthetic data
    x_train, y_train = generate_synthetic_data(n_samples=100)
    x_test = jnp.linspace(-3, 3, 100)

    # Train the model
    stein, stein_result = train_steinvi(x_train, y_train, hidden_dim=10)

    # Make predictions
    predictions = make_predictions(stein, stein_result, x_test, hidden_dim=10)

    # Plot results
    plot_results(x_train, y_train, x_test, predictions)

    # Report MSE on training data
    train_predictions = make_predictions(stein, stein_result, x_train, hidden_dim=10)
    train_mse = compute_mse(y_train, train_predictions)
    print(f"Train MSE: {train_mse:.4f}")

    # Report MSE on test data
    test_predictions = make_predictions(stein, stein_result, x_test, hidden_dim=10)
    test_mse = compute_mse(y_train, test_predictions)
    print(f"Test MSE: {test_mse:.4f}")
