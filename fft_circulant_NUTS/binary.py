import numpy as np
import jax.numpy as jnp
from jax import random, vmap
import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


# Generate synthetic binary classification data
def generate_binary_classification_data(n_samples=500, noise=0.1):
    """
    Generate synthetic binary classification data.
    """
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)  # Only 2 features for visualization
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 + noise * np.random.randn(n_samples)) > 1.5
    y = y.astype(int)  # Convert to 0/1
    return jnp.array(X), jnp.array(y)


# Perform circulant matrix multiplication using FFT
def circulant_matrix_multiply(first_row, X):
    """
    Perform circulant matrix multiplication using FFT.
    """
    first_row_fft = jnp.fft.fft(first_row, axis=-1)
    X_fft = jnp.fft.fft(X, axis=-1)
    result_fft = first_row_fft[None, :] * X_fft
    result = jnp.fft.ifft(result_fft, axis=-1).real
    return result


# Define the Bayesian Neural Network for binary classification
def bnn_circulant(X, y=None):
    """
    Bayesian Neural Network with Circulant Matrix Layer for Binary Classification.
    """
    input_size = X.shape[1]

    # Priors for circulant matrix first row and biases
    first_row = numpyro.sample("first_row", dist.Normal(0, 1).expand([input_size]))
    bias_circulant = numpyro.sample("bias_circulant", dist.Normal(0, 1))

    # Circulant layer
    hidden = circulant_matrix_multiply(first_row, X) + bias_circulant
    hidden = jax.nn.relu(hidden)

    # Output layer
    weights_out = numpyro.sample(
        "weights_out", dist.Normal(0, 1).expand([hidden.shape[1], 1])
    )
    bias_out = numpyro.sample("bias_out", dist.Normal(0, 1))

    logits = jnp.matmul(hidden, weights_out).squeeze() + bias_out

    # Likelihood (Bernoulli for binary classification)
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)


# Run MCMC inference using NUTS
def run_inference(model, rng_key, X, y, num_samples=1000, num_warmup=500):
    """
    Run MCMC using NUTS.
    """
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, X=X, y=y)
    mcmc.print_summary()
    return mcmc


# Evaluate the BNN on input data
def evaluate_bnn(mcmc, X):
    """
    Evaluate the Bayesian Neural Network to predict probabilities.
    """
    posterior_samples = mcmc.get_samples()
    rng_keys = random.split(random.PRNGKey(1), posterior_samples["bias_out"].shape[0])

    def single_prediction(sample, rng_key):
        # Substitute samples into the model for a single forward pass
        model = numpyro.handlers.substitute(
            numpyro.handlers.seed(bnn_circulant, rng_key), sample
        )
        trace = numpyro.handlers.trace(model).get_trace(X=X, y=None)
        logits = trace["obs"]["fn"].logits
        return jax.nn.sigmoid(logits)  # Convert logits to probabilities

    predictions = []
    for i, rng_key in enumerate(rng_keys):
        single_sample = {
            k: v[i] for k, v in posterior_samples.items()
        }  # Extract single sample
        predictions.append(single_prediction(single_sample, rng_key))

    predictions = jnp.array(predictions)
    return predictions.mean(axis=0)  # Return the mean probability


# Plot decision boundary
def plot_decision_boundary(mcmc, X_train, y_train, X_test, y_test):
    """
    Plot decision boundary for the Bayesian Neural Network.
    """
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = jnp.c_[xx.ravel(), yy.ravel()]

    # Predict probabilities for the grid
    mean_probs = evaluate_bnn(mcmc, grid)
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


def compute_accuracy(mcmc, X, y_true, threshold=0.5):
    """
    Compute accuracy for the Bayesian Neural Network.
    """
    # Predict probabilities
    mean_probs = evaluate_bnn(mcmc, X)

    # Convert probabilities to binary predictions using the threshold
    y_pred = (mean_probs >= threshold).astype(int)

    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


if __name__ == "__main__":
    # Generate synthetic data
    X, y = generate_binary_classification_data(n_samples=500, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # Run Bayesian inference
    rng_key = random.PRNGKey(0)
    mcmc = run_inference(
        bnn_circulant, rng_key, X_train, y_train, num_samples=1000, num_warmup=500
    )

    # Compute accuracy
    train_accuracy = compute_accuracy(mcmc, X_train, y_train)
    test_accuracy = compute_accuracy(mcmc, X_test, y_test)

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot decision boundary
    plot_decision_boundary(mcmc, X_train, y_train, X_test, y_test)
