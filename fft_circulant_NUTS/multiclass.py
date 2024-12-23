import numpy as np
import jax.numpy as jnp
from jax import random, vmap
import jax
import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


# Generate synthetic multiclass classification data
def generate_multiclass_data(n_samples=500, n_classes=3, noise=0.2):
    """
    Generate synthetic multiclass classification data.
    """
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = np.random.randint(0, n_classes, size=n_samples)

    # Add nonlinear decision boundaries
    y[X[:, 0] ** 2 + X[:, 1] ** 2 < 1.5] = 0
    y[(X[:, 0] ** 2 + X[:, 1] ** 2 >= 1.5) & (X[:, 1] > 0)] = 1
    y[(X[:, 0] ** 2 + X[:, 1] ** 2 >= 1.5) & (X[:, 1] <= 0)] = 2

    # Add noise
    X += noise * np.random.randn(*X.shape)
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


# Define the Bayesian Neural Network for multiclass classification
def bnn_circulant_multiclass(X, y=None, n_classes=3):
    """
    Bayesian Neural Network with Circulant Matrix Layer for Multiclass Classification.
    """
    input_size = X.shape[1]

    # Priors for circulant matrix and biases
    first_row = numpyro.sample("first_row", dist.Normal(0, 1).expand([input_size]))
    bias_circulant = numpyro.sample("bias_circulant", dist.Normal(0, 1))

    # Circulant layer
    hidden = circulant_matrix_multiply(first_row, X) + bias_circulant
    hidden = jax.nn.relu(hidden)

    # Output layer (logits for each class)
    weights_out = numpyro.sample(
        "weights_out", dist.Normal(0, 1).expand([hidden.shape[1], n_classes])
    )
    bias_out = numpyro.sample("bias_out", dist.Normal(0, 1).expand([n_classes]))

    logits = jnp.matmul(hidden, weights_out) + bias_out

    # Likelihood (Categorical for multiclass classification)
    numpyro.sample("obs", dist.Categorical(logits=logits), obs=y)


# Run MCMC inference using NUTS
def run_inference_multiclass(
    model, rng_key, X, y, num_samples=1000, num_warmup=500, n_classes=3
):
    """
    Run MCMC using NUTS for multiclass classification.
    """
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, X=X, y=y, n_classes=n_classes)
    mcmc.print_summary()
    return mcmc


# Evaluate the BNN for multiclass classification
def evaluate_bnn_multiclass(mcmc, X, n_classes=3):
    """
    Evaluate the Bayesian Neural Network to predict class probabilities.
    """
    posterior_samples = mcmc.get_samples()
    rng_keys = random.split(random.PRNGKey(1), posterior_samples["bias_out"].shape[0])

    def single_prediction(sample, rng_key):
        # Substitute samples into the model for a single forward pass
        model = numpyro.handlers.substitute(
            numpyro.handlers.seed(bnn_circulant_multiclass, rng_key), sample
        )
        trace = numpyro.handlers.trace(model).get_trace(
            X=X, y=None, n_classes=n_classes
        )
        logits = trace["obs"]["fn"].logits
        return jax.nn.softmax(logits, axis=1)  # Convert logits to probabilities

    predictions = []
    for i, rng_key in enumerate(rng_keys):
        single_sample = {
            k: v[i] for k, v in posterior_samples.items()
        }  # Extract single sample
        predictions.append(single_prediction(single_sample, rng_key))

    predictions = jnp.array(predictions)
    return predictions.mean(axis=0)  # Return mean probabilities


# Compute accuracy
def compute_accuracy_multiclass(mcmc, X, y_true, n_classes=3):
    """
    Compute accuracy for the Bayesian Neural Network.
    """
    mean_probs = evaluate_bnn_multiclass(mcmc, X, n_classes)
    y_pred = jnp.argmax(mean_probs, axis=1)  # Class with highest probability
    return accuracy_score(y_true, y_pred)


# Plot decision boundary for multiclass classification
def plot_decision_boundary_multiclass(
    mcmc, X_train, y_train, X_test, y_test, n_classes=3
):
    """
    Plot decision boundary for the Bayesian Neural Network (Multiclass).
    """
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = jnp.c_[xx.ravel(), yy.ravel()]

    # Predict probabilities for the grid
    mean_probs = evaluate_bnn_multiclass(mcmc, grid, n_classes=n_classes)
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
        X_test[:, 0], X_test[:, 1], c=y_test, cmap="viridis", marker="x", label="Test"
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


if __name__ == "__main__":
    # Generate synthetic data
    X, y = generate_multiclass_data(n_samples=500, n_classes=3, noise=0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # Run Bayesian inference
    rng_key = random.PRNGKey(0)
    mcmc = run_inference_multiclass(
        bnn_circulant_multiclass, rng_key, X_train, y_train, n_classes=3
    )

    # Compute accuracy
    train_accuracy = compute_accuracy_multiclass(mcmc, X_train, y_train, n_classes=3)
    test_accuracy = compute_accuracy_multiclass(mcmc, X_test, y_test, n_classes=3)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot decision boundary
    plot_decision_boundary_multiclass(
        mcmc, X_train, y_train, X_test, y_test, n_classes=3
    )
