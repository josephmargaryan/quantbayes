import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import jax
from numpyro import sample, deterministic
from numpyro.contrib.einstein import SteinVI, RBFKernel, MixtureGuidePredictive
from numpyro.distributions import Normal, Categorical, Gamma
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoNormal
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score


# Step 1: Generate Synthetic Data for Multiclass Classification
def generate_multiclass_data(n_samples=500, n_classes=3, noise=0.3):
    """
    Generate synthetic data for multiclass classification.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        cluster_std=noise,
        random_state=42,
    )
    return jnp.array(X), jnp.array(y)


# Step 2: FFT Circulant Matrix Multiplication
def circulant_matrix_multiply(first_row, X):
    """
    Perform circulant matrix multiplication using FFT.
    """
    n_features = first_row.shape[0]

    # Pad or truncate X to match the size of first_row
    if X.shape[1] < n_features:
        X = jnp.pad(X, ((0, 0), (0, n_features - X.shape[1])))
    elif X.shape[1] > n_features:
        X = X[:, :n_features]

    first_row_fft = jnp.fft.fft(first_row, axis=-1)
    X_fft = jnp.fft.fft(X, axis=-1)
    result_fft = first_row_fft[None, :] * X_fft
    result = jnp.fft.ifft(result_fft, axis=-1).real
    return result


# Step 3: Bayesian Neural Network Model for Multiclass Classification
def bnn_model_fft(x, y=None, hidden_dim=10, n_classes=3):
    """
    Bayesian Neural Network with FFT-based circulant layers for multiclass classification.
    """
    input_dim = x.shape[1]

    # Priors for input projection layer
    first_row_proj = sample("first_row_proj", Normal(0, 1).expand([hidden_dim]))
    bias_proj = sample("bias_proj", Normal(0, 1).expand([hidden_dim]))

    # Project input to hidden_dim
    x_proj = (
        circulant_matrix_multiply(first_row_proj, x) + bias_proj
    )  # Shape: (n_samples, hidden_dim)

    # Priors for circulant layers
    first_row_1 = sample("first_row_1", Normal(0, 1).expand([hidden_dim]))
    bias_circulant_1 = sample("bias_circulant_1", Normal(0, 1).expand([hidden_dim]))

    first_row_2 = sample("first_row_2", Normal(0, 1).expand([hidden_dim]))
    bias_circulant_2 = sample("bias_circulant_2", Normal(0, 1).expand([n_classes]))

    # Circulant Layer 1
    hidden = circulant_matrix_multiply(first_row_1, x_proj) + bias_circulant_1
    hidden = jnp.maximum(hidden, 0)  # ReLU activation

    # Circulant Layer 2 (logits for each class)
    logits = (
        circulant_matrix_multiply(first_row_2, hidden)[:, :n_classes] + bias_circulant_2
    )

    # Explicitly define logits as deterministic
    deterministic("logits", logits)

    # Likelihood
    sample("y", Categorical(logits=logits), obs=y)


# Step 4: Train the Model Using SteinVI
def train_steinvi(x_train, y_train, hidden_dim=10, n_classes=3):
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
        rng_key,
        num_steps=2000,
        x=x_train,
        y=y_train,
        hidden_dim=hidden_dim,
        n_classes=n_classes,
    )
    return stein, stein_result


# Step 5: Make Predictions
def make_predictions(stein, stein_result, x_test, hidden_dim=10, n_classes=3):
    """
    Generate predictions using the trained SteinVI model.
    """
    predictive = MixtureGuidePredictive(
        bnn_model_fft,
        stein.guide,
        params=stein.get_params(stein_result.state),
        guide_sites=stein.guide_sites,
        num_samples=100,
    )
    rng_key = random.PRNGKey(1)
    pred_samples = predictive(
        rng_key, x=x_test, hidden_dim=hidden_dim, n_classes=n_classes
    )

    # Extract logits and compute class probabilities
    logits = pred_samples["logits"]
    class_probs = jax.nn.softmax(logits, axis=-1).mean(axis=0)  # Mean over samples
    class_preds = jnp.argmax(class_probs, axis=-1)  # Predicted classes

    return class_preds, class_probs


# Step 6: Compute Accuracy
def compute_accuracy(y_true, y_pred):
    """
    Compute accuracy for multiclass classification.
    """
    return accuracy_score(y_true, y_pred)


def plot_decision_boundary_multiclass(
    stein, stein_result, x_train, y_train, hidden_dim=10, n_classes=3
):
    """
    Visualize the decision boundary of the trained BNN for multiclass classification.
    """
    # Define grid for plotting
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = jnp.c_[xx.ravel(), yy.ravel()]

    # Predict probabilities for the grid
    class_preds, _ = make_predictions(
        stein, stein_result, grid, hidden_dim=hidden_dim, n_classes=n_classes
    )
    zz = class_preds.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(
        xx, yy, zz, levels=np.arange(n_classes + 1) - 0.5, cmap="viridis", alpha=0.8
    )

    # Add training points
    scatter = plt.scatter(
        x_train[:, 0],
        x_train[:, 1],
        c=y_train,
        edgecolor="k",
        cmap="viridis",
        s=50,
        label="Training Data",
    )
    plt.colorbar(scatter, ticks=np.arange(n_classes))
    plt.title("Multiclass Classification Decision Boundary with FFT BNN")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


# Main script
if __name__ == "__main__":
    # Generate synthetic data
    x_train, y_train = generate_multiclass_data(n_samples=500, n_classes=3)
    x_test, y_test = generate_multiclass_data(n_samples=200, n_classes=3)

    # Train the model
    stein, stein_result = train_steinvi(x_train, y_train, hidden_dim=10, n_classes=3)

    # Predict on training and test data
    train_preds, _ = make_predictions(
        stein, stein_result, x_train, hidden_dim=10, n_classes=3
    )
    test_preds, _ = make_predictions(
        stein, stein_result, x_test, hidden_dim=10, n_classes=3
    )

    # Compute accuracy
    train_accuracy = compute_accuracy(y_train, train_preds)
    test_accuracy = compute_accuracy(y_test, test_preds)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot decision boundary
    plot_decision_boundary_multiclass(
        stein, stein_result, x_train, y_train, hidden_dim=10, n_classes=3
    )
