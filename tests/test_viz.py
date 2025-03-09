import jax
import jax.nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def analyze_pre_activations(X_pre):
    """
    Computes and visualizes the empirical covariance matrix from pre-activations.

    Parameters:
      X_pre: jnp.ndarray of shape (N, in_features)

    Returns:
      cov_matrix: jnp.ndarray of shape (N, N)
    """
    # Compute the empirical covariance matrix (across the data points)
    # Each row is a data point's feature representation.
    X_centered = X_pre - X_pre.mean(axis=0)
    cov_matrix = (X_centered @ X_centered.T) / (X_pre.shape[1] - 1)

    # Visualize the covariance matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(jax.device_get(cov_matrix), cmap="viridis")
    plt.colorbar()
    plt.title("Empirical Covariance of Pre-Activations")
    plt.xlabel("Data Point Index")
    plt.ylabel("Data Point Index")
    plt.tight_layout()
    plt.show()

    return cov_matrix


def compute_binary_entropy(probs, eps=1e-8):
    """
    Compute the entropy for binary predictions.

    Parameters:
        probs (np.ndarray): Array of predicted probabilities (shape: [n_samples]).
        eps (float): Small constant to avoid log(0).

    Returns:
        np.ndarray: Entropy values for each sample.
    """
    return -probs * np.log(probs + eps) - (1 - probs) * np.log(1 - probs + eps)


def compute_multiclass_entropy(probs, eps=1e-8):
    """
    Compute the entropy for multiclass predictions.

    Parameters:
        probs (np.ndarray): Array of predicted probabilities (shape: [n_samples, n_classes]).
        eps (float): Small constant to avoid log(0).

    Returns:
        np.ndarray: Entropy values for each sample.
    """
    return -np.sum(probs * np.log(probs + eps), axis=1)


def visualize_uncertainty_binary(
    model, X, rng_key, posterior="logits", num_samples=None
):
    """
    Visualizes predictive uncertainty for binary classification.

    It draws predictive samples from the model, converts logits to probabilities,
    computes the entropy for each sample, and plots a histogram of these entropies.

    Parameters:
        model: The Bayesian model (instance of your base class).
        X (np.ndarray): Input data.
        rng_key: Random key for sampling predictions.
        posterior (str): The posterior to sample from.
        num_samples (int): Number of samples from the predictive distribution.
    """
    # Obtain predictive samples (assumed shape: [num_samples, n_data])
    preds = model.predict(X, rng_key, posterior=posterior, num_samples=num_samples)
    preds = np.array(preds)

    # Convert logits to probabilities (binary classification)
    probs = jax.nn.sigmoid(preds)
    # Average over samples for a single probability per data point
    mean_probs = np.mean(probs, axis=0)

    # Compute predictive uncertainty (entropy)
    uncertainty = compute_binary_entropy(mean_probs)

    # Visualize the uncertainty
    plt.figure(figsize=(8, 6))
    plt.hist(uncertainty, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Predictive Uncertainty (Entropy)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Predictive Uncertainty for Binary Classification")
    plt.show()


def visualize_uncertainty_multiclass(
    model, X, rng_key, posterior="logits", num_samples=None
):
    """
    Visualizes predictive uncertainty for multiclass classification.

    It draws predictive samples from the model, computes softmax to get probabilities,
    calculates the entropy for each sample, and plots a histogram of these entropies.

    Parameters:
        model: The Bayesian model (instance of your base class).
        X (np.ndarray): Input data.
        rng_key: Random key for sampling predictions.
        posterior (str): The posterior to sample from.
        num_samples (int): Number of samples from the predictive distribution.
    """
    # Obtain predictive samples (assumed shape: [num_samples, n_data, n_classes])
    preds = model.predict(X, rng_key, posterior=posterior, num_samples=num_samples)
    preds = np.array(preds)

    # Compute the mean logits across samples and convert to probabilities
    pred_mean_logits = np.mean(preds, axis=0)
    probs = jax.nn.softmax(pred_mean_logits, axis=-1)

    # Compute predictive uncertainty (entropy) for each sample
    uncertainty = compute_multiclass_entropy(probs)

    # Visualize the uncertainty
    plt.figure(figsize=(8, 6))
    plt.hist(uncertainty, bins=20, color="mediumseagreen", edgecolor="black")
    plt.xlabel("Predictive Uncertainty (Entropy)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Predictive Uncertainty for Multiclass Classification")
    plt.show()


# --------------------------------------------------------------------
# Test / Simulation
# --------------------------------------------------------------------
if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    from quantbayes import bnn
    from quantbayes.fake_data import (
        generate_multiclass_classification_data,
    )

    df = generate_multiclass_classification_data(n_categorical=2, n_continuous=2)
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)

    class Net(bnn.Module):
        def __init__(self):
            super().__init__(task_type="multiclass", method="nuts")

        def __call__(self, X, y=None):
            N, in_feature = X.shape
            X = bnn.Linear(in_features=in_feature, out_features=3, name="layer")(X)
            logits = X
            numpyro.deterministic("logits", logits)
            with numpyro.plate("data", N):
                numpyro.sample("likelihood", dist.Categorical(logits=logits), obs=y)

    key = jax.random.PRNGKey(42)
    model = Net()
    model.compile()
    model.fit(X, y, key)
    preds = model.predict(X, key, posterior="likelihood")
    visualize_uncertainty_multiclass(model, X, key)
    model.visualize(X, y)
