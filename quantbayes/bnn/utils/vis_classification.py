import jax
import jax.nn
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["visualize_uncertainty_multiclass", "visualize_uncertainty_binary"]


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
