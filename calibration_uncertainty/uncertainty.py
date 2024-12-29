import numpy as np
import jax
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    """
    Plot an ROC curve for binary classification.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True binary labels (0 or 1).
    - y_scores: array-like of shape (n_samples,)
        Target scores, which can be probabilities or confidence values.
    - title: str
        Title for the plot (default: "ROC Curve").

    Returns:
    - None. Displays the plot.
    """
    # Calculate false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculate AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)  # Diagonal line
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()


def plot_calibration_curve(y_true, y_prob, num_bins=10, plot_type="binary"):
    """
    Plots calibration curves for binary or multiclass classification.

    Parameters:
    - y_true (array-like): True labels.
    - y_prob (array-like): Predicted probabilities.
    - num_bins (int): Number of bins for the calibration curve.
    - plot_type (str): "binary" or "multiclass".

    Returns:
    None (plots are shown).
    """
    if plot_type == "binary":
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=num_bins, strategy="uniform"
        )
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker="o", label="Calibration Curve")
        plt.plot([0, 1], [0, 1], "--", label="Perfect Calibration")
        plt.title("Calibration Curve for Binary Classification")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend()
        plt.grid()
        plt.show()

    elif plot_type == "multiclass":
        num_classes = y_prob.shape[1]
        plt.figure(figsize=(12, 6))

        # Plot calibration curve for each class
        for class_idx in range(num_classes):
            prob_true, prob_pred = calibration_curve(
                y_true == class_idx,
                y_prob[:, class_idx],
                n_bins=num_bins,
                strategy="uniform",
            )
            plt.plot(prob_pred, prob_true, marker="o", label=f"Class {class_idx}")

        plt.plot([0, 1], [0, 1], "--", label="Perfect Calibration")
        plt.title("Calibration Curve for Multiclass Classification (Per Class)")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot average calibration curve
        avg_prob_true = np.zeros(num_bins)
        avg_prob_pred = np.zeros(num_bins)
        for class_idx in range(num_classes):
            prob_true, prob_pred = calibration_curve(
                y_true == class_idx,
                y_prob[:, class_idx],
                n_bins=num_bins,
                strategy="uniform",
            )
            avg_prob_true += np.interp(
                np.linspace(0, 1, num_bins), prob_pred, prob_true, left=0, right=0
            )
            avg_prob_pred += np.linspace(0, 1, num_bins)

        avg_prob_true /= num_classes
        avg_prob_pred /= num_classes

        plt.figure(figsize=(8, 6))
        plt.plot(
            avg_prob_pred, avg_prob_true, marker="o", label="Average Calibration Curve"
        )
        plt.plot([0, 1], [0, 1], "--", label="Perfect Calibration")
        plt.title("Average Calibration Curve for Multiclass Classification")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend()
        plt.grid()
        plt.show()

    else:
        raise ValueError("Invalid plot_type. Choose either 'binary' or 'multiclass'.")


def expected_calibration_error(y_true, y_prob, num_bins=10):
    """
    Computes the Expected Calibration Error (ECE) for binary or multiclass classification.

    Parameters:
    - y_true (array-like): True labels.
    - y_prob (array-like): Predicted probabilities.
    - num_bins (int): Number of bins for calibration.

    Returns:
    - ece (float): Expected Calibration Error.
    """
    if len(y_prob.shape) == 1:  # Binary classification
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=num_bins, strategy="uniform"
        )
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges, right=True) - 1
        bin_counts = np.bincount(bin_indices, minlength=num_bins)
        valid_bins = bin_counts > 0  # Use only bins with samples
        ece = np.sum(
            np.abs(prob_true - prob_pred) * (bin_counts[valid_bins] / len(y_true))
        )
    else:  # Multiclass classification
        num_classes = y_prob.shape[1]
        ece_total = 0
        for class_idx in range(num_classes):
            prob_true, prob_pred = calibration_curve(
                y_true == class_idx,
                y_prob[:, class_idx],
                n_bins=num_bins,
                strategy="uniform",
            )
            bin_edges = np.linspace(0, 1, num_bins + 1)
            bin_indices = np.digitize(y_prob[:, class_idx], bin_edges, right=True) - 1
            bin_counts = np.bincount(bin_indices, minlength=num_bins)
            valid_bins = bin_counts > 0  # Use only bins with samples
            ece_total += np.sum(
                np.abs(prob_true - prob_pred) * (bin_counts[valid_bins] / len(y_true))
            )
        ece = ece_total / num_classes
    return ece


def compute_entropy_multiclass(probs):
    """Compute entropy for a multiclass probability distribution."""
    return -np.sum(probs * np.log(probs + 1e-10), axis=-1)


def compute_mutual_information_multiclass(probs):
    """
    Compute mutual information (MI) for multiclass classification.

    Args:
        probs: Array of shape (num_samples, num_test_points, num_classes) containing probabilities.

    Returns:
        MI: Mutual information for each test point.
        Predictive entropy for each test point.
    """
    mean_probs = probs.mean(axis=0)  # Mean probabilities (point estimate)
    predictive_entropy = compute_entropy_multiclass(mean_probs)  # Predictive entropy

    sample_entropies = compute_entropy_multiclass(probs)  # Per-sample entropies
    expected_entropy = np.mean(sample_entropies, axis=0)  # Mean entropy over samples

    MI = predictive_entropy - expected_entropy
    return MI, predictive_entropy


def compute_entropy_binary(probs):
    """Compute entropy for binary classification."""
    return -probs * np.log(probs + 1e-10) - (1 - probs) * np.log(1 - probs + 1e-10)


def compute_mutual_information_binary(probs):
    """
    Compute mutual information (MI) for binary classification.

    Args:
        logits: Array of shape (num_samples, num_test_points) containing logits.

    Returns:
        MI: Mutual information for each test point.
    """

    mean_probs = probs.mean(axis=0)  # Mean probabilities (point estimate)

    predictive_entropy = compute_entropy_binary(mean_probs)  # Predictive entropy
    sample_entropies = compute_entropy_binary(probs)  # Per-sample entropies
    expected_entropy = sample_entropies.mean(axis=0)  # Mean entropy over samples

    MI = predictive_entropy - expected_entropy
    return MI, predictive_entropy


def compute_mutual_information_regression(samples):
    """
    Compute mutual information for regression tasks.

    Args:
        samples: Array of shape (num_samples, num_test_points) containing predictive samples.

    Returns:
        MI: Mutual information for each test point.
        Predictive entropy for each test point.
    """
    mean_predictions = samples.mean(axis=0)  # Mean predictions
    variance_predictions = samples.var(axis=0)  # Variance of predictions

    predictive_entropy = 0.5 * np.log(
        2 * np.pi * np.e * variance_predictions + 1e-10
    )  # Predictive entropy

    # Expected entropy assumes per-sample variance
    sample_variances = samples.var(axis=0)
    expected_entropy = 0.5 * np.log(2 * np.pi * np.e * sample_variances + 1e-10)

    MI = predictive_entropy - expected_entropy
    return MI, predictive_entropy


def compute_entropy_regression(samples):
    """
    Compute predictive entropy for regression tasks.

    Args:
        samples: Array of shape (num_samples, num_test_points) containing predictive samples.

    Returns:
        Predictive entropy for each test point.
    """
    # Variance of the posterior predictive distribution
    variance_predictions = samples.var(axis=0)

    # Predictive entropy formula for Gaussian distributions
    predictive_entropy = 0.5 * np.log(2 * np.pi * np.e * variance_predictions + 1e-10)

    return predictive_entropy


def visualize_entropy_and_mi(MI, predictive_entropy, task_type="binary"):
    """
    Visualize Mutual Information (MI) and Predictive Entropy for a single model or case.

    Args:
        MI: Array of mutual information values for test samples.
        predictive_entropy: Array of predictive entropy values for test samples.
        task_type: Type of the task - "binary", "multiclass", or "regression".
    """
    num_samples = len(MI)
    x_axis = np.arange(num_samples)

    plt.figure(figsize=(12, 6))

    plt.scatter(
        x_axis, MI, label="Mutual Information (MI)", alpha=0.7, marker="o", color="blue"
    )
    plt.scatter(
        x_axis,
        predictive_entropy,
        label="Predictive Entropy",
        alpha=0.7,
        marker="x",
        color="orange",
    )

    plt.title(
        f"Mutual Information and Predictive Entropy ({task_type.capitalize()} Task)",
        fontsize=16,
    )
    plt.xlabel("Test Sample Index", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.4)

    plt.show()


def visualize_entropy_and_mi_with_average(
    MI, predictive_entropy, avg_entropy, task_type="multiclass"
):
    """
    Visualize Mutual Information (MI), Predictive Entropy, and Average Entropy for a single model or case.

    Args:
        MI: Array of mutual information values for test samples.
        predictive_entropy: Array of predictive entropy values for test samples.
        avg_entropy: Array of average entropy values for test samples.
        task_type: Type of the task - "binary", "multiclass", or "regression".
    """
    num_samples = len(MI)
    x_axis = np.arange(num_samples)

    plt.figure(figsize=(12, 6))

    plt.scatter(
        x_axis, MI, label="Mutual Information (MI)", alpha=0.7, marker="o", color="blue"
    )
    plt.scatter(
        x_axis,
        predictive_entropy,
        label="Predictive Entropy",
        alpha=0.7,
        marker="x",
        color="orange",
    )
    plt.scatter(
        x_axis,
        avg_entropy,
        label="Average Entropy (Aleatoric)",
        alpha=0.7,
        marker="s",
        color="green",
    )

    plt.title(f"Uncertainty Analysis ({task_type.capitalize()} Task)", fontsize=16)
    plt.xlabel("Test Sample Index", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.4)

    plt.show()
