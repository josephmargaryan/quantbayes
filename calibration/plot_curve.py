import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


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


# Example usage:
"""y_true_binary = np.array([0, 1, 0, 1, 1])
y_prob_binary = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
plot_calibration_curve(y_true_binary, y_prob_binary, plot_type="binary")
ece_binary = expected_calibration_error(y_true_binary, y_prob_binary)
print(f"ECE (Binary): {ece_binary}")"""


"""y_true_multiclass = np.array([0, 1, 2, 1, 0])
y_prob_multiclass = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.1, 0.7],
    [0.1, 0.6, 0.3],
    [0.8, 0.1, 0.1]
])
plot_calibration_curve(y_true_multiclass, y_prob_multiclass, plot_type="multiclass")
ece_multiclass = expected_calibration_error(y_true_multiclass, y_prob_multiclass)
print(f"ECE (Multiclass): {ece_multiclass}")"""
