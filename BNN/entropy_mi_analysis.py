import numpy as np
import matplotlib.pyplot as plt


class EntropyAndMutualInformation:
    def __init__(self, task_type="binary"):
        """
        Initialize the class for computing and visualizing entropy and mutual information.

        Args:
            task_type: Type of the task - "binary", "multiclass", or "regression".
        """
        self.task_type = task_type

    @staticmethod
    def compute_entropy_multiclass(probs):
        """Compute entropy for a multiclass probability distribution."""
        return -np.sum(probs * np.log(probs + 1e-10), axis=-1)

    @staticmethod
    def compute_entropy_binary(probs):
        """Compute entropy for binary classification."""
        return -probs * np.log(probs + 1e-10) - (1 - probs) * np.log(1 - probs + 1e-10)

    @staticmethod
    def compute_entropy_regression(samples):
        """
        Compute predictive entropy for regression tasks.

        Args:
            samples: Array of shape (num_samples, num_test_points) containing predictive samples.

        Returns:
            Predictive entropy for each test point.
        """
        variance_predictions = samples.var(axis=0)
        return 0.5 * np.log(2 * np.pi * np.e * variance_predictions + 1e-10)

    def compute_mutual_information(self, probs_or_samples):
        """
        Compute mutual information for the specified task type.

        Args:
            probs_or_samples: Array of probabilities (classification) or samples (regression).

        Returns:
            MI: Mutual information for each test point.
            Predictive entropy for each test point.
        """
        if self.task_type == "binary":
            mean_probs = probs_or_samples.mean(axis=0)
            predictive_entropy = self.compute_entropy_binary(mean_probs)
            sample_entropies = self.compute_entropy_binary(probs_or_samples)
            expected_entropy = sample_entropies.mean(axis=0)
        elif self.task_type == "multiclass":
            mean_probs = probs_or_samples.mean(axis=0)
            predictive_entropy = self.compute_entropy_multiclass(mean_probs)
            sample_entropies = self.compute_entropy_multiclass(probs_or_samples)
            expected_entropy = sample_entropies.mean(axis=0)
        elif self.task_type == "regression":
            mean_predictions = probs_or_samples.mean(axis=0)
            variance_predictions = probs_or_samples.var(axis=0)
            predictive_entropy = 0.5 * np.log(
                2 * np.pi * np.e * variance_predictions + 1e-10
            )
            expected_entropy = 0.5 * np.log(
                2 * np.pi * np.e * probs_or_samples.var(axis=0) + 1e-10
            )
        else:
            raise ValueError(
                "Unsupported task type. Choose from 'binary', 'multiclass', or 'regression'."
            )

        MI = predictive_entropy - expected_entropy
        return MI, predictive_entropy

    def visualize(self, MI, predictive_entropy):
        """
        Visualize Mutual Information (MI) and Predictive Entropy.

        Args:
            MI: Array of mutual information values for test samples.
            predictive_entropy: Array of predictive entropy values for test samples.
        """
        num_samples = len(MI)
        x_axis = np.arange(num_samples)

        plt.figure(figsize=(12, 6))

        plt.scatter(
            x_axis,
            MI,
            label="Mutual Information (MI)",
            alpha=0.7,
            marker="o",
            color="blue",
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
            f"Mutual Information and Predictive Entropy ({self.task_type.capitalize()} Task)",
            fontsize=16,
        )
        plt.xlabel("Test Sample Index", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.4)

        plt.show()
