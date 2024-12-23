import matplotlib.pyplot as plt
import numpy as np


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_predictions_with_uncertainty(ground_truth, means, stds, num_examples=3):
    # ground_truth, means, stds: [N, T, x_dim]
    # Plot a few examples
    for i in range(min(num_examples, means.shape[0])):
        time_steps = np.arange(means.shape[1])
        mean_series = means[i, :, 0]
        std_series = stds[i, :, 0]
        gt = ground_truth[i, :, 0]

        plt.figure(figsize=(10, 5))
        plt.plot(time_steps, gt, label="Ground Truth", linewidth=2)
        plt.plot(time_steps, mean_series, label="Prediction (mean)", linestyle="--")
        plt.fill_between(
            time_steps,
            mean_series - 2 * std_series,
            mean_series + 2 * std_series,
            color="gray",
            alpha=0.3,
            label="±2 Std Dev",
        )
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"Sequence {i+1}: Predictions with Uncertainty")
        plt.legend()
        plt.tight_layout()
        plt.show()
