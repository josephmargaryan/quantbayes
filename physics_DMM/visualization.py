import matplotlib.pyplot as plt
import torch
import numpy as np
import pyro.distributions as dist


def plot_losses(train_losses, val_losses):
    """
    Plot training and validation loss curves.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_predictions(dmm, data_loader, device, num_examples=1, num_samples=50):
    """
    Run inference on the validation loader and plot predictions vs. ground truth,
    including uncertainty intervals by drawing multiple samples.

    Args:
        dmm: trained DMM model
        data_loader: DataLoader for validation set
        device: torch device (cpu or cuda)
        num_examples: how many sequences to plot
        num_samples: how many posterior samples to draw for uncertainty estimation
    """
    dmm.eval()
    all_ground_truth = []
    all_means = []
    all_stds = []  # To store standard deviation across samples

    with torch.no_grad():
        for padded_sequences, masks, targets in data_loader:
            padded_sequences = padded_sequences.to(device)

            # Get q(z_t|x) parameters
            loc_seq, scale_seq = dmm.guide_rnn(padded_sequences)  # [B, T, z_dim]

            # We'll store samples for each batch
            batch_samples = []
            for _ in range(num_samples):
                # Sample latent states: z ~ q(z|x)
                z_samples = dist.Normal(loc_seq, scale_seq).sample()  # [B, T, z_dim]

                # Compute emission predictions
                preds = []
                for t in range(z_samples.size(1)):
                    z_t = z_samples[:, t, :]
                    x_loc, x_scale = dmm.emission(z_t)
                    # Instead of just storing x_loc, we can store samples or just the mean
                    # If we want to sample from emission as well:
                    #   emission_sample = dist.Normal(x_loc, x_scale).sample()
                    #   preds.append(emission_sample)
                    # For now let's just store the emission mean for simplicity
                    preds.append(x_loc)
                preds = torch.stack(preds, dim=1)  # [B, T, x_dim]

                batch_samples.append(preds)

            # batch_samples is a list of [B, T, x_dim] of length num_samples
            batch_samples = torch.stack(
                batch_samples, dim=0
            )  # [num_samples, B, T, x_dim]

            # Compute mean and std across samples
            mean_preds = batch_samples.mean(dim=0)  # [B, T, x_dim]
            std_preds = batch_samples.std(dim=0)  # [B, T, x_dim]

            all_means.append(mean_preds.cpu().numpy())
            all_stds.append(std_preds.cpu().numpy())
            all_ground_truth.append(padded_sequences.cpu().numpy())

    all_means = np.concatenate(all_means, axis=0)  # [N, T, x_dim]
    all_stds = np.concatenate(all_stds, axis=0)  # [N, T, x_dim]
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)

    # Plot a few examples
    import matplotlib.pyplot as plt

    for i in range(min(num_examples, all_means.shape[0])):
        time_steps = np.arange(all_means.shape[1])
        mean_series = all_means[i, :, 0]
        std_series = all_stds[i, :, 0]
        ground_truth = all_ground_truth[i, :, 0]

        plt.figure(figsize=(10, 5))
        plt.plot(time_steps, ground_truth, label="Ground Truth", linewidth=2)
        plt.plot(time_steps, mean_series, label="Prediction (mean)", linestyle="--")

        # Plot uncertainty as a shaded region
        plt.fill_between(
            time_steps,
            mean_series - 2 * std_series,
            mean_series + 2 * std_series,
            color="gray",
            alpha=0.3,
            label="Uncertainty (±2σ)",
        )

        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"Sequence {i+1}: Ground Truth vs Predictions with Uncertainty")
        plt.legend()
        plt.tight_layout()
        plt.show()


import matplotlib.pyplot as plt
import numpy as np


def visualize_full_predictions(dmm, data, device):
    """
    Visualize predictions and uncertainty for the entire dataset.

    Args:
        dmm: Trained DMM model.
        data: Entire dataset as a Pandas DataFrame.
        device: Torch device.
    """
    dmm.eval()
    ground_truth = data["Close"].values  # Target variable
    features = data.drop(columns=["Close"]).values  # Input features

    # Convert to Torch tensor
    features_tensor = (
        torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        # Get predictions from the guide
        loc, scale = dmm.guide_rnn(features_tensor)
        loc = loc.squeeze(0).cpu().numpy()  # Predicted means
        scale = scale.squeeze(0).cpu().numpy()  # Predicted standard deviations

    # Plot ground truth, predictions, and uncertainty bounds
    time_steps = np.arange(len(ground_truth))
    plt.figure(figsize=(15, 8))

    # Ground truth
    plt.plot(time_steps, ground_truth, label="Ground Truth", color="blue")

    # Predictions
    plt.plot(time_steps, loc[:, 0], label="Predictions (Mean)", color="orange")

    # Uncertainty bounds (±2 stddev)
    plt.fill_between(
        time_steps,
        loc[:, 0] - 2 * scale[:, 0],
        loc[:, 0] + 2 * scale[:, 0],
        color="orange",
        alpha=0.3,
        label="Uncertainty (±2 stddev)",
    )

    plt.xlabel("Time Step")
    plt.ylabel("Close Price")
    plt.title("Predictions, Ground Truth, and Uncertainty")
    plt.legend()
    plt.show()
