# binary_script_torch.py

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, roc_curve


# ----------------------------------------------------------------
# 1. Model Definition
#    MLP for Binary Classification:
#    Output dimension = 1 (logits for BCEWithLogitsLoss)
# ----------------------------------------------------------------
class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits, shape: (batch_size, 1)
        return self.net(x)


# ----------------------------------------------------------------
# 2. Training Function
# ----------------------------------------------------------------
def train_binary(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> nn.Module:
    """
    Train a binary classification MLP using BCEWithLogitsLoss.
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []

    def batch_generator(X, y, bs):
        n = len(X)
        indices = torch.randperm(n)
        for start in range(0, n, bs):
            end = start + bs
            batch_idx = indices[start:end]
            yield X[batch_idx], y[batch_idx]

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []

        for batch_x, batch_y in batch_generator(X_train_t, y_train_t, batch_size):
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)  # (batch_size,)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))

        # Validation: run in evaluation mode
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).squeeze(-1)
            val_loss = loss_fn(val_logits, y_val_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % max(1, (num_epochs // 10)) == 0:
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Plot training and validation loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


def visualize_binary(model: nn.Module, X: np.ndarray, y: np.ndarray):
    """
    Visualizes performance for a deterministic PyTorch binary classification model.

    This function obtains raw logits, applies sigmoid to get probabilities,
    and then produces:
      - A histogram of predicted probabilities.
      - An ROC curve with AUC.
      - A calibration plot.

    Parameters:
        model: A PyTorch nn.Module that returns logits.
        X: Input features as a NumPy array of shape (n_samples, n_features).
        y: True binary labels as a NumPy array of shape (n_samples,) (0 or 1).
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_t).squeeze()  # Ensure shape is (n_samples,)
        probs = torch.sigmoid(logits)

    probs_np = probs.cpu().numpy()
    y_np = y_t.cpu().numpy()

    # ROC curve and AUC using sklearn
    fpr, tpr, _ = roc_curve(y_np, probs_np)
    roc_auc = auc(fpr, tpr)

    # Calibration curve using sklearn
    prob_true, prob_pred = calibration_curve(y_np, probs_np, n_bins=10)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram of predicted probabilities
    axs[0].hist(probs_np, bins=20, color="skyblue", edgecolor="black", alpha=0.8)
    axs[0].set_xlabel("Predicted Probability")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Predicted Probability Histogram")

    # ROC Curve
    axs[1].plot(fpr, tpr, color="darkred", lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
    axs[1].plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 1.05])
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate")
    axs[1].set_title("ROC Curve")
    axs[1].legend(loc="lower right")

    # Calibration Plot
    axs[2].plot(
        prob_pred, prob_true, marker="o", linewidth=1, label="Calibration Curve"
    )
    axs[2].plot([0, 1], [0, 1], linestyle="--", label="Ideal Calibration")
    axs[2].set_xlabel("Mean Predicted Probability")
    axs[2].set_ylabel("Fraction of Positives")
    axs[2].set_title("Calibration Plot")
    axs[2].legend()

    plt.suptitle("Binary Classification Performance")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ----------------------------------------------------------------
# 4. Example Test
# ----------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic binary data: two clusters
    np.random.seed(0)
    N = 200
    X0 = np.random.normal(loc=[-2, 0], scale=1.0, size=(N // 2, 2))
    X1 = np.random.normal(loc=[2, 0], scale=1.0, size=(N // 2, 2))
    X_all = np.vstack([X0, X1]).astype(np.float32)
    y_all = np.array([0] * (N // 2) + [1] * (N // 2), dtype=np.float32)

    # Shuffle data
    perm = np.random.permutation(N)
    X_all = X_all[perm]
    y_all = y_all[perm]

    # Split data
    train_size = int(0.8 * N)
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    y_train, y_val = y_all[:train_size], y_all[train_size:]

    # Initialize model
    model = MLPBinaryClassifier(input_dim=2, hidden_dim=32)

    # Train model
    model = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-3,
    )
