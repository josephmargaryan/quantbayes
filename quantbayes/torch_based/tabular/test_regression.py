# regression_script_torch.py

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------------------------------------------
# 1. Model Definition
#    Simple MLP: input_dim -> hidden -> hidden -> 1 output
# ----------------------------------------------------------------
class MLPRegression(nn.Module):
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
        return self.net(x).squeeze()  # shape: (batch_size, 1)


# ----------------------------------------------------------------
# 2. Training Function
# ----------------------------------------------------------------
def train_regressor(
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
    Train a regression MLP using MSE loss.
    """
    # Convert data to torch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    # Simple mini-batch generator
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
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Training loss for this epoch
        train_loss = float(np.mean(epoch_losses))

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = loss_fn(val_preds, y_val_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % max(1, (num_epochs // 10)) == 0:
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Plot train/val losses
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


def visualize_regressor(model: nn.Module, X: np.ndarray, y: np.ndarray):
    """
    Visualizes regression performance for a deterministic PyTorch model.

    For a single feature, it plots a scatter of true values and a line plot of predictions.
    For multiple features, it creates:
      (1) A scatter plot of predicted vs. true values (with a 45° reference line)
      (2) A residual plot.

    Parameters:
        model: A PyTorch nn.Module that returns predictions.
        X: Input features as a NumPy array of shape (n_samples,) or (n_samples, n_features).
        y: True target values as a NumPy array of shape (n_samples,).
    """
    model.eval()
    # Convert data to tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    with torch.no_grad():
        # Ensure predictions are squeezed to 1D if possible
        preds = model(X_t).squeeze().cpu().numpy()
    y_np = y_t.cpu().numpy()

    # Check if we have a single feature
    if (X.ndim == 1) or (X.ndim == 2 and X.shape[1] == 1):
        X_flat = X.flatten()
        sorted_idx = np.argsort(X_flat)
        plt.figure(figsize=(10, 6))
        plt.scatter(X_flat, y_np, color="black", label="True Values", alpha=0.7)
        plt.plot(
            X_flat[sorted_idx],
            preds[sorted_idx],
            color="blue",
            lw=2,
            label="Predictions",
        )
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.title("Regression Predictions (Single Feature)")
        plt.legend()
        plt.show()
    else:
        # Multi-feature: Scatter plot of predicted vs. true and a residual plot.
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        # Scatter plot: predicted vs. true values with 45° reference line.
        axs[0].scatter(preds, y_np, alpha=0.7)
        min_val = min(y_np.min(), preds.min())
        max_val = max(y_np.max(), preds.max())
        axs[0].plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
        axs[0].set_xlabel("Predicted")
        axs[0].set_ylabel("True")
        axs[0].set_title("Predicted vs. True")

        # Residual plot: errors (true - predicted) vs. predicted.
        residuals = y_np - preds
        axs[1].scatter(preds, residuals, alpha=0.7)
        axs[1].axhline(0, color="k", linestyle="--", lw=2)
        axs[1].set_xlabel("Predicted")
        axs[1].set_ylabel("Residuals")
        axs[1].set_title("Residual Plot")

        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------
# 4. Example Test
# ----------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from quantbayes.fake_data import generate_regression_data

    df = generate_regression_data(n_categorical=1, n_continuous=2)

    X, y = df.drop("target", axis=1), df["target"]
    X, y = torch.tensor(X.values, dtype=torch.float32), torch.tensor(
        y.values, dtype=torch.long
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X.clone(), y.clone(), test_size=0.2, random_state=24
    )

    # Define model
    model = MLPRegression(input_dim=X_train.shape[-1], hidden_dim=32)

    # Train
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
