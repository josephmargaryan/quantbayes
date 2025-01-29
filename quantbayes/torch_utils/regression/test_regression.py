# regression_script_torch.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Optional


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
        return self.net(x)  # shape: (batch_size, 1)


# ----------------------------------------------------------------
# 2. Training Function
# ----------------------------------------------------------------
def train_model(
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


# ----------------------------------------------------------------
# 3. Evaluation with MC Sampling
# ----------------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_samples: int = 50,
    feature: Optional[int] = 0,
) -> dict:
    """
    Perform Monte Carlo sampling (by forcing dropout layers to 'train' mode if present).
    Plots predictions vs. one specified feature (x-axis) with uncertainty.

    Returns a dictionary with mean_preds, std_preds, mse, etc.
    """
    # Convert to torch
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).squeeze(-1)  # shape: (N,)

    # We'll store all the predictions
    preds_list = []

    # If the model has dropout layers, we want them "on" for MC sampling
    # so let's force "model.train()" for multiple forward passes.
    for _ in range(num_samples):
        model.train()
        with torch.no_grad():
            preds = model(X_val_t).squeeze(-1)  # shape: (N,)
        preds_list.append(preds.detach().cpu().numpy())

    # (num_samples, N)
    preds_stacked = np.stack(preds_list, axis=0)
    mean_preds = np.mean(preds_stacked, axis=0)
    std_preds = np.std(preds_stacked, axis=0)

    # MSE w.r.t. the *mean* predictions
    mse_val = np.mean((mean_preds - y_val_t.cpu().numpy()) ** 2)

    # Visualization: Feature vs. target
    X_feature = X_val[:, feature]  # (N,)
    idx_sorted = np.argsort(X_feature)
    sorted_x = X_feature[idx_sorted]
    sorted_mean = mean_preds[idx_sorted]
    sorted_std = std_preds[idx_sorted]
    sorted_y = y_val_t.numpy()[idx_sorted]

    plt.figure(figsize=(8, 5))
    plt.scatter(sorted_x, sorted_y, alpha=0.6, label="True", color="black")
    plt.plot(sorted_x, sorted_mean, label="Mean Prediction", color="blue")
    plt.fill_between(
        sorted_x,
        sorted_mean - sorted_std,
        sorted_mean + sorted_std,
        alpha=0.2,
        color="blue",
        label="Mean ± 1 std",
    )
    plt.xlabel(f"Feature {feature}")
    plt.ylabel("Target")
    plt.title("Regression Predictions with Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        "mean_predictions": mean_preds,
        "std_predictions": std_preds,
        "mse": mse_val,
        "all_samples": preds_stacked,
    }


# ----------------------------------------------------------------
# 4. Example Test
# ----------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic Data: y = 3x + noise
    np.random.seed(42)
    N = 200
    X_all = np.random.uniform(-5, 5, (N, 1)).astype(np.float32)
    y_all = (3.0 * X_all + np.random.normal(0, 2, (N, 1))).astype(np.float32)

    # Split
    train_size = int(0.8 * N)
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    y_train, y_val = y_all[:train_size], y_all[train_size:]

    # Define model
    model = MLPRegression(input_dim=1, hidden_dim=32)

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

    # Evaluate
    results = evaluate_model(model, X_val, y_val, num_samples=50, feature=0)
    print("Final Regression MSE:", results["mse"])
