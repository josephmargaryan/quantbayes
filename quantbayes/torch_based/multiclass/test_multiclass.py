# multiclass_script_torch.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Optional, Tuple


# ----------------------------------------------------------------
# 1. Model Definition for Multiclass
#    output_dim = num_classes
# ----------------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits of shape (batch_size, num_classes)
        return self.net(x)


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
    Train a multiclass classification model using CrossEntropyLoss.
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)  # class labels as long
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

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
            logits = model(batch_x)  # (batch_size, num_classes)
            loss = loss_fn(logits, batch_y)  # cross entropy loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % max(1, (num_epochs // 10)) == 0:
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Plot training & validation loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


# ----------------------------------------------------------------
# 3. Deterministic Evaluation and Decision Boundary Visualization
# ----------------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    features: Optional[Tuple[int, int]] = (0, 1),
):
    """
    Perform deterministic evaluation for multiclass classification.
    Plots 2D decision boundary and an uncertainty measure (predictive entropy).
    """
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    # Set model to evaluation mode for deterministic predictions
    model.eval()
    with torch.no_grad():
        logits = model(X_val_t)
        probs = torch.softmax(logits, dim=-1)  # (N, num_classes)
        mean_probs = probs.cpu().numpy()

    pred_classes = np.argmax(mean_probs, axis=-1)
    accuracy = np.mean(pred_classes == y_val)

    # Create grid for decision boundary visualization (2D)
    f1, f2 = features
    x_min, x_max = X_val[:, f1].min() - 0.5, X_val[:, f1].max() + 0.5
    y_min, y_max = X_val[:, f2].min() - 0.5, X_val[:, f2].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100),
    )
    grid_points = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    gp_t = torch.tensor(grid_points, dtype=torch.float32)

    with torch.no_grad():
        logits_grid = model(gp_t)
        probs_grid = torch.softmax(logits_grid, dim=-1).cpu().numpy()

    grid_pred_classes = np.argmax(probs_grid, axis=-1)
    Z = grid_pred_classes.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.tab10)
    plt.scatter(
        X_val[:, f1],
        X_val[:, f2],
        c=y_val,
        cmap=plt.cm.tab10,
        edgecolor="k",
        alpha=0.7,
    )
    plt.colorbar()
    plt.title(f"Multiclass Decision Boundary (Accuracy={accuracy:.2f})")
    plt.xlabel(f"Feature {f1}")
    plt.ylabel(f"Feature {f2}")
    plt.show()

    # Uncertainty visualization via predictive entropy
    # entropy = -sum(p * log(p))
    eps = 1e-9
    mean_entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=-1)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        X_val[:, f1],
        X_val[:, f2],
        c=mean_entropy,
        cmap="viridis",
        edgecolor="k",
    )
    plt.colorbar(sc, label="Predictive Entropy")
    plt.title("Uncertainty (Entropy) in Predictions")
    plt.xlabel(f"Feature {f1}")
    plt.ylabel(f"Feature {f2}")
    plt.grid(True)
    plt.show()

    return {
        "predicted_classes": pred_classes,
        "accuracy": accuracy,
        "mean_probs": mean_probs,
        "entropy": mean_entropy,
    }


# ----------------------------------------------------------------
# 4. Example Test
# ----------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data for 3 classes in 2D
    np.random.seed(42)
    N = 300
    # Create 3 clusters
    X_c0 = np.random.normal(loc=[-2, -2], scale=1.0, size=(N // 3, 2))
    X_c1 = np.random.normal(loc=[2, 2], scale=1.0, size=(N // 3, 2))
    X_c2 = np.random.normal(loc=[-2, 2], scale=1.0, size=(N // 3, 2))

    X_all = np.vstack([X_c0, X_c1, X_c2]).astype(np.float32)
    y_all = np.array([0] * (N // 3) + [1] * (N // 3) + [2] * (N // 3), dtype=np.int32)

    # Shuffle the dataset
    perm = np.random.permutation(N)
    X_all = X_all[perm]
    y_all = y_all[perm]

    # Split into training and validation sets
    train_size = int(0.8 * N)
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    y_train, y_val = y_all[:train_size], y_all[train_size:]

    num_classes = 3
    model = MLPClassifier(input_dim=2, num_classes=num_classes, hidden_dim=32)

    # Train the model
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

    # Evaluate the model deterministically
    results = evaluate_model(
        model,
        X_val,
        y_val,
        num_classes=num_classes,
        features=(0, 1),
    )
    print("Final Accuracy:", results["accuracy"])
