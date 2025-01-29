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
            loss = loss_fn(logits, batch_y)  # cross entropy
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

    # Plot
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
# 3. Evaluation with MC Sampling (Dropout) + Decision Boundary
# ----------------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    num_samples: int = 50,
    features: Optional[Tuple[int, int]] = (0, 1),
):
    """
    Perform MC sampling for multiclass classification.
    Plots 2D decision boundary + uncertainty measure (entropy).
    """
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # We'll store predicted probabilities from multiple forward passes
    def predict_logits(batch_x: torch.Tensor) -> torch.Tensor:
        # shape: (batch_size, num_classes)
        return model(batch_x)

    # Accumulate all passes
    model.train()  # ensure dropout is "on" if present
    logits_list = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = predict_logits(X_val_t)
            # convert to probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, num_classes)
            logits_list.append(probs.cpu().numpy())

    # (num_samples, N, num_classes)
    probs_stacked = np.stack(logits_list, axis=0)
    mean_probs = np.mean(probs_stacked, axis=0)  # (N, num_classes)
    pred_classes = np.argmax(mean_probs, axis=-1)  # (N,)

    # Accuracy
    accuracy = np.mean(pred_classes == y_val)

    # Decision boundary in 2D
    f1, f2 = features
    x_min, x_max = X_val[:, f1].min() - 0.5, X_val[:, f1].max() + 0.5
    y_min, y_max = X_val[:, f2].min() - 0.5, X_val[:, f2].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100),
    )
    grid_points = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

    grid_probs_list = []
    with torch.no_grad():
        for _ in range(num_samples):
            gp_t = torch.tensor(grid_points, dtype=torch.float32)
            logits_gp = predict_logits(gp_t)
            probs_gp = torch.softmax(logits_gp, dim=-1)
            grid_probs_list.append(probs_gp.cpu().numpy())

    # (num_samples, 10000, num_classes)
    grid_probs_stacked = np.stack(grid_probs_list, axis=0)
    grid_mean_probs = np.mean(grid_probs_stacked, axis=0)  # (10000, num_classes)
    grid_pred_classes = np.argmax(grid_mean_probs, axis=-1)  # (10000,)
    Z = grid_pred_classes.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.tab10)
    # Plot the validation points
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

    # Let's also visualize uncertainty as predictive entropy:
    # entropy = -sum(p * log(p)) of the mean distribution
    eps = 1e-9
    mean_entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=-1)  # (N,)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        X_val[:, f1],
        X_val[:, f2],
        c=mean_entropy,
        cmap="viridis",
        edgecolor="k",
    )
    plt.colorbar(sc, label="Predictive Entropy")
    plt.title("Uncertainty (Entropy) in Mean Prediction")
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
    # Make 3 clusters
    X_c0 = np.random.normal(loc=[-2, -2], scale=1.0, size=(N // 3, 2))
    X_c1 = np.random.normal(loc=[2, 2], scale=1.0, size=(N // 3, 2))
    X_c2 = np.random.normal(loc=[-2, 2], scale=1.0, size=(N // 3, 2))

    X_all = np.vstack([X_c0, X_c1, X_c2]).astype(np.float32)
    y_all = np.array([0] * (N // 3) + [1] * (N // 3) + [2] * (N // 3), dtype=np.int32)

    # Shuffle
    perm = np.random.permutation(N)
    X_all = X_all[perm]
    y_all = y_all[perm]

    # Split
    train_size = int(0.8 * N)
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    y_train, y_val = y_all[:train_size], y_all[train_size:]

    num_classes = 3
    model = MLPClassifier(input_dim=2, num_classes=num_classes, hidden_dim=32)

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
    results = evaluate_model(
        model,
        X_val,
        y_val,
        num_classes=num_classes,
        num_samples=50,
        features=(0, 1),
    )
    print("Final Accuracy:", results["accuracy"])
