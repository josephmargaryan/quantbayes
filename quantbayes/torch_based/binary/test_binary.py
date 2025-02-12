# binary_script_torch.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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


def visualize_binary_torch(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: tuple = (0, 1),
    unique_threshold: int = 10,
    resolution: int = 100,
    title: str = "Binary Decision Boundary",
):
    """
    Visualize a binary classification decision boundary with automatic checks for
    continuous vs. categorical features.

    Parameters
    ----------
    model : torch.nn.Module
        A trained binary classifier that outputs raw logits.
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Binary class labels (n_samples,).
    feature_indices : tuple of int, optional
        Indices of the two features to visualize.
    unique_threshold : int, optional
        Maximum number of unique values for a feature to be considered categorical.
    resolution : int, optional
        Number of points in the grid (for continuous features).
    title : str, optional
        Title for the plot.
    """
    # Ensure data are numpy arrays.
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    f1, f2 = feature_indices

    # Determine if features are categorical.
    unique_f1 = np.unique(X_np[:, f1])
    unique_f2 = np.unique(X_np[:, f2])
    is_f1_cat = len(unique_f1) < unique_threshold
    is_f2_cat = len(unique_f2) < unique_threshold

    # Put model in evaluation mode and get its device.
    model.eval()
    device = next(model.parameters()).device

    # Helper: given a numpy array of inputs, predict probabilities.
    def model_predict(X_input):
        with torch.no_grad():
            X_tensor = torch.tensor(X_input, dtype=torch.float32, device=device)
            logits = model(X_tensor).squeeze(-1)  # shape: (n_samples,)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    # --- Case 1: Both features are continuous ---
    if not is_f1_cat and not is_f2_cat:
        x_min, x_max = X_np[:, f1].min() - 0.5, X_np[:, f1].max() + 0.5
        y_min, y_max = X_np[:, f2].min() - 0.5, X_np[:, f2].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )
        n_features = X_np.shape[1]
        # Build grid: for the two visualized features use the mesh grid; for others use their mean.
        grid_list = []
        for i in range(n_features):
            if i == f1:
                grid_list.append(xx.ravel())
            elif i == f2:
                grid_list.append(yy.ravel())
            else:
                grid_list.append(np.full(xx.ravel().shape, X_np[:, i].mean()))
        grid_arr = np.stack(grid_list, axis=1)
        probs = model_predict(grid_arr)
        # Predict class (using 0.5 threshold)
        class_preds = (probs > 0.5).astype(np.int32).reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, class_preds, alpha=0.3, cmap=plt.cm.Paired)
        plt.scatter(X_np[:, f1], X_np[:, f2], c=y_np, edgecolor="k", cmap=plt.cm.Paired)
        plt.xlabel(f"Feature {f1}")
        plt.ylabel(f"Feature {f2}")
        plt.title(title)
        plt.grid(True)
        plt.show()

    # --- Case 2: One feature categorical, one continuous ---
    elif (is_f1_cat and not is_f2_cat) or (not is_f1_cat and is_f2_cat):
        # Let the categorical feature be the one with fewer unique values.
        if is_f1_cat:
            cat_idx, cont_idx = f1, f2
        else:
            cat_idx, cont_idx = f2, f1

        unique_cats = np.unique(X_np[:, cat_idx])
        num_cats = len(unique_cats)
        fig, axes = plt.subplots(1, num_cats, figsize=(5 * num_cats, 5), squeeze=False)
        for j, cat in enumerate(unique_cats):
            ax = axes[0, j]
            mask = X_np[:, cat_idx] == cat
            # Define grid along the continuous feature.
            cont_vals = X_np[:, cont_idx]
            c_min, c_max = cont_vals.min() - 0.5, cont_vals.max() + 0.5
            cont_grid = np.linspace(c_min, c_max, resolution)
            n_features = X_np.shape[1]
            grid_list = []
            for i in range(n_features):
                if i == cont_idx:
                    grid_list.append(cont_grid)
                elif i == cat_idx:
                    grid_list.append(np.full(cont_grid.shape, cat))
                else:
                    grid_list.append(np.full(cont_grid.shape, X_np[:, i].mean()))
            grid_arr = np.stack(grid_list, axis=1)
            probs = model_predict(grid_arr)
            class_preds = (probs > 0.5).astype(np.int32)
            ax.plot(cont_grid, class_preds, label="Decision boundary")
            ax.scatter(
                X_np[mask, cont_idx], y_np[mask], c="k", edgecolors="w", label="Data"
            )
            ax.set_title(f"Feature {cat_idx} = {cat}")
            ax.set_xlabel(f"Feature {cont_idx}")
            ax.set_ylabel("Predicted class")
            ax.legend()
        plt.suptitle(title)
        plt.show()

    # --- Case 3: Both features categorical ---
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(
            X_np[:, f1], X_np[:, f2], c=y_np, cmap=plt.cm.Paired, edgecolors="k"
        )
        plt.xlabel(f"Feature {f1}")
        plt.ylabel(f"Feature {f2}")
        plt.title(title + " (Both features categorical)")
        plt.grid(True)
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

    # Evaluate model
    visualize_binary_torch(model, X_val, y_val)
