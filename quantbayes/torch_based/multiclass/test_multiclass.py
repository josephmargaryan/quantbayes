# multiclass_script_torch.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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


def visualize_multiclass_torch(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: tuple = (0, 1),
    unique_threshold: int = 10,
    resolution: int = 100,
    title: str = "Multiclass Decision Boundary",
):
    """
    Visualize a multiclass classification decision boundary with automatic checks for
    continuous vs. categorical features.

    Parameters
    ----------
    model : torch.nn.Module
        A trained multiclass classifier that outputs raw logits.
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        True class labels (n_samples,).
    feature_indices : tuple of int, optional
        Indices of the two features to visualize.
    unique_threshold : int, optional
        Maximum number of unique values for a feature to be considered categorical.
    resolution : int, optional
        Number of points in the grid (for continuous features).
    title : str, optional
        Title for the plot.
    """
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    f1, f2 = feature_indices

    unique_f1 = np.unique(X_np[:, f1])
    unique_f2 = np.unique(X_np[:, f2])
    is_f1_cat = len(unique_f1) < unique_threshold
    is_f2_cat = len(unique_f2) < unique_threshold

    model.eval()
    device = next(model.parameters()).device

    def model_predict(X_input):
        with torch.no_grad():
            X_tensor = torch.tensor(X_input, dtype=torch.float32, device=device)
            logits = model(X_tensor)  # shape: (n_samples, num_classes)
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    # --- Case 1: Both features continuous ---
    if not is_f1_cat and not is_f2_cat:
        x_min, x_max = X_np[:, f1].min() - 0.5, X_np[:, f1].max() + 0.5
        y_min, y_max = X_np[:, f2].min() - 0.5, X_np[:, f2].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )
        n_features = X_np.shape[1]
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
        class_preds = np.argmax(probs, axis=-1).reshape(xx.shape)

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
            class_preds = np.argmax(probs, axis=-1)
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
    from quantbayes.fake_data import generate_multiclass_classification_data
    from sklearn.model_selection import train_test_split

    df = generate_multiclass_classification_data(n_categorical=1, n_continuous=2)

    X, y = df.drop("target", axis=1), df["target"]
    X, y = torch.tensor(X.values, dtype=torch.float32), torch.tensor(
        y.values, dtype=torch.long
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X.clone(), y.clone(), test_size=0.2, random_state=24
    )

    num_classes = 3
    model = MLPClassifier(input_dim=X.shape[-1], num_classes=num_classes, hidden_dim=32)

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

    visualize_multiclass_torch(model, X_val, y_val)
