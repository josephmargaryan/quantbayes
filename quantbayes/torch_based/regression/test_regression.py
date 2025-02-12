# regression_script_torch.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns


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


def visualize_regression(
    model,
    X: np.ndarray,
    y: np.ndarray,
    grid_points: int = 100,
    unique_threshold: int = 10,
):
    """
    Unified visualization for deterministic regression models.
    Produces separate visualizations for a continuous feature and a categorical feature
    if both are available. Otherwise, visualizes the available type.
    """

    # Ensure X and y are NumPy arrays.
    X_np = np.asarray(X)
    y_np = np.asarray(y).squeeze()
    n_samples, n_features = X_np.shape

    # Classify features.
    continuous_features = []
    categorical_features = []
    for i in range(n_features):
        unique_vals = np.unique(X_np[:, i])
        if len(unique_vals) < unique_threshold:
            categorical_features.append(i)
        else:
            continuous_features.append(i)

    # Helper function for model prediction.
    def model_predict(x_np):
        # x_np should be of shape (n_samples, n_features)
        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            preds = model(x_tensor).squeeze(-1).cpu().numpy()
        return np.atleast_1d(preds)

    # Decide on which features to visualize.
    cont_idx = continuous_features[0] if continuous_features else None
    cat_idx = categorical_features[0] if categorical_features else None

    # Decide on subplot grid.
    # Cases:
    #   - Both types: use 2 rows, 2 columns (each feature gets its own PDP and ICE).
    #   - Only one type: use 1 row, 2 columns.
    if cont_idx is not None and cat_idx is not None:
        nrows = 2
    else:
        nrows = 1
    ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
    # If axes is 1D, convert it to 2D for consistent indexing.
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    # -------------------------------
    # Continuous Feature Visualization
    # -------------------------------
    if cont_idx is not None:
        cont_vals = X_np[:, cont_idx]
        cont_min = np.min(cont_vals)
        cont_max = np.max(cont_vals)
        cont_grid = np.linspace(cont_min, cont_max, grid_points)

        # PDP for continuous feature.
        baseline = np.mean(X_np, axis=0)
        X_pdp = np.tile(baseline, (grid_points, 1))
        X_pdp[:, cont_idx] = cont_grid
        pdp_preds = model_predict(X_pdp)

        ax_pdp = axes[0, 0]
        # KDE plot for data density.
        sns.kdeplot(
            x=cont_vals,
            y=y_np,
            ax=ax_pdp,
            cmap="Blues",
            fill=True,
            alpha=0.5,
            thresh=0.05,
        )
        ax_pdp.plot(cont_grid, pdp_preds, color="red", label="PDP")
        ax_pdp.set_xlabel(f"Feature {cont_idx} (Continuous)")
        ax_pdp.set_ylabel("Target")
        ax_pdp.set_title("Continuous PDP")
        ax_pdp.legend()

        # ICE plot for continuous feature.
        rng = np.random.default_rng(42)
        ice_indices = rng.choice(n_samples, size=min(20, n_samples), replace=False)
        ax_ice = axes[0, 1]
        for idx in ice_indices:
            sample = X_np[idx, :].copy()
            X_ice = np.tile(sample, (grid_points, 1))
            X_ice[:, cont_idx] = cont_grid
            ice_preds = model_predict(X_ice)
            ax_ice.plot(cont_grid, ice_preds, alpha=0.5)
        ax_ice.set_xlabel(f"Feature {cont_idx} (Continuous)")
        ax_ice.set_ylabel("Target")
        ax_ice.set_title("Continuous ICE")
    else:
        # Hide continuous subplots if no continuous feature exists.
        for ax in axes[0]:
            ax.axis("off")

    # -------------------------------
    # Categorical Feature Visualization
    # -------------------------------
    if cat_idx is not None:
        cat_vals = X_np[:, cat_idx]
        unique_cats = np.unique(cat_vals)

        # Categorical PDP: bar plot.
        baseline = np.mean(X_np, axis=0)
        cat_pdp_means = []
        for cat in unique_cats:
            sample = baseline.copy()
            sample[cat_idx] = cat
            pred = model_predict(sample[None, :])[0]
            cat_pdp_means.append(pred)
        # Place categorical PDP in second row, first column (if both types exist)
        # or first row if only categorical exists.
        ax_pdp_cat = axes[1, 0] if nrows == 2 else axes[0, 0]
        ax_pdp_cat.bar(unique_cats, cat_pdp_means, alpha=0.7)
        ax_pdp_cat.set_xlabel(f"Feature {cat_idx} (Categorical)")
        ax_pdp_cat.set_ylabel("Predicted Target")
        ax_pdp_cat.set_title("Categorical PDP")

        # Categorical ICE: box plot.
        cat_predictions = {}
        for cat in unique_cats:
            mask = cat_vals == cat
            X_cat = X_np[mask, :]
            if X_cat.shape[0] > 0:
                preds_cat = model_predict(X_cat)
                cat_predictions[cat] = preds_cat
        ax_ice_cat = axes[1, 1] if nrows == 2 else axes[0, 1]
        box_data = [
            cat_predictions[cat] for cat in unique_cats if cat in cat_predictions
        ]
        ax_ice_cat.boxplot(box_data, tick_labels=unique_cats)
        ax_ice_cat.set_xlabel(f"Feature {cat_idx} (Categorical)")
        ax_ice_cat.set_ylabel("Predicted Target")
        ax_ice_cat.set_title("Categorical ICE")
    else:
        # Hide categorical subplots if no categorical feature exists.
        if nrows == 2:
            for ax in axes[1]:
                ax.axis("off")

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------
# 4. Example Test
# ----------------------------------------------------------------
if __name__ == "__main__":
    from quantbayes.fake_data import generate_regression_data
    from sklearn.model_selection import train_test_split

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

    visualize_regression(model, X_val, y_val)
