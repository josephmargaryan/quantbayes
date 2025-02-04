"""
regression_mlp.py

1) Regression with an MLP.
Data: X ∈ ℝ^(N×D), y ∈ ℝ^(N,).

Requirements:
- MLP regression in Equinox
- MSE loss, early stopping
- Plot predictions vs. ground truth
- Plot training & validation loss curves
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------
# Model Definition
# ----------------------------
class MLPRegression(eqx.Module):
    """A simple MLP for regression."""
    layers: list

    def __init__(self, in_size, hidden_sizes, out_size, *, key):
        # We'll create a list of eqx.nn.Linear layers
        keys = jax.random.split(key, num=len(hidden_sizes) + 1)
        current_size = in_size
        lyrs = []
        for i, h in enumerate(hidden_sizes):
            lyrs.append(eqx.nn.Linear(current_size, h, use_bias=True, key=keys[i]))
            current_size = h
        # final layer
        lyrs.append(eqx.nn.Linear(current_size, out_size, use_bias=True, key=keys[-1]))
        self.layers = lyrs

    def __call__(self, x):
        # Apply ReLU on all but last layer
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x).squeeze()


# ----------------------------
# Data Handling
# ----------------------------
def prepare_data(key, N=200, D=2):
    """Generate a synthetic dataset for regression."""
    key1, key2 = jax.random.split(key)
    X = jax.random.normal(key1, shape=(N, D))
    # A simple linear relationship plus noise
    w_true = jnp.arange(1, D + 1, dtype=jnp.float32) * 1.5  # e.g. [1.5, 3.0] for D=2
    noise = 0.1 * jax.random.normal(key2, shape=(N,))
    y = X @ w_true + 1.0 + noise
    return X, y


def train_val_split(X, y, val_ratio=0.2, seed=42):
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(N)
    rng.shuffle(indices)
    split = int(N * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ----------------------------
# Training Pipeline
# ----------------------------
def mse_loss(model, x, y):
    pred = jax.vmap(model)(x)
    return jnp.mean((pred - y) ** 2)


@eqx.filter_jit
def make_step(model, x, y, opt_state, optimizer):
    """Single training step with JIT compilation."""
    loss_value, grads = eqx.filter_value_and_grad(mse_loss)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss_value, model, opt_state


def train_model(model, X_train, y_train, X_val, y_val, lr=1e-3, num_epochs=200, patience=20):
    """Trains the model with early stopping."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    best_val_loss = float("inf")
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        loss_train, model, opt_state = make_step(model, X_train, y_train, opt_state, optimizer)
        loss_val = mse_loss(model, X_val, y_val)
        train_losses.append(loss_train)
        val_losses.append(loss_val)

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {loss_train:.4f}, Val Loss: {loss_val:.4f}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return model, train_losses, val_losses


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(model, X_test, y_test, train_losses, val_losses, feature_index=0):
    test_loss = mse_loss(model, X_test, y_test)
    print(f"Test MSE Loss: {test_loss:.4f}")

    # Sort X_test by selected feature for visualization
    order = jnp.argsort(X_test[:, feature_index])
    sorted_x = X_test[order]
    sorted_y = y_test[order]
    preds = jax.vmap(model)(sorted_x)

    # Plot predictions vs ground truth
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(np.array(sorted_x[:, feature_index]), np.array(sorted_y), label="Ground Truth", alpha=0.7)
    plt.scatter(np.array(sorted_x[:, feature_index]), np.array(preds), label="Predictions", alpha=0.7)
    plt.xlabel(f"Feature {feature_index}")
    plt.ylabel("Target")
    plt.legend()
    plt.title("Predictions vs Ground Truth")

    # Plot the training and validation loss curves
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    key = jax.random.PRNGKey(0)

    # Prepare data
    X, y = prepare_data(key, N=200, D=2)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.2)

    # Initialize model
    model_key = jax.random.split(key, 1)[0]
    model = MLPRegression(in_size=X.shape[1], hidden_sizes=[32, 16], out_size=1, key=model_key)

    # Train model
    model, train_losses, val_losses = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        lr=1e-3,
        num_epochs=200,
        patience=20
    )

    # Evaluate
    evaluate_model(model, X_val, y_val, train_losses, val_losses, feature_index=0)


if __name__ == "__main__":
    main()
