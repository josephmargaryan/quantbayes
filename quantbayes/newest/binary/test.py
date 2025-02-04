"""
binary_classification_mlp.py

2) Binary classification with an MLP.
Data: X ∈ ℝ^(N×D), y ∈ {0,1}^(N,).

Requirements:
- MLP classifier with sigmoid output
- Binary cross-entropy loss
- Train/val split
- Possibly dropout or batchnorm
- Evaluate accuracy, precision, recall, F1-score
- 2D decision boundary plot
- Plot train & val losses
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
class BinaryClassifier(eqx.Module):
    """An MLP for binary classification with a final sigmoid."""
    layers: list
    dropout_rate: float
    training: bool  # Will control dropout usage

    def __init__(self, in_size, hidden_sizes, out_size=1, dropout_rate=0.0, *, key):
        keys = jax.random.split(key, num=len(hidden_sizes) + 1)
        lyrs = []
        current_size = in_size
        for i, h in enumerate(hidden_sizes):
            lyrs.append(eqx.nn.Linear(current_size, h, use_bias=True, key=keys[i]))
            current_size = h
        # final layer
        lyrs.append(eqx.nn.Linear(current_size, out_size, use_bias=True, key=keys[-1]))
        self.layers = lyrs
        self.dropout_rate = dropout_rate
        self.training = True  # set to False during evaluation

    def __call__(self, x, *, key=None):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
            # Dropout
            if self.dropout_rate > 0.0 and self.training:
                if key is None:
                    raise ValueError("Key must be provided when using dropout in training mode.")
                keep_prob = 1.0 - self.dropout_rate
                drop_mask = jax.random.bernoulli(key, keep_prob, shape=x.shape)
                x = jnp.where(drop_mask, x / keep_prob, 0)
                # Update key
                key, _ = jax.random.split(key)
        logits = self.layers[-1](x)
        return jax.nn.sigmoid(logits).squeeze()

# ----------------------------
# Data Handling
# ----------------------------
def prepare_data(key, N=500, D=2):
    """Generate two classes in 2D for easy visualization."""
    key1, key2 = jax.random.split(key)
    N1 = N // 2
    # Class 0: around (-2, -2)
    class0 = jax.random.normal(key1, shape=(N1, D)) + jnp.array([-2.0, -2.0])
    # Class 1: around (+2, +2)
    class1 = jax.random.normal(key2, shape=(N - N1, D)) + jnp.array([2.0, 2.0])
    X = jnp.concatenate([class0, class1], axis=0)
    y = jnp.concatenate([jnp.zeros(N1), jnp.ones(N - N1)], axis=0)
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
# Loss and Metric
# ----------------------------
def binary_cross_entropy_loss(model, x, y, key=None):
    """Compute BCE loss. Model outputs sigmoid(...) in [0,1]."""
    preds = jax.vmap(lambda xx, kk: model(xx, key=kk))(x, jax.random.split(key, x.shape[0])) if key else jax.vmap(model)(x)
    eps = 1e-7
    return -jnp.mean(y * jnp.log(preds + eps) + (1 - y) * jnp.log(1 - preds + eps))


def compute_metrics(model, X, y):
    """Computes accuracy, precision, recall, F1."""
    preds = jax.vmap(model)(X)  # (N,)
    pred_labels = (preds > 0.5).astype(jnp.float32)
    y_true = y.astype(jnp.float32)
    accuracy = jnp.mean(pred_labels == y_true)

    tp = jnp.sum((pred_labels == 1) & (y_true == 1))
    fp = jnp.sum((pred_labels == 1) & (y_true == 0))
    fn = jnp.sum((pred_labels == 0) & (y_true == 1))
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return accuracy, precision, recall, f1


# ----------------------------
# Training Pipeline
# ----------------------------
@eqx.filter_jit
def make_step(model, x, y, opt_state, optimizer, key):
    # We'll handle dropout key usage in the loss function
    loss_value, grads = eqx.filter_value_and_grad(binary_cross_entropy_loss)(model, x, y, key=key)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss_value, model, opt_state


def train_model(model, X_train, y_train, X_val, y_val, lr=1e-3, epochs=100, patience=10):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    best_val_loss = float("inf")
    patience_counter = 0

    train_losses = []
    val_losses = []

    key_seq = jax.random.split(jax.random.PRNGKey(0), epochs)

    for epoch in range(epochs):
        # Switch model to training mode
        model.training = True
        loss_train, model, opt_state = make_step(model, X_train, y_train, opt_state, optimizer, key_seq[epoch])
        model.training = False
        loss_val = binary_cross_entropy_loss(model, X_val, y_val)
        train_losses.append(loss_train)
        val_losses.append(loss_val)

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train BCE: {loss_train:.4f}, Val BCE: {loss_val:.4f}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return model, train_losses, val_losses


# ----------------------------
# Evaluation & Visualization
# ----------------------------
def plot_decision_boundary(model, X, y, steps=100):
    """Plot 2D decision boundary for a user-specified (x1, x2)."""
    x_min, x_max = float(X[:, 0].min()) - 1, float(X[:, 0].max()) + 1
    y_min, y_max = float(X[:, 1].min()) - 1, float(X[:, 1].max()) + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps),
                         np.linspace(y_min, y_max, steps))
    grid = jnp.stack([jnp.ravel(jnp.array(xx)), jnp.ravel(jnp.array(yy))], axis=-1)

    # Evaluate model for the grid
    model.training = False
    zz = jax.vmap(model)(grid)
    zz = zz.reshape(xx.shape)

    plt.contourf(xx, yy, zz, alpha=0.4, levels=[-0.1, 0.5, 1.1], colors=["blue", "red"])


def evaluate_model(model, X_test, y_test, train_losses, val_losses):
    acc, prec, rec, f1 = compute_metrics(model, X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    plt.figure(figsize=(12, 5))
    # Decision boundary
    plt.subplot(1, 2, 1)
    plot_decision_boundary(model, X_test, y_test)
    plt.scatter(np.array(X_test[:, 0]), np.array(X_test[:, 1]), c=np.array(y_test), edgecolors="k")
    plt.title("Decision Boundary")

    # Loss curves
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    key = jax.random.PRNGKey(1)
    X, y = prepare_data(key, N=500, D=2)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.3)

    model_key = jax.random.split(key, 1)[0]
    model = BinaryClassifier(in_size=2, hidden_sizes=[16, 16], out_size=1, dropout_rate=0.2, key=model_key)

    model, train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val,
                                                  lr=1e-3, epochs=100, patience=10)

    evaluate_model(model, X_val, y_val, train_losses, val_losses)


if __name__ == "__main__":
    main()
