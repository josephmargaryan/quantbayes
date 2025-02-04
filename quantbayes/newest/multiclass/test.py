"""
multiclass_classification_mlp.py

3) Multiclass classification with an MLP.
Data: X ∈ ℝ^(N×D), y ∈ ℝ^(N×num_classes) (one-hot).

Requirements:
- MLP with softmax
- Cross-entropy loss
- Train/val split
- LR scheduling
- Evaluate accuracy & confusion matrix
- Visualize decision boundary
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
class MulticlassClassifier(eqx.Module):
    """MLP for multiclass classification, output = logits -> softmax externally."""
    layers: list

    def __init__(self, in_size, hidden_sizes, out_size, *, key):
        keys = jax.random.split(key, num=len(hidden_sizes) + 1)
        lyrs = []
        current_size = in_size
        for i, h in enumerate(hidden_sizes):
            lyrs.append(eqx.nn.Linear(current_size, h, use_bias=True, key=keys[i]))
            current_size = h
        # final
        lyrs.append(eqx.nn.Linear(current_size, out_size, use_bias=True, key=keys[-1]))
        self.layers = lyrs

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        logits = self.layers[-1](x)
        return logits


# ----------------------------
# Data Handling
# ----------------------------
def prepare_data(key, N=600, num_classes=3):
    """Generate synthetic data with 3 clusters in 2D, one-hot targets."""
    N_per_class = N // num_classes
    keys = jax.random.split(key, num_classes)
    centers = [(0, 0), (3, 3), (-3, 3)]  # just for variety
    X, y = [], []
    for i in range(num_classes):
        x_i = jax.random.normal(keys[i], shape=(N_per_class, 2)) + jnp.array(centers[i])
        X.append(x_i)
        one_hot = jnp.zeros(num_classes)
        one_hot = one_hot.at[i].set(1.0)
        y_i = jnp.tile(one_hot, (N_per_class, 1))
        y.append(y_i)
    X = jnp.concatenate(X, axis=0)
    Y = jnp.concatenate(y, axis=0)
    return X, Y


def train_val_split(X, Y, val_ratio=0.2, seed=42):
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    split = int(N * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]
    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]


# ----------------------------
# Loss and LR scheduling
# ----------------------------
def cross_entropy_loss(model, x, y):
    # y is one-hot
    logits = jax.vmap(model)(x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(y * log_probs, axis=-1))


def accuracy_and_confusion_matrix(model, x, y):
    """Returns accuracy and confusion matrix (numpy)."""
    logits = jax.vmap(model)(x)
    preds = jnp.argmax(logits, axis=-1)
    targets = jnp.argmax(y, axis=-1)
    acc = jnp.mean(preds == targets)

    # Confusion matrix
    num_classes = y.shape[1]
    cm = jnp.zeros((num_classes, num_classes), dtype=jnp.int32)
    def update_cm(c, p):
        return c.at[targets[p], preds[p]].add(1)
    indices = jnp.arange(len(preds))
    cm = jax.lax.fori_loop(0, len(preds), lambda i, c: c.at[targets[i], preds[i]].add(1), cm)
    return acc, cm


# ----------------------------
# Training
# ----------------------------
@eqx.filter_jit
def make_step(model, x, y, opt_state, optimizer):
    loss_value, grads = eqx.filter_value_and_grad(cross_entropy_loss)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss_value, model, opt_state


def train_model(model, X_train, Y_train, X_val, Y_val,
                lr=1e-3, epochs=100, lr_schedule_gamma=0.99):
    # Example: exponential decay in LR each epoch
    scheduler = optax.exponential_decay(init_value=lr, transition_steps=1, decay_rate=lr_schedule_gamma)
    optimizer = optax.adam(scheduler)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        loss_train, model, opt_state = make_step(model, X_train, Y_train, opt_state, optimizer)
        loss_val = cross_entropy_loss(model, X_val, Y_val)
        train_losses.append(loss_train)
        val_losses.append(loss_val)

        if (epoch + 1) % 10 == 0:
            lr_current = scheduler(epoch)
            print(f"Epoch {epoch+1}/{epochs} - LR: {lr_current:.6f} - Train CE: {loss_train:.4f} - Val CE: {loss_val:.4f}")

    return model, train_losses, val_losses


# ----------------------------
# Visualization
# ----------------------------
def plot_decision_boundary(model, X, num_classes=3, steps=100, feature_index=0):
    x_min, x_max = float(X[:, 0].min()) - 1, float(X[:, 0].max()) + 1
    y_min, y_max = float(X[:, 1].min()) - 1, float(X[:, 1].max()) + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
    grid = jnp.stack([jnp.ravel(jnp.array(xx)), jnp.ravel(jnp.array(yy))], axis=-1)

    logits = jax.vmap(model)(grid)
    preds = jnp.argmax(logits, axis=-1)
    preds = preds.reshape(xx.shape)
    plt.contourf(xx, yy, preds, alpha=0.3, levels=np.arange(-0.5, num_classes, 1))


def evaluate_model(model, X_test, Y_test, train_losses, val_losses):
    acc, cm = accuracy_and_confusion_matrix(model, X_test, Y_test)
    print(f"Test Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(np.array(cm))

    # Plots
    plt.figure(figsize=(12, 5))
    # Decision boundary
    plt.subplot(1, 2, 1)
    plot_decision_boundary(model, X_test, num_classes=Y_test.shape[1])
    # color by actual class
    class_colors = jnp.argmax(Y_test, axis=-1)
    plt.scatter(np.array(X_test[:, 0]), np.array(X_test[:, 1]), c=np.array(class_colors), edgecolors="k")
    plt.title("Decision Boundary")

    # Loss curves
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CE Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    key = jax.random.PRNGKey(2)
    X, Y = prepare_data(key, N=600, num_classes=3)
    X_train, Y_train, X_val, Y_val = train_val_split(X, Y, val_ratio=0.25)

    model_key = jax.random.split(key, 1)[0]
    model = MulticlassClassifier(in_size=2, hidden_sizes=[32, 32], out_size=3, key=model_key)

    model, train_losses, val_losses = train_model(model, X_train, Y_train, X_val, Y_val,
                                                  lr=1e-3, epochs=100, lr_schedule_gamma=0.99)
    evaluate_model(model, X_val, Y_val, train_losses, val_losses)


if __name__ == "__main__":
    main()
