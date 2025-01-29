# binary_script.py
import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
from flax.training import train_state
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, Optional


# -----------------------------------------------------------
# 1. Example MLP for Binary Classification
# -----------------------------------------------------------
class MLPBinaryClassifier(nn.Module):
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, x, **kwargs):
        """
        x : (batch_size, input_dim)
        Output: (batch_size, 1) (logits)
        We'll apply a sigmoid externally or interpret as logits for BCE.
        """
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)  # logits for binary classification
        return x


# -----------------------------------------------------------
# 2. Train State
# -----------------------------------------------------------
def create_train_state(
    rng: jax.random.PRNGKey,
    model: nn.Module,
    learning_rate: float,
    example_input: jnp.ndarray,
) -> train_state.TrainState:
    params = model.init(rng, example_input)["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# -----------------------------------------------------------
# 3. Loss & Train Step
# -----------------------------------------------------------
def binary_cross_entropy_loss(params, apply_fn, x, y, apply_fn_kwargs=None):
    """
    y: (batch_size,) or (batch_size,1) with values in {0,1}.
    We'll interpret apply_fn(...) as logits, so we do:
        pred_prob = sigmoid(logits)
        loss = -[y*log(prob) + (1-y)*log(1-prob)]
    """
    if apply_fn_kwargs is None:
        apply_fn_kwargs = {}
    logits = apply_fn({"params": params}, x, **apply_fn_kwargs)  # (batch_size,1)
    logits = logits.squeeze(-1)  # (batch_size,)

    # we can use optax sigmoid BCE
    loss_val = optax.sigmoid_binary_cross_entropy(logits, y).mean()
    return loss_val


@jax.jit
def train_step(
    state: train_state.TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    apply_fn_kwargs: Dict[str, Any],
):
    loss, grads = jax.value_and_grad(binary_cross_entropy_loss)(
        state.params, state.apply_fn, x, y, apply_fn_kwargs
    )
    state = state.apply_gradients(grads=grads)
    return state, loss


# -----------------------------------------------------------
# 4. Data Generator
# -----------------------------------------------------------
def data_generator(
    rng: jax.random.PRNGKey,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
):
    num_samples = X.shape[0]
    if shuffle:
        indices = jax.random.permutation(rng, num_samples)
        X = X[indices]
        y = y[indices]

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]


# -----------------------------------------------------------
# 5. Training Function
# -----------------------------------------------------------
def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    rng: jax.random.PRNGKey,
    apply_fn_kwargs_train: Dict[str, Any],
    apply_fn_kwargs_val: Optional[Dict[str, Any]] = None,
):
    if apply_fn_kwargs_val is None:
        apply_fn_kwargs_val = {}

    input_dim = X_train.shape[1]
    example_input = jnp.ones((1, input_dim), dtype=jnp.float32)

    state = create_train_state(rng, model, learning_rate, example_input)

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        rng, data_rng = jax.random.split(rng)
        epoch_losses = []

        for batch_X, batch_y in data_generator(data_rng, X_train, y_train, batch_size):
            batch_X = jnp.array(batch_X, dtype=jnp.float32)
            batch_y = jnp.array(batch_y, dtype=jnp.float32)
            state, loss = train_step(state, batch_X, batch_y, apply_fn_kwargs_train)
            epoch_losses.append(loss)

        train_loss = jnp.mean(jnp.array(epoch_losses))

        val_loss = evaluate_loss(
            state.params,
            state.apply_fn,
            X_val,
            y_val,
            batch_size,
            rng,
            apply_fn_kwargs_val,
        )

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

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
    plt.ylabel("Binary Cross Entropy")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return state, train_losses, val_losses


def evaluate_loss(
    params,
    apply_fn,
    X,
    y,
    batch_size,
    rng,
    apply_fn_kwargs=None,
):
    if apply_fn_kwargs is None:
        apply_fn_kwargs = {}

    losses = []
    for batch_X, batch_y in data_generator(rng, X, y, batch_size, shuffle=False):
        batch_X = jnp.array(batch_X, dtype=jnp.float32)
        batch_y = jnp.array(batch_y, dtype=jnp.float32)
        loss_val = binary_cross_entropy_loss(
            params, apply_fn, batch_X, batch_y, apply_fn_kwargs
        )
        losses.append(loss_val)
    return float(jnp.mean(jnp.array(losses)))


# -----------------------------------------------------------
# 6. Evaluation Function (MC Sampling + Decision Boundary)
# -----------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    params,
    X_val: np.ndarray,
    y_val: np.ndarray,
    rng: jax.random.PRNGKey,
    num_samples: int,
    features: Optional[Tuple[int, int]] = (0, 1),
    apply_fn_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Perform MC sampling to visualize binary decision boundary + uncertainty.
    We'll pick 2 features (features=(0,1)) from X to plot in 2D.
    If X has more dims, they won't be visualized.
    """
    if apply_fn_kwargs is None:
        apply_fn_kwargs = {}

    # Probability predictions from multiple passes
    def predict_probs(params, X, **kwargs):
        logits = model.apply({"params": params}, X, **kwargs)  # (batch_size,1)
        return jax.nn.sigmoid(logits).squeeze(-1)  # (batch_size,)

    preds_list = []
    for _ in range(num_samples):
        rng, subkey = jax.random.split(rng)
        # if dropout: apply_fn_kwargs['rngs'] = {'dropout': subkey}
        p = predict_probs(params, X_val, **apply_fn_kwargs)
        preds_list.append(p)
    # (num_samples, N)
    preds_stacked = jnp.stack(preds_list, axis=0)
    mean_preds = jnp.mean(preds_stacked, axis=0)  # (N,)
    std_preds = jnp.std(preds_stacked, axis=0)  # (N,)

    # Classification accuracy using mean_preds>0.5
    y_val = y_val.squeeze(-1) if y_val.ndim > 1 else y_val
    acc = jnp.mean((mean_preds > 0.5) == (y_val > 0.5))

    # Decision boundary in 2D space for chosen features
    f1, f2 = features
    x_min, x_max = X_val[:, f1].min() - 0.5, X_val[:, f1].max() + 0.5
    y_min, y_max = X_val[:, f2].min() - 0.5, X_val[:, f2].max() + 0.5
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100),
    )
    # Flatten for model input
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)

    # MC sampling on the grid
    grid_preds_list = []
    for _ in range(num_samples):
        rng, subkey = jax.random.split(rng)
        gp = predict_probs(params, grid_points, **apply_fn_kwargs)
        grid_preds_list.append(gp)
    # (num_samples, 10000)
    grid_preds_stacked = jnp.stack(grid_preds_list, axis=0)
    grid_mean = jnp.mean(grid_preds_stacked, axis=0)  # (10000,)
    grid_std = jnp.std(grid_preds_stacked, axis=0)  # (10000,)

    # Plot data points
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_val[:, f1][y_val == 0],
        X_val[:, f2][y_val == 0],
        c="red",
        edgecolor="k",
        label="Class 0",
    )
    plt.scatter(
        X_val[:, f1][y_val == 1],
        X_val[:, f2][y_val == 1],
        c="blue",
        edgecolor="k",
        label="Class 1",
    )

    # Decision boundary (mean ~ 0.5)
    # Reshape grid for contour
    grid_mean_2d = grid_mean.reshape(grid_x.shape)
    contour = plt.contourf(
        grid_x,
        grid_y,
        grid_mean_2d,
        levels=[0.0, 0.5, 1.0],
        alpha=0.3,
        colors=["red", "blue"],
    )
    plt.colorbar(contour, label="Mean Probability (Class=1)")

    plt.xlabel(f"Feature {f1}")
    plt.ylabel(f"Feature {f2}")
    plt.title(f"Binary Classification - Mean Decision Boundary\nAccuracy={acc:.2f}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # We can also plot uncertainty as a heatmap
    plt.figure(figsize=(8, 6))
    grid_std_2d = grid_std.reshape(grid_x.shape)
    cs = plt.contourf(grid_x, grid_y, grid_std_2d, cmap="viridis")
    plt.scatter(
        X_val[:, f1][y_val == 0],
        X_val[:, f2][y_val == 0],
        c="red",
        edgecolor="k",
    )
    plt.scatter(
        X_val[:, f1][y_val == 1],
        X_val[:, f2][y_val == 1],
        c="blue",
        edgecolor="k",
    )
    plt.colorbar(cs, label="Std Dev of Predicted Probability")
    plt.xlabel(f"Feature {f1}")
    plt.ylabel(f"Feature {f2}")
    plt.title("Uncertainty (Std Dev) in Predictions")
    plt.grid(True)
    plt.show()

    return {
        "mean_preds": mean_preds,
        "std_preds": std_preds,
        "accuracy": float(acc),
        "all_samples": preds_stacked,
    }


# -----------------------------------------------------------
# 7. Example Test
# -----------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data for a binary classification: two clusters
    rng_np = np.random.RandomState(0)
    N = 200
    X0 = rng_np.normal(loc=[-2, 0], scale=1.0, size=(N // 2, 2))
    X1 = rng_np.normal(loc=[2, 0], scale=1.0, size=(N // 2, 2))
    X_all = np.vstack([X0, X1]).astype(np.float32)
    y_all = np.array([0] * (N // 2) + [1] * (N // 2), dtype=np.float32)

    # Shuffle
    perm = rng_np.permutation(N)
    X_all = X_all[perm]
    y_all = y_all[perm]

    # Split
    train_size = int(0.8 * N)
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    y_train, y_val = y_all[:train_size], y_all[train_size:]

    model = MLPBinaryClassifier(hidden_dim=32)
    rng = jax.random.PRNGKey(0)

    # Train
    state, train_losses, val_losses = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-3,
        rng=rng,
        apply_fn_kwargs_train={},
        apply_fn_kwargs_val={},
    )

    # Evaluate
    results = evaluate_model(
        model,
        state.params,
        X_val,
        y_val,
        rng,
        num_samples=50,
        features=(0, 1),  # We only have 2 features in this example
        apply_fn_kwargs={},
    )
    print("Final Accuracy on Val:", results["accuracy"])
