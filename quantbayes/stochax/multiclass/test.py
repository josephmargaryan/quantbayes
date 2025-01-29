# multiclass_script.py
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
# 1. MLP for Multiclass (C classes)
# -----------------------------------------------------------
class MLPClassifier(nn.Module):
    num_classes: int
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, x, **kwargs):
        """
        x: (batch_size, input_dim)
        outputs: (batch_size, num_classes) logits
        """
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)  # logits
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
def cross_entropy_loss(params, apply_fn, x, y, apply_fn_kwargs=None):
    """
    y: (batch_size,) containing integer class labels in [0, num_classes-1].
    logits: (batch_size, num_classes)
    We'll use optax's softmax_cross_entropy.
    """
    if apply_fn_kwargs is None:
        apply_fn_kwargs = {}
    logits = apply_fn({"params": params}, x, **apply_fn_kwargs)
    # compute cross-entropy
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    loss_val = optax.softmax_cross_entropy(logits, one_hot).mean()
    return loss_val


@jax.jit
def train_step(
    state: train_state.TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    apply_fn_kwargs: Dict[str, Any],
):
    loss, grads = jax.value_and_grad(cross_entropy_loss)(
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
            batch_y = jnp.array(batch_y, dtype=jnp.int32)
            state, loss_val = train_step(state, batch_X, batch_y, apply_fn_kwargs_train)
            epoch_losses.append(loss_val)

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

    # Plot training and validation losses
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")
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
        batch_y = jnp.array(batch_y, dtype=jnp.int32)
        loss_val = cross_entropy_loss(
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
    Perform MC sampling for multiclass. Plot 2D decision boundaries for chosen features,
    color-coded by predicted class. We'll also visualize uncertainty as the predicted
    class probabilities' max std or entropy.
    """
    if apply_fn_kwargs is None:
        apply_fn_kwargs = {}

    # We'll do multiple forward passes and average the probabilities
    def predict_logits(params, x, **kwargs):
        return model.apply({"params": params}, x, **kwargs)  # (batch_size, num_classes)

    # We'll convert logits to probabilities: softmax
    # shape => (num_samples, batch_size, num_classes)
    logits_list = []
    for _ in range(num_samples):
        rng, subkey = jax.random.split(rng)
        # if using dropout, pass subkey in apply_fn_kwargs
        logits_list.append(predict_logits(params, X_val, **apply_fn_kwargs))
    logits_stacked = jnp.stack(logits_list, axis=0)
    probs_stacked = jax.nn.softmax(
        logits_stacked, axis=-1
    )  # (num_samples, N, num_classes)
    mean_probs = jnp.mean(probs_stacked, axis=0)  # (N, num_classes)
    preds = jnp.argmax(mean_probs, axis=-1)  # (N,)

    # Accuracy
    acc = jnp.mean(preds == y_val)

    # Decision boundary in 2D for chosen features
    f1, f2 = features
    x_min, x_max = X_val[:, f1].min() - 0.5, X_val[:, f1].max() + 0.5
    y_min, y_max = X_val[:, f2].min() - 0.5, X_val[:, f2].max() + 0.5
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100),
    )
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)

    # Evaluate MC on grid
    grid_logits_list = []
    for _ in range(num_samples):
        rng, subkey = jax.random.split(rng)
        grid_logits = predict_logits(params, grid_points, **apply_fn_kwargs)
        grid_logits_list.append(grid_logits)
    # (num_samples, 10000, num_classes)
    grid_logits_stacked = jnp.stack(grid_logits_list, axis=0)
    grid_probs_stacked = jax.nn.softmax(
        grid_logits_stacked, axis=-1
    )  # (num_samples, 10000, C)
    grid_mean_probs = jnp.mean(grid_probs_stacked, axis=0)  # (10000, C)
    grid_preds = jnp.argmax(grid_mean_probs, axis=-1)  # (10000,)

    # Reshape
    zz = grid_preds.reshape(grid_x.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x, grid_y, zz, alpha=0.3, cmap=plt.cm.tab10)

    # Plot data points
    plt.scatter(
        X_val[:, f1],
        X_val[:, f2],
        c=y_val,
        edgecolor="k",
        cmap=plt.cm.tab10,
        alpha=0.7,
    )
    plt.xlabel(f"Feature {f1}")
    plt.ylabel(f"Feature {f2}")
    plt.title(f"Multiclass Decision Boundary (Accuracy={acc:.2f})")
    plt.colorbar()
    plt.show()

    # Optionally visualize uncertainty (e.g., max(prob) std or predictive entropy)
    # Let's do predictive entropy: -sum(p * log(p))
    # We'll compute the mean entropy across the MC samples
    # or the entropy of the mean distribution. Let's do both.

    # Entropy of the mean distribution:
    # mean_probs: (N, num_classes)
    epsilon = 1e-10
    mean_entropy = -jnp.sum(mean_probs * jnp.log(mean_probs + epsilon), axis=-1)  # (N,)
    # Let's quickly plot that for the data points.
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        X_val[:, f1],
        X_val[:, f2],
        c=mean_entropy,
        cmap="viridis",
        edgecolor="k",
    )
    plt.colorbar(sc, label="Predictive Entropy of Mean Probs")
    plt.xlabel(f"Feature {f1}")
    plt.ylabel(f"Feature {f2}")
    plt.title("Uncertainty (Entropy) in Mean Prediction")
    plt.grid(True)
    plt.show()

    return {
        "pred_classes": preds,
        "accuracy": float(acc),
        "mean_probs": mean_probs,  # (N, num_classes)
        "entropy": mean_entropy,  # (N,)
    }


# -----------------------------------------------------------
# 7. Example Test
# -----------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data for 3 classes in 2D
    rng_np = np.random.RandomState(42)
    N = 300
    X_class0 = rng_np.normal(loc=[-2, -2], scale=1.0, size=(N // 3, 2))
    X_class1 = rng_np.normal(loc=[2, 2], scale=1.0, size=(N // 3, 2))
    X_class2 = rng_np.normal(loc=[-2, 2], scale=1.0, size=(N // 3, 2))

    X_all = np.vstack([X_class0, X_class1, X_class2]).astype(np.float32)
    y_all = np.array([0] * (N // 3) + [1] * (N // 3) + [2] * (N // 3), dtype=np.int32)

    # Shuffle
    perm = rng_np.permutation(N)
    X_all = X_all[perm]
    y_all = y_all[perm]

    # Split
    train_size = int(0.8 * N)
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    y_train, y_val = y_all[:train_size], y_all[train_size:]

    # Model
    num_classes = 3
    model = MLPClassifier(num_classes=num_classes, hidden_dim=32)
    rng = jax.random.PRNGKey(0)

    # Train
    state, train_losses, val_losses = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
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
        features=(0, 1),
        apply_fn_kwargs={},
    )
    print("Final Accuracy:", results["accuracy"])
