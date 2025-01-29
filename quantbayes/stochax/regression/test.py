# regression_script.py
import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
from flax.training import train_state
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, Optional
import time


# -----------------------------------------------------------
# 1. Example MLP for Regression
# -----------------------------------------------------------
class MLPRegression(nn.Module):
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, x, **kwargs):
        """
        x : (batch_size, input_dim)
        Output : (batch_size, 1)

        kwargs: extra model arguments (e.g. 'deterministic', 'rngs', etc.)
        """
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)  # output shape: (batch_size, 1)
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
# 3. Loss and Single Train Step
# -----------------------------------------------------------
def mse_loss(params, apply_fn, x, y, apply_fn_kwargs=None):
    if apply_fn_kwargs is None:
        apply_fn_kwargs = {}
    preds = apply_fn({"params": params}, x, **apply_fn_kwargs)
    return jnp.mean((preds - y) ** 2)


@jax.jit
def train_step(
    state: train_state.TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    apply_fn_kwargs: Dict[str, Any],
):
    loss, grads = jax.value_and_grad(mse_loss)(
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
    """
    Train an MLP regression model with MSE. Returns the final state, and lists of train/val losses.
    """
    if apply_fn_kwargs_val is None:
        apply_fn_kwargs_val = {}

    input_dim = X_train.shape[1]
    example_input = jnp.ones((1, input_dim), dtype=jnp.float32)

    state = create_train_state(rng, model, learning_rate, example_input)

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        rng, subkey = jax.random.split(rng)
        epoch_losses = []
        for batch_X, batch_y in data_generator(subkey, X_train, y_train, batch_size):
            batch_X = jnp.array(batch_X, dtype=jnp.float32)
            batch_y = jnp.array(batch_y, dtype=jnp.float32)
            state, loss = train_step(state, batch_X, batch_y, apply_fn_kwargs_train)
            epoch_losses.append(loss)

        train_loss = jnp.mean(jnp.array(epoch_losses))

        # Validation
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

    # Plot the training and validation losses
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
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
        loss_val = mse_loss(params, apply_fn, batch_X, batch_y, apply_fn_kwargs)
        losses.append(loss_val)
    return float(jnp.mean(jnp.array(losses)))


# -----------------------------------------------------------
# 6. Evaluation Function with MC Sampling
# -----------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    params,
    X_val: np.ndarray,
    y_val: np.ndarray,
    rng: jax.random.PRNGKey,
    num_samples: int,
    feature: Optional[int] = 0,  # which feature to plot on x-axis
    apply_fn_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Perform MC sampling (for dropout, etc.) for regression.
    Plot predictions vs. one specified feature, plus uncertainty.
    Returns dictionary with mean_preds, std_preds, mse, all_samples.
    """
    if apply_fn_kwargs is None:
        apply_fn_kwargs = {}

    predictions_list = []
    for _ in range(num_samples):
        rng, subkey = jax.random.split(rng)
        # apply_fn_kwargs["rngs"] = {"dropout": subkey}  # if needed
        preds = model.apply({"params": params}, X_val, **apply_fn_kwargs)  # (N,1)
        predictions_list.append(preds)

    # (num_samples, N, 1)
    preds_stacked = jnp.stack(predictions_list, axis=0).squeeze(-1)  # (num_samples, N)

    mean_preds = jnp.mean(preds_stacked, axis=0)  # (N,)
    std_preds = jnp.std(preds_stacked, axis=0)

    y_val = y_val.squeeze(-1)  # (N,)
    mse_val = jnp.mean((mean_preds - y_val) ** 2)

    # Visualization: Feature vs. Predictions
    X_feature = X_val[:, feature]  # shape: (N,)
    # Sort by X_feature for a nicer line plot
    sorted_indices = jnp.argsort(X_feature)
    sorted_X = X_feature[sorted_indices]
    sorted_mean = mean_preds[sorted_indices]
    sorted_std = std_preds[sorted_indices]
    sorted_y = y_val[sorted_indices]

    plt.figure(figsize=(8, 5))
    plt.scatter(sorted_X, sorted_y, alpha=0.6, label="True", color="black")
    plt.plot(sorted_X, sorted_mean, label="Mean Pred", color="blue")
    plt.fill_between(
        sorted_X,
        sorted_mean - sorted_std,
        sorted_mean + sorted_std,
        alpha=0.2,
        color="blue",
        label="Std. Dev.",
    )
    plt.xlabel(f"Feature {feature}")
    plt.ylabel("Target")
    plt.title("Regression Predictions with Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        "mean_predictions": mean_preds,
        "std_predictions": std_preds,
        "mse": float(mse_val),
        "all_samples": preds_stacked,
    }


# -----------------------------------------------------------
# 7. Example Test
# -----------------------------------------------------------
if __name__ == "__main__":
    # Synthetic dataset: y = 3*x + noise
    rng_np = np.random.RandomState(42)
    N = 200
    X_all = rng_np.uniform(-5, 5, size=(N, 1)).astype(np.float32)
    y_all = (3.0 * X_all + rng_np.normal(0, 2, size=(N, 1))).astype(np.float32)

    # Split
    train_size = int(0.8 * N)
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    y_train, y_val = y_all[:train_size], y_all[train_size:]

    model = MLPRegression(hidden_dim=32)
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

    # Evaluate with MC sampling
    results = evaluate_model(
        model,
        state.params,
        X_val,
        y_val,
        rng,
        num_samples=50,
        feature=0,  # we only have 1 feature anyway
        apply_fn_kwargs={},
    )
    print("Final Regression MSE:", results["mse"])
