# image_classification_script.py

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
# 1. CNN Classifier for Image Classification
# -----------------------------------------------------------
class CNNClassifier(nn.Module):
    num_classes: int
    # For more advanced networks, you might add batchnorm, more conv layers, etc.

    @nn.compact
    def __call__(self, x, **kwargs):
        """
        x: (batch_size, height, width, channels)
        Returns: (batch_size, num_classes) logits
        """
        # First conv
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)

        # Second conv
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # Dense
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)

        # Final classifier layer
        x = nn.Dense(features=self.num_classes)(x)
        return x


# -----------------------------------------------------------
# 2. Create Train State
# -----------------------------------------------------------
def create_train_state(
    rng: jax.random.PRNGKey,
    model: nn.Module,
    learning_rate: float,
    example_input: jnp.ndarray,
) -> train_state.TrainState:
    # model.init(...) returns a dict with various collections (params, batch_stats, etc. if used)
    params = model.init(rng, example_input)["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# -----------------------------------------------------------
# 3. Loss (Cross-Entropy) & Train Step
# -----------------------------------------------------------
def cross_entropy_loss(
    params,
    apply_fn,
    x: jnp.ndarray,
    y: jnp.ndarray,
    apply_fn_kwargs: Dict[str, Any] = None,
) -> jnp.ndarray:
    """
    y: (batch_size,) integer labels in [0, num_classes-1]
    logits: (batch_size, num_classes)
    """
    if apply_fn_kwargs is None:
        apply_fn_kwargs = {}
    logits = apply_fn({"params": params}, x, **apply_fn_kwargs)
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    return loss


@jax.jit
def train_step(
    state: train_state.TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    apply_fn_kwargs: Dict[str, Any],
) -> Tuple[train_state.TrainState, jnp.ndarray]:
    """
    Single train step:
      - compute cross-entropy
      - compute grads
      - update state
    """
    loss, grads = jax.value_and_grad(cross_entropy_loss)(
        state.params, state.apply_fn, x, y, apply_fn_kwargs
    )
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


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
# 5. Train Function
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
    Trains a CNN model for image classification.

    Args:
      model: Flax module
      X_train, y_train, X_val, y_val: datasets
      apply_fn_kwargs_train: e.g. {"deterministic": False, "rngs": {"dropout": dropout_key}}
      apply_fn_kwargs_val:   e.g. {"deterministic": True}
    """
    if apply_fn_kwargs_val is None:
        apply_fn_kwargs_val = {}

    # Example input for shape initialization
    example_input = jnp.ones((1,) + X_train.shape[1:], dtype=jnp.float32)
    state = create_train_state(rng, model, learning_rate, example_input)

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        rng, data_rng = jax.random.split(rng)
        batch_losses = []

        for batch_X, batch_y in data_generator(data_rng, X_train, y_train, batch_size):
            batch_X = jnp.array(batch_X, dtype=jnp.float32)
            batch_y = jnp.array(batch_y, dtype=jnp.int32)

            state, loss_val = train_step(state, batch_X, batch_y, apply_fn_kwargs_train)
            batch_losses.append(loss_val)

        train_loss = float(jnp.mean(jnp.array(batch_losses)))
        val_loss = evaluate_loss(
            state.params,
            state.apply_fn,
            X_val,
            y_val,
            batch_size,
            rng,
            apply_fn_kwargs_val,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % max(1, (num_epochs // 10)) == 0:
            print(
                f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Plot training and validation loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

    return state, train_losses, val_losses


def evaluate_loss(
    params,
    apply_fn,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: jax.random.PRNGKey,
    apply_fn_kwargs: Dict[str, Any],
):
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
# 6. Evaluation (Accuracy + Visualization)
# -----------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    params,
    X_test: np.ndarray,
    y_test: np.ndarray,
    rng: jax.random.PRNGKey,
    num_samples: int,
    apply_fn_kwargs: Optional[Dict[str, Any]] = None,
    num_display: int = 8,
):
    """
    Perform multiple forward passes (e.g. for dropout) to estimate uncertainty
    and visualize some predictions.
    """
    if apply_fn_kwargs is None:
        apply_fn_kwargs = {}

    def forward(params, x, **kwargs):
        return model.apply({"params": params}, x, **kwargs)

    # Do multiple stochastic forward passes if needed
    logits_list = []
    batch_size = 64
    # We'll break X_test into mini-batches for memory
    for start in range(0, X_test.shape[0], batch_size):
        end = start + batch_size
        x_batch = jnp.array(X_test[start:end], dtype=jnp.float32)

        # For MC sampling, we do num_samples forward passes on this batch
        local_logits_list = []
        for _ in range(num_samples):
            rng, subkey = jax.random.split(rng)
            local_logits_list.append(forward(params, x_batch, **apply_fn_kwargs))
        # shape = (num_samples, batch_size, num_classes)
        local_logits_stacked = jnp.stack(local_logits_list, axis=0)
        logits_list.append(local_logits_stacked)

    # (num_batches, num_samples, current_batch_size, num_classes)
    all_logits = jnp.concatenate(logits_list, axis=1)  # gather on axis=1 for batch
    # Reshape to (num_samples, N, num_classes)
    all_logits = all_logits[:, : X_test.shape[0], :]  # handle last batch if mismatch
    # Average over MC samples => shape: (N, num_classes)
    mean_logits = jnp.mean(all_logits, axis=0)
    mean_probs = jax.nn.softmax(mean_logits, axis=-1)
    predictions = jnp.argmax(mean_probs, axis=-1)  # shape (N,)

    # Compute accuracy
    accuracy = jnp.mean(predictions == jnp.array(y_test, dtype=jnp.int32))
    print(f"Test Accuracy: {accuracy:.4f}")

    # Visualize a few predictions
    plt.figure(figsize=(12, 3))
    indices = np.random.choice(X_test.shape[0], size=num_display, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(1, num_display, i + 1)
        plt.imshow(np.clip(X_test[idx], 0, 1))  # if data is in [0,1] or similar range
        pred_label = int(predictions[idx])
        true_label = int(y_test[idx])
        plt.title(f"Pred: {pred_label}\nGT: {true_label}")
        plt.axis("off")
    plt.suptitle("Sample Predictions")
    plt.show()

    return {
        "accuracy": float(accuracy),
        "predictions": np.array(predictions),
        "mean_probs": np.array(mean_probs),
    }


# -----------------------------------------------------------
# 7. Example Usage
# -----------------------------------------------------------
if __name__ == "__main__":
    # Synthetic dataset: shape (N, 32, 32, 3), labels in [0..9]
    rng_np = np.random.RandomState(0)
    N = 1000
    num_classes = 10

    X_data = rng_np.rand(N, 32, 32, 3).astype(np.float32)
    y_data = rng_np.randint(0, num_classes, size=(N,)).astype(np.int32)

    # Train/Val split
    train_size = int(0.8 * N)
    X_train, y_train = X_data[:train_size], y_data[:train_size]
    X_val, y_val = X_data[train_size:], y_data[train_size:]

    # Define model
    model = CNNClassifier(num_classes=num_classes)
    rng = jax.random.PRNGKey(42)

    # We pass dropout rng and set deterministic=False for training
    apply_fn_kwargs_train = {"deterministic": False, "rngs": {"dropout": rng}}
    apply_fn_kwargs_val = {"deterministic": True}  # e.g., no dropout active

    # Train
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-3

    print("Starting training...")
    state, train_losses, val_losses = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        rng=rng,
        apply_fn_kwargs_train=apply_fn_kwargs_train,
        apply_fn_kwargs_val=apply_fn_kwargs_val,
    )

    print("Evaluating model with MC sampling (dropout)...")
    results = evaluate_model(
        model=model,
        params=state.params,
        X_test=X_val,
        y_test=y_val,
        rng=rng,
        num_samples=10,  # multiple forward passes
        apply_fn_kwargs=apply_fn_kwargs_train,  # test with dropout to see if it changes
        num_display=8,
    )
