import copy
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

# -------------------------------
# Data Loading Utilities
# -------------------------------


def data_loader(X, y, batch_size, shuffle=True, key=jr.PRNGKey(0)):
    """
    Generator that yields minibatches from X and y.

    Args:
        X: Input data (JAX array).
        y: Target data (JAX array).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the indices.
        key: JAX PRNG key for shuffling.

    Yields:
        Tuples (X_batch, y_batch)
    """
    n = X.shape[0]
    indices = jnp.arange(n)
    if shuffle:
        indices = jr.permutation(key, indices)
    for i in range(0, n, batch_size):
        batch_idx = indices[i : i + batch_size]
        yield X[batch_idx], y[batch_idx]


# -------------------------------
# Example Model Definition
# -------------------------------


class EQNet(eqx.Module):
    """
    Example Equinox model analogous to a TorchNet:
      BatchNorm1d(5) -> Linear(5,10) -> Dropout(0.1) -> ReLU -> Linear(10,1)

    If your model does not require stochasticity or state (e.g. no dropout or batchnorm),
    simply ignore the `key` and `state` inputs.
    """

    bn: eqx.nn.BatchNorm
    dropout: eqx.nn.Dropout
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, key):
        key1, key2, key3, key4 = jr.split(key, 4)
        self.bn = eqx.nn.BatchNorm(input_size=5, axis_name="batch")
        self.dropout = eqx.nn.Dropout(0.1)
        self.fc1 = eqx.nn.Linear(in_features=5, out_features=10, key=key1)
        self.fc2 = eqx.nn.Linear(in_features=10, out_features=1, key=key2)

    def __call__(self, x, key, state):
        # If your model does not use state, you can simply pass None and ignore it.
        x, state = self.bn(x, state)
        x = self.fc1(x)
        x = self.dropout(x, key=key)
        x = jax.nn.relu(x)
        x = self.fc2(x)
        return x, state


# -------------------------------
# Generic Loss Function & Steps
# -------------------------------


def binary_loss(model, state, x, y, key):
    """
    Binary cross-entropy loss with logits.
    Assumes y is of shape (batch, 1) and contains binary targets.
    """
    batch_size = x.shape[0]
    keys = jr.split(key, batch_size)
    batched_model = jax.vmap(
        model, in_axes=(0, 0, None), out_axes=(0, None), axis_name="batch"
    )
    logits, state = batched_model(x, keys, state)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))
    return loss, state


def multiclass_loss(model, state, x, y, key):
    """
    Softmax cross-entropy loss.
    Assumes y is of shape (batch,) with integer class labels.
    """
    batch_size = x.shape[0]
    keys = jr.split(key, batch_size)
    batched_model = jax.vmap(
        model, in_axes=(0, 0, None), out_axes=(0, None), axis_name="batch"
    )
    logits, state = batched_model(x, keys, state)
    # Convert integer labels to one-hot assuming logits.shape[-1] gives the number of classes.
    num_classes = logits.shape[-1]
    y_onehot = jax.nn.one_hot(y, num_classes)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, y_onehot))
    return loss, state


def regression_loss(model, state, x, y, key):
    """
    Mean squared error loss.
    Assumes y is of shape (batch, output_dim) and is a continuous target.
    """
    batch_size = x.shape[0]
    keys = jr.split(key, batch_size)
    batched_model = jax.vmap(
        model, in_axes=(0, 0, None), out_axes=(0, None), axis_name="batch"
    )
    preds, state = batched_model(x, keys, state)
    loss = jnp.mean((preds - y) ** 2)
    return loss, state


@eqx.filter_jit
def train_step(model, state, opt_state, x, y, key, loss_fn, optimizer):
    """
    Generic training step.

    Args:
        model: Equinox model.
        state: Model state (e.g. BatchNorm statistics); use None if unused.
        opt_state: Optimizer state.
        x, y: Batch data.
        key: PRNG key.
        loss_fn: A function with signature (model, state, x, y, key) -> (loss, state).
        optimizer: An optax optimizer.

    Returns:
        Updated (model, state, opt_state).
    """
    grad_fn = eqx.filter_grad(loss_fn, has_aux=True)
    grads, state = grad_fn(model, state, x, y, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state


@eqx.filter_jit
def eval_step(model, state, x, y, key, loss_fn):
    """
    Generic evaluation step returning the loss.

    Args:
        model: Equinox model.
        state: Model state.
        x, y: Batch data.
        key: PRNG key.
        loss_fn: Loss function.

    Returns:
        Loss value.
    """
    loss, _ = loss_fn(model, state, x, y, key)
    return loss


# -------------------------------
# Generic Training and Evaluation Loops
# -------------------------------


def train_equinox(
    model,
    state,
    opt_state,
    optimizer,
    loss_fn,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size,
    num_epochs,
    patience,
    key,
):
    """
    Generic training loop with early stopping and checkpointing.

    Args:
        model: Equinox model.
        state: Model state.
        opt_state: Optimizer state.
        optimizer: An optax optimizer.
        loss_fn: Loss function.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        batch_size: Batch size.
        num_epochs: Maximum number of epochs.
        patience: Patience for early stopping.
        key: PRNG key.

    Returns:
        best_model: The best model (in training mode).
        best_state: Corresponding best state.
        train_losses, val_losses: Lists of per-epoch losses.
    """
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model = None
    best_state = None
    patience_counter = 0

    # Split key for training and evaluation.
    train_key, eval_key = jr.split(key)
    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        total_train_samples = 0
        train_key, loader_key = jr.split(train_key)
        # Training loop
        for xb, yb in data_loader(
            X_train, y_train, batch_size, shuffle=True, key=loader_key
        ):
            train_key, subkey = jr.split(train_key)
            model, state, opt_state = train_step(
                model, state, opt_state, xb, yb, subkey, loss_fn, optimizer
            )
            loss_val, _ = loss_fn(model, state, xb, yb, subkey)
            epoch_train_loss += loss_val * xb.shape[0]
            total_train_samples += xb.shape[0]
        epoch_train_loss /= total_train_samples

        # Evaluation loop
        epoch_val_loss = 0.0
        total_val_samples = 0
        for xb, yb in data_loader(
            X_val, y_val, batch_size, shuffle=False, key=eval_key
        ):
            eval_key, subkey = jr.split(eval_key)
            loss_val = eval_step(model, state, xb, yb, subkey, loss_fn)
            epoch_val_loss += loss_val * xb.shape[0]
            total_val_samples += xb.shape[0]
        epoch_val_loss /= total_val_samples

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        print(
            f"Epoch {epoch+1:4d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = copy.deepcopy(model)
            best_state = copy.deepcopy(state)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return best_model, best_state, train_losses, val_losses


def eval_equinox(model, state, loss_fn, X_val, y_val, batch_size, key, metric_fn):
    """
    Generic evaluation loop.

    Args:
        model: Equinox model.
        state: Model state.
        loss_fn: Loss function.
        X_val, y_val: Validation data.
        batch_size: Batch size.
        key: PRNG key.
        metric_fn: A function that accepts (targets, predictions) and returns a metric value.

    Returns:
        Computed metric over the validation set.
    """
    # When in inference mode, stochastic layers are deactivated.
    dummy_key = jr.PRNGKey(0)
    all_logits = []
    all_targets = []
    batched_inference = jax.vmap(
        model, in_axes=(0, None, None), out_axes=(0, None), axis_name="batch"
    )
    for xb, yb in data_loader(X_val, y_val, batch_size, shuffle=False, key=key):
        logits, _ = batched_inference(xb, dummy_key, state)
        all_logits.append(logits)
        all_targets.append(yb)
    all_logits = jnp.concatenate(all_logits, axis=0)
    all_targets = jnp.concatenate(all_targets, axis=0)
    # For binary classification, use sigmoid activation.
    probs = jax.nn.sigmoid(all_logits)
    return metric_fn(np.array(all_targets), np.array(probs))


# -------------------------------
# Main Routine (Demonstration)
# -------------------------------

if __name__ == "__main__":
    # For demonstration, we use synthetic binary classification data.
    # In practice, you can replace this with any dataset.
    num_samples = 1000
    features = 5
    rng = np.random.RandomState(0)
    X_np = rng.randn(num_samples, features)
    # Create binary targets via a random linear process.
    true_w = rng.randn(features, 1)
    logits = X_np @ true_w
    p = 1 / (1 + np.exp(-logits))
    y_np = (rng.rand(num_samples, 1) < p).astype(np.float32)

    X = jnp.array(X_np, dtype=jnp.float32)
    y = jnp.array(y_np, dtype=jnp.float32)
    train_size = int(0.8 * X.shape[0])
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Hyperparameters.
    learning_rate = 1e-3
    num_epochs = 1000
    patience = 100
    batch_size = 800
    key = jr.PRNGKey(42)

    # Set up model, state, optimizer, and opt_state.
    model_key, _, train_key, eval_key = jr.split(key, 4)
    model, state = eqx.nn.make_with_state(EQNet)(model_key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Use our generic loss; here we use binary cross-entropy with logits.
    loss_fn = binary_loss

    # Train the model.
    best_model, best_state, train_losses, val_losses = train_equinox(
        model,
        state,
        opt_state,
        optimizer,
        loss_fn,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size,
        num_epochs,
        patience,
        train_key,
    )

    # Switch to inference mode.
    inference_model = eqx.nn.inference_mode(best_model)
    # Evaluate using log_loss as metric (you could also use mean_squared_error for regression, etc.)
    final_metric = eval_equinox(
        inference_model,
        best_state,
        loss_fn,
        X_val,
        y_val,
        batch_size,
        eval_key,
        log_loss,
    )
    print(f"Final validation log loss: {final_metric:.3f}")

    # Plot training curves.
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(1, len(train_losses) + 1), np.array(train_losses), label="Train Loss"
    )
    plt.plot(
        np.arange(1, len(val_losses) + 1), np.array(val_losses), label="Validation Loss"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()
