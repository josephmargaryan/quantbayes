import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

__all__ = [
    "spectral_penalty",
    "data_loader",
    "binary_loss",
    "multiclass_loss",
    "regression_loss",
    "train",
    "predict",
]
# -------------------------------
# Data Loading Utilities
# -------------------------------


def spectral_penalty(model):
    """
    Finds any sub‑module with .w_real, .w_imag and .alpha,
    and returns
      ∑ₖ (w_real[k]² + w_imag[k]²) * (1 + k**alpha)
    """
    penalty = 0.0
    leaves, _ = jax.tree_flatten(model)
    for leaf in leaves:
        if (
            hasattr(leaf, "w_real")
            and hasattr(leaf, "w_imag")
            and hasattr(leaf, "alpha")
        ):
            w_r = leaf.w_real
            w_i = leaf.w_imag
            α = leaf.alpha
            K = w_r.shape[0]
            k = jnp.arange(K)
            decay = 1.0 + k**α
            penalty += jnp.sum((w_r**2 + w_i**2) * decay)
    return penalty


def data_loader(X, y, batch_size, shuffle=True, key=None):
    """
    Generator that yields minibatches from X and y.

    Args:
        X: Input data (JAX array) with shape [N, ...].
        y: Target data (JAX array) with shape [N, ...].
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the indices.
        key: JAX PRNG key for shuffling. If None, no shuffling is done.

    Yields:
        Tuples (X_batch, y_batch)
    """
    n = X.shape[0]
    indices = jnp.arange(n)
    if shuffle:
        if key is None:
            raise ValueError("Shuffling requested but no key provided.")
        indices = jr.permutation(key, indices)
    for i in range(0, n, batch_size):
        batch_idx = indices[i : i + batch_size]
        yield X[batch_idx], y[batch_idx]


# -------------------------------
# Generic Loss Function & Steps
# -------------------------------


def binary_loss(model, state, x, y, key):
    """
    Binary cross-entropy loss with logits.

    This loss function computes the elementwise binary cross-entropy
    between the logits produced by the model and the binary targets.

    Expected inputs:
      - x: Input features of shape [batch_size, feature_dim].
      - y: Targets of shape [batch_size, 1] as float32 values (with values 0. or 1.).
      - key: A PRNG key for randomness, used for any stochastic operations in the model.
      - model: A callable that accepts inputs x along with a per-example key and state,
               and returns logits of shape [batch_size, 1] (or broadcastable to y) and updated state.
      - state: Optional model state (e.g., for BatchNorm), can be None if unused.

    Returns:
      - loss: A scalar representing the mean binary cross-entropy loss over the batch.
      - state: The updated state after processing the batch.
    """
    batch_size = x.shape[0]
    keys = jr.split(key, batch_size)
    batched_model = jax.vmap(
        model, in_axes=(0, 0, None), out_axes=(0, None), axis_name="batch"
    )
    logits, state = batched_model(x, keys, state)
    # The loss function expects logits and y to be float and of matching shape.
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))
    return loss, state


def multiclass_loss(model, state, x, y, key):
    """
    Softmax cross-entropy loss for multiclass classification.

    This loss function computes the categorical cross-entropy loss between the
    logits and the integer class labels.

    Expected inputs:
      - x: Input features of shape [batch_size, feature_dim].
      - y: Integer targets of shape [batch_size] (each entry is in the range 0 to num_classes-1).
      - key: A PRNG key for any stochastic operations in the model.
      - model: A callable that accepts inputs x along with a per-example key and state,
               and returns logits of shape [batch_size, num_classes] and updated state.
      - state: Optional model state (e.g., for BatchNorm), can be None if unused.

    Note:
      The loss function `optax.softmax_cross_entropy_with_integer_labels` is used,
      which expects integer labels (not one-hot). Therefore, do NOT convert y to one-hot.

    Returns:
      - loss: A vector of shape [batch_size] with the loss for each example,
              or a scalar mean loss if reduced (here we take the mean).
      - state: The updated state after processing the batch.
    """
    batch_size = x.shape[0]
    keys = jr.split(key, batch_size)
    batched_model = jax.vmap(
        model, in_axes=(0, 0, None), out_axes=(0, None), axis_name="batch"
    )
    logits, state = batched_model(x, keys, state)
    # Use integer labels directly; optax.softmax_cross_entropy_with_integer_labels expects:
    # logits: [batch_size, num_classes] and labels: [batch_size] integers.
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
    return loss, state


def regression_loss(model, state, x, y, key):
    """
    Mean Squared Error (MSE) loss for regression tasks.

    This loss function computes the mean squared error between the model's predictions
    and the continuous targets.

    Expected inputs:
      - x: Input features of shape [batch_size, feature_dim].
      - y: Continuous target values of shape [batch_size, output_dim] as float32.
           For single-output regression, output_dim should be 1.
      - key: A PRNG key for any stochastic operations in the model.
      - model: A callable that accepts inputs x along with a per-example key and state,
               and returns predictions of shape [batch_size, output_dim] and updated state.
      - state: Optional model state, can be None if unused.

    Returns:
      - loss: A scalar representing the mean squared error loss over the batch.
      - state: The updated state after processing the batch.
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
def train_step(
    model,
    state,
    opt_state,
    x,
    y,
    key,
    loss_fn,
    optimizer,
    lambda_spec: float = 0.0,
):
    """
    One backward pass through the Pytree
    """

    # wrap the base loss to include spectral regularization
    def total_loss_fn(m, s, xb, yb, k):
        loss, new_s = loss_fn(m, s, xb, yb, k)
        if lambda_spec > 0.0:
            loss = loss + lambda_spec * spectral_penalty(m)
        return loss, new_s

    grad_fn = eqx.filter_grad(total_loss_fn, has_aux=True)
    grads, new_state = grad_fn(model, state, x, y, key)
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    model = eqx.apply_updates(model, updates)
    return model, new_state, opt_state


# -------------------------------
# Evaluation Step
# -------------------------------
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
# Training Loop
# -------------------------------
def train(
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
    lambda_spec: float = 0.0,
):
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model = None
    best_state = None
    patience_counter = 0

    train_key, eval_key = jax.random.split(key)
    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        total_train_samples = 0
        train_key, loader_key = jax.random.split(train_key)
        for xb, yb in data_loader(
            X_train, y_train, batch_size, shuffle=True, key=loader_key
        ):
            train_key, subkey = jax.random.split(train_key)
            model, state, opt_state = train_step(
                model,
                state,
                opt_state,
                xb,
                yb,
                subkey,
                loss_fn,
                optimizer,
                lambda_spec=lambda_spec,
            )
            loss_val, _ = loss_fn(model, state, xb, yb, subkey)
            epoch_train_loss += loss_val * xb.shape[0]
            total_train_samples += xb.shape[0]
        epoch_train_loss /= total_train_samples

        epoch_val_loss = 0.0
        total_val_samples = 0
        for xb, yb in data_loader(
            X_val, y_val, batch_size, shuffle=False, key=eval_key
        ):
            eval_key, subkey = jax.random.split(eval_key)
            loss_val = eval_step(model, state, xb, yb, subkey, loss_fn)
            epoch_val_loss += loss_val * xb.shape[0]
            total_val_samples += xb.shape[0]
        epoch_val_loss /= total_val_samples

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == num_epochs - 1:
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


# -------------------------------
# Prediction / Inference
# -------------------------------
def predict(model, state, X, key):
    """
    Predict function for computing model outputs (logits) for input data.

    Args:
        model: Equinox model.
        state: Model state.
        X: Input data as a JAX array.
        key: PRNG key.

    Returns:
        logits: The model outputs (before sigmoid), vectorized over the batch.
    """
    # For inference, if your model uses dropout, you may wish to disable it (or use a fixed key).
    batched_inference = jax.vmap(model, in_axes=(0, None, None))
    logits, _ = batched_inference(X, key, state)
    return logits


# -------------------------------
# Main Routine (Demonstration)
# -------------------------------

if __name__ == "__main__":

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
    # Define a linear learning rate schedule that decays from 1e-3 to 1e-4 over 3000 steps.
    lr_schedule = optax.linear_schedule(
        init_value=1e-3,
        end_value=1e-4,
        transition_steps=1000,
    )

    # Create the optimizer by chaining gradient clipping and AdamW with the learning rate schedule and weight decay.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients with a global norm of 1.0
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4),
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Use our generic loss; here we use binary cross-entropy with logits.
    loss_fn = binary_loss

    # Train the model.
    best_model, best_state, train_losses, val_losses = train(
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
