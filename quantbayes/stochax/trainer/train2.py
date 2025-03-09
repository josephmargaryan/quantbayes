"""
A complete Equinox training framework for neural networks.

Features:
  - Generic Trainer class to train any Equinox-based model.
  - Support for stateful operations (e.g. BatchNorm) via an extra state parameter.
  - Proper PRNGKey management for stochastic layers (e.g. Dropout).
  - Mini-batch training.
  - Flexible loss function selection (regression, binary, multiclass).
  - Simple save and load functions (using pickle).
  
To test the framework, a dummy regression model is defined using a linear layer,
a dropout layer, and a second linear layer. A synthetic regression dataset is used.
"""

import functools
import pickle
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import numpy as np

# ----------------------------
# Loss functions
# ----------------------------

def mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error for regression."""
    return jnp.mean((predictions - targets) ** 2)

def binary_cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Element-wise binary cross-entropy loss. Assumes logits for the positive class."""
    # Numerically stable implementation using log-sigmoid.
    p = jax.nn.sigmoid(logits)
    eps = 1e-7
    loss = - (labels * jnp.log(p + eps) + (1 - labels) * jnp.log(1 - p + eps))
    return jnp.mean(loss)

def softmax_cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Softmax cross entropy loss for multiclass tasks. Assumes one-hot labels."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(labels * log_probs, axis=-1)
    return jnp.mean(loss)

# ----------------------------
# Utility: PRNGKey update
# ----------------------------

def update_key(key: jax.random.PRNGKey) -> Tuple[jax.random.PRNGKey, jax.random.PRNGKey]:
    """Splits and returns an updated key and a subkey."""
    new_key, subkey = jr.split(key)
    return new_key, subkey

# ----------------------------
# Trainer class
# ----------------------------

class Trainer:
    """
    A trainer for Equinox-based neural networks.
    
    Attributes:
      model: The Equinox model (an eqx.Module).
      state: The extra state (e.g. running statistics) or None if the model is stateless.
      opt: The optax optimizer.
      opt_state: The optimizer state.
      loss_fn: The loss function.
      task: A string indicating the task ("regression", "binary", "multiclass").
      key: The current PRNGKey (for dropout etc.).
      batch_size: Batch size for training and evaluation.
    """
    def __init__(
        self,
        model: eqx.Module,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        task: str,
        state: Optional[Any] = None,
        key: jax.random.PRNGKey = jr.PRNGKey(0),
        batch_size: int = 32,
    ):
        self.model = model
        self.state = state  # For stateful layers (e.g. BatchNorm). Use None if not needed.
        self.opt = optimizer
        # We only optimize parameters (filtering arrays that are inexact)
        self.opt_state = self.opt.init(eqx.filter(self.model, eqx.is_inexact_array))
        self.loss_fn = loss_fn
        self.task = task
        self.key = key
        self.batch_size = batch_size

    def train_step(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Performs one gradient update on a batch.
        Uses a helper function `compute_loss` that accepts a PRNGKey for dropout.
        """
        def compute_loss(model, state, x, y, key):
            # If the model is stateful then its call returns (output, new_state)
            if state is None:
                predictions = model(x, key=key)
                return self.loss_fn(predictions, y)
            else:
                predictions, new_state = model(x, state, key=key)
                # Return a tuple (loss, new_state) so we can update state
                return self.loss_fn(predictions, y), new_state

        self.key, subkey = update_key(self.key)
        if self.state is None:
            # For stateless models
            grad_fn = eqx.filter_grad(compute_loss)
            grads = grad_fn(self.model, None, x, y, key=subkey)
            updates, self.opt_state = self.opt.update(grads, self.opt_state)
            self.model = eqx.apply_updates(self.model, updates)
            loss = compute_loss(self.model, None, x, y, key=subkey)
            return loss
        else:
            # For stateful models: use has_aux to return new state
            grad_fn = eqx.filter_grad(compute_loss, has_aux=True)
            grads, new_state = grad_fn(self.model, self.state, x, y, key=subkey)
            updates, self.opt_state = self.opt.update(grads, self.opt_state)
            self.model = eqx.apply_updates(self.model, updates)
            self.state = new_state
            # Compute loss again (auxiliary value not needed here)
            loss, _ = compute_loss(self.model, self.state, x, y, key=subkey)
            return loss

    def train_epoch(self, X_train: jnp.ndarray, y_train: jnp.ndarray) -> float:
        """
        Runs one epoch of training on the full training dataset (shuffled into batches).
        """
        num_samples = X_train.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        losses = []
        for start in range(0, num_samples, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            x_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            loss = self.train_step(x_batch, y_batch)
            losses.append(loss)
        return float(np.mean(losses))

    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        Evaluates the current model on the given data (using a fixed key and inference mode).
        For dropout and BatchNorm we switch to inference mode.
        """
        # Switch to inference mode
        model_inf = eqx.nn.inference_mode(self.model)
        dummy_key = jr.PRNGKey(0)  # dummy key; dropout is inactive in inference mode
        def eval_step(x, y):
            if self.state is None:
                predictions = model_inf(x, key=dummy_key)
            else:
                predictions, _ = model_inf(x, self.state, key=dummy_key)
            return self.loss_fn(predictions, y)
        losses = []
        num_samples = X.shape[0]
        for start in range(0, num_samples, self.batch_size):
            x_batch = X[start:start + self.batch_size]
            y_batch = y[start:start + self.batch_size]
            loss = eval_step(x_batch, y_batch)
            losses.append(loss)
        return float(np.mean(losses))

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Runs inference on input X.
        For stateful models, the state is not updated.
        """
        model_inf = eqx.nn.inference_mode(self.model)
        dummy_key = jr.PRNGKey(0)
        if self.state is None:
            preds = jax.vmap(lambda x: model_inf(x, key=dummy_key))(X)
        else:
            preds, _ = jax.vmap(lambda x: model_inf(x, self.state, key=dummy_key))(X)
        return preds

    def save(self, path: str) -> None:
        """Saves the model and state (if any) to a file using pickle."""
        with open(path, "wb") as f:
            pickle.dump((self.model, self.state), f)

    def load(self, path: str) -> None:
        """Loads the model and state (if any) from a file."""
        with open(path, "rb") as f:
            self.model, self.state = pickle.load(f)

# ----------------------------
# Test example: a simple regression model with dropout
# ----------------------------

# Define a simple Equinox model with a linear layer, dropout and a final linear layer.
class SimpleModel(eqx.Module):
    linear1: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    linear2: eqx.nn.Linear

    def __init__(self, key: jax.random.PRNGKey):
        # Split the key for each layer
        key1, key2, key3 = jr.split(key, 3)
        self.linear1 = eqx.nn.Linear(in_features=1, out_features=10, key=key1)
        # Dropout with probability 0.1 during training
        self.dropout = eqx.nn.Dropout(p=0.1, inference=False)
        self.linear2 = eqx.nn.Linear(in_features=10, out_features=1, key=key2)

    def __call__(self, x: jnp.ndarray, state: Any = None, *, key: jax.random.PRNGKey) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Any]]:
        # For this stateless model, we require a key (for dropout)
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.dropout(x, key=key)
        x = self.linear2(x)
        return x  # stateless: no extra state is returned

if __name__ == "__main__":
    # Set a seed and create a key
    key = jr.PRNGKey(42)

    # Create synthetic regression data: y = 3*x + 2 + noise
    N = 100
    X_train = jnp.linspace(-1, 1, N).reshape(-1, 1)
    # Use a fixed noise key for reproducibility
    key, noise_key = jr.split(key)
    y_train = 3 * X_train + 2 + 0.1 * jr.normal(noise_key, (N, 1))

    # Instantiate the model and (for this stateless example) state is None.
    model = SimpleModel(key)
    state = None

    # Create an optimizer (Adam with learning rate 1e-3)
    optimizer = optax.adam(1e-3)

    # Choose the regression loss (MSE)
    loss_fn = mse_loss

    # Create a Trainer with a chosen batch size.
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        task="regression",
        state=state,
        key=key,
        batch_size=16,
    )

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(X_train, y_train)
        val_loss = trainer.evaluate(X_train, y_train)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Run predictions on the training data
    preds = trainer.predict(X_train)
    print("Sample predictions:", np.array(preds)[:5])

    # Save the model to disk and then load it back
    save_path = "equinox_model.pkl"
    trainer.save(save_path)
    # For demonstration, create a new Trainer instance and load the saved model.
    new_trainer = Trainer(
        model=None,  # Will be replaced by load()
        optimizer=optimizer,
        loss_fn=loss_fn,
        task="regression",
        state=None,
        key=key,
        batch_size=16,
    )
    new_trainer.load(save_path)
    new_preds = new_trainer.predict(X_train)
    print("Predictions from loaded model:", np.array(new_preds)[:5])
