import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
import optax
import matplotlib.pyplot as plt


class BaseModel(ABC):
    def __init__(self, batch_size=None, key=None):
        """
        Initialize the base model.
        Args:
            batch_size (int): Size of the batches for training. If None, no batching is used.
            key (jax.random.PRNGKey): Random key for stochastic operations.
        """
        self.batch_size = batch_size
        self.key = key if key is not None else jax.random.PRNGKey(0)
        self.train_losses = []
        self.val_losses = []
        self._compiled = False  # Ensure user compiles model before training


    def compile(self, optimizer, loss_fn):
        """
        Configures the model with optimizer and loss function before training.
        Args:
            optimizer (optax.GradientTransformation): Optimizer (e.g., optax.adam).
            loss_fn (callable): Loss function.
        """
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self._compiled = True

    @staticmethod
    def dataloader(arrays, batch_size):
        """
        Utility function to create a dataloader for batch training.
        Args:
            arrays (list of np.ndarray): List of arrays to batch.
            batch_size (int): Size of the batches.

        Yields:
            tuple: A batch of arrays.
        """
        dataset_size = arrays[0].shape[0]
        assert all(array.shape[0] == dataset_size for array in arrays), "Arrays must have the same size."
        indices = np.arange(dataset_size)
        while True:
            perm = np.random.permutation(indices)
            start = 0
            end = batch_size
            while end <= dataset_size:
                batch_perm = perm[start:end]
                yield tuple(array[batch_perm] for array in arrays)
                start = end
                end = start + batch_size

    def fit(self, model, state, X_train, y_train, X_val=None, y_val=None, epochs=100):
        """
        Training method for the model. Uses batching if batch_size is specified.
        Args:
            model (eqx.Module): Equinox model.
            state (eqx.nn.State or None): State object for BatchNorm layers.
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray or None): Validation features.
            y_val (np.ndarray or None): Validation labels.
            epochs (int): Number of epochs to train.

        Returns:
            tuple: Updated model and state.
        """
        if not self._compiled:
            raise ValueError("Model must be compiled before training. Call model.compile() first.")

        updated_model, updated_state = model, state  # Track updates

        for epoch in range(epochs):
            epoch_train_loss = []
            if self.batch_size:
                data_gen = self.dataloader([X_train, y_train], self.batch_size)
                for X_batch, y_batch in data_gen:
                    self.key, subkey = jax.random.split(self.key)
                    updated_model, updated_state, loss = self.train_step(
                        updated_model, updated_state, X_batch, y_batch, key=subkey
                    )
                    epoch_train_loss.append(loss)
            else:
                self.key, subkey = jax.random.split(self.key)
                updated_model, updated_state, loss = self.train_step(
                    updated_model, updated_state, X_train, y_train, key=subkey
                )
                epoch_train_loss.append(loss)

            self.train_losses.append(jnp.mean(jnp.array(epoch_train_loss)))

            # Validation step
            if X_val is not None and y_val is not None:
                self.key, subkey = jax.random.split(self.key)
                val_loss = self.evaluate(updated_model, updated_state, X_val, y_val, key=subkey)
                self.val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {self.train_losses[-1]:.4f}", 
                  f"Val Loss: {self.val_losses[-1]:.4f}" if X_val is not None else "")

        return updated_model, updated_state

    def evaluate(self, model, state, X, y, key):
        """Compute loss on validation/test data."""
        loss, _ = self.loss_fn(model, state, X, y, key)
        return loss

    def predict(self, model, state, X, key):
        """Runs model inference."""
        return self.predict_step(model, state, X, key)

    @property
    def training_history(self):
        """Returns the recorded training and validation losses."""
        return {"train_loss": self.train_losses, "val_loss": self.val_losses}

    @abstractmethod
    def train_step(self, model, state, X, y, key):
        """Abstract method for a single training step."""
        pass

    @abstractmethod
    def predict_step(self, model, state, X, key):
        """Abstract method for making predictions."""
        pass

    @abstractmethod
    def visualize(self, X_test, y_test, y_pred, title="Model Predictions"):
        """Abstract method for task-specific visualization."""
        pass
