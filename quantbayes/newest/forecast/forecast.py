# forecasting.py
import math
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
from quantbayes.newest.base import BaseModel

# Define a simple forecasting network using a GRU cell.
class ForecastNet(eqx.Module):
    cell: eqx.nn.GRUCell
    fc: eqx.nn.Linear

    def __init__(self, input_dim: int, hidden_size: int, key):
        # Split key for the GRU cell and the final linear layer.
        cell_key, fc_key = jax.random.split(key)
        # Create a GRU cell. The GRU cell processes a single time step.
        self.cell = eqx.nn.GRUCell(input_size=input_dim, hidden_size=hidden_size, key=cell_key)
        # The final linear layer maps the hidden state to one prediction.
        self.fc = eqx.nn.Linear(hidden_size, 1, key=fc_key)

    def __call__(self, x):
        """
        x: a JAX array of shape [seq_len, input_dim] representing one time series.
        Returns: a prediction for the last time step, shape [1].
        """
        # Initialize hidden state as zeros.
        init_state = jnp.zeros(self.cell.hidden_size)
        # Define a scan function that applies the GRU cell to each time step.
        scan_fn = lambda carry, x: (self.cell(x, carry), None)
        final_state, _ = jax.lax.scan(scan_fn, init_state, x)
        # Apply the final linear layer to the last hidden state.
        return self.fc(final_state)

# Forecasting model subclass based on BaseModel.
class ForecastingModel(BaseModel):
    def __init__(self, batch_size=None, key=None):
        """
        Initialize the forecasting model.
        """
        super().__init__(batch_size, key)
        self.opt_state = None

    @staticmethod
    def mse_loss(model, x, y):
        """
        Compute mean squared error over a batch.
          x: JAX array with shape [batch, seq_len, input_dim]
          y: JAX array with shape [batch, 1]
        """
        preds = jax.vmap(model)(x)
        return jnp.mean((preds - y) ** 2)

    def train_step(self, model, state, X, y, key):
        """
        A single training step.
        """
        def loss_fn(m):
            return ForecastingModel.mse_loss(m, X, y)
        loss, grads = jax.value_and_grad(loss_fn)(model)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_model = eqx.apply_updates(model, updates)
        return loss, new_model, new_opt_state

    def predict_step(self, model, state, X, key):
        """
        Vectorize model prediction over the batch.
        """
        return jax.vmap(model)(X)

    def visualize(self, X_test, y_test, y_pred, title="Forecasting Predictions vs Ground Truth"):
        """
        Visualizes forecasting results by plotting predicted vs. ground truth values.
        Here we assume y_test and y_pred are of shape [batch, 1] (i.e. one prediction per series).
        """
        # Convert to 1D arrays for plotting.
        y_test = np.array(y_test).flatten()
        y_pred = np.array(y_pred).flatten()

        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label="Ground Truth", marker="o")
        plt.plot(y_pred, label="Predictions", marker="x")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.title(title)
        plt.legend()
        plt.show()

    def fit(self, model, X_train, y_train, X_val, y_val, num_epochs=100, lr=1e-3, patience=10):
        """
        Train the forecasting model using early stopping.
          X_train: [batch, seq_len, input_dim]
          y_train: [batch, 1]
        """
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

        best_val_loss = float("inf")
        patience_counter = 0

        # Reset training history.
        self.train_losses = []
        self.val_losses = []

        for epoch in range(num_epochs):
            self.key, subkey = jax.random.split(self.key)
            train_loss, model, self.opt_state = self.train_step(model, None, X_train, y_train, subkey)
            val_loss = ForecastingModel.mse_loss(model, X_val, y_val)

            self.train_losses.append(float(train_loss))
            self.val_losses.append(float(val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Plot the training and validation loss curves.
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curve")
        plt.legend()
        plt.show()

        return model

if __name__ == "__main__":

    from quantbayes.fake_data import create_synthetic_time_series

    seq_len = 10  # sequence length
    input_dim = 1

    X_train, X_test, y_train, y_test = create_synthetic_time_series()

    key = jax.random.PRNGKey(42)
    # Instantiate the forecasting network.
    forecast_model = ForecastNet(input_dim=input_dim, hidden_size=16, key=key)

    # Create an instance of our ForecastingModel subclass.
    forecasting_model = ForecastingModel(key=key)

    # Train the model.
    trained_model = forecasting_model.fit(
        forecast_model,
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        num_epochs=200,
        lr=1e-3,
        patience=20
    )

    # Make predictions on the test set.
    preds = jax.vmap(trained_model)(X_test)
    preds = forecasting_model.predict_step(trained_model, None, X_test, None)

    # Visualize predictions vs. ground truth.
    forecasting_model.visualize(X_test, y_test, preds)
