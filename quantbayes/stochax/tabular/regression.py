import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from quantbayes.stochax.base import BaseModel

class SimpleRegressor(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, key, in_features, out_features=1):
        self.linear = eqx.nn.Linear(in_features, out_features, key=key)

    def __call__(self, x):
        out = self.linear(x)
        return jnp.squeeze(out)

class RegressionModel(BaseModel):
    def __init__(self, batch_size=None, key=None):
        """
        Initialize the regression model.
        """
        super().__init__(batch_size, key)
        self.opt_state = None  # This will be set when training begins

    @staticmethod
    def mse_loss(model, x, y):
        """
        Compute the mean squared error over a dataset.
        """
        preds = jax.vmap(model)(x)
        return jnp.mean((preds - y) ** 2)

    # The train_step is now decorated with filter_jit so that it is compiled,
    # and we use filter_value_and_grad within it for efficient gradient computation.
    @eqx.filter_jit
    def train_step(self, model, state, X, y, key):
        # Define a loss function and compute its value and gradient.
        # The filter_value_and_grad decorator automatically filters out nondifferentiable parts.
        @eqx.filter_value_and_grad(has_aux=False)
        def loss_fn(m):
            return RegressionModel.mse_loss(m, X, y)
        loss, grads = loss_fn(model)
        # Use the stored optimizer to update parameters.
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_model = eqx.apply_updates(model, updates)
        return loss, new_model, new_opt_state

    # JIT-compiled predict_step for fast inference.
    @eqx.filter_jit
    def predict_step(self, model, state, X, key):
        return jax.vmap(model)(X)

    def visualize(self, X_test, y_test, y_pred, feature_index=0, title="Predictions vs Ground Truth"):
        """
        Visualizes predictions versus the ground truth by sorting according to one selected feature.
        """
        order = jnp.argsort(X_test[:, feature_index])
        sorted_x = X_test[order]
        sorted_y = y_test[order]
        sorted_preds = y_pred[order]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(np.array(sorted_x[:, feature_index]), np.array(sorted_y), label="Ground Truth", alpha=0.7)
        plt.scatter(np.array(sorted_x[:, feature_index]), np.array(sorted_preds), label="Predictions", alpha=0.7)
        plt.xlabel(f"Feature {feature_index}")
        plt.ylabel("Target")
        plt.legend()
        plt.title(title)
        plt.show()

    def fit(self, model, X_train, y_train, X_val, y_val, num_epochs=100, lr=1e-3, patience=10):
        """
        Trains the regression model using early stopping. This method initializes the optimizer,
        runs the training loop, tracks losses, and plots the training curves.

        Args:
            model (eqx.Module): The Equinox model to train.
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.
            num_epochs (int): Maximum number of epochs.
            lr (float): Learning rate.
            patience (int): Number of epochs to wait for improvement before early stopping.
        
        Returns:
            eqx.Module: The updated model after training.
        """
        # Initialize the optimizer and its state.
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

        best_val_loss = float("inf")
        patience_counter = 0

        # Reset training history
        self.train_losses = []
        self.val_losses = []

        for epoch in range(num_epochs):
            # For this example we assume full-batch training.
            # (You could also iterate over mini-batches if self.batch_size is specified.)
            self.key, subkey = jax.random.split(self.key)
            train_loss, model, self.opt_state = self.train_step(model, None, X_train, y_train, subkey)
            val_loss = RegressionModel.mse_loss(model, X_val, y_val)

            self.train_losses.append(float(train_loss))
            self.val_losses.append(float(val_loss))

            # Early stopping logic
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

        # Plot training and validation loss curves
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
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    from quantbayes.fake_data import generate_regression_data
    
    df = generate_regression_data() 
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

    key = jax.random.PRNGKey(42)
    in_features = X_train.shape[1]
    model = SimpleRegressor(key, in_features)

    # Create an instance of our RegressionModel subclass.
    reg_model = RegressionModel(key=key)

    # Train the model using the regression fit method.
    trained_model = reg_model.fit(
        model,
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        num_epochs=1000,
        lr=1e-3,
        patience=10
    )

    # Use the trained model to make predictions on the test set.
    preds = jax.vmap(trained_model)(X_test)
    preds = reg_model.predict_step(trained_model, None, X_test, None)

    # Visualize predictions vs. ground truth.
    reg_model.visualize(X_test, y_test, preds, feature_index=0)
