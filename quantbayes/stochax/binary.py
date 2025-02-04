import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from quantbayes.newest.base import BaseModel

# A simple binary classifier: a linear layer with sigmoid activation.
class SimpleBinaryClassifier(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, key, in_features, out_features=1):
        self.linear = eqx.nn.Linear(in_features, out_features, key=key)

    def __call__(self, x):
        # Compute logits then apply sigmoid.
        logits = self.linear(x)
        return jax.nn.sigmoid(jnp.squeeze(logits))

# Binary classification model subclass inheriting from BaseModel.
class BinaryClassificationModel(BaseModel):
    def __init__(self, batch_size=None, key=None):
        """
        Initialize the binary classification model.
        """
        super().__init__(batch_size, key)
        self.opt_state = None  # To be set during training

    @staticmethod
    def binary_cross_entropy_loss(model, x, y):
        """
        Compute binary cross-entropy loss for a batch.
        """
        preds = jax.vmap(model)(x)
        epsilon = 1e-7
        preds = jnp.clip(preds, epsilon, 1 - epsilon)
        return -jnp.mean(y * jnp.log(preds) + (1 - y) * jnp.log(1 - preds))

    # Use filter_jit and filter_value_and_grad for training.
    @eqx.filter_jit
    def train_step(self, model, state, X, y, key):
        @eqx.filter_value_and_grad(has_aux=False)
        def loss_fn(m):
            return BinaryClassificationModel.binary_cross_entropy_loss(m, X, y)
        loss, grads = loss_fn(model)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_model = eqx.apply_updates(model, updates)
        return loss, new_model, new_opt_state

    # JIT-compiled predict step.
    @eqx.filter_jit
    def predict_step(self, model, state, X, key):
        preds = jax.vmap(model)(X)
        # Return class labels based on a 0.5 threshold.
        return (preds > 0.5).astype(jnp.int32)

    def visualize(self, X_test, y_test, y_pred, feature_indices: tuple = (0, 1), title="Decision Boundary"):
        """
        Visualizes decision boundaries for binary classification.
        The user specifies which two features (by index) to use for the x and y axes.
        For features not visualized, their mean value is used.
        """
        f1, f2 = feature_indices
        x_min, x_max = X_test[:, f1].min() - 0.5, X_test[:, f1].max() + 0.5
        y_min, y_max = X_test[:, f2].min() - 0.5, X_test[:, f2].max() + 0.5

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        num_features = X_test.shape[1]
        # Build grid input where features not visualized are fixed to their mean.
        X_grid = []
        for i in range(num_features):
            if i == f1:
                X_grid.append(xx.ravel())
            elif i == f2:
                X_grid.append(yy.ravel())
            else:
                X_grid.append(np.full(xx.ravel().shape, float(X_test[:, i].mean())))
        X_grid = np.stack(X_grid, axis=1)
        X_grid = jnp.array(X_grid)

        # Use the model's forward pass (wrapped with sigmoid) to get predicted probabilities.
        # Here we assume that the model, when called, outputs the logit.
        # We apply jax.nn.sigmoid to convert logits to probabilities, then threshold.
        preds_grid = jax.vmap(lambda x: (jax.nn.sigmoid(model(x)) > 0.5).astype(jnp.int32))(X_grid)
        preds_grid = np.array(preds_grid).reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, preds_grid, alpha=0.3, levels=[-0.5, 0.5, 1.5], cmap=plt.cm.Paired)
        plt.scatter(np.array(X_test[:, f1]), np.array(X_test[:, f2]), c=np.array(y_test), edgecolors='k')
        plt.xlabel(f"Feature {f1}")
        plt.ylabel(f"Feature {f2}")
        plt.title(title)
        plt.show()

    def fit(self, model, X_train, y_train, X_val, y_val, num_epochs=100, lr=1e-3, patience=10):
        """
        Trains the binary classification model using early stopping. Initializes the optimizer,
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
        # Initialize optimizer and its state.
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
            val_loss = BinaryClassificationModel.binary_cross_entropy_loss(model, X_val, y_val)

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

        # Plot training and validation loss curves.
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
    from quantbayes.fake_data import generate_binary_classification_data

    # Generate data.
    df = generate_binary_classification_data()
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

    key = jax.random.PRNGKey(42)
    in_features = X_train.shape[1]
    # Instantiate the simple binary classifier.
    model = SimpleBinaryClassifier(key, in_features)

    # Create an instance of our BinaryClassificationModel.
    binary_model = BinaryClassificationModel(key=key)

    # Train the model.
    trained_model = binary_model.fit(
        model,
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        num_epochs=1000,
        lr=1e-3,
        patience=10
    )

    # Make predictions on the test set.
    preds = jax.vmap(trained_model)(X_test)
    preds = binary_model.predict_step(trained_model, None, X_test, None)
    # Save the trained model so it can be used in visualization.
    binary_model.model = trained_model
    binary_model.visualize(X_test, y_test, preds, feature_indices=(0, 1))
