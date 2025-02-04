import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from quantbayes.newest.base import BaseModel

# A simple multiclass classifier: a linear layer returning logits.
class SimpleMulticlassClassifier(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, key, in_features, num_classes):
        self.linear = eqx.nn.Linear(in_features, num_classes, key=key)

    def __call__(self, x):
        # Return logits (no softmax here so that we can use optax’s cross entropy loss).
        return self.linear(x)

# Multiclass classification model subclass inheriting from BaseModel.
class MulticlassClassificationModel(BaseModel):
    def __init__(self, batch_size=None, key=None):
        """
        Initialize the multiclass classification model.
        """
        super().__init__(batch_size, key)
        self.opt_state = None

    @staticmethod
    def cross_entropy_loss(model, x, y):
        """
        Compute cross-entropy loss for multiclass classification.
        Assumes that y contains integer labels and model outputs logits.
        """
        logits = jax.vmap(model)(x)  # Shape: (N, num_classes)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return jnp.mean(loss)

    # Use filter_jit and filter_value_and_grad in the training step.
    @eqx.filter_jit
    def train_step(self, model, state, X, y, key):
        @eqx.filter_value_and_grad(has_aux=False)
        def loss_fn(m):
            return MulticlassClassificationModel.cross_entropy_loss(m, X, y)
        loss, grads = loss_fn(model)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_model = eqx.apply_updates(model, updates)
        return loss, new_model, new_opt_state

    # JIT-compiled predict step that returns class labels.
    @eqx.filter_jit
    def predict_step(self, model, state, X, key):
        logits = jax.vmap(model)(X)
        return jnp.argmax(logits, axis=-1)

    def visualize(self, X_test, y_test, y_pred, feature_indices: tuple = (0, 1), title="Decision Boundary"):
        """
        Visualizes decision boundaries for multiclass classification.
        The user specifies two feature indices to visualize; non-selected features are fixed to their mean.
        """
        f1, f2 = feature_indices
        x_min, x_max = X_test[:, f1].min() - 0.5, X_test[:, f1].max() + 0.5
        y_min, y_max = X_test[:, f2].min() - 0.5, X_test[:, f2].max() + 0.5

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        num_features = X_test.shape[1]
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

        # Here we use a lambda that directly calls the model.
        # Note: In this visualization function, we assume that the variable 'model' is available.
        # In practice, you might pass the model as an argument.
        preds_grid = jax.vmap(lambda x: jnp.argmax(model(x)))(X_grid)
        preds_grid = np.array(preds_grid).reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, preds_grid, alpha=0.3, cmap=plt.cm.Paired)
        plt.scatter(np.array(X_test[:, f1]), np.array(X_test[:, f2]), c=np.array(y_test), edgecolors='k')
        plt.xlabel(f"Feature {f1}")
        plt.ylabel(f"Feature {f2}")
        plt.title(title)
        plt.show()

    def fit(self, model, X_train, y_train, X_val, y_val, num_epochs=100, lr=1e-3, patience=10):
        """
        Trains the multiclass classification model using early stopping.
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

        self.train_losses = []
        self.val_losses = []

        for epoch in range(num_epochs):
            self.key, subkey = jax.random.split(self.key)
            train_loss, model, self.opt_state = self.train_step(model, None, X_train, y_train, subkey)
            val_loss = MulticlassClassificationModel.cross_entropy_loss(model, X_val, y_val)

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
    from quantbayes.fake_data import generate_multiclass_classification_data

    # Generate data.
    df = generate_multiclass_classification_data()
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

    key = jax.random.PRNGKey(42)
    in_features = X_train.shape[1]
    num_classes = int(jnp.unique(y_train).shape[0])
    # Instantiate the simple multiclass classifier.
    model = SimpleMulticlassClassifier(key, in_features, num_classes)

    # Create an instance of our MulticlassClassificationModel.
    multiclass_model = MulticlassClassificationModel(key=key)

    # Train the model.
    trained_model = multiclass_model.fit(
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
    preds = multiclass_model.predict_step(trained_model, None, X_test, None)

    multiclass_model.model = trained_model
    multiclass_model.visualize(X_test, y_test, preds, feature_indices=(0, 1))
