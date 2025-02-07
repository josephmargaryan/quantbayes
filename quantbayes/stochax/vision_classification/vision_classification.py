# vision_classification.py
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
from quantbayes.stochax.base import BaseModel


# Define a simple vision classifier.
class SimpleVisionClassifier(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    fc: eqx.nn.Linear

    def __init__(self, key, in_channels, num_classes):
        # Split key for each layer.
        key1, key2, key3 = jax.random.split(key, 3)
        # Use SAME padding so that spatial dims are preserved.
        self.conv1 = eqx.nn.Conv2d(
            in_channels, 16, kernel_size=3, padding="SAME", key=key1
        )
        self.conv2 = eqx.nn.Conv2d(16, 32, kernel_size=3, padding="SAME", key=key2)
        # For simplicity, assume images are 32x32. After two conv layers (with SAME padding)
        # we flatten 32x32x32 features.
        self.fc = eqx.nn.Linear(32 * 32 * 32, num_classes, key=key3)

    def __call__(self, x):
        """
        x: an image of shape (H, W, C) with H=W=32 in this example.
        """
        # Convert from (H, W, C) to (C, H, W) for Conv2d.
        x = jnp.transpose(x, (2, 0, 1))
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        # Flatten features.
        x = jnp.reshape(x, (-1,))
        logits = self.fc(x)
        return logits


# Vision Classification Model subclass of BaseModel.
class VisionClassificationModel(BaseModel):
    def __init__(self, batch_size=None, key=None):
        """
        Initialize the vision classification model.
        """
        super().__init__(batch_size, key)
        self.opt_state = None

    @staticmethod
    def cross_entropy_loss(model, x, y):
        """
        Compute cross-entropy loss for vision classification.
        Args:
            x: A batch of images of shape (batch, H, W, C).
            y: Integer labels of shape (batch,).
        """
        logits = jax.vmap(model)(x)  # (batch, num_classes)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return jnp.mean(loss)

    # JIT-compiled training step.
    @eqx.filter_jit
    def train_step(self, model, state, X, y, key):
        @eqx.filter_value_and_grad(has_aux=False)
        def loss_fn(m):
            return VisionClassificationModel.cross_entropy_loss(m, X, y)

        loss, grads = loss_fn(model)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_model = eqx.apply_updates(model, updates)
        return loss, new_model, new_opt_state

    # JIT-compiled prediction step.
    @eqx.filter_jit
    def predict_step(self, model, state, X, key):
        logits = jax.vmap(model)(X)
        return jnp.argmax(logits, axis=-1)

    def visualize(
        self,
        X_test,
        y_test,
        y_pred,
        rows=3,
        cols=3,
        title="Vision Classification Results",
    ):
        """
        Visualizes a grid of sample images with the predicted and ground truth labels.
        
        Args:
            X_test: (batch, C, H, W) array of images.
            y_test: (batch,) ground truth integer labels.
            y_pred: (batch,) predicted labels.
            rows: Number of rows in the grid.
            cols: Number of columns in the grid.
            title: Title for the plot.
        """
        num_samples = rows * cols
        # Select the first num_samples from the test set.
        images = np.array(X_test[:num_samples])
        true_labels = np.array(y_test[:num_samples])
        pred_labels = np.array(y_pred[:num_samples])

        plt.figure(figsize=(cols * 3, rows * 3))
        for i in range(num_samples):
            # Get the image in (C, H, W) format.
            img = images[i]
            # Convert to (H, W, C) if needed.
            if img.ndim == 3:
                if img.shape[0] == 1:
                    img = img.squeeze(0)  # Grayscale image becomes (H, W)
                elif img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))  # RGB image
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.title(f"GT: {true_labels[i]}\nPred: {pred_labels[i]}")
            plt.axis("off")
        plt.suptitle(title)
        plt.show()


    def fit(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=100,
        lr=1e-3,
        patience=10,
    ):
        """
        Trains the vision classification model.
        Args:
            model (eqx.Module): The Equinox model to train.
            X_train: (batch, H, W, C) training images.
            y_train: (batch,) training integer labels.
            X_val: (batch, H, W, C) validation images.
            y_val: (batch,) validation labels.
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
            train_loss, model, self.opt_state = self.train_step(
                model, None, X_train, y_train, subkey
            )
            val_loss = VisionClassificationModel.cross_entropy_loss(model, X_val, y_val)

            self.train_losses.append(float(train_loss))
            self.val_losses.append(float(val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

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

    # For demonstration purposes, generate synthetic image classification data.
    # Here we generate N random images of shape (32, 32, 3) and random integer labels.
    N = 200
    H, W, C = 32, 32, 3
    num_classes = 5
    # Create synthetic images (values in [0, 1]).
    X_np = np.random.rand(N, H, W, C)
    # Create synthetic labels (integers in [0, num_classes-1]).
    y_np = np.random.randint(0, num_classes, size=(N,))

    X = jnp.array(X_np)
    y = jnp.array(y_np)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    key = jax.random.PRNGKey(42)
    # Instantiate the simple vision classifier.
    model = SimpleVisionClassifier(key, in_channels=C, num_classes=num_classes)

    # Create an instance of our VisionClassificationModel subclass.
    vision_model = VisionClassificationModel(key=key)

    # Train the model.
    trained_model = vision_model.fit(
        model,
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        num_epochs=100,
        lr=1e-3,
        patience=10,
    )

    # Get predictions on the test set using the predict_step.
    preds = vision_model.predict_step(trained_model, None, X_test, None)
    vision_model.visualize(X_test, y_test, preds, rows=3, cols=3)
