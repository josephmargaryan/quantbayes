# segmentation.py
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
from quantbayes.stochax.base import BaseModel


# Define a simple segmentation network using Conv2d.
# We assume images come in as (H, W, C) and we convert them to (C, H, W)
# for processing. The network outputs a single-channel mask (H, W) after applying
# a sigmoid activation.
class SimpleSegmentationNet(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d

    def __init__(self, key, in_channels, out_channels=1):
        key1, key2 = jax.random.split(key)
        # Use SAME padding so that output spatial dimensions match input.
        self.conv1 = eqx.nn.Conv2d(
            in_channels, 16, kernel_size=3, padding="SAME", key=key1
        )
        self.conv2 = eqx.nn.Conv2d(
            16, out_channels, kernel_size=3, padding="SAME", key=key2
        )

    def __call__(self, x):
        # x is assumed to have shape (H, W, C)
        x = jnp.transpose(x, (2, 0, 1))  # Convert to (C, H, W)
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.sigmoid(x)  # output in [0,1]
        # Squeeze the channel dimension (assumed to be 1) -> (H, W)
        x = jnp.squeeze(x, axis=0)
        return x


# Define Dice loss.
def dice_loss(pred, target, eps=1e-6):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = jnp.sum(pred * target)
    union = jnp.sum(pred) + jnp.sum(target)
    return 1.0 - (2.0 * intersection + eps) / (union + eps)


# Define binary cross-entropy loss.
def bce_loss(pred, target, eps=1e-7):
    pred = jnp.clip(pred, eps, 1 - eps)
    return -jnp.mean(target * jnp.log(pred) + (1 - target) * jnp.log(1 - pred))


# Define overall segmentation loss functions.
def segmentation_loss_dice(model, X, Y):
    preds = jax.nn.sigmoid(preds)
    preds = jax.vmap(model)(X)
    losses = jax.vmap(dice_loss)(preds, Y)
    return jnp.mean(losses)


def segmentation_loss_bce(model, X, Y):
    preds = jax.nn.sigmoid(preds)
    preds = jax.vmap(model)(X)
    losses = jax.vmap(lambda pred, target: bce_loss(pred, target))(preds, Y)
    return jnp.mean(losses)


# Define the segmentation model subclass (for binary segmentation).
# This class extends BaseModel and adds a choice of loss function.
class SegmentationModel(BaseModel):
    def __init__(self, batch_size=None, key=None, loss_type="dice"):
        """
        Initialize the segmentation model.
        Args:
            batch_size (int): Batch size for training.
            key (jax.random.PRNGKey): Random key.
            loss_type (str): Either "dice" or "bce".
        """
        super().__init__(batch_size, key)
        self.opt_state = None
        self.loss_type = loss_type

    def loss_fn(self, model, X, Y):
        """
        Selects the segmentation loss.
        Args:
            X: Input images.
            Y: Ground truth masks.
        Returns:
            Computed loss.
        """
        if self.loss_type == "dice":
            return segmentation_loss_dice(model, X, Y)
        elif self.loss_type == "bce":
            return segmentation_loss_bce(model, X, Y)
        else:
            raise ValueError("Invalid loss type. Use 'dice' or 'bce'.")

    # JIT-compiled training step.
    @eqx.filter_jit
    def train_step(self, model, state, X, Y, key):
        @eqx.filter_value_and_grad(has_aux=False)
        def loss_func(m):
            return self.loss_fn(m, X, Y)

        loss, grads = loss_func(model)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_model = eqx.apply_updates(model, updates)
        return loss, new_model, new_opt_state

    # JIT-compiled prediction step.
    @eqx.filter_jit
    def predict_step(self, model, state, X, key):
        preds = jax.vmap(model)(X)
        # Threshold predictions to produce binary masks.
        return (preds > 0.5).astype(jnp.int32)

    def visualize(self, X_test, Y_test, Y_pred, title="Segmentation Results"):
        """
        Visualizes segmentation results for one sample.
        
        Args:
            X_test: (batch, C, H, W) original images.
            Y_test: (batch, H, W) true masks.
            Y_pred: (batch, H, W) predicted masks.
            title (str): Plot title.
        """
        # For simplicity, display the first sample.
        orig = np.array(X_test[0])
        true_mask = np.array(Y_test[0])
        pred_mask = np.array(Y_pred[0])
        
        # Convert the image from (C, H, W) to (H, W, C) if necessary.
        if orig.ndim == 3:
            if orig.shape[0] == 1:
                # If grayscale, squeeze out the channel dimension.
                orig = orig.squeeze(0)
            elif orig.shape[0] == 3:
                # Convert from channels-first to channels-last.
                orig = np.transpose(orig, (1, 2, 0))
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(orig)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(true_mask, cmap="gray")
        plt.title("True Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.suptitle(title)
        plt.show()

    def fit(
        self,
        model,
        X_train,
        Y_train,
        X_val,
        Y_val,
        num_epochs=100,
        lr=1e-3,
        patience=10,
    ):
        """
        Trains the segmentation model.
        Args:
            model (eqx.Module): The Equinox model to train.
            X_train: (batch, H, W, C) training images.
            Y_train: (batch, H, W) training masks.
            X_val: (batch, H, W, C) validation images.
            Y_val: (batch, H, W) validation masks.
            num_epochs (int): Maximum number of epochs.
            lr (float): Learning rate.
            patience (int): Epochs to wait for improvement before early stopping.
        Returns:
            eqx.Module: The updated model after training.
        """
        # Initialize optimizer and its state.
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

        best_val_loss = float("inf")
        patience_counter = 0

        self.train_losses = []
        self.val_losses = []

        for epoch in range(num_epochs):
            self.key, subkey = jax.random.split(self.key)
            train_loss, model, self.opt_state = self.train_step(
                model, None, X_train, Y_train, subkey
            )
            val_loss = self.loss_fn(model, X_val, Y_val)

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

    # For demonstration, create synthetic segmentation data.
    # Let’s assume images of shape (64, 64, 3) and binary masks of shape (64, 64).
    N = 100
    H, W, C = 64, 64, 3
    X_np = np.random.rand(N, H, W, C)  # synthetic images
    # Create synthetic masks, e.g., by thresholding a grayscale conversion.
    gray = np.mean(X_np, axis=-1)
    Y_np = (gray > 0.5).astype(np.float32)  # binary mask

    X = jnp.array(X_np)
    Y = jnp.array(Y_np)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    key = jax.random.PRNGKey(42)
    in_channels = C
    # Instantiate the segmentation network.
    seg_net = SimpleSegmentationNet(key, in_channels, out_channels=1)

    # Create an instance of our SegmentationModel subclass; choose loss_type "dice" or "bce"
    seg_model = SegmentationModel(key=key, loss_type="dice")
    trained_model = seg_model.fit(
        seg_net, X_train, Y_train, X_val, Y_val, num_epochs=100, lr=1e-3, patience=10
    )

    # Get predictions on the validation set.
    preds = seg_model.predict_step(trained_model, None, X_val, None)
    seg_model.visualize(X_val, Y_val, preds)
