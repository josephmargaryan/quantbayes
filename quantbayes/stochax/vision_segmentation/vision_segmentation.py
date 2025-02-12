from typing import Tuple
import numpy as np
import optax
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import jax.numpy as jnp
import equinox as eqx


class SegmentationModel:
    """
    A training wrapper that handles:
      - applying the selected segmentation loss
      - training loop with an optimizer
      - usage of eqx.nn.State for BatchNorm
      - usage of eqx.inference_mode to toggle training vs. inference
    """

    def __init__(self, loss_type="dice", lr=1e-3):
        self.loss_type = loss_type
        self.lr = lr
        self.optimizer = optax.adam(lr)
        self.opt_state = None
        self.train_losses = []
        self.val_losses = []

    def _loss_function(self, logits, labels):
        if self.loss_type == "dice":
            # Convert logits to probabilities for dice loss.
            probs = jax.nn.sigmoid(logits)
            # Your custom dice loss function.
            def dice_loss(pred, target, eps=1e-6):
                pred = pred.reshape(-1)
                target = target.reshape(-1)
                intersection = jnp.sum(pred * target)
                union = jnp.sum(pred) + jnp.sum(target)
                return 1.0 - (2.0 * intersection + eps) / (union + eps)

            return dice_loss(probs, labels)
        elif self.loss_type == "bce":
            # Use optax's BCE loss on logits.
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
        elif self.loss_type == "focal":
            # Use optax's focal loss on logits.
            return jnp.mean(
                optax.sigmoid_focal_loss(logits, labels, alpha=0.25, gamma=2.0)
            )
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

    def _batch_forward_train(self, model, state, x, key):
        """
        Forward pass for an entire batch in "training mode" => BN updates
        """
        # Toggle model to training mode
        train_model = eqx.tree_inference(model, value=False)
        keys = jax.random.split(key, x.shape[0])

        # We'll create a function that handles a single sample:
        def _single_forward(m, s, img, key):
            out, new_s = m(img, state=s, key=key)
            return out, new_s

        # vmap across the batch dimension
        batch_out, new_state = jax.vmap(
            _single_forward,
            axis_name="batch",
            in_axes=(None, None, 0, 0),
            out_axes=(0, None),
        )(train_model, state, x, keys)

        return batch_out, new_state

    def _batch_forward_inference(self, model, state, x, key):
        """
        Forward pass for an entire batch in "inference mode" => BN does not update
        """
        inf_model = eqx.tree_inference(model, value=True)  # *important fix*

        def _single_forward(m, s, img):
            out, new_s = m(img, state=s, key=key)
            return out, new_s

        batch_out, new_state = jax.vmap(
            _single_forward,
            axis_name="batch",
            in_axes=(None, None, 0),
            out_axes=(0, None),
        )(inf_model, state, x)

        return batch_out, new_state

    def _compute_batch_loss(self, model, state, x, y, key, training=True):
        if training:
            preds, new_state = self._batch_forward_train(model, state, x, key)
        else:
            preds, new_state = self._batch_forward_inference(model, state, x, key)

        loss_vec = jax.vmap(self._loss_function)(preds, y)
        loss = jnp.mean(loss_vec)
        return loss, new_state

    @eqx.filter_jit
    def _train_step(self, model, state, x, y, key):
        def _loss_wrapper(m, s):
            loss_val, new_s = self._compute_batch_loss(m, s, x, y, key, training=True)
            return loss_val, new_s

        (loss_val, new_state), grads = eqx.filter_value_and_grad(
            _loss_wrapper, has_aux=True
        )(model, state)

        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_state, new_opt_state, loss_val

    def _eval_step(self, model, state, x, y, key):
        loss_val, _ = self._compute_batch_loss(model, state, x, y, key, training=False)
        return loss_val

    def fit(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        X_train: jnp.ndarray,
        Y_train: jnp.ndarray,
        X_val: jnp.ndarray,
        Y_val: jnp.ndarray,
        num_epochs=50,
        patience=10,
        key=jr.PRNGKey(0),
    ):
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            key, subkey = jr.split(key)
            model, state, self.opt_state, train_loss = self._train_step(
                model, state, X_train, Y_train, subkey
            )
            self.train_losses.append(float(train_loss))

            key, subkey = jr.split(key)
            val_loss = self._eval_step(model, state, X_val, Y_val, subkey)
            self.val_losses.append(float(val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        return model, state

    def predict(self, model, state, X, key=jr.PRNGKey(123)):
        """
        Predict binary masks (0 or 1) from input images X in inference mode
        """
        preds, _ = self._batch_forward_inference(model, state, X, key)
        preds = jax.nn.sigmoid(preds)
        return (preds > 0.5).astype(jnp.float32)

    def visualize(
        self, X, Y_true, Y_pred, title="Segmentation Visualization", num_samples=3
    ):

        samples = min(num_samples, X.shape[0])
        plt.figure(figsize=(5 * samples, 6))
        for i in range(samples):
            img = np.array(X[i])
            mask_true = np.array(Y_true[i])
            mask_pred = np.array(Y_pred[i])

            # If image is channel-first and has 3 channels, transpose to (H, W, C)
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
                plt.subplot(3, samples, i + 1)
                plt.imshow(img)
            else:
                plt.subplot(3, samples, i + 1)
                plt.imshow(img.squeeze(), cmap="gray")
            plt.axis("off")
            if i == 0:
                plt.title("Image")

            # Visualize the true mask (assumed to be single-channel)
            plt.subplot(3, samples, samples + i + 1)
            plt.imshow(mask_true.squeeze(), cmap="gray")
            plt.axis("off")
            if i == 0:
                plt.title("True Mask")

            # Visualize the predicted mask (assumed to be single-channel)
            plt.subplot(3, samples, 2 * samples + i + 1)
            plt.imshow(mask_pred.squeeze(), cmap="gray")
            plt.axis("off")
            if i == 0:
                plt.title("Predicted Mask")

        plt.suptitle(title)
        plt.show()


if __name__ == "__main__":
    from quantbayes.stochax.vision_segmentation import UNet

    print("=== Demo: SegmentationModel with UNet ===")
    key = jr.PRNGKey(42)

    # Suppose we create a UNet
    unet_init_key, data_key = jr.split(key, 2)
    # eqx.nn.make_with_state(...) will create (model, state) from a constructor
    model_unet, state_unet = eqx.nn.make_with_state(UNet)(
        in_channels=3, out_channels=1, base_channels=8, key=unet_init_key
    )

    # Create synthetic data for segmentation
    # We'll do a small "batch" of images  => shape (B, C, H, W)
    N = 16
    B, C, H, W = N, 3, 32, 32
    X_seg = jr.uniform(data_key, (B, C, H, W))
    # Synthetic ground-truth masks: threshold the image
    Y_seg = (X_seg.mean(axis=1, keepdims=True) > 0.5).astype(jnp.float32)

    # Split into train/val
    X_train_seg, X_val_seg = X_seg[:8], X_seg[8:]
    Y_train_seg, Y_val_seg = Y_seg[:8], Y_seg[8:]

    seg_model = SegmentationModel(loss_type="dice", lr=1e-3)
    # Fit
    model_unet, state_unet = seg_model.fit(
        model_unet,
        state_unet,
        X_train_seg,
        Y_train_seg,
        X_val_seg,
        Y_val_seg,
        num_epochs=20,
        patience=5,
        key=jr.PRNGKey(999),
    )

    # Predict
    preds_seg = seg_model.predict(model_unet, state_unet, X_val_seg)
    # Visualize
    seg_model.visualize(
        X_val_seg, Y_val_seg, preds_seg, title="Segmentation Demo", num_samples=3
    )
