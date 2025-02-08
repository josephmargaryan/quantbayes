from typing import Tuple
import numpy as np
import optax 
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import jax.numpy as jnp 
import equinox as eqx

class VisionClassificationModel:
    """
    Vision Classification
    """

    def __init__(self, lr=1e-3, problem_type="multiclass"):
        self.lr = lr
        self.problem_type = problem_type
        self.optimizer = optax.adam(lr)
        self.opt_state = None
        self.train_losses = []
        self.val_losses = []

    def _loss_function(self, logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        if self.problem_type == "multiclass":
            return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        elif self.problem_type == "binary":
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
        else:
            raise ValueError("Unsupported problem type")

    def _batch_forward_train(
        self, model, state, x, key
    ) -> Tuple[jnp.ndarray, eqx.nn.State]:
        """
        Single forward pass in training mode, returning logits for each sample in the batch.
        x shape: (B, C, H, W), or you adapt as needed.
        """
        train_model = eqx.tree_inference(model, value=False)
        keys = jax.random.split(key, x.shape[0])

        def _single_forward(m, s, inp, key):
            out, new_s = m(inp, state=s, key=key)  # returns (logits, new_state)
            return out, new_s

        logits, new_state = jax.vmap(
            _single_forward,
            axis_name="batch",
            in_axes=(None, None, 0, 0),
            out_axes=(0, None),
        )(train_model, state, x, keys)

        return logits, new_state

    def _batch_forward_inference(
        self, model, state, x, key
    ) -> Tuple[jnp.ndarray, eqx.nn.State]:
        """
        Single forward pass in inference mode, returning logits.
        """
        inf_model = eqx.tree_inference(model, value=True)

        def _single_forward(m, s, inp):
            out, new_s = m(inp, state=s, key=key)
            return out, new_s

        logits, new_state = jax.vmap(
            _single_forward,
            axis_name="batch",
            in_axes=(None, None, 0),
            out_axes=(0, None),
        )(inf_model, state, x)

        return logits, new_state

    def _compute_batch_loss(self, model, state, x, y, key, training=True):
        if training:
            logits, new_state = self._batch_forward_train(model, state, x, key)
        else:
            logits, new_state = self._batch_forward_inference(model, state, x, key)

        loss_val = self._loss_function(logits, y)
        return loss_val, new_state, logits


    @eqx.filter_jit
    def _train_step(self, model, state, x, y, key):
        def _loss_wrapper(m, s):
            loss_val, new_s, _ = self._compute_batch_loss(m, s, x, y, key, training=True)
            return loss_val, new_s

        (loss_val, new_state), grads = eqx.filter_value_and_grad(
            _loss_wrapper, has_aux=True
        )(model, state)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_state, new_opt_state, loss_val

    def _eval_step(self, model, state, x, y, key):
        loss_val, _, _ = self._compute_batch_loss(model, state, x, y, key, training=False)
        return loss_val

    def fit(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        X_val: jnp.ndarray,
        y_val: jnp.ndarray,
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
                model, state, X_train, y_train, subkey
            )
            self.train_losses.append(float(train_loss))

            key, subkey = jr.split(key)
            val_loss = self._eval_step(model, state, X_val, y_val, subkey)
            self.val_losses.append(float(val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        return model, state

    def predict(self, model, state, X, key=jr.PRNGKey(123)):
        logits, _ = self._batch_forward_inference(model, state, X, key)
        
        if self.problem_type == "multiclass":
            # No need to apply softmax; argmax on logits gives the predicted class.
            return jnp.argmax(logits, axis=-1)
        elif self.problem_type == "binary":
            # For binary, convert logits to probabilities and threshold.
            probs = jax.nn.sigmoid(logits)
            return (probs > 0.5).astype(jnp.float32)
        else:
            raise ValueError("Unsupported problem type")


    def visualize(self, X, y_true, y_pred, rows=2, cols=3, title="Classification Results"):
        """
        Visualize a grid of images + (true_label, pred_label).
        X is assumed (B, C, H, W) or (B, H, W, C). Adapt as needed.
        """
        num_samples = rows * cols
        plt.figure(figsize=(cols * 3, rows * 3))
        for i in range(num_samples):
            if i >= X.shape[0]:
                break
            img = np.array(X[i])
            # If channel-first (C,H,W) for e.g. C=3 => reorder
            if img.ndim == 3 and img.shape[0] <= 4:
                img = np.transpose(img, (1, 2, 0))  # (H,W,C)

            true_label = y_true[i]
            pred_label = y_pred[i]

            plt.subplot(rows, cols, i + 1)
            plt.imshow(img.astype(np.float32))
            plt.title(f"GT: {true_label}, Pred: {pred_label}")
            plt.axis("off")

        plt.suptitle(title)
        plt.show()