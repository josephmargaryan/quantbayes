import equinox as eqx
import pickle
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


class MulticlassModel:
    """
    Trains a model that outputs raw logits for multiclass classification.
    Uses cross-entropy with integer labels.
    Allows BN/Dropout if your model has them (via eqx.inference_mode).
    """

    def __init__(self, lr=1e-3):
        self.lr = lr
        self.optimizer = optax.adam(lr)
        self.opt_state = None
        self.train_losses = []
        self.val_losses = []
        self.key = jr.PRNGKey(0)

    @staticmethod
    def cross_entropy_loss(logits, labels):
        """
        logits: shape (batch, num_classes)
        labels: shape (batch,) with integer class indices
        """
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    def _compute_loss(self, model, state, X, y, key, training=True):
        """
        Single forward pass + cross-entropy.
        We split the provided key so that each sample gets its own subkey.
        Expects model(...) => (logits, new_state).
        """
        if training:
            mod = eqx.tree_inference(model, value=False)
        else:
            mod = eqx.tree_inference(model, value=True)

        # Split the key into one key per sample
        keys = jr.split(key, X.shape[0])

        def forward_one(x, subkey):
            logits, new_state = mod(x, state=state, key=subkey)
            return logits

        # Use vmap to apply forward_one over the batch
        all_logits = jax.vmap(forward_one, in_axes=(0, 0))(
            X, keys
        )  # shape [batch, num_classes]
        loss_val = self.cross_entropy_loss(all_logits, y)
        return loss_val, all_logits

    @eqx.filter_jit
    def _train_step(self, model, state, X, y, key):
        """
        A single training step that uses the provided key.
        """

        def loss_wrapper(m):
            loss_val, _ = self._compute_loss(m, state, X, y, key, training=True)
            return loss_val

        loss_val, grads = eqx.filter_value_and_grad(loss_wrapper)(model)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_model = eqx.apply_updates(model, updates)
        return loss_val, new_model, new_opt_state, state

    def _eval_step(self, model, state, X, y, key):
        """
        A single evaluation step that uses the provided key.
        """
        loss_val, _ = self._compute_loss(model, state, X, y, key, training=False)
        return loss_val

    def fit(
        self,
        model,
        state,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=100,
        patience=10,
        key=None,
    ):
        if key is not None:
            self.key = key
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        best_val_loss = float("inf")
        patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        for epoch in range(num_epochs):
            self.key, subkey = jr.split(self.key)
            train_loss, model, self.opt_state, state = self._train_step(
                model, state, X_train, y_train, subkey
            )
            self.train_losses.append(float(train_loss))
            self.key, subkey = jr.split(self.key)
            val_loss = self._eval_step(model, state, X_val, y_val, subkey)
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

        # Plot training curves
        plt.figure()
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.legend()
        plt.title("Multiclass Classification Training Curve")
        plt.show()

        return model, state

    def predict(self, model, state, X, key=jr.PRNGKey(123)):
        """
        Return predicted classes from raw logits.
        Splits the key so that each sample gets its own subkey.
        """
        inf_model = eqx.tree_inference(model, value=True)
        keys = jr.split(key, X.shape[0])

        def forward_one(x, subkey):
            logits, _ = inf_model(x, state=state, key=subkey)
            return logits

        preds = jax.vmap(forward_one, in_axes=(0, 0))(X, keys)
        return preds

    def visualize(self, model, state, X, y, key=jr.PRNGKey(0)):
        """
        Visualizes performance for a deterministic multiclass classification Equinox model.

        The function computes the raw logits, applies softmax to get predicted probabilities,
        determines the predicted class for each sample, and then shows:
        - A confusion matrix.
        - A bar chart of average predicted probabilities per class.

        Parameters:
            model: Equinox model returning (logits, state).
            state: Model state.
            X (jnp.ndarray): Input features.
            y (jnp.ndarray): True class labels (as integers).
            key: JAX random key.
        """
        inf_model = jax.tree_util.tree_map(lambda x: x, model)
        keys = jr.split(key, X.shape[0])
        logits = jax.vmap(lambda x, k: inf_model(x, state=state, key=k)[0])(X, keys)
        logits = np.array(logits)
        probs = jax.nn.softmax(logits, axis=-1)
        probs = np.array(probs)
        pred_classes = np.argmax(probs, axis=-1)

        # Confusion matrix.
        cm = confusion_matrix(y, pred_classes)

        # Plot confusion matrix and average predicted probabilities.
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Confusion matrix heatmap.
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[0])
        axs[0].set_xlabel("Predicted")
        axs[0].set_ylabel("True")
        axs[0].set_title("Confusion Matrix")

        # Bar chart of average predicted probabilities per class.
        avg_probs = np.mean(probs, axis=0)
        num_classes = probs.shape[1]
        axs[1].bar(
            range(num_classes), avg_probs, color="mediumseagreen", edgecolor="black"
        )
        axs[1].set_xlabel("Class")
        axs[1].set_ylabel("Average Predicted Probability")
        axs[1].set_title("Average Predicted Probabilities")
        axs[1].set_xticks(range(num_classes))
        axs[1].set_xticklabels([f"Class {i}" for i in range(num_classes)])

        plt.suptitle("Multiclass Classification Performance")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def save(self, path, model, state):
        """Save the model, state, and optimizer state to disk."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "state": state,
                    "opt_state": self.opt_state,
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses,
                },
                f,
            )
        print(f"Saved model to {path}")

    @classmethod
    def load(cls, path):
        """Load the model, state, and optimizer state from disk.

        Returns a tuple: (wrapper, model, state)
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Reconstruct the wrapper.
        wrapper = cls()
        wrapper.opt_state = data["opt_state"]
        wrapper.train_losses = data.get("train_losses", [])
        wrapper.val_losses = data.get("val_losses", [])
        print(f"Loaded model from {path}")
        return wrapper, data["model"], data["state"]


if __name__ == "__main__":
    import jax
    import jax.random as jr
    import jax.numpy as jnp
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss

    from quantbayes.fake_data import generate_multiclass_classification_data

    df = generate_multiclass_classification_data(n_categorical=1, n_continuous=1)
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=24, test_size=0.2
    )

    # Simple eqx model that returns (logits, state) with shape (num_classes,)
    class SimpleMulti(eqx.Module):
        fc1: eqx.nn.Linear
        fc2: eqx.nn.Linear
        fc3: eqx.nn.Linear

        def __init__(self, input_dim, hidden_dim, key):
            k1, k2, k3 = jr.split(key, 3)
            self.fc1 = eqx.nn.Linear(input_dim, hidden_dim, key=k1)
            self.fc2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k2)
            self.fc3 = eqx.nn.Linear(hidden_dim, 3, key=k3)

        def __call__(self, x, state=None, *, key=None):
            x = jax.nn.relu(self.fc1(x))
            x = jax.nn.relu(self.fc2(x))
            pred = self.fc3(x)
            return pred, state

    model = SimpleMulti(X.shape[1], 64, jr.PRNGKey(1))
    state = None

    trainer = MulticlassModel(lr=1e-2)
    model, state = trainer.fit(
        model,
        state,
        jnp.array(X_train),
        jnp.array(y_train),
        jnp.array(X_test),
        jnp.array(y_test),
        num_epochs=1000,
        patience=5,
        key=jr.PRNGKey(999),
    )

    logits = trainer.predict(model, state, jnp.array(X_test))
    probs = jax.nn.softmax(logits, axis=-1)
    print(f"Log loss: {log_loss(np.array(y_test), np.array(probs))}")
    trainer.visualize(model, state, X_train, y_train, jax.random.key(35))
