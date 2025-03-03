import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
import optax
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


class BinaryModel:
    """
    Trains a model that outputs raw logits for binary classification.
    Uses BCE loss with a sigmoid inside the loss function.
    Allows BN/Dropout if your model includes them.
    """

    def __init__(self, lr=1e-3):
        self.lr = lr
        self.optimizer = optax.adam(lr)
        self.opt_state = None
        self.train_losses = []
        self.val_losses = []
        self.key = jax.random.PRNGKey(0)

    @staticmethod
    def bce_loss_with_logits(logits, labels, eps=1e-7):
        """
        logits: (batch, ) raw outputs
        labels: (batch, ) in {0, 1}
        We'll apply sigmoid to logits.
        """
        preds = jax.nn.sigmoid(logits)
        preds = jnp.clip(preds, eps, 1 - eps)
        return -jnp.mean(labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds))

    def _compute_loss(self, model, state, X, y, key, training=True):
        """
        Single forward pass + BCE loss. We vmap over the batch dimension.
        If your model has BN or dropout, we set inference_mode accordingly.
        """
        if training:
            mod = eqx.tree_inference(model, value=False)
        else:
            mod = eqx.tree_inference(model, value=True)

        # Split the key so that each sample gets a subkey.
        keys = jr.split(key, X.shape[0])

        def forward_one(x, subkey):
            # Model should return raw logits.
            logits, new_state = mod(x, state=state, key=subkey)
            return logits

        all_logits = jax.vmap(forward_one, in_axes=(0, 0))(X, keys)  # shape [batch]
        loss_val = self.bce_loss_with_logits(all_logits, y)
        return loss_val, all_logits  # new_state is ignored for brevity

    @eqx.filter_jit
    def _train_step(self, model, state, X, y, key):
        def loss_wrapper(m):
            loss_val, _ = self._compute_loss(m, state, X, y, key, training=True)
            return loss_val

        loss_val, grads = eqx.filter_value_and_grad(loss_wrapper)(model)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_model = eqx.apply_updates(model, updates)
        return loss_val, new_model, new_opt_state, state

    def _eval_step(self, model, state, X, y, key):
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
            self.key, subkey = jax.random.split(self.key)
            train_loss, model, self.opt_state, state = self._train_step(
                model, state, X_train, y_train, subkey
            )
            self.train_losses.append(float(train_loss))

            self.key, subkey = jax.random.split(self.key)
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

        # Plot
        plt.figure()
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.legend()
        plt.title("Binary Classification Training Curve")
        plt.show()

        return model, state

    def predict(self, model, state, X, key=jr.PRNGKey(123)):
        inf_model = eqx.tree_inference(model, value=True)
        keys = jr.split(key, X.shape[0])

        def forward_one(x, subkey):
            logits, _ = inf_model(x, state=state, key=subkey)
            # Apply sigmoid to get probabilities.
            return logits

        probs = jax.vmap(forward_one, in_axes=(0, 0))(X, keys)
        return probs

    def visualize(self, model, state, X, y, key=jr.PRNGKey(0)):
        """
        Visualizes performance for a deterministic binary classification Equinox model.

        This utility computes predictions (raw logits), converts them to probabilities
        using a sigmoid, and then displays:
        - A ROC curve with AUC.
        - A calibration plot.
        - Optionally, a histogram of the predicted probabilities.

        Parameters:
            model: Equinox model returning (logits, state).
            state: Model state.
            X (jnp.ndarray): Input features.
            y (jnp.ndarray): True binary labels.
            key: JAX random key.
        """
        inf_model = jax.tree_util.tree_map(lambda x: x, model)
        keys = jr.split(key, X.shape[0])
        logits = jax.vmap(lambda x, k: inf_model(x, state=state, key=k)[0])(X, keys)
        logits = np.array(logits)
        probs = jax.nn.sigmoid(logits)
        probs = np.array(probs)

        # ROC curve & AUC.
        fpr, tpr, _ = roc_curve(y, probs)
        roc_auc = auc(fpr, tpr)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Histogram of predicted probabilities.
        axs[0].hist(probs, bins=20, color="skyblue", edgecolor="black", alpha=0.8)
        axs[0].set_xlabel("Predicted Probability")
        axs[0].set_ylabel("Frequency")
        axs[0].set_title("Predicted Probability Histogram")

        # ROC Curve.
        axs[1].plot(fpr, tpr, color="darkred", lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
        axs[1].plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
        axs[1].set_xlim([0.0, 1.0])
        axs[1].set_ylim([0.0, 1.05])
        axs[1].set_xlabel("False Positive Rate")
        axs[1].set_ylabel("True Positive Rate")
        axs[1].set_title("ROC Curve")
        axs[1].legend(loc="lower right")

        # Calibration plot.
        prob_true, prob_pred = calibration_curve(y, probs, n_bins=10)
        axs[2].plot(prob_pred, prob_true, marker="o", linewidth=1, label="Calibration")
        axs[2].plot([0, 1], [0, 1], linestyle="--", label="Ideal")
        axs[2].set_xlabel("Mean Predicted Probability")
        axs[2].set_ylabel("Fraction of Positives")
        axs[2].set_title("Calibration Plot")
        axs[2].legend()

        plt.suptitle("Binary Classification Performance")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


if __name__ == "__main__":
    import jax
    import jax.random as jr
    import jax.numpy as jnp
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss

    from quantbayes.fake_data import generate_binary_classification_data

    df = generate_binary_classification_data(n_categorical=1, n_continuous=1)
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )

    # Simple eqx model that outputs raw logits
    class SimpleBinaryModel(eqx.Module):
        fc1: eqx.nn.Linear
        fc2: eqx.nn.Linear
        fc3: eqx.nn.Linear

        def __init__(self, input_dim, hidden_dim, key):
            k1, k2, k3 = jr.split(key, 3)
            self.fc1 = eqx.nn.Linear(input_dim, hidden_dim, key=k1)
            self.fc2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k2)
            self.fc3 = eqx.nn.Linear(hidden_dim, 1, key=k3)

        def __call__(self, x, state=None, *, key=None):
            x = jax.nn.relu(self.fc1(x))
            x = jax.nn.relu(self.fc2(x))
            pred = self.fc3(x).squeeze()
            return pred, state

    model = SimpleBinaryModel(X.shape[1], 32, jr.PRNGKey(1))
    state = None  # no BN/Dropout => no eqx.nn.State

    # Create the training wrapper
    trainer = BinaryModel(lr=1e-2)
    model, state = trainer.fit(
        model,
        state,
        X_train,
        y_train,
        X_test,
        y_test,
        num_epochs=1000,
        patience=5,
        key=jr.PRNGKey(123),
    )

    # Make predictions
    preds = trainer.predict(model, state, X_test)
    probs = jax.nn.sigmoid(preds)
    loss = log_loss(np.array(y_test), np.array(probs))
    print(f"Log loss for jax model: {loss}")
    trainer.visualize(model, state, X_train, y_train, jax.random.key(35))
