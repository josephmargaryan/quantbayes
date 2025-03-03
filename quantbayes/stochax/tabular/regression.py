import equinox as eqx
import pickle
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib.pyplot as plt
import numpy as np


class RegressionModel:
    """
    Simple MSE-based regression.
    If the user has a model with BN or Dropout, we toggle inference_mode.
    Otherwise it's just a no-op.

    This implementation uses a separate random key for each sample and
    supports a state variable.
    """

    def __init__(self, lr=1e-3):
        self.lr = lr
        self.optimizer = optax.adam(lr)
        self.opt_state = None
        self.train_losses = []
        self.val_losses = []
        self.key = jr.PRNGKey(0)

    def mse_loss(self, model, state, X, y, key, training=True):
        """
        Computes MSE loss over a batch.

        Expects model(...) => (prediction, new_state).
        Splits the key so that each sample gets its own subkey.
        """
        if training:
            mod = eqx.tree_inference(model, value=False)
        else:
            mod = eqx.tree_inference(model, value=True)

        # Split key into one per sample (assumes X has shape [B, ...])
        keys = jr.split(key, X.shape[0])

        def forward_one(x, label, subkey):
            pred, _ = mod(x, state=state, key=subkey)
            return (pred - label) ** 2

        mse_vals = jax.vmap(forward_one, in_axes=(0, 0, 0))(X, y, keys)
        return jnp.mean(mse_vals)

    def _compute_loss(self, model, state, X, y, key, training=True):
        """
        Computes the loss for a batch.
        """
        loss_val = self.mse_loss(model, state, X, y, key, training=training)
        return loss_val

    @eqx.filter_jit
    def _train_step(self, model, state, X, y, key):
        """
        A single training step.
        """

        def loss_wrapper(m):
            return self._compute_loss(m, state, X, y, key, training=True)

        loss_val, grads = eqx.filter_value_and_grad(loss_wrapper)(model)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_model = eqx.apply_updates(model, updates)
        return loss_val, new_model, new_opt_state, state

    def _eval_step(self, model, state, X, y, key):
        """
        A single evaluation step.
        """
        loss_val = self.mse_loss(model, state, X, y, key, training=False)
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
            # For training step: split key and pass to _train_step.
            self.key, subkey = jr.split(self.key)
            train_loss, model, self.opt_state, state = self._train_step(
                model, state, X_train, y_train, subkey
            )
            self.train_losses.append(float(train_loss))
            # For evaluation step: also split the key.
            self.key, subkey = jr.split(self.key)
            val_loss = self._eval_step(model, state, X_val, y_val, subkey)
            self.val_losses.append(float(val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Plot training curves.
        plt.figure()
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.legend()
        plt.title("Regression Training Curve")
        plt.show()

        return model, state

    def predict(self, model, state, X, key=jr.PRNGKey(123)):
        """
        Return predictions in inference mode.
        Splits the key so that each sample gets its own subkey.
        """
        inf_model = eqx.tree_inference(model, value=True)
        keys = jr.split(key, X.shape[0])

        def forward_one(x, subkey):
            pred, _ = inf_model(x, state=state, key=subkey)
            return pred

        preds = jax.vmap(forward_one, in_axes=(0, 0))(X, keys)
        return preds

    def visualize(self, model, state, X, y, key=jr.PRNGKey(0)):
        """
        Visualizes regression performance for a deterministic Equinox model.

        For a single feature, it plots the true values along with a line
        connecting the predicted values (sorted by the feature value). For multiple
        features, it plots a scatter plot of predicted vs. true values along with a
        45-degree reference line and a residual plot.

        Parameters:
            model: Equinox model with a `__call__` method that returns (prediction, state)
            state: Model state.
            X (jnp.ndarray): Input features.
            y (jnp.ndarray): True target values.
            key: JAX random key.
        """
        inf_model = jax.tree_util.tree_map(
            lambda x: x, model
        )  # For clarity; no dropout/BN in inference.
        # Split key: one key per sample
        keys = jr.split(key, X.shape[0])

        # Compute predictions via vmap.
        preds = jax.vmap(lambda x, k: inf_model(x, state=state, key=k)[0])(X, keys)
        preds = np.array(preds)
        y = np.array(y)

        if (X.ndim == 1) or (X.ndim == 2 and X.shape[1] == 1):
            # Single-feature: sort by the feature value for a smooth line.
            X_ = np.array(X).flatten()
            sorted_idx = np.argsort(X_)
            plt.figure(figsize=(10, 6))
            plt.scatter(X_, y, color="black", label="True values", alpha=0.7)
            plt.plot(
                X_[sorted_idx],
                preds.flatten()[sorted_idx],
                color="blue",
                lw=2,
                label="Predictions",
            )
            plt.xlabel("Feature")
            plt.ylabel("Target")
            plt.title("Regression Predictions (Single Feature)")
            plt.legend()
            plt.show()
        else:
            # Multi-feature: scatter plot predicted vs. true and a residual plot.
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))

            # Scatter: predicted vs. true.
            axs[0].scatter(preds, y, alpha=0.7)
            axs[0].plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)
            axs[0].set_xlabel("Predicted")
            axs[0].set_ylabel("True")
            axs[0].set_title("Predicted vs. True")

            # Residual plot.
            residuals = y - preds.flatten()
            axs[1].scatter(preds, residuals, alpha=0.7)
            axs[1].axhline(0, color="k", linestyle="--", lw=2)
            axs[1].set_xlabel("Predicted")
            axs[1].set_ylabel("Residuals")
            axs[1].set_title("Residual Plot")

            plt.tight_layout()
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
    from sklearn.metrics import mean_squared_error

    from quantbayes.fake_data import generate_regression_data

    df = generate_regression_data(n_categorical=0, n_continuous=1)
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Simple eqx model returning (prediction, state)
    class MLPRegression(eqx.Module):
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

    model = MLPRegression(X.shape[1], 32, jr.PRNGKey(1))
    state = None
    key = jax.random.key(4)
    trainer = RegressionModel(lr=1e-2)
    model, state = trainer.fit(
        model,
        state,
        X_train,
        Y_train,
        X_test,
        Y_test,
        num_epochs=250,
        patience=5,
        key=jr.PRNGKey(999),
    )

    preds = trainer.predict(model, state, X_test)
    print(f"MSE: {mean_squared_error(np.array(y_test), np.array(preds))}")
    trainer.visualize(model, state, X_train, y_train, key)

    ##### Example loading and saving
    # model, state = ...  # assume these are obtained after training
    # regression_model = RegressionModel(lr=1e-3)
    # regression_model.save("model_state.pkl", model, state)
    # regression_model, model, state = RegressionModel.load("model_state.pkl")
