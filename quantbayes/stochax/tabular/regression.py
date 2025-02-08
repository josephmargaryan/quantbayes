import equinox as eqx
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

    def fit(self, model, state, X_train, y_train, X_val, y_val,
            num_epochs=100, patience=10, key=None):
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
            train_loss, model, self.opt_state, state = self._train_step(model, state, X_train, y_train, subkey)
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
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

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

    def visualize(self, X_test, y_test, y_pred, feature_index=0, title="Predictions vs Ground Truth"):
        """
        Plots predictions vs. ground truth, sorted by one feature.
        """
        X_test_np = np.array(X_test)
        y_test_np = np.array(y_test)
        y_pred_np = np.array(y_pred)

        order = np.argsort(X_test_np[:, feature_index])
        sorted_x = X_test_np[order]
        sorted_y = y_test_np[order]
        sorted_preds = y_pred_np[order]

        plt.figure(figsize=(10, 5))
        plt.scatter(sorted_x[:, feature_index], sorted_y, label="Ground Truth", alpha=0.7)
        plt.scatter(sorted_x[:, feature_index], sorted_preds, label="Predictions", alpha=0.7)
        plt.xlabel(f"Feature {feature_index}")
        plt.ylabel("Target")
        plt.legend()
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    import jax
    import jax.random as jr
    import jax.numpy as jnp
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error 

    from quantbayes.fake_data import generate_regression_data

    df = generate_regression_data()
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Simple eqx model returning (prediction, state)
    class SimpleReg(eqx.Module):
        w: jnp.ndarray
        b: jnp.ndarray
        def __init__(self, in_features, key):
            k1, _ = jr.split(key)
            self.w = jr.normal(k1, (in_features,))
            self.b = jnp.array(0.0)
        def __call__(self, x, state=None, *, key=None):
            pred = jnp.dot(x, self.w) + self.b
            return pred, state

    model = SimpleReg(X.shape[1], jr.PRNGKey(1))
    state = None

    trainer = RegressionModel(lr=1e-2)
    model, state = trainer.fit(
        model, state,
        X_train, Y_train,
        X_test, Y_test,
        num_epochs=250,
        patience=5,
        key=jr.PRNGKey(999)
    )

    preds = trainer.predict(model, state, X_test)
    print(f"MSE: {mean_squared_error(np.array(y_test), np.array(preds))}")

    trainer.visualize(X_test, Y_test, preds, feature_index=0, title="Simple Regression Demo")
