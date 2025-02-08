import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib.pyplot as plt
import numpy as np


class MulticlassClassificationModel:
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

    def visualize(
        self,
        model,
        state,
        X_test,
        y_test,
        feature_indices=(0, 1),
        title="Decision Boundary",
    ):
        """
        Illustrate a 2D decision boundary by scanning across a grid for the specified feature indices.
        Expects the model to output raw logits, which are then converted to predicted classes.
        """
        f1, f2 = feature_indices
        x_min, x_max = X_test[:, f1].min() - 0.5, X_test[:, f1].max() + 0.5
        y_min, y_max = X_test[:, f2].min() - 0.5, X_test[:, f2].max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)
        )
        num_features = X_test.shape[1]
        grid_list = []
        for i in range(num_features):
            if i == f1:
                grid_list.append(xx.ravel())
            elif i == f2:
                grid_list.append(yy.ravel())
            else:
                grid_list.append(np.full(xx.ravel().shape, float(X_test[:, i].mean())))
        grid_arr = np.stack(grid_list, axis=1)
        grid_jnp = jnp.array(grid_arr)

        inf_model = eqx.tree_inference(model, value=True)

        def forward_one(x, subkey):
            logits, _ = inf_model(x, state=state, key=subkey)
            return jnp.argmax(logits, axis=-1)

        # Use a fixed key for the grid evaluation (or split if stochastic operations are used)
        grid_key = jr.PRNGKey(42)
        keys = jr.split(grid_key, grid_jnp.shape[0])
        preds_grid = jax.vmap(forward_one, in_axes=(0, 0))(grid_jnp, keys)
        preds_grid = np.array(preds_grid).reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, preds_grid, alpha=0.3, cmap=plt.cm.Paired)
        plt.scatter(
            np.array(X_test[:, f1]),
            np.array(X_test[:, f2]),
            c=np.array(y_test),
            edgecolors="k",
        )
        plt.xlabel(f"Feature {f1}")
        plt.ylabel(f"Feature {f2}")
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    import jax
    import jax.random as jr
    import jax.numpy as jnp
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss

    from quantbayes.fake_data import generate_multiclass_classification_data

    df = generate_multiclass_classification_data()
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=24, test_size=0.2
    )

    # Simple eqx model that returns (logits, state) with shape (num_classes,)
    class SimpleMulti(eqx.Module):
        w: jnp.ndarray
        b: jnp.ndarray

        def __init__(self, in_features, out_features, key):
            k1, _ = jr.split(key)
            self.w = jr.normal(k1, (out_features, in_features))
            self.b = jnp.zeros((out_features,))

        def __call__(self, x, state=None, *, key=None):
            logits = jnp.dot(self.w, x) + self.b
            return logits, state

    model = SimpleMulti(X.shape[1], 3, jr.PRNGKey(1))
    state = None

    trainer = MulticlassClassificationModel(lr=1e-2)
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

    trainer.visualize(
        model,
        state,
        jnp.array(X_test),
        np.array(y_test),
        feature_indices=(0, 1),
        title="Multiclass Classification Demo",
    )
