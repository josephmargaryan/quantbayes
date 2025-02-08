import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np


class BinaryClassificationModel:
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
        Illustrate a 2D decision boundary by scanning across a grid for feature_indices.
        Expects the model to output raw logits, which we convert to a predicted class.
        """
        f1, f2 = feature_indices
        x_min, x_max = X_test[:, f1].min() - 0.5, X_test[:, f1].max() + 0.5
        y_min, y_max = X_test[:, f2].min() - 0.5, X_test[:, f2].max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)
        )
        num_features = X_test.shape[1]

        # Build grid input
        X_grid = []
        for i in range(num_features):
            if i == f1:
                X_grid.append(xx.ravel())
            elif i == f2:
                X_grid.append(yy.ravel())
            else:
                X_grid.append(np.full(xx.ravel().shape, float(X_test[:, i].mean())))
        X_grid = np.stack(X_grid, axis=1)
        X_grid_jnp = jnp.array(X_grid)

        # Evaluate model in inference mode
        inf_model = eqx.tree_inference(model, value=True)

        def forward_one(x):
            logits, _ = inf_model(x, state=state, key=None)
            p = jax.nn.sigmoid(logits)
            return (p > 0.5).astype(jnp.int32)

        preds_grid = jax.vmap(forward_one)(X_grid_jnp)
        preds_grid = np.array(preds_grid).reshape(xx.shape)

        # Plot decision boundary
        plt.figure(figsize=(8, 6))
        plt.contourf(
            xx, yy, preds_grid, alpha=0.3, levels=[-0.5, 0.5, 1.5], cmap=plt.cm.Paired
        )
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

    from quantbayes.fake_data import generate_binary_classification_data

    df = generate_binary_classification_data()
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )

    # Simple eqx model that outputs raw logits
    class SimpleBinaryModel(eqx.Module):
        w: jnp.ndarray
        b: jnp.ndarray

        def __init__(self, in_features, key):
            k1, _ = jr.split(key)
            self.w = jr.normal(k1, (in_features,))
            self.b = jnp.array(0.0)

        def __call__(self, x, state=None, *, key=None):
            # Return raw logits (scalar) => shape ()
            logits = jnp.dot(self.w, x) + self.b
            return logits, state

    model = SimpleBinaryModel(X.shape[1], jr.PRNGKey(1))
    state = None  # no BN/Dropout => no eqx.nn.State

    # Create the training wrapper
    trainer = BinaryClassificationModel(lr=1e-2)
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

    # Visualize 2D decision boundary
    trainer.visualize(
        model, state, X_test, y_test, feature_indices=(0, 1), title="Binary Class Demo"
    )

    ############
    # Test scikit learn
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression().fit(np.array(X_train), np.array(y_train))
    preds = model.predict_proba(X_test)
    print(f"Log loss for scikit learns model: {log_loss(np.array(y_test), preds)}")
