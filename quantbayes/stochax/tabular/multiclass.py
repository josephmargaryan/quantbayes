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
        X_test: jnp.ndarray,
        y_test: jnp.ndarray,
        feature_indices: tuple = (0, 1),
        unique_threshold: int = 10,
        resolution: int = 200,
        title: str = "Multiclass Decision Boundary",
    ):
        """
        Visualize the decision boundary for a multiclass classifier.
        Automatically checks if the two selected features are continuous or categorical.

        Parameters
        ----------
        model : Equinox model
            A trained multiclass classifier that outputs raw logits.
        state : any
            The model's state.
        X_test : jnp.ndarray
            Input features of shape (n_samples, n_features).
        y_test : jnp.ndarray
            True class labels (n_samples,).
        feature_indices : tuple, optional
            The two feature indices to visualize.
        unique_threshold : int, optional
            Maximum number of unique values for a feature to be considered categorical.
        resolution : int, optional
            Grid resolution for continuous features.
        title : str, optional
            Plot title.
        """
        # Convert inputs to NumPy for plotting and inspection.
        X_np = np.array(X_test)
        y_np = np.array(y_test)
        f1, f2 = feature_indices
        unique_f1 = np.unique(X_np[:, f1])
        unique_f2 = np.unique(X_np[:, f2])
        is_f1_cat = len(unique_f1) < unique_threshold
        is_f2_cat = len(unique_f2) < unique_threshold

        # Create an inference version of the model.
        inf_model = eqx.tree_inference(model, value=True)

        def predict_batch(X_input, key):
            keys = jr.split(key, X_input.shape[0])

            def forward_one(x, subkey):
                logits, _ = inf_model(x, state=state, key=subkey)
                return logits

            logits = jax.vmap(forward_one, in_axes=(0, 0))(X_input, keys)
            # Apply softmax to get class probabilities.
            probs = jax.nn.softmax(logits, axis=-1)
            return probs

        # --- Case 1: Both features continuous ---
        if not is_f1_cat and not is_f2_cat:
            x_min, x_max = X_np[:, f1].min() - 0.5, X_np[:, f1].max() + 0.5
            y_min, y_max = X_np[:, f2].min() - 0.5, X_np[:, f2].max() + 0.5
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, resolution),
                np.linspace(y_min, y_max, resolution),
            )
            n_features = X_np.shape[1]
            grid_list = []
            for i in range(n_features):
                if i == f1:
                    grid_list.append(xx.ravel())
                elif i == f2:
                    grid_list.append(yy.ravel())
                else:
                    grid_list.append(np.full(xx.ravel().shape, X_np[:, i].mean()))
            grid_arr = np.stack(grid_list, axis=1)
            grid_key = jr.PRNGKey(42)
            probs = predict_batch(jnp.array(grid_arr), grid_key)
            # For multiclass, take the class with maximum probability.
            class_preds = np.argmax(np.array(probs), axis=-1).reshape(xx.shape)
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, class_preds, alpha=0.3, cmap=plt.cm.Paired)
            plt.scatter(
                X_np[:, f1], X_np[:, f2], c=y_np, edgecolor="k", cmap=plt.cm.Paired
            )
            plt.xlabel(f"Feature {f1}")
            plt.ylabel(f"Feature {f2}")
            plt.title(title)
            plt.grid(True)
            plt.show()

        # --- Case 2: One feature categorical, one continuous ---
        elif (is_f1_cat and not is_f2_cat) or (not is_f1_cat and is_f2_cat):
            if is_f1_cat:
                cat_idx, cont_idx = f1, f2
            else:
                cat_idx, cont_idx = f2, f1
            unique_cats = np.unique(X_np[:, cat_idx])
            num_cats = len(unique_cats)
            fig, axes = plt.subplots(
                1, num_cats, figsize=(5 * num_cats, 5), squeeze=False
            )
            for j, cat in enumerate(unique_cats):
                ax = axes[0, j]
                mask = X_np[:, cat_idx] == cat
                cont_vals = X_np[:, cont_idx]
                c_min, c_max = cont_vals.min() - 0.5, cont_vals.max() + 0.5
                cont_grid = np.linspace(c_min, c_max, resolution)
                n_features = X_np.shape[1]
                grid_list = []
                for i in range(n_features):
                    if i == cont_idx:
                        grid_list.append(cont_grid)
                    elif i == cat_idx:
                        grid_list.append(np.full(cont_grid.shape, cat))
                    else:
                        grid_list.append(np.full(cont_grid.shape, X_np[:, i].mean()))
                grid_arr = np.stack(grid_list, axis=1)
                grid_key = jr.PRNGKey(42)
                probs = predict_batch(jnp.array(grid_arr), grid_key)
                class_preds = np.argmax(np.array(probs), axis=-1)
                ax.plot(cont_grid, class_preds, label="Decision Boundary")
                ax.scatter(
                    X_np[mask, cont_idx],
                    y_np[mask],
                    c="k",
                    edgecolors="w",
                    label="Data",
                )
                ax.set_title(f"Feature {cat_idx} = {cat}")
                ax.set_xlabel(f"Feature {cont_idx}")
                ax.set_ylabel("Predicted class")
                ax.legend()
            plt.suptitle(title)
            plt.show()

        # --- Case 3: Both features categorical ---
        else:
            plt.figure(figsize=(8, 6))
            plt.scatter(
                X_np[:, f1], X_np[:, f2], c=y_np, cmap=plt.cm.Paired, edgecolors="k"
            )
            plt.xlabel(f"Feature {f1}")
            plt.ylabel(f"Feature {f2}")
            plt.title(title + " (Both features categorical)")
            plt.grid(True)
            plt.show()


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
