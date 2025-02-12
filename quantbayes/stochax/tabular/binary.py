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
        X_test: jnp.ndarray,
        y_test: jnp.ndarray,
        feature_indices: tuple = (0, 1),
        unique_threshold: int = 10,
        resolution: int = 200,
        title: str = "Binary Decision Boundary",
    ):
        """
        Visualize the decision boundary for a binary classifier.
        Automatically checks if the two selected features are continuous or categorical.

        Parameters
        ----------
        model : Equinox model
            A trained binary classifier whose forward pass returns raw logits.
        state : any
            The model's state.
        X_test : jnp.ndarray
            Input features of shape (n_samples, n_features).
        y_test : jnp.ndarray
            Binary labels (n_samples,).
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
            # Split the key for each sample.
            keys = jr.split(key, X_input.shape[0])

            def forward_one(x, subkey):
                logits, _ = inf_model(x, state=state, key=subkey)
                return logits

            logits = jax.vmap(forward_one, in_axes=(0, 0))(X_input, keys)
            # Apply sigmoid to get probabilities.
            probs = jax.nn.sigmoid(logits)
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
            # For binary classification, threshold probability at 0.5.
            class_preds = (np.array(probs) > 0.5).astype(np.int32).reshape(xx.shape)
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
            # Let the categorical feature be the one with fewer unique values.
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
                class_preds = (np.array(probs) > 0.5).astype(np.int32)
                ax.plot(cont_grid, class_preds, label="Decision boundary")
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
