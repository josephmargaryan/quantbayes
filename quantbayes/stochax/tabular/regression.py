import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import seaborn as sns
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

    def visualize(
        self,
        model,
        state,
        X: jnp.ndarray,
        y: jnp.ndarray,
        grid_points: int = 100,
        unique_threshold: int = 10,
    ):
        """
        Unified visualization for regression tasks in Equinox.

        This function inspects the input data `X` and target `y` to decide whether to
        generate single-feature plots (using either continuous or categorical PDP/ICE)
        or multi-feature plots. For each feature it automatically determines if it is
        continuous (many unique values) or categorical (few unique values), and it plots:

        - For continuous features: a Partial Dependence Plot (PDP) with a KDE overlay
            and Individual Conditional Expectation (ICE) curves.
        - For categorical features: a bar plot PDP and a box-plot ICE.

        Parameters
        ----------
        model : Equinox model
            The trained regression model.
        state : any
            The state associated with the model (if any).
        X : jnp.ndarray
            Input data of shape (n_samples, n_features).
        y : jnp.ndarray
            Target values of shape (n_samples,).
        grid_points : int, optional
            Number of grid points used for continuous feature visualizations.
        unique_threshold : int, optional
            Maximum number of unique values for a feature to be considered categorical.
        """
        # Convert to NumPy arrays for inspection and plotting.
        X_np = np.asarray(X)
        y_np = np.asarray(y).squeeze()
        n_samples, n_features = X_np.shape

        # Define a helper to call our Equinox predict method.
        def model_predict(X_input):
            # We use a fixed key for visualization so that plots are reproducible.
            key = jr.PRNGKey(0)
            preds = self.predict(model, state, jnp.array(X_input), key=key)
            return np.asarray(preds)

        # ----------- Single-Feature Case -----------
        if n_features == 1:
            # Decide if the single feature is categorical.
            unique_vals = np.unique(X_np[:, 0])
            is_categorical = len(unique_vals) < unique_threshold

            if is_categorical:
                # --- Categorical PDP ---
                baseline = np.mean(X_np, axis=0)
                pdp_values = []
                for cat in unique_vals:
                    sample = baseline.copy()
                    sample[0] = cat
                    pred = model_predict(sample[None, :])[0]
                    pdp_values.append(float(np.array(pred).squeeze()))
                plt.figure(figsize=(10, 6))
                plt.bar(unique_vals, pdp_values, alpha=0.7, capsize=5)
                plt.xlabel("Feature (Categorical)")
                plt.ylabel("Predicted Target")
                plt.title("Categorical Partial Dependence Plot (PDP)")
                plt.show()

                # --- Categorical ICE (box plot) ---
                cat_preds = {}
                for cat in unique_vals:
                    mask = X_np[:, 0] == cat
                    if np.sum(mask) > 0:
                        preds = model_predict(X_np[mask, :])
                        # If predictions are sampled multiple times, average over them.
                        if preds.ndim > 1:
                            preds = preds.mean(axis=0).squeeze()
                        cat_preds[cat] = preds
                plt.figure(figsize=(10, 6))
                box_data = [cat_preds[cat] for cat in unique_vals if cat in cat_preds]
                plt.boxplot(box_data, tick_labels=unique_vals)
                plt.xlabel("Feature (Categorical)")
                plt.ylabel("Predicted Target")
                plt.title("Categorical ICE")
                plt.show()

            else:
                # --- Continuous PDP ---
                cont_vals = X_np[:, 0]
                cont_min, cont_max = float(np.min(cont_vals)), float(np.max(cont_vals))
                cont_grid = np.linspace(cont_min, cont_max, grid_points)
                baseline = np.mean(X_np, axis=0)
                X_pdp = np.tile(baseline, (grid_points, 1))
                X_pdp[:, 0] = cont_grid
                pdp_preds = model_predict(X_pdp)
                # In case there are multiple posterior samples, average them.
                if pdp_preds.ndim > 1:
                    pdp_mean = pdp_preds.mean(axis=0).squeeze()
                else:
                    pdp_mean = pdp_preds.squeeze()

                plt.figure(figsize=(10, 6))
                sns.kdeplot(
                    x=cont_vals, y=y_np, cmap="Blues", fill=True, alpha=0.5, thresh=0.05
                )
                plt.plot(
                    cont_grid, pdp_mean, color="red", label="Mean Prediction (PDP)"
                )
                plt.xlabel("Feature (Continuous)")
                plt.ylabel("Target")
                plt.title("Continuous PDP")
                plt.legend()
                plt.show()

                # --- Continuous ICE ---
                rng = np.random.default_rng(42)
                ice_indices = rng.choice(
                    n_samples, size=min(20, n_samples), replace=False
                )
                plt.figure(figsize=(10, 6))
                for idx in ice_indices:
                    sample = X_np[idx, :].copy()
                    X_ice = np.tile(sample, (grid_points, 1))
                    X_ice[:, 0] = cont_grid
                    ice_preds = model_predict(X_ice)
                    if ice_preds.ndim > 1:
                        for curve in ice_preds:
                            plt.plot(
                                cont_grid,
                                curve,
                                color="dodgerblue",
                                linewidth=0.5,
                                alpha=0.7,
                            )
                    else:
                        plt.plot(
                            cont_grid,
                            ice_preds,
                            color="dodgerblue",
                            linewidth=0.5,
                            alpha=0.7,
                        )
                plt.xlabel("Feature (Continuous)")
                plt.ylabel("Target")
                plt.title("Continuous ICE")
                plt.show()

        # ----------- Multi-Feature Case -----------
        else:
            # Determine which features are continuous and which are categorical.
            continuous_features = []
            categorical_features = []
            for i in range(n_features):
                if len(np.unique(X_np[:, i])) < unique_threshold:
                    categorical_features.append(i)
                else:
                    continuous_features.append(i)

            # Decide on a subplot grid.
            if continuous_features and categorical_features:
                nrows = 2
            else:
                nrows = 1
            ncols = 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
            if nrows == 1:
                axes = np.expand_dims(axes, axis=0)

            # --- Continuous Feature Visualization ---
            if continuous_features:
                cont_idx = continuous_features[
                    0
                ]  # visualize the first continuous feature
                cont_vals = X_np[:, cont_idx]
                cont_min, cont_max = float(np.min(cont_vals)), float(np.max(cont_vals))
                cont_grid = np.linspace(cont_min, cont_max, grid_points)
                baseline = np.mean(X_np, axis=0)
                X_pdp = np.tile(baseline, (grid_points, 1))
                X_pdp[:, cont_idx] = cont_grid
                pdp_preds = model_predict(X_pdp)
                if pdp_preds.ndim > 1:
                    pdp_mean = pdp_preds.mean(axis=0).squeeze()
                else:
                    pdp_mean = pdp_preds.squeeze()

                ax_cont_pdp = axes[0, 0]
                sns.kdeplot(
                    x=cont_vals,
                    y=y_np,
                    ax=ax_cont_pdp,
                    cmap="Blues",
                    fill=True,
                    alpha=0.5,
                    thresh=0.05,
                )
                ax_cont_pdp.plot(
                    cont_grid, pdp_mean, color="red", label="Mean Prediction"
                )
                ax_cont_pdp.set_xlabel(f"Feature {cont_idx} (Continuous)")
                ax_cont_pdp.set_ylabel("Target")
                ax_cont_pdp.set_title("Continuous PDP")
                ax_cont_pdp.legend()

                # ICE for the continuous feature.
                rng = np.random.default_rng(42)
                ice_indices = rng.choice(
                    n_samples, size=min(20, n_samples), replace=False
                )
                ax_cont_ice = axes[0, 1]
                for idx in ice_indices:
                    sample = X_np[idx, :].copy()
                    X_ice = np.tile(sample, (grid_points, 1))
                    X_ice[:, cont_idx] = cont_grid
                    ice_preds = model_predict(X_ice)
                    if ice_preds.ndim > 1:
                        for curve in ice_preds:
                            ax_cont_ice.plot(
                                cont_grid,
                                curve,
                                color="dodgerblue",
                                linewidth=0.5,
                                alpha=0.7,
                            )
                    else:
                        ax_cont_ice.plot(
                            cont_grid,
                            ice_preds,
                            color="dodgerblue",
                            linewidth=0.5,
                            alpha=0.7,
                        )
                ax_cont_ice.set_xlabel(f"Feature {cont_idx} (Continuous)")
                ax_cont_ice.set_ylabel("Target")
                ax_cont_ice.set_title("Continuous ICE")
            else:
                # If there is no continuous feature, hide the top row.
                for j in range(ncols):
                    axes[0, j].axis("off")

            # --- Categorical Feature Visualization ---
            if categorical_features:
                cat_idx = categorical_features[
                    0
                ]  # visualize the first categorical feature
                cat_vals = X_np[:, cat_idx]
                unique_cats = np.unique(cat_vals)
                baseline = np.mean(X_np, axis=0)
                pdp_values = []
                for cat in unique_cats:
                    sample = baseline.copy()
                    sample[cat_idx] = cat
                    pred = model_predict(sample[None, :])[0]
                    pdp_values.append(float(np.array(pred).squeeze()))
                # Decide which row to plot on.
                cat_row = 1 if (continuous_features and categorical_features) else 0
                ax_cat_pdp = axes[cat_row, 0]
                ax_cat_pdp.bar(unique_cats, pdp_values, alpha=0.7, capsize=5)
                ax_cat_pdp.set_xlabel(f"Feature {cat_idx} (Categorical)")
                ax_cat_pdp.set_ylabel("Predicted Target")
                ax_cat_pdp.set_title("Categorical PDP")

                # ICE for categorical features via box plots.
                cat_predictions = {}
                for cat in unique_cats:
                    mask = cat_vals == cat
                    X_cat = X_np[mask, :]
                    if X_cat.shape[0] > 0:
                        preds_cat = model_predict(X_cat)
                        if preds_cat.ndim > 1:
                            preds_cat = preds_cat.mean(axis=0).squeeze()
                        cat_predictions[cat] = preds_cat
                ax_cat_ice = axes[cat_row, 1]
                boxplot_data = [
                    cat_predictions[cat]
                    for cat in unique_cats
                    if cat in cat_predictions
                ]
                ax_cat_ice.boxplot(boxplot_data, tick_labels=unique_cats)
                ax_cat_ice.set_xlabel(f"Feature {cat_idx} (Categorical)")
                ax_cat_ice.set_ylabel("Predicted Target")
                ax_cat_ice.set_title("Categorical ICE")
            else:
                # If there is no categorical feature, hide the bottom row.
                if nrows == 2:
                    for j in range(ncols):
                        axes[1, j].axis("off")

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    import jax
    import jax.random as jr
    import jax.numpy as jnp
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    from quantbayes.fake_data import generate_regression_data

    df = generate_regression_data(n_categorical=1, n_continuous=1)
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
    trainer.visualize(model, state, X_train, y_train)
