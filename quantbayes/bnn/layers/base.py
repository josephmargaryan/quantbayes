from typing import Optional, Tuple
import jax
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from numpyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal
from numpyro.contrib.einstein import SteinVI, RBFKernel, MixtureGuidePredictive
from numpyro.optim import Adam, Adagrad, SGD
import pickle
import numpy as np

from quantbayes.stochax.utils.fft_direct_prior import (
    visualize_circulant_kernel,
    plot_fft_spectrum,
)


class Module:
    """
    Base class for probabilistic models with support for modularized inference,
    prediction, and visualization.
    """

    def __init__(self, method="nuts", task_type="regression"):
        """
        Initialize the BaseModel.

        :param method: str
            Inference method ('nuts', 'svi', or 'steinvi').
        :param task_type: str
            Task type ('regression', 'binary', or 'multiclass').
        """
        self.method = method
        self.task_type = task_type
        self.inference = None
        self.samples = None
        self.params = None
        self.losses = None
        self.stein_result = None

    def compile(self, **kwargs):
        """
        Compiles the model for the specified inference method.
        """
        if self.method == "nuts":
            kernel = NUTS(self.__call__)
            self.inference = MCMC(
                kernel,
                num_warmup=kwargs.get("num_warmup", 500),
                num_samples=kwargs.get("num_samples", 1000),
                num_chains=kwargs.get("num_chains", 1),
            )
        elif self.method == "svi":
            guide = AutoNormal(self.__call__)
            optimizer = Adam(kwargs.get("learning_rate", 0.01))
            self.inference = SVI(self.__call__, guide, optimizer, loss=Trace_ELBO())
        elif self.method == "steinvi":
            guide = AutoNormal(self.__call__)
            self.inference = SteinVI(
                model=self.__call__,
                guide=guide,
                optim=Adagrad(kwargs.get("learning_rate", 0.01)),
                kernel_fn=RBFKernel(),
                repulsion_temperature=kwargs.get("repulsion_temperature", 1.0),
                num_stein_particles=kwargs.get("num_stein_particles", 10),
                num_elbo_particles=kwargs.get("num_elbo_particles", 1),
            )
        else:
            raise ValueError(f"Unknown inference method: {self.method}")

    def fit(self, X_train, y_train, rng_key, **kwargs):
        """
        Fits the model using the selected inference method.
        """
        if isinstance(self.inference, MCMC):
            self.inference.run(rng_key, X_train, y_train)
            self.samples = self.inference.get_samples()
        elif isinstance(self.inference, SVI):
            svi_state = self.inference.init(rng_key, X_train, y_train)
            self.losses = []
            for step in range(kwargs.get("num_steps", 1000)):
                svi_state, loss = self.inference.update(svi_state, X_train, y_train)
                self.losses.append(loss)
                if step % 100 == 0:
                    print(f"Step {step}, Loss: {loss:.4f}")
            self.params = self.inference.get_params(svi_state)
        elif isinstance(self.inference, SteinVI):
            self.stein_result = self.inference.run(
                rng_key,
                kwargs.get("num_steps", 1000),
                X_train,
                y_train,
                progress_bar=kwargs.get("progress_bar", True),
            )
        else:
            raise ValueError("Inference method not initialized. Call `compile` first.")

    def predict(self, X_test, rng_key, posterior="logits", num_samples=None):
        """
        Generates predictions using the specified posterior.

        :param X_test: jnp.ndarray
            Test data for prediction.
        :param rng_key: jax.random.PRNGKey
            Random key for sampling predictions.
        :param posterior: str
            Name of the posterior to sample from (default: 'logits').
        :param num_samples: int
            Number of posterior samples to use (only for SVI/SteinVI).
        """
        if isinstance(self.inference, MCMC):
            predictive = Predictive(self.__call__, posterior_samples=self.samples)
        elif isinstance(self.inference, SVI):
            assert (
                self.params is not None
            ), "SVI parameters are not available. Ensure `fit` was called."
            predictive = Predictive(
                self.__call__,
                guide=self.inference.guide,
                params=self.params,
                num_samples=num_samples or 100,
            )
        elif isinstance(self.inference, SteinVI):
            assert (
                self.stein_result is not None
            ), "SteinVI results are not available. Ensure `fit` was called."
            params = self.inference.get_params(self.stein_result.state)
            predictive = MixtureGuidePredictive(
                model=self.__call__,
                guide=self.inference.guide,
                params=params,
                num_samples=num_samples or 100,
                guide_sites=self.inference.guide_sites,
            )
        else:
            raise ValueError("Inference method not initialized. Call `compile` first.")

        preds = predictive(rng_key, X_test)
        if posterior not in preds:
            raise ValueError(f"The posterior '{posterior}' is not available. ")
        return preds.get(posterior, preds)

    @property
    def get_samples(self):
        if isinstance(self.inference, MCMC):
            if self.samples is None:
                raise ValueError(
                    "MCMC samples are not available. Ensure `fit` was called."
                )
            return self.samples
        raise ValueError("MCMC is not the selected inference method.")

    @property
    def get_params(self):
        if isinstance(self.inference, SVI):
            if self.params is None:
                raise ValueError(
                    "SVI parameters are not available. Ensure `fit` was called."
                )
            return self.params
        raise ValueError("SVI is not the selected inference method.")

    @property
    def get_losses(self):
        if isinstance(self.inference, SVI):
            if self.losses is None:
                raise ValueError(
                    "SVI losses are not available. Ensure `fit` was called."
                )
            return self.losses
        raise ValueError("SVI is not the selected inference method.")

    @property
    def get_stein_result(self):
        if isinstance(self.inference, SteinVI):
            if self.stein_result is None:
                raise ValueError(
                    "SteinVI results are not available. Ensure `fit` was called."
                )
            return self.stein_result
        raise ValueError("SteinVI is not the selected inference method.")

    def visualize(
        self,
        X,
        y=None,
        num_classes=None,
        posterior="logits",
        features=(0, 1),
        resolution=100,
        feature_index=None,
    ):
        """
        Visualizes model outputs based on the task type.

        For regression, if X contains a single feature, it calls the single-feature
        visualization. If X contains multiple features, it calls the multi-feature
        visualization function.
        """
        if self.task_type == "multiclass":
            if num_classes is None:
                raise ValueError(
                    "num_classes must be provided for multiclass visualization."
                )
            self._visualize_multiclass(
                X, y, num_classes, features, resolution, posterior
            )
        elif self.task_type == "binary":
            self._visualize_binary(X, y, features, resolution, posterior)
        elif self.task_type == "regression":
            self._visualize_regression(
                X, y, posterior=posterior, grid_points=100, unique_threshold=10
            )
        elif self.task_type == "image_classification":
            self._visualize_image_classification(X, y, num_classes, posterior)
        elif self.task_type == "image_segmentation":
            self._visualize_image_segmentation(X, y, posterior)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def visualize_image_classification(
        self,
        X,
        y,
        posterior: str = "logits",
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
    ):
        """
        Visualizes predictions for image classification tasks in a general way.

        Args:
            X (jnp.ndarray): Input images (batch_size, channels, height, width)
                            or preprocessed patches.
            y (jnp.ndarray): True labels (optional, for comparison).
            num_classes (int): Number of output classes.
            predict_fn (callable): A function that takes (X, rng_key, posterior) and returns
                                posterior samples. For instance, this can be `self.predict`.
            posterior (str): The posterior to use (default: "logits").
            image_size (Optional[int]): If X is not 4D, a fallback image size for a placeholder.
            in_channels (Optional[int]): If X is not 4D, a fallback number of channels for a placeholder.
        """
        # Get predictions from the provided function.
        pred_samples = self.predict(X, jax.random.PRNGKey(0), posterior=posterior)
        # Compute mean predictions using softmax.
        mean_preds = jax.nn.softmax(
            pred_samples.mean(axis=0), axis=-1
        )  # [batch_size, num_classes]
        predicted_classes = jnp.argmax(mean_preds, axis=-1)  # [batch_size]

        # Set up the visualization grid.
        batch_size = X.shape[0]
        rows = int(jnp.ceil(batch_size / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(12, rows * 3))

        # Ensure axes is 2D.
        if rows == 1:
            axes = axes[np.newaxis, :]

        for i, ax in enumerate(axes.flatten()):
            if i >= batch_size:
                ax.axis("off")
                continue

            # If input is image data (4D), assume it is in the format (batch, channels, height, width)
            if X.ndim == 4:
                # Transpose to (height, width, channels) for visualization.
                img = X[i].transpose(1, 2, 0)
            else:
                # If not, try to use provided image_size and in_channels to create a placeholder.
                if image_size is not None and in_channels is not None:
                    img = jnp.zeros((image_size, image_size, in_channels))
                else:
                    raise ValueError(
                        "Input X is not 4D and no fallback image_size/in_channels provided."
                    )

            true_label = y[i] if y is not None else None
            pred_label = predicted_classes[i]
            pred_probs = mean_preds[i]

            # Display the image.
            ax.imshow(
                img.squeeze(),
                cmap="gray" if (X.ndim == 4 and X.shape[1] == 1) else None,
            )
            ax.axis("off")

            # Title with prediction and probability.
            title = f"Pred: {pred_label} ({pred_probs[pred_label]:.2f})"
            if y is not None:
                title += f"\nTrue: {true_label}"
            ax.set_title(title, fontsize=10)

        plt.tight_layout()
        plt.show()

    def _visualize_multiclass(
        self,
        X,
        y,
        num_classes: int,
        features: Tuple[int, int] = (0, 1),
        resolution: int = 100,
        posterior: str = "logits",
        unique_threshold: int = 10,
    ):
        """
        Unified visualization for multiclass classification.

        If both selected features are continuous, a 2D grid is used to plot the decision boundary
        along with uncertainty (entropy). If one of the features is categorical, a subplot is generated
        for each category value; if both are categorical, a scatter plot is shown.

        Parameters
        ----------
        X : jnp.ndarray
            Input data, shape (n_samples, n_features).
        y : jnp.ndarray
            True labels.
        num_classes : int
            Number of classes.
        features : tuple of int, optional
            Indices of the two features to visualize.
        resolution : int, optional
            Grid resolution.
        posterior : str, optional
            Which posterior mode to use for predictions.
        unique_threshold : int, optional
            Maximum number of unique values for a feature to be treated as categorical.
        """
        # Convert to NumPy for inspection and plotting.
        X_np = np.array(X)
        y_np = np.array(y)
        f1, f2 = features
        unique_f1 = np.unique(X_np[:, f1])
        unique_f2 = np.unique(X_np[:, f2])
        is_f1_cat = len(unique_f1) < unique_threshold
        is_f2_cat = len(unique_f2) < unique_threshold

        # CASE 1: Both features continuous.
        if not is_f1_cat and not is_f2_cat:
            x_min, x_max = X_np[:, f1].min() - 1, X_np[:, f1].max() + 1
            y_min, y_max = X_np[:, f2].min() - 1, X_np[:, f2].max() + 1
            xx, yy = jnp.meshgrid(
                jnp.linspace(x_min, x_max, resolution),
                jnp.linspace(y_min, y_max, resolution),
            )
            # Build a full-grid input: for nonvisualized features, use their mean.
            grid_points = jnp.c_[xx.ravel(), yy.ravel()]
            grid_points_full = jnp.zeros((grid_points.shape[0], X.shape[1]))
            grid_points_full = grid_points_full.at[:, f1].set(grid_points[:, 0])
            grid_points_full = grid_points_full.at[:, f2].set(grid_points[:, 1])
            # Predict on grid points.
            pred_samples = self.predict(
                grid_points_full, jax.random.PRNGKey(0), posterior=posterior
            )  # shape: [num_samples, resolution^2, num_classes]
            grid_mean_predictions = jax.nn.softmax(pred_samples.mean(axis=0), axis=-1)
            grid_classes = jnp.argmax(grid_mean_predictions, axis=-1).reshape(xx.shape)
            grid_uncertainty = -jnp.sum(
                grid_mean_predictions * jnp.log(grid_mean_predictions + 1e-9), axis=-1
            ).reshape(xx.shape)

            plt.figure(figsize=(10, 6))
            plt.contourf(
                np.array(xx),
                np.array(yy),
                np.array(grid_classes),
                levels=num_classes,
                cmap=plt.cm.RdYlBu,
                alpha=0.6,
            )
            plt.colorbar(label="Predicted Class")
            plt.contourf(
                np.array(xx),
                np.array(yy),
                np.array(grid_uncertainty),
                levels=20,
                cmap="gray",
                alpha=0.4,
            )
            plt.colorbar(label="Uncertainty (Entropy)")
            plt.scatter(
                X_np[:, f1],
                X_np[:, f2],
                c=y_np,
                edgecolor="k",
                cmap=plt.cm.RdYlBu,
                s=20,
            )
            plt.title("Multiclass Decision Boundaries with Uncertainty")
            plt.xlabel(f"Feature {f1 + 1}")
            plt.ylabel(f"Feature {f2 + 1}")
            plt.grid(True)
            plt.show()

        # CASE 2: One feature categorical, one continuous.
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
                c_min, c_max = cont_vals.min() - 1, cont_vals.max() + 1
                cont_grid = jnp.linspace(c_min, c_max, resolution)
                n_features = X_np.shape[1]
                # For each value in the continuous grid, fix the categorical feature.
                grid_list = []
                for i in range(n_features):
                    if i == cont_idx:
                        grid_list.append(cont_grid)
                    elif i == cat_idx:
                        grid_list.append(jnp.full(cont_grid.shape, cat))
                    else:
                        grid_list.append(
                            jnp.full(cont_grid.shape, jnp.array(X_np[:, i]).mean())
                        )
                grid_arr = jnp.stack(
                    grid_list, axis=1
                )  # shape: (resolution, n_features)
                pred_samples = self.predict(
                    grid_arr, jax.random.PRNGKey(42), posterior=posterior
                )  # shape: [num_samples, resolution, num_classes]
                probs = jax.nn.softmax(pred_samples.mean(axis=0), axis=-1)
                class_preds = jnp.argmax(probs, axis=-1)
                ax.plot(
                    np.array(cont_grid),
                    np.array(class_preds),
                    label="Decision Boundary",
                )
                ax.scatter(
                    X_np[mask, cont_idx], y_np[mask], c="k", edgecolor="w", label="Data"
                )
                ax.set_title(f"Feature {cat_idx} = {cat}")
                ax.set_xlabel(f"Feature {cont_idx}")
                ax.set_ylabel("Predicted class")
                ax.legend()
            plt.suptitle("Multiclass Decision Boundary with Categorical Feature")
            plt.show()

        # CASE 3: Both features categorical.
        else:
            plt.figure(figsize=(8, 6))
            plt.scatter(
                X_np[:, f1], X_np[:, f2], c=y_np, cmap=plt.cm.RdYlBu, edgecolor="k"
            )
            plt.xlabel(f"Feature {f1}")
            plt.ylabel(f"Feature {f2}")
            plt.title("Multiclass Decision Boundary (Both features categorical)")
            plt.grid(True)
            plt.show()

    def _visualize_binary(
        self,
        X,
        y,
        features: Tuple[int, int] = (0, 1),
        resolution: int = 100,
        posterior: str = "logits",
        unique_threshold: int = 10,
    ):
        """
        Unified visualization for binary classification.

        If both selected features are continuous, a 2D grid is used to display the decision boundary
        (probability map) and uncertainty (via binary entropy). If one of the features is categorical,
        a subplot is generated for each category; if both are categorical, a scatter plot is shown.

        Parameters
        ----------
        X : jnp.ndarray
            Input data, shape (n_samples, n_features).
        y : jnp.ndarray
            Binary labels.
        features : tuple of int, optional
            The two features to visualize.
        resolution : int, optional
            Grid resolution.
        posterior : str, optional
            Which posterior mode to use for predictions.
        unique_threshold : int, optional
            Maximum number of unique values for a feature to be considered categorical.
        """
        X_np = np.array(X)
        y_np = np.array(y)
        f1, f2 = features
        unique_f1 = np.unique(X_np[:, f1])
        unique_f2 = np.unique(X_np[:, f2])
        is_f1_cat = len(unique_f1) < unique_threshold
        is_f2_cat = len(unique_f2) < unique_threshold

        # CASE 1: Both features continuous.
        if not is_f1_cat and not is_f2_cat:
            x_min, x_max = X_np[:, f1].min() - 1, X_np[:, f1].max() + 1
            y_min, y_max = X_np[:, f2].min() - 1, X_np[:, f2].max() + 1
            xx, yy = jnp.meshgrid(
                jnp.linspace(x_min, x_max, resolution),
                jnp.linspace(y_min, y_max, resolution),
            )
            grid_points = jnp.c_[xx.ravel(), yy.ravel()]
            grid_points_full = jnp.zeros((grid_points.shape[0], X.shape[1]))
            grid_points_full = grid_points_full.at[:, f1].set(grid_points[:, 0])
            grid_points_full = grid_points_full.at[:, f2].set(grid_points[:, 1])
            pred_samples = self.predict(
                grid_points_full, jax.random.PRNGKey(0), posterior=posterior
            )  # shape: [num_samples, grid_points]
            grid_mean_predictions = jax.nn.sigmoid(pred_samples.mean(axis=0))
            grid_uncertainty = -(
                grid_mean_predictions * jnp.log(grid_mean_predictions + 1e-9)
                + (1 - grid_mean_predictions)
                * jnp.log(1 - grid_mean_predictions + 1e-9)
            )
            grid_mean_predictions = grid_mean_predictions.reshape(xx.shape)
            grid_uncertainty = grid_uncertainty.reshape(xx.shape)
            plt.figure(figsize=(10, 6))
            plt.contourf(
                np.array(xx),
                np.array(yy),
                np.array(grid_mean_predictions),
                levels=100,
                cmap=plt.cm.RdYlBu,
                alpha=0.8,
            )
            plt.colorbar(label="Probability of Class 1")
            plt.contourf(
                np.array(xx),
                np.array(yy),
                np.array(grid_uncertainty),
                levels=20,
                cmap="gray",
                alpha=0.4,
            )
            plt.colorbar(label="Uncertainty (Entropy)")
            plt.scatter(
                X_np[:, f1],
                X_np[:, f2],
                c=y_np,
                edgecolor="k",
                cmap=plt.cm.RdYlBu,
                s=20,
            )
            plt.title("Binary Decision Boundary with Uncertainty")
            plt.xlabel(f"Feature {f1 + 1}")
            plt.ylabel(f"Feature {f2 + 1}")
            plt.grid(True)
            plt.show()

        # CASE 2: One feature categorical, one continuous.
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
                c_min, c_max = cont_vals.min() - 1, cont_vals.max() + 1
                cont_grid = jnp.linspace(c_min, c_max, resolution)
                n_features = X_np.shape[1]
                grid_list = []
                for i in range(n_features):
                    if i == cont_idx:
                        grid_list.append(cont_grid)
                    elif i == cat_idx:
                        grid_list.append(jnp.full(cont_grid.shape, cat))
                    else:
                        grid_list.append(
                            jnp.full(cont_grid.shape, jnp.array(X_np[:, i]).mean())
                        )
                grid_arr = jnp.stack(grid_list, axis=1)
                pred_samples = self.predict(
                    grid_arr, jax.random.PRNGKey(42), posterior=posterior
                )
                grid_mean_predictions = jax.nn.sigmoid(pred_samples.mean(axis=0))
                class_preds = (grid_mean_predictions > 0.5).astype(jnp.int32)
                ax.plot(
                    np.array(cont_grid),
                    np.array(class_preds),
                    label="Decision Boundary",
                )
                ax.scatter(
                    X_np[mask, cont_idx], y_np[mask], c="k", edgecolor="w", label="Data"
                )
                ax.set_title(f"Feature {cat_idx} = {cat}")
                ax.set_xlabel(f"Feature {cont_idx}")
                ax.set_ylabel("Predicted class")
                ax.legend()
            plt.suptitle("Binary Decision Boundary with Categorical Feature")
            plt.show()

        # CASE 3: Both features categorical.
        else:
            plt.figure(figsize=(8, 6))
            plt.scatter(
                X_np[:, f1], X_np[:, f2], c=y_np, cmap=plt.cm.RdYlBu, edgecolors="k"
            )
            plt.xlabel(f"Feature {f1}")
            plt.ylabel(f"Feature {f2}")
            plt.title("Binary Decision Boundary (Both features categorical)")
            plt.grid(True)
            plt.show()

    def _visualize_regression(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        posterior: str = "logits",
        grid_points: int = 100,
        unique_threshold: int = 10,
    ):
        """
        Unified visualization for regression tasks.

        This function automatically inspects the input data X and y to decide whether to
        produce a single-feature or multi-feature visualization. For each feature, it
        further determines whether it is continuous (many unique values) or categorical
        (few unique values) and applies the appropriate plots:

        - Continuous features: PDP (with a KDE density overlay) and ICE curves.
        - Categorical features: Bar plot for PDP and box plot for ICE.

        Parameters
        ----------
        X : jnp.ndarray
            Input data of shape (n_samples, n_features).
        y : jnp.ndarray
            Target values of shape (n_samples,).
        posterior : str, optional
            The posterior mode to use for predictions.
        grid_points : int, optional
            Number of grid points used for continuous plots.
        unique_threshold : int, optional
            Maximum number of unique values for a feature to be treated as categorical.
        """
        # Convert to NumPy arrays for inspection and plotting.
        X_np = np.asarray(X)
        y_np = np.asarray(y).squeeze()
        n_samples, n_features = X_np.shape

        # Helper function to predict with our model.
        def model_predict(X_input):
            key = jr.PRNGKey(0)
            preds = self.predict(jnp.array(X_input), key, posterior=posterior)
            return np.asarray(preds)

        # --- Single-Feature Case ---
        if n_features == 1:
            # Determine if the single feature is categorical.
            unique_vals = np.unique(X_np[:, 0])
            is_categorical = len(unique_vals) < unique_threshold

            if is_categorical:
                # Categorical single-feature visualization.
                baseline = np.mean(X_np, axis=0)
                cat_means = []
                for cat in unique_vals:
                    sample = baseline.copy()
                    sample[0] = cat
                    pred = model_predict(sample[None, :])[0]
                    pred_scalar = float(np.array(pred).squeeze())
                    cat_means.append(pred_scalar)
                plt.figure(figsize=(10, 6))
                plt.bar(unique_vals, cat_means, alpha=0.7, capsize=5)
                plt.xlabel("Feature (Categorical)")
                plt.ylabel("Predicted Target")
                plt.title("Categorical PDP")
                plt.show()

                # For ICE, collect predictions for each category and plot as boxplots.
                cat_preds = {}
                for cat in unique_vals:
                    mask = X_np[:, 0] == cat
                    if np.sum(mask) > 0:
                        preds = model_predict(
                            X_np[mask, :]
                        )  # shape: (n_posterior, n_samples_in_cat)
                        preds_mean = preds.mean(
                            axis=0
                        ).squeeze()  # shape: (n_samples_in_cat,)
                        cat_preds[cat] = preds_mean
                plt.figure(figsize=(10, 6))
                box_data = [cat_preds[cat] for cat in unique_vals if cat in cat_preds]
                plt.boxplot(box_data, labels=unique_vals)
                plt.xlabel("Feature (Categorical)")
                plt.ylabel("Predicted Target")
                plt.title("Categorical ICE")
                plt.show()

            else:
                # Continuous single-feature visualization.
                cont_vals = X_np[:, 0]
                cont_min, cont_max = float(np.min(cont_vals)), float(np.max(cont_vals))
                cont_grid = np.linspace(cont_min, cont_max, grid_points)
                baseline = np.mean(X_np, axis=0)
                X_pdp = np.tile(baseline, (grid_points, 1))
                X_pdp[:, 0] = cont_grid
                pdp_preds = model_predict(X_pdp)  # shape: (n_posterior, grid_points)

                plt.figure(figsize=(10, 6))
                sns.kdeplot(
                    x=cont_vals, y=y_np, cmap="Blues", fill=True, alpha=0.5, thresh=0.05
                )
                # Plot PDP (mean over posterior samples)
                plt.plot(
                    cont_grid,
                    pdp_preds.mean(axis=0).squeeze(),
                    color="red",
                    label="Mean Prediction (PDP)",
                )
                plt.xlabel("Feature (Continuous)")
                plt.ylabel("Target")
                plt.title("Continuous PDP")
                plt.legend()
                plt.show()

                # ICE for continuous feature.
                # For clarity, we select one (or a few) individuals.
                rng = np.random.default_rng(42)
                # Here, we choose one random individual from the dataset.
                idx = rng.choice(n_samples)
                sample = X_np[idx, :].copy()
                X_ice = np.tile(sample, (grid_points, 1))
                X_ice[:, 0] = cont_grid
                ice_preds = model_predict(X_ice)  # shape: (n_posterior, grid_points)

                # Compute summary statistics over all posterior samples.
                mean_ice = np.mean(ice_preds, axis=0)
                lower_ice = np.percentile(ice_preds, 5, axis=0)
                upper_ice = np.percentile(ice_preds, 95, axis=0)

                # Randomly select up to 30 posterior samples to display.
                n_posterior = ice_preds.shape[0]
                if n_posterior > 30:
                    sample_indices = rng.choice(n_posterior, size=30, replace=False)
                else:
                    sample_indices = np.arange(n_posterior)

                plt.figure(figsize=(10, 6))
                # Shaded credible interval from all posterior samples.
                plt.fill_between(
                    cont_grid,
                    lower_ice,
                    upper_ice,
                    color="lightblue",
                    alpha=0.5,
                    label="90% Credible Interval",
                )
                # Plot a subset of individual ICE curves.
                for i in sample_indices:
                    plt.plot(
                        cont_grid,
                        ice_preds[i],
                        color="dodgerblue",
                        linewidth=0.5,
                        alpha=0.7,
                    )
                # Overlay the mean ICE curve.
                plt.plot(
                    cont_grid,
                    mean_ice,
                    color="darkred",
                    linewidth=2,
                    label="Mean Prediction (ICE)",
                )
                plt.xlabel("Feature (Continuous)")
                plt.ylabel("Target")
                plt.title("Continuous ICE")
                plt.legend()
                plt.show()

        # --- Multi-Feature Case ---
        else:
            # Classify features.
            continuous_features = []
            categorical_features = []
            for i in range(n_features):
                if len(np.unique(X_np[:, i])) < unique_threshold:
                    categorical_features.append(i)
                else:
                    continuous_features.append(i)

            # Decide on subplot grid:
            # If both continuous and categorical representative features exist, use 2x2;
            # otherwise, use 1x2.
            if continuous_features and categorical_features:
                nrows = 2
            else:
                nrows = 1
            ncols = 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
            if nrows == 1:
                axes = np.expand_dims(axes, axis=0)

            # --- Continuous Visualization (if any continuous features exist) ---
            if continuous_features:
                cont_idx = continuous_features[0]
                cont_vals = X_np[:, cont_idx]
                cont_min = float(np.min(cont_vals))
                cont_max = float(np.max(cont_vals))
                cont_grid = np.linspace(cont_min, cont_max, grid_points)
                baseline = np.mean(X_np, axis=0)
                X_pdp = np.tile(baseline, (grid_points, 1))
                X_pdp[:, cont_idx] = cont_grid
                pdp_preds = model_predict(X_pdp)  # shape: (n_posterior, grid_points)
                mean_pdp_preds = pdp_preds.mean(
                    axis=0
                ).squeeze()  # shape: (grid_points,)
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
                    cont_grid, mean_pdp_preds, color="red", label="Mean Prediction"
                )
                ax_cont_pdp.set_xlabel(f"Feature {cont_idx} (Continuous)")
                ax_cont_pdp.set_ylabel("Target")
                ax_cont_pdp.set_title("Continuous PDP")
                ax_cont_pdp.legend()

                # ICE for continuous feature.
                # Here, we select one random individual for illustration.
                rng = np.random.default_rng(42)
                idx = rng.choice(n_samples)
                sample = X_np[idx, :].copy()
                X_ice = np.tile(sample, (grid_points, 1))
                X_ice[:, cont_idx] = cont_grid
                ice_preds = model_predict(X_ice)  # shape: (n_posterior, grid_points)
                mean_ice = np.mean(ice_preds, axis=0)
                lower_ice = np.percentile(ice_preds, 5, axis=0)
                upper_ice = np.percentile(ice_preds, 95, axis=0)
                n_posterior = ice_preds.shape[0]
                if n_posterior > 30:
                    sample_indices = rng.choice(n_posterior, size=30, replace=False)
                else:
                    sample_indices = np.arange(n_posterior)
                ax_cont_ice = axes[0, 1]
                ax_cont_ice.fill_between(
                    cont_grid,
                    lower_ice,
                    upper_ice,
                    color="lightblue",
                    alpha=0.5,
                    label="90% Credible Interval",
                )
                for i in sample_indices:
                    ax_cont_ice.plot(
                        cont_grid,
                        ice_preds[i],
                        color="dodgerblue",
                        linewidth=0.5,
                        alpha=0.7,
                    )
                ax_cont_ice.plot(
                    cont_grid,
                    mean_ice,
                    color="darkred",
                    linewidth=2,
                    label="Mean Prediction",
                )
                ax_cont_ice.set_xlabel(f"Feature {cont_idx} (Continuous)")
                ax_cont_ice.set_ylabel("Target")
                ax_cont_ice.set_title("Continuous ICE")
                ax_cont_ice.legend()
            else:
                # Hide the top row if no continuous features.
                for j in range(ncols):
                    axes[0, j].axis("off")

            # --- Categorical Visualization (if any categorical features exist) ---
            if categorical_features:
                cat_idx = categorical_features[0]
                cat_vals = X_np[:, cat_idx]
                unique_cats = np.unique(cat_vals)
                baseline = np.mean(X_np, axis=0)
                cat_means = []
                for cat in unique_cats:
                    sample = baseline.copy()
                    sample[cat_idx] = cat
                    pred = model_predict(sample[None, :])[0]
                    pred_scalar = float(np.array(pred).squeeze())
                    cat_means.append(pred_scalar)
                cat_row = 1 if (continuous_features and categorical_features) else 0
                ax_cat_pdp = axes[cat_row, 0]
                ax_cat_pdp.bar(unique_cats, cat_means, alpha=0.7, capsize=5)
                ax_cat_pdp.set_xlabel(f"Feature {cat_idx} (Categorical)")
                ax_cat_pdp.set_ylabel("Predicted Target")
                ax_cat_pdp.set_title("Categorical PDP")

                # Categorical ICE using boxplots.
                cat_predictions = {}
                for cat in unique_cats:
                    mask = cat_vals == cat
                    X_cat = X_np[mask, :]
                    if X_cat.shape[0] > 0:
                        preds_cat = model_predict(
                            X_cat
                        )  # shape: (n_posterior, n_cat_samples)
                        preds_cat_mean = preds_cat.mean(
                            axis=0
                        ).squeeze()  # shape: (n_cat_samples,)
                        cat_predictions[cat] = preds_cat_mean
                ax_cat_ice = axes[cat_row, 1]
                boxplot_data = [
                    cat_predictions[cat]
                    for cat in unique_cats
                    if cat in cat_predictions
                ]
                ax_cat_ice.boxplot(boxplot_data, labels=unique_cats)
                ax_cat_ice.set_xlabel(f"Feature {cat_idx} (Categorical)")
                ax_cat_ice.set_ylabel("Predicted Target")
                ax_cat_ice.set_title("Categorical ICE")
            else:
                if nrows == 2:
                    for j in range(ncols):
                        axes[1, j].axis("off")

            plt.tight_layout()
            plt.show()

    def _visualize_image_segmentation(self, X, y=None, posterior: str = "logits"):
        """
        Visualizes predictions for image segmentation tasks.

        :param X: jnp.ndarray
            Input images (batch_size, channels, height, width).
        :param y: jnp.ndarray
            Ground truth segmentation masks (optional, for comparison).
        :param num_samples: int
            Number of posterior samples for uncertainty estimation.
        """
        # Predict on input images
        pred_samples = self.predict(
            X, jax.random.PRNGKey(0), posterior=posterior
        )  # [num_samples, batch, 1, H, W]
        mean_preds = jax.nn.sigmoid(pred_samples.mean(axis=0))  # [batch, 1, H, W]
        uncertainty = -(
            mean_preds * jnp.log(mean_preds + 1e-9)
            + (1 - mean_preds) * jnp.log(1 - mean_preds + 1e-9)
        )  # Entropy [batch, 1, H, W]

        # Visualization
        batch_size = X.shape[0]
        fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))

        if batch_size == 1:
            axes = axes[np.newaxis, :]  # Ensure axes is always 2D for consistency

        for i in range(batch_size):
            # Original image
            img = X[i].squeeze()  # (H, W) for visualization
            pred_mask = mean_preds[i].squeeze()  # (H, W) predicted mask
            unc_map = uncertainty[i].squeeze()  # (H, W) uncertainty map

            # Ground truth mask (optional)
            if y is not None:
                true_mask = y[i].squeeze()

            # Plot original image
            axes[i, 0].imshow(img, cmap="gray")
            axes[i, 0].axis("off")
            axes[i, 0].set_title("Original Image")

            # Plot predicted mask
            axes[i, 1].imshow(pred_mask, cmap="gray")
            axes[i, 1].axis("off")
            axes[i, 1].set_title("Predicted Segmentation")

            # Overlay ground truth on predicted mask (optional)
            if y is not None:
                axes[i, 1].contour(
                    true_mask, colors="red", linewidths=1, alpha=0.7, levels=[0.5]
                )

            # Plot uncertainty heatmap
            im = axes[i, 2].imshow(unc_map, cmap="viridis")
            axes[i, 2].axis("off")
            axes[i, 2].set_title("Uncertainty (Entropy)")
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    def visualize_fourier_spectrum(self, layer_obj, show=True):
        """
        After the model is fit, we do a forward pass with a *concrete set of parameters*,
        then call get_fourier_coeffs(). That should be a device array we can host-convert safely.
        """
        fft_full = layer_obj.get_fourier_coeffs()  # jnp.array of shape (n,)?
        # The code below does jnp -> np conversion for plotting, so as long as we do NOT run
        # inside a jit/scan, it should be safe.
        fig1 = plot_fft_spectrum(fft_full, show=False)
        fig2 = visualize_circulant_kernel(fft_full, show=False)
        if show:
            plt.show()
        return fig1, fig2

    def save_params(self, file_path):
        """
        Saves trained model parameters to a file.

        :param file_path: str
            Path to save the parameters.
        """
        if isinstance(self.inference, SVI):
            if self.params is None:
                raise ValueError(
                    "SVI parameters are not available. Ensure `fit` was called."
                )
            params_to_save = self.params

        elif isinstance(self.inference, MCMC):
            if self.samples is None:
                raise ValueError(
                    "MCMC samples are not available. Ensure `fit` was called."
                )
            params_to_save = self.samples

        elif isinstance(self.inference, SteinVI):
            if self.stein_result is None:
                raise ValueError(
                    "SteinVI results are not available. Ensure `fit` was called."
                )
            params_to_save = self.stein_result.state  # Save the entire state

        else:
            raise ValueError("Inference method not initialized. Call `compile` first.")

        # Save parameters as a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(params_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✅ Model parameters successfully saved to {file_path}")

    def load_params(self, file_path):
        """
        Loads trained model parameters from a file.

        :param file_path: str
            Path to load the parameters from.
        """
        with open(file_path, "rb") as f:
            loaded_params = pickle.load(f)

        # Assign to the correct inference method
        if isinstance(self.inference, SVI):
            self.params = loaded_params

        elif isinstance(self.inference, MCMC):
            self.samples = loaded_params

        elif isinstance(self.inference, SteinVI):
            if self.stein_result is None:
                # Initialize SteinVI with a dummy run if it hasn't been set up
                self.stein_result = self.inference.run(
                    jax.random.PRNGKey(0), num_steps=1, progress_bar=False
                )

            # Overwrite SteinVI's internal state
            self.stein_result.state = loaded_params  # Corrected assignment

        else:
            raise ValueError("Inference method not initialized. Call `compile` first.")

        print(f"✅ Model parameters successfully loaded from {file_path}")
