from typing import Optional, Typing, Tuple
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal
from numpyro.contrib.einstein import SteinVI, RBFKernel, MixtureGuidePredictive
from numpyro.optim import Adam, Adagrad, SGD
import pickle
import numpy as np


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
        """
        if self.task_type == "multiclass":
            assert (
                num_classes is not None
            ), "num_classes must be provided for multiclass visualization."
            self._visualize_multiclass(
                X, y, num_classes, features, resolution, posterior
            )
        elif self.task_type == "binary":
            self._visualize_binary(X, y, features, resolution, posterior)
        elif self.task_type == "regression":
            self._visualize_regression(X, y, feature_index, posterior)
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
    ):
        feature_1, feature_2 = features
        X_selected = X[:, [feature_1, feature_2]]
        x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
        y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
        xx, yy = jnp.meshgrid(
            jnp.linspace(x_min, x_max, resolution),
            jnp.linspace(y_min, y_max, resolution),
        )
        grid_points = jnp.c_[xx.ravel(), yy.ravel()]
        grid_points_full = jnp.zeros((grid_points.shape[0], X.shape[1]))
        grid_points_full = grid_points_full.at[:, feature_1].set(grid_points[:, 0])
        grid_points_full = grid_points_full.at[:, feature_2].set(grid_points[:, 1])

        # Predict on grid points
        pred_samples = self.predict(
            grid_points_full, jax.random.PRNGKey(0), posterior=posterior
        )  # Shape: [num_samples, resolution^2, num_classes]
        grid_mean_predictions = jax.nn.softmax(
            pred_samples.mean(axis=0), axis=-1
        )  # Shape: [resolution^2, num_classes]

        # Compute uncertainty (entropy)
        grid_uncertainty = -jnp.sum(
            grid_mean_predictions * jnp.log(grid_mean_predictions + 1e-9), axis=1
        )  # Shape: [resolution^2]
        grid_uncertainty = grid_uncertainty.reshape(
            xx.shape
        )  # Shape: [resolution, resolution]

        # Get predicted classes
        grid_classes = jnp.argmax(grid_mean_predictions, axis=1).reshape(
            xx.shape
        )  # Shape: [resolution, resolution]

        # Plot decision boundaries and uncertainty
        plt.figure(figsize=(10, 6))
        plt.contourf(
            xx, yy, grid_classes, levels=num_classes, cmap=plt.cm.RdYlBu, alpha=0.6
        )
        plt.colorbar(label="Predicted Class")
        plt.contourf(xx, yy, grid_uncertainty, levels=20, cmap="gray", alpha=0.4)
        plt.colorbar(label="Uncertainty (Entropy)")
        plt.scatter(
            X_selected[:, 0],
            X_selected[:, 1],
            c=y,
            edgecolor="k",
            cmap=plt.cm.RdYlBu,
            s=20,
        )
        plt.title("Multiclass Decision Boundaries with Uncertainty")
        plt.xlabel(f"Feature {feature_1 + 1}")
        plt.ylabel(f"Feature {feature_2 + 1}")
        plt.grid(True)
        plt.show()

    def _visualize_binary(
        self,
        X,
        y,
        features: Tuple[int, int] = (0, 1),
        resolution: int = 100,
        posterior: str = "logits",
    ):
        feature_1, feature_2 = features
        X_selected = X[:, [feature_1, feature_2]]
        x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
        y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
        xx, yy = jnp.meshgrid(
            jnp.linspace(x_min, x_max, resolution),
            jnp.linspace(y_min, y_max, resolution),
        )
        grid_points = jnp.c_[xx.ravel(), yy.ravel()]
        grid_points_full = jnp.zeros((grid_points.shape[0], X.shape[1]))
        grid_points_full = grid_points_full.at[:, feature_1].set(grid_points[:, 0])
        grid_points_full = grid_points_full.at[:, feature_2].set(grid_points[:, 1])

        # Predict probabilities
        pred_samples = self.predict(
            grid_points_full, jax.random.PRNGKey(0), posterior=posterior
        )  # Shape: [num_samples, num_grid_points]
        grid_mean_predictions = jax.nn.sigmoid(
            pred_samples.mean(axis=0)
        )  # Shape: [num_grid_points]

        # Compute uncertainty (entropy)
        grid_uncertainty = -(
            grid_mean_predictions * jnp.log(grid_mean_predictions + 1e-9)
            + (1 - grid_mean_predictions) * jnp.log(1 - grid_mean_predictions + 1e-9)
        )  # Shape: [num_grid_points]

        # Reshape predictions and uncertainty
        grid_mean_predictions = grid_mean_predictions.reshape(xx.shape)
        grid_uncertainty = grid_uncertainty.reshape(xx.shape)

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.contourf(
            xx, yy, grid_mean_predictions, levels=100, cmap=plt.cm.RdYlBu, alpha=0.8
        )
        plt.colorbar(label="Probability of Class 1")
        plt.contourf(xx, yy, grid_uncertainty, levels=20, cmap="gray", alpha=0.4)
        plt.colorbar(label="Uncertainty (Entropy)")
        plt.scatter(
            X_selected[:, 0],
            X_selected[:, 1],
            c=y,
            edgecolor="k",
            cmap=plt.cm.RdYlBu,
            s=20,
        )
        plt.title("Binary Classification Decision Boundary with Uncertainty")
        plt.xlabel(f"Feature {feature_1 + 1}")
        plt.ylabel(f"Feature {feature_2 + 1}")
        plt.grid(True)
        plt.show()

    def _visualize_regression(
        self, X, y, feature_index: int = 0, posterior: str = "logits"
    ):
        preds = self.predict(X, jax.random.PRNGKey(0), posterior=posterior)
        mean_preds = preds.mean(axis=0).squeeze()
        lower_bound = jnp.percentile(preds, 2.5, axis=0).squeeze()
        upper_bound = jnp.percentile(preds, 97.5, axis=0).squeeze()
        feature = X[:, feature_index].squeeze()
        y = y.squeeze()

        sorted_indices = jnp.argsort(feature)
        feature = feature[sorted_indices]
        mean_preds = mean_preds[sorted_indices]
        lower_bound = lower_bound[sorted_indices]
        upper_bound = upper_bound[sorted_indices]
        y = y[sorted_indices]

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.scatter(feature, y, color="blue", alpha=0.6, label="True Targets")
        plt.plot(feature, mean_preds, color="red", label="Mean Predictions")
        plt.fill_between(
            feature,
            lower_bound,
            upper_bound,
            color="gray",
            alpha=0.3,
            label="Uncertainty Bounds",
        )
        plt.legend()
        plt.xlabel(f"Feature {feature_index}")
        plt.ylabel("Target")
        plt.title("Regression Visualization with Uncertainty Bounds")
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
