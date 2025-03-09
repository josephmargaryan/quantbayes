import pickle
from typing import Any, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpyro.contrib.einstein import MixtureGuidePredictive, RBFKernel, SteinVI
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adagrad, Adam
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, roc_curve


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
        X: jnp.ndarray,
        y: Optional[jnp.ndarray] = None,
        rng_key: Any = jax.random.PRNGKey(235),
        posterior: str = "logits",
        num_samples: Optional[int] = None,
        credible_interval: float = 90,
        **kwargs,
    ) -> None:
        """
        Unified visualization method that delegates to task-specific methods.

        Parameters:
            X (jnp.ndarray): Input data.
            y (Optional[jnp.ndarray]): Ground truth labels (if available).
            rng_key: Random key for sampling.
            posterior (str): Posterior type to use.
            num_samples (Optional[int]): Number of samples to draw.
            credible_interval (float): Credible interval percentage (used in regression).
            **kwargs: Additional keyword arguments for specialized visualizations.
        """
        if self.task_type == "multiclass":
            self._visualize_multiclass(X, y, rng_key, posterior, num_samples)
        elif self.task_type == "binary":
            self._visualize_binary(X, y, rng_key, posterior, num_samples)
        elif self.task_type == "regression":
            self._visualize_regression(
                X, y, rng_key, posterior, num_samples, credible_interval
            )
        elif self.task_type == "image_classification":
            self._visualize_image_classification(X, y, posterior, **kwargs)
        elif self.task_type == "image_segmentation":
            self._visualize_image_segmentation(X, y, posterior, **kwargs)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _visualize_image_classification(
        self,
        X: jnp.ndarray,
        y: Optional[jnp.ndarray],
        posterior: str = "logits",
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Visualizes predictions for image classification tasks.

        Parameters:
            X (jnp.ndarray): Input images (batch_size, channels, height, width)
            y (Optional[jnp.ndarray]): True labels (if available).
            posterior (str): Posterior type (default: "logits").
            image_size (Optional[int]): Fallback image size if X is not 4D.
            in_channels (Optional[int]): Fallback number of channels if X is not 4D.
            **kwargs: Additional parameters.
        """
        # Get predictions and compute mean probabilities.
        pred_samples = self.predict(X, jax.random.PRNGKey(0), posterior=posterior)
        mean_preds = jax.nn.softmax(
            pred_samples.mean(axis=0), axis=-1
        )  # [batch_size, num_classes]
        predicted_classes = jnp.argmax(mean_preds, axis=-1)

        # Setup grid for visualization.
        batch_size = X.shape[0]
        rows = int(jnp.ceil(batch_size / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(12, rows * 3))
        if rows == 1:
            axes = axes[np.newaxis, :]

        for i, ax in enumerate(axes.flatten()):
            if i >= batch_size:
                ax.axis("off")
                continue

            if X.ndim == 4:
                img = X[i].transpose(
                    1, 2, 0
                )  # Convert (batch, channels, height, width) -> (height, width, channels)
            else:
                if image_size is not None and in_channels is not None:
                    img = jnp.zeros((image_size, image_size, in_channels))
                else:
                    raise ValueError(
                        "Input X is not 4D and no fallback image_size/in_channels provided."
                    )

            true_label = y[i] if y is not None else None
            pred_label = int(predicted_classes[i])
            pred_probs = mean_preds[i]

            ax.imshow(
                img.squeeze(),
                cmap="gray" if (X.ndim == 4 and X.shape[1] == 1) else None,
            )
            ax.axis("off")
            title = f"Pred: {pred_label} ({pred_probs[pred_label]:.2f})"
            if y is not None:
                title += f"\nTrue: {true_label}"
            ax.set_title(title, fontsize=10)

        plt.tight_layout()
        plt.show()

    def _visualize_regression(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        rng_key: Any,
        posterior: str = "logits",
        num_samples: Optional[int] = None,
        credible_interval: float = 90,
    ) -> None:
        """
        Visualizes regression predictions.

        Depending on the input dimensionality, uses either a line plot with credible intervals
        (for single-feature data) or a scatter plot with error bars and residual plot (for multi-feature data).

        Parameters:
            X (jnp.ndarray): Input features. For 1D regression, shape (n_samples,) or (n_samples, 1);
                             for multi-feature, shape (n_samples, n_features) with n_features > 1.
            y (jnp.ndarray): True target values of shape (n_samples,).
            rng_key: Random key for sampling predictions.
            posterior (str): Posterior type to use.
            num_samples (Optional[int]): Number of samples to draw.
            credible_interval (float): Credible interval percentage.
        """
        preds = self.predict(X, rng_key, posterior=posterior, num_samples=num_samples)
        preds = np.array(preds)
        pred_mean = np.mean(preds, axis=0)
        lower_bound = np.percentile(preds, (100 - credible_interval) / 2, axis=0)
        upper_bound = np.percentile(preds, 100 - (100 - credible_interval) / 2, axis=0)

        # Single-feature regression: line plot.
        if (X.ndim == 1) or (X.ndim == 2 and X.shape[1] == 1):
            plt.figure(figsize=(10, 6))
            plt.scatter(X, y, color="black", label="True values", alpha=0.7)
            sorted_idx = np.argsort(X.flatten())
            plt.plot(
                X.flatten()[sorted_idx],
                pred_mean.flatten()[sorted_idx],
                color="blue",
                lw=2,
                label="Predictive mean",
            )
            plt.fill_between(
                X.flatten()[sorted_idx],
                lower_bound.flatten()[sorted_idx],
                upper_bound.flatten()[sorted_idx],
                color="blue",
                alpha=0.2,
                label=f"{credible_interval}% Credible Interval",
            )
            plt.xlabel("X")
            plt.ylabel("y")
            plt.title("Regression Predictions with Uncertainty")
            plt.legend()
            plt.show()
        else:
            # Multi-feature regression: scatter plot and residual plot.
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            axs[0].errorbar(
                pred_mean,
                y,
                yerr=[y - lower_bound, upper_bound - y],
                fmt="o",
                alpha=0.6,
                ecolor="gray",
                capsize=3,
            )
            min_val = min(y.min(), pred_mean.min())
            max_val = max(y.max(), pred_mean.max())
            axs[0].plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
            axs[0].set_xlabel("Predicted")
            axs[0].set_ylabel("True")
            axs[0].set_title("Predicted vs. True Values")

            residuals = y - pred_mean
            axs[1].scatter(pred_mean, residuals, alpha=0.6)
            axs[1].axhline(0, color="k", linestyle="--", lw=2)
            axs[1].set_xlabel("Predicted")
            axs[1].set_ylabel("Residuals")
            axs[1].set_title("Residual Plot")
            plt.tight_layout()
            plt.show()

    def _visualize_binary(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        rng_key: Any,
        posterior: str = "logits",
        num_samples: Optional[int] = None,
    ) -> None:
        """
        Visualizes binary classification predictions by displaying:
          1. Histogram of predicted probabilities.
          2. ROC curve with AUC.
          3. Calibration plot.

        Parameters:
            X (jnp.ndarray): Input data.
            y (jnp.ndarray): True binary labels (0 or 1).
            rng_key: Random key for sampling predictions.
            posterior (str): Posterior type (default: 'logits').
            num_samples (Optional[int]): Number of samples to draw.
        """
        preds = self.predict(X, rng_key, posterior=posterior, num_samples=num_samples)
        preds = np.array(preds)
        preds_prob = jax.nn.sigmoid(preds)
        preds_prob = np.array(preds_prob)
        pred_mean_prob = np.mean(preds_prob, axis=0)

        fpr, tpr, _ = roc_curve(y, pred_mean_prob)
        roc_auc = auc(fpr, tpr)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].hist(
            pred_mean_prob, bins=20, color="skyblue", edgecolor="black", alpha=0.8
        )
        axs[0].set_title("Histogram of Predicted Probabilities")
        axs[0].set_xlabel("Predicted Probability")
        axs[0].set_ylabel("Frequency")

        axs[1].plot(
            fpr, tpr, color="darkred", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        axs[1].plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
        axs[1].set_xlim([0.0, 1.0])
        axs[1].set_ylim([0.0, 1.05])
        axs[1].set_xlabel("False Positive Rate")
        axs[1].set_ylabel("True Positive Rate")
        axs[1].set_title("Receiver Operating Characteristic")
        axs[1].legend(loc="lower right")

        prob_true, prob_pred = calibration_curve(y, pred_mean_prob, n_bins=10)
        axs[2].plot(
            prob_pred, prob_true, marker="o", linewidth=1, label="Calibration curve"
        )
        axs[2].plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        axs[2].set_xlabel("Mean Predicted Probability")
        axs[2].set_ylabel("Fraction of Positives")
        axs[2].set_title("Calibration Plot")
        axs[2].legend()

        plt.suptitle("Binary Classification Visualizations")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _visualize_multiclass(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        rng_key: Any,
        posterior: str = "logits",
        num_samples: Optional[int] = None,
    ) -> None:
        """
        Visualizes multiclass classification predictions by showing:
          1. A confusion matrix.
          2. A bar chart of average predicted probabilities per class.

        Parameters:
            X (jnp.ndarray): Input data.
            y (jnp.ndarray): True class labels (0, 1, ..., num_classes-1).
            rng_key: Random key for sampling predictions.
            posterior (str): Posterior type.
            num_samples (Optional[int]): Number of samples to draw.
        """
        preds = self.predict(X, rng_key, posterior=posterior, num_samples=num_samples)
        preds = np.array(preds)
        pred_mean_logits = np.mean(preds, axis=0)
        pred_mean_probs = jax.nn.softmax(pred_mean_logits, axis=-1)
        pred_classes = np.argmax(pred_mean_probs, axis=-1)

        # Compute confusion matrix.

        cm = confusion_matrix(y, pred_classes)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[0])
        axs[0].set_xlabel("Predicted")
        axs[0].set_ylabel("True")
        axs[0].set_title("Confusion Matrix")

        num_classes = pred_mean_probs.shape[1]
        avg_probs = np.mean(pred_mean_probs, axis=0)
        axs[1].bar(
            range(num_classes), avg_probs, color="mediumseagreen", edgecolor="black"
        )
        axs[1].set_xlabel("Class")
        axs[1].set_ylabel("Average Predicted Probability")
        axs[1].set_title("Average Predicted Probabilities")
        axs[1].set_xticks(range(num_classes))
        axs[1].set_xticklabels([f"Class {i}" for i in range(num_classes)])

        plt.suptitle("Multiclass Classification Visualizations")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _visualize_image_segmentation(
        self,
        X: jnp.ndarray,
        y: Optional[jnp.ndarray] = None,
        posterior: str = "logits",
        **kwargs,
    ) -> None:
        """
        Visualizes predictions for image segmentation tasks.

        Parameters:
            X (jnp.ndarray): Input images (batch_size, channels, height, width).
            y (Optional[jnp.ndarray]): Ground truth segmentation masks.
            posterior (str): Posterior type.
            **kwargs: Additional keyword arguments.
        """
        pred_samples = self.predict(X, jax.random.PRNGKey(0), posterior=posterior)
        mean_preds = jax.nn.sigmoid(pred_samples.mean(axis=0))
        uncertainty = -(
            mean_preds * jnp.log(mean_preds + 1e-9)
            + (1 - mean_preds) * jnp.log(1 - mean_preds + 1e-9)
        )

        batch_size = X.shape[0]
        fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
        if batch_size == 1:
            axes = axes[np.newaxis, :]

        for i in range(batch_size):
            img = X[i].squeeze()
            pred_mask = mean_preds[i].squeeze()
            unc_map = uncertainty[i].squeeze()

            axes[i, 0].imshow(img, cmap="gray")
            axes[i, 0].axis("off")
            axes[i, 0].set_title("Original Image")

            axes[i, 1].imshow(pred_mask, cmap="gray")
            axes[i, 1].axis("off")
            axes[i, 1].set_title("Predicted Segmentation")

            if y is not None:
                true_mask = y[i].squeeze()
                axes[i, 1].contour(
                    true_mask, colors="red", linewidths=1, alpha=0.7, levels=[0.5]
                )

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
