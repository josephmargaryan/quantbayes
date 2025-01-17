import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal
from numpyro.contrib.einstein import SteinVI, RBFKernel, MixtureGuidePredictive
from numpyro.optim import Adam, Adagrad, SGD
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from bnn.utils.generalization_bound import BayesianAnalysis
from bnn.layers.testing.modules import *
from fake_data import *


class BaseModel:
    def __init__(self, task_type="regression", method="nuts", **kwargs):
        self.task_type = task_type.lower()
        self.method = method.lower()
        self.inference = None
        self.samples = None
        self.params = None
        self.losses = None
        self.stein_result = None  # Store SteinVI results
        self.kwargs = kwargs

    def model(self, X, y=None):
        """
        Defines the probabilistic model. Relies on `forward` and `define_observation`.
        """
        assert hasattr(
            self, "forward"
        ), "Derived class must implement a `forward` method."
        logits = self.forward(X)
        numpyro.deterministic("logits", logits)  # Store raw logits
        if y is not None:
            self.define_observation(logits, y)

    def compile(self, **kwargs):
        """
        Compiles the model for the specified inference method.
        """
        if self.method == "nuts":
            kernel = NUTS(self.model)
            self.inference = MCMC(
                kernel,
                num_warmup=kwargs.get("num_warmup", 500),
                num_samples=kwargs.get("num_samples", 1000),
                num_chains=kwargs.get("num_chains", 1),
            )
        elif self.method == "svi":
            guide = AutoNormal(self.model)
            optimizer = Adam(kwargs.get("learning_rate", 0.01))
            self.inference = SVI(self.model, guide, optimizer, loss=Trace_ELBO())
        elif self.method == "steinvi":
            guide = AutoNormal(self.model)
            self.inference = SteinVI(
                model=self.model,
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

    def predict(self, X_test, rng_key, num_samples=None):
        """
        Generates predictions using raw logits.
        """
        if isinstance(self.inference, MCMC):
            predictive = Predictive(self.model, posterior_samples=self.samples)
        elif isinstance(self.inference, SVI):
            # Ensure num_samples is specified for SVI
            assert (
                self.params is not None
            ), "SVI parameters are not available. Ensure `fit` was called."
            predictive = Predictive(
                self.model,
                guide=self.inference.guide,
                params=self.params,
                num_samples=num_samples
                or 100,  # Default to 100 samples if not specified
            )
        elif isinstance(self.inference, SteinVI):
            assert (
                self.stein_result is not None
            ), "SteinVI results are not available. Ensure `fit` was called."
            params = self.inference.get_params(self.stein_result.state)
            predictive = MixtureGuidePredictive(
                model=self.model,
                guide=self.inference.guide,
                params=params,
                num_samples=num_samples or 100,
                guide_sites=self.inference.guide_sites,
            )
        else:
            raise ValueError("Inference method not initialized. Call `compile` first.")

        preds = predictive(rng_key, X_test)
        return preds["logits"]

    @property
    def get_samples(self):
        """
        Retrieve posterior samples for MCMC.
        Returns:
            dict: Posterior samples from MCMC.
        """
        if isinstance(self.inference, MCMC):
            if self.samples is None:
                raise ValueError(
                    "MCMC samples are not available. Ensure `fit` was called."
                )
            return self.samples
        raise ValueError("MCMC is not the selected inference method.")

    @property
    def get_params(self):
        """
        Retrieve SVI parameters.
        Returns:
            dict: SVI guide parameters.
        """
        if isinstance(self.inference, SVI):
            if self.params is None:
                raise ValueError(
                    "SVI parameters are not available. Ensure `fit` was called."
                )
            return self.params
        raise ValueError("SVI is not the selected inference method.")

    @property
    def get_losses(self):
        """
        Retrieve SVI losses during optimization.
        Returns:
            list: Loss values recorded during SVI optimization.
        """
        if isinstance(self.inference, SVI):
            if self.losses is None:
                raise ValueError(
                    "SVI losses are not available. Ensure `fit` was called."
                )
            return self.losses
        raise ValueError("SVI is not the selected inference method.")

    @property
    def get_stein_result(self):
        """
        Retrieve SteinVI results.
        """
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
        features=(0, 1),
        resolution=100,
        feature_index=None,
    ):
        if self.task_type == "multiclass":
            assert (
                num_classes is not None
            ), "num_classes must be provided for multiclass visualization."
            self._visualize_multiclass(X, y, num_classes, features, resolution)
        elif self.task_type == "binary":
            self._visualize_binary(X, y, features, resolution)
        elif self.task_type == "regression":
            self._visualize_regression(X, y, feature_index)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def visualize(
        self,
        X,
        y=None,
        num_classes=None,
        features=(0, 1),
        resolution=100,
        feature_index=None,
    ):
        """
        Visualizes model outputs based on the task type.

        Args:
            X: Input data (shape depends on the task).
            y: True labels or targets (optional).
            num_classes: Number of classes (used for multiclass tasks).
            features: Tuple specifying the indices of the two features to visualize (for binary/multiclass tasks).
            resolution: Resolution for the grid in decision boundary plots.
            feature_index: Feature index for regression visualization.
        """
        if self.task_type == "multiclass":
            assert (
                num_classes is not None
            ), "num_classes must be provided for multiclass visualization."
            self._visualize_multiclass(X, y, num_classes, features, resolution)
        elif self.task_type == "binary":
            self._visualize_binary(X, y, features, resolution)
        elif self.task_type == "regression":
            self._visualize_regression(X, y, feature_index)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _visualize_multiclass(self, X, y, num_classes, features, resolution):
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
            grid_points_full, jax.random.PRNGKey(0)
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

    def _visualize_binary(self, X, y, features, resolution):
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
            grid_points_full, jax.random.PRNGKey(0)
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

    def _visualize_regression(self, X, y, feature_index):
        preds = self.predict(X, jax.random.PRNGKey(0))
        mean_preds = preds.mean(axis=0).squeeze()  # Ensure the mean is 1D
        lower_bound = jnp.percentile(
            preds, 2.5, axis=0
        ).squeeze()  # Ensure lower bound is 1D
        upper_bound = jnp.percentile(
            preds, 97.5, axis=0
        ).squeeze()  # Ensure upper bound is 1D
        feature = X[:, feature_index].squeeze()  # Ensure the feature is 1D
        y = y.squeeze()  # Ensure the target is 1D

        # Sort by feature values for a smooth plot
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


class MulticlassModel(BaseModel):
    def __init__(self, in_features, hidden_dim, num_classes):
        super().__init__(task_type="multiclass", method="svi")
        self.linear = Linear(in_features, hidden_dim, name="dense 1")
        self.linear2 = Linear(hidden_dim, num_classes, name="dense 2")

    def forward(self, x):
        x = jax.nn.tanh(self.linear(x))
        return self.linear2(x)

    def define_observation(self, logits, y):
        # Use logits to define the categorical distribution
        numpyro.sample("obs", dist.Categorical(logits=logits), obs=y)


# Generate sample data
df = generate_multiclass_classification_data()
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Training and evaluation
key = jr.key(0)

model = MulticlassModel(in_features=X.shape[1], hidden_dim=10, num_classes=3)
model.compile(num_steps=100)
model.fit(X_train, y_train, key, num_steps=100)

# Predict and visualize
preds = model.predict(X_test, key)
probs = jax.nn.softmax(preds, axis=-1)
mean = probs.mean(axis=0)
loss = log_loss(np.array(y_test), np.array(mean))
print(f"Loss: {loss:.3f}")
bound = BayesianAnalysis(
    len(X_train),
    delta=0.05,
    task_type="multiclass",
    inference_type="svi",
    posterior_samples=model.get_params,
)
bound.compute_pac_bayesian_bound(probs, y_test)

model.visualize(X, y, feature_index=(0, 1), num_classes=3)
