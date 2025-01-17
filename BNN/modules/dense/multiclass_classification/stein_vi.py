from bnn.core.base_task import BaseTask
from bnn.core.base_inference import BaseInference
from numpyro.contrib.einstein import SteinVI, RBFKernel, MixtureGuidePredictive
from numpyro.optim import Adam, Adagrad, SGD
from numpyro.infer.autoguide import AutoNormal
from typing import Callable
import jax.random as jr
import jax
import jax.numpy as jnp
import numpyro
import numpy as np
import matplotlib.pyplot as plt
import numpyro.distributions as dist


class DenseMultiClassSteinVI(BaseTask, BaseInference):
    """
    Dense multi-class classification model using Stein Variational Inference with flexible architecture.
    """

    def __init__(
        self,
        num_steps: int = 1000,
        model_type: str = "shallow",  # New argument
        num_classes: int = 3,
        hidden_size: int = 10,
        num_particles: int = 10,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_particles = num_particles
        self.model_type = model_type.lower()
        self.stein = None
        self.stein_result = None

        # Validate the model_type
        if self.model_type not in ["shallow", "deep"]:
            raise ValueError("model_type must be 'shallow' or 'deep'.")

    def get_default_model(self) -> Callable:
        """
        Return the appropriate model based on the model_type.
        """
        if self.model_type == "shallow":
            return lambda X, y=None, **kwargs: self.multiclass_model(X, y)
        elif self.model_type == "deep":
            return lambda X, y=None, **kwargs: self.deep_multiclass_model(X, y)

    def multiclass_model(self, X, y=None):
        """
        Shallow Bayesian multi-class classification model with one dense layer.
        """
        num_classes = self.num_classes
        input_size = X.shape[1]
        hidden_size = self.hidden_size
        num_particles = self.num_particles

        # Dense layer weights and biases
        weights_dense = numpyro.sample(
            "weights_dense",
            dist.Normal(0, 1).expand([num_particles, input_size, hidden_size]),
        )
        bias_dense = numpyro.sample(
            "bias_dense", dist.Normal(0, 1).expand([num_particles, hidden_size])
        )

        hidden_list = []
        for i in range(num_particles):
            hidden = jnp.matmul(X, weights_dense[i]) + bias_dense[i]
            hidden = jax.nn.silu(hidden)
            hidden_list.append(hidden)

        hidden = jnp.stack(hidden_list, axis=0)

        # Output layer weights and biases
        weights_out = numpyro.sample(
            "weights_out",
            dist.Normal(0, 1).expand([num_particles, hidden_size, num_classes]),
        )
        bias_out = numpyro.sample(
            "bias_out", dist.Normal(0, 1).expand([num_particles, num_classes])
        )

        logits = jnp.einsum("pbi,pio->pbo", hidden, weights_out) + bias_out[:, None, :]
        mean_logits = jnp.mean(logits, axis=0)
        logits = jnp.clip(mean_logits, a_min=-10, a_max=10)

        numpyro.deterministic("logits", logits)
        numpyro.sample("obs", dist.Categorical(logits=logits), obs=y)

    def deep_multiclass_model(self, X, y=None):
        """
        Deep Bayesian multi-class classification model with two dense layers.
        """
        num_classes = self.num_classes
        input_size = X.shape[1]
        hidden_size = self.hidden_size
        num_particles = self.num_particles

        # First dense layer weights and biases
        weights_dense1 = numpyro.sample(
            "weights_dense1",
            dist.Normal(0, 1).expand([num_particles, input_size, hidden_size]),
        )
        bias_dense1 = numpyro.sample(
            "bias_dense1", dist.Normal(0, 1).expand([num_particles, hidden_size])
        )

        hidden_list1 = []
        for i in range(num_particles):
            hidden1 = jnp.matmul(X, weights_dense1[i]) + bias_dense1[i]
            hidden1 = jax.nn.silu(hidden1)
            hidden_list1.append(hidden1)

        hidden1 = jnp.stack(hidden_list1, axis=0)

        # Second dense layer weights and biases
        weights_dense2 = numpyro.sample(
            "weights_dense2",
            dist.Normal(0, 1).expand([num_particles, hidden_size, hidden_size]),
        )
        bias_dense2 = numpyro.sample(
            "bias_dense2", dist.Normal(0, 1).expand([num_particles, hidden_size])
        )

        hidden_list2 = []
        for i in range(num_particles):
            hidden2 = jnp.matmul(hidden1[i], weights_dense2[i]) + bias_dense2[i]
            hidden2 = jax.nn.silu(hidden2)
            hidden_list2.append(hidden2)

        hidden2 = jnp.stack(hidden_list2, axis=0)

        # Output layer weights and biases
        weights_out = numpyro.sample(
            "weights_out",
            dist.Normal(0, 1).expand([num_particles, hidden_size, num_classes]),
        )
        bias_out = numpyro.sample(
            "bias_out", dist.Normal(0, 1).expand([num_particles, num_classes])
        )

        logits = jnp.einsum("pbi,pio->pbo", hidden2, weights_out) + bias_out[:, None, :]
        mean_logits = jnp.mean(logits, axis=0)
        logits = jnp.clip(mean_logits, a_min=-10, a_max=10)

        numpyro.deterministic("logits", logits)
        numpyro.sample("obs", dist.Categorical(logits=logits), obs=y)

    def bayesian_inference(
        self,
        X_train,
        y_train,
        rng_key,
        repulsion_temperature=1.0,
        num_stein_particles=5,
        num_elbo_particles=50,
    ):
        """
        Train a Bayesian Neural Network for multiclass classification using SteinVI.
        """
        model = self.get_default_model()
        guide = AutoNormal(model)

        self.stein = SteinVI(
            model=model,
            guide=guide,
            optim=Adagrad(0.5),
            kernel_fn=RBFKernel(),
            repulsion_temperature=repulsion_temperature,
            num_stein_particles=num_stein_particles,
            num_elbo_particles=num_elbo_particles,
        )

        rng_key = jr.PRNGKey(0)
        self.stein_result = self.stein.run(
            rng_key,
            self.num_steps,
            X_train,
            y_train,
            num_classes=self.num_classes,
            progress_bar=True,
        )

    def retrieve_results(self):
        """
        Retrieve the posterior samples.
        """
        if not self.stein:
            raise RuntimeError("No inference results available. Fit the model first.")
        return self.stein_result

    def fit(self, X_train, y_train, rng_key, **kwargs):
        """
        Fit the model using MCMC.
        """
        self.bayesian_inference(X_train, y_train, rng_key, **kwargs)
        self.fitted = True

    def predict(self, X_test, rng_key):
        """
        Generate predictions for regression using a trained Bayesian Neural Network.
        """
        model = self.get_default_model()
        predictive = MixtureGuidePredictive(
            model,
            self.stein.guide,
            params=self.stein.get_params(self.stein_result.state),
            num_samples=100,
            guide_sites=self.stein.guide_sites,
        )

        rng_key = jr.PRNGKey(1)
        predictions = predictive(rng_key, X_test)["logits"]

        return predictions

    def visualize(self, X, y, num_classes, features=(0, 1), resolution=100):
        """
        Visualizes multiclass decision boundaries with uncertainty.

        Args:
            X: Input data (shape: (N, D)).
            y: True class labels (shape: (N,)).
            num_classes: Number of classes.
            features: Tuple specifying the indices of the two features to visualize.
            resolution: Grid resolution for decision boundary visualization.
        """
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
        pred_samples = self.predict(grid_points_full, jax.random.PRNGKey(0))
        grid_mean_predictions = jax.nn.softmax(pred_samples.mean(axis=0), axis=-1)

        # Compute entropy (uncertainty)
        grid_uncertainty = -jnp.sum(
            grid_mean_predictions * jnp.log(grid_mean_predictions + 1e-9), axis=1
        )

        grid_classes = jnp.argmax(grid_mean_predictions, axis=1).reshape(xx.shape)
        grid_uncertainty = grid_uncertainty.reshape(xx.shape)

        # Plot decision boundaries and uncertainty
        plt.figure(figsize=(12, 8))
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
        plt.title(
            f"Multiclass Decision Boundaries with Uncertainty (Features {features[0]} and {features[1]})"
        )
        plt.xlabel(f"Feature {feature_1 + 1}")
        plt.ylabel(f"Feature {feature_2 + 1}")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":

    ############# Example Usage #############
    from bnn.modules.dense.multiclass_classification.stein_vi import (
        DenseMultiClassSteinVI,
    )
    from bnn.utils.entropy_analysis import EntropyAndMutualInformation
    from bnn.utils.generalization_bound import BayesianAnalysis
    from fake_data import generate_multiclass_classification_data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss
    import jax.random as jr
    import jax.numpy as jnp

    df = generate_multiclass_classification_data()
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )
    key = jr.key(0)
    classifier = DenseMultiClassSteinVI(
        num_classes=3, num_particles=5, num_steps=100, hidden_size=3, model_type="deep"
    )
    classifier.fit(X_train, y_train, key)
    posteriors_preds = classifier.predict(X_test, key)
    print(posteriors_preds.shape)
    probabilities = jax.nn.softmax(posteriors_preds, axis=-1)

    stein_results = classifier.retrieve_results()
    mean_preds = probabilities.mean(axis=0)
    loss = log_loss(np.array(y_test), np.array(mean_preds))
    print(f"Logloss: {loss}")
    classifier.visualize(X_test, y_test, num_classes=3, features=(0, 1), resolution=100)
    analysis = BayesianAnalysis(len(X_train), delta=0.05, task_type="multiclass")
    # Compute PAC-Bayesian bound for MCMC
    from jax.tree_util import tree_flatten

    def extract_layer_names(stein_result):
        param_tree = stein_result.params
        flattened, _ = tree_flatten(param_tree)
        return [name for name in param_tree if not name.startswith(("prec", "sigma"))]

    layer_names = extract_layer_names(stein_results)

    bound = analysis.compute_pac_bayesian_bound(
        predictions=probabilities,
        y_true=y_test,
        posterior_samples=stein_results,
        layer_names=layer_names,
        inference_type="steinvi",
        prior_mean=0,
        prior_std=1,
    )
    print("PAC-Bayesian Bound (SteinVI):", bound)
    mi = analysis.compute_mutual_information_bound(
        posterior_samples=stein_results,
        layer_names=layer_names,
        inference_type="steinvi",
    )
    print(f"Mutual Information bound: {mi}")
    uncertainty_quantification = EntropyAndMutualInformation("multiclass")
    mi, mi1 = uncertainty_quantification.compute_mutual_information(probabilities)
    uncertainty_quantification.visualize(mi, mi1)
