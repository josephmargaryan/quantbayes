from bnn.core.base_task import BaseTask
from bnn.core.base_inference import BaseInference
from numpyro.infer import MCMC, NUTS, Predictive
from typing import Callable
import jax.random as jr
import jax
import jax.numpy as jnp
import numpyro
import numpy as np
import matplotlib.pyplot as plt
import numpyro.distributions as dist


class DenseBinaryMCMC(BaseTask, BaseInference):
    """
    Dense regression model using MCMC inference with flexibility for shallow or deep networks.
    """

    def __init__(
        self,
        num_warmup: int = 500,
        num_samples: int = 1000,
        model_type: str = "shallow",  # New argument
        hidden_dim: int = 10,
    ):
        super().__init__()
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.hidden_dim = hidden_dim
        self.model_type = model_type.lower()
        self.mcmc = None

        # Validate the model_type
        if self.model_type not in ["shallow", "deep"]:
            raise ValueError("model_type must be 'shallow' or 'deep'.")

    def get_default_model(self) -> Callable:
        """
        Return the appropriate model based on the model_type.
        """
        if self.model_type == "shallow":
            return lambda X, y=None: self.binary_model(X, y)
        elif self.model_type == "deep":
            return lambda X, y=None: self.deep_binary_model(X, y)

    def binary_model(self, X, y=None):
        """
        Single-layer Bayesian Neural Network for binary classification.
        """
        input_dim = X.shape[1]
        hidden_dim = self.hidden_dim

        w_hidden = numpyro.sample(
            "w_hidden", dist.Normal(0, 1).expand([input_dim, hidden_dim])
        )
        b_hidden = numpyro.sample("b_hidden", dist.Normal(0, 1).expand([hidden_dim]))

        w_out = numpyro.sample("w_out", dist.Normal(0, 1).expand([hidden_dim, 1]))
        b_out = numpyro.sample("b_out", dist.Normal(0, 1).expand([1]))

        hidden = jax.nn.tanh(jnp.dot(X, w_hidden) + b_hidden)
        logits = jnp.dot(hidden, w_out).squeeze() + b_out
        logits = jnp.clip(logits, a_min=-10, a_max=10)
        numpyro.deterministic("logits", logits)
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

    def deep_binary_model(self, X, y=None):
        """
        Two-hidden-layer Bayesian Neural Network for binary classification.
        """
        input_dim = X.shape[1]
        hidden_dim = self.hidden_dim

        # First hidden layer
        w_hidden1 = numpyro.sample(
            "w_hidden1", dist.Normal(0, 1).expand([input_dim, hidden_dim])
        )
        b_hidden1 = numpyro.sample("b_hidden1", dist.Normal(0, 1).expand([hidden_dim]))
        hidden1 = jax.nn.tanh(jnp.dot(X, w_hidden1) + b_hidden1)

        # Second hidden layer
        w_hidden2 = numpyro.sample(
            "w_hidden2", dist.Normal(0, 1).expand([hidden_dim, hidden_dim])
        )
        b_hidden2 = numpyro.sample("b_hidden2", dist.Normal(0, 1).expand([hidden_dim]))
        hidden2 = jax.nn.tanh(jnp.dot(hidden1, w_hidden2) + b_hidden2)

        # Output layer
        w_out = numpyro.sample("w_out", dist.Normal(0, 1).expand([hidden_dim, 1]))
        b_out = numpyro.sample("b_out", dist.Normal(0, 1).expand([1]))
        logits = jnp.dot(hidden2, w_out).squeeze() + b_out
        logits = jnp.clip(logits, a_min=-10, a_max=10)
        numpyro.deterministic("logits", logits)
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

    def bayesian_inference(self, X_train, y_train, rng_key):
        """
        Perform MCMC inference.
        """
        model = self.get_default_model()
        kernel = NUTS(model)
        self.mcmc = MCMC(
            kernel, num_warmup=self.num_warmup, num_samples=self.num_samples
        )
        self.mcmc.run(rng_key, X=X_train, y=y_train)
        self.mcmc.print_summary()

    def retrieve_results(self):
        """
        Retrieve the posterior samples.
        """
        if not self.mcmc:
            raise RuntimeError("No inference results available. Fit the model first.")
        return self.mcmc.get_samples()

    def fit(self, X_train, y_train, rng_key):
        """
        Fit the model using MCMC.
        """
        self.bayesian_inference(X_train, y_train, rng_key)
        self.fitted = True

    def predict(self, X_test, rng_key):
        """
        Predict regression values using the posterior.
        """
        model = self.get_default_model()
        if not self.fitted or self.mcmc is None:
            raise RuntimeError("Model must be fitted before making predictions.")
        posterior_samples = self.mcmc.get_samples()
        predictive = Predictive(model, posterior_samples)
        preds = predictive(rng_key=rng_key, X=X_test)
        return preds["logits"]

    def visualize(self, X, y, feature_indices=(0, 1), grid_resolution=100):
        """
        Visualize binary classification decision boundary with uncertainty.

        Args:
            X (jnp.ndarray): Input features.
            y (jnp.ndarray): Target labels (binary).
            mcmc: numpyro.infer.MCMC
                Trained MCMC object containing posterior samples.
            predict: Callable
                Prediction function for binary classification.
            binary_model: Callable
                The binary classification model to use for predictions.
            feature_indices (tuple): Indices of the two features to visualize (x and y axes).
            grid_resolution (int): Number of points for each grid axis (higher means finer grid).

        Returns:
            None. Displays the plot.
        """
        X = np.array(X)
        y = np.array(y)
        feature1_idx, feature2_idx = feature_indices
        feature1, feature2 = X[:, feature1_idx], X[:, feature2_idx]

        x_min, x_max = feature1.min() - 1, feature1.max() + 1
        y_min, y_max = feature2.min() - 1, feature2.max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_resolution),
            np.linspace(y_min, y_max, grid_resolution),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        X_for_grid = np.zeros((grid.shape[0], X.shape[1]))
        X_for_grid[:, feature1_idx] = grid[:, 0]
        X_for_grid[:, feature2_idx] = grid[:, 1]
        for i in range(X.shape[1]):
            if i not in feature_indices:
                X_for_grid[:, i] = X[:, i].mean()

        grid_preds = self.predict(X_for_grid, jr.key(1))
        grid_preds = jax.nn.sigmoid(grid_preds)
        grid_mean = grid_preds.mean(axis=0).reshape(xx.shape)
        grid_uncertainty = grid_preds.var(axis=0).reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(
            xx, yy, grid_mean, levels=20, cmap="RdBu", alpha=0.8, vmin=0, vmax=1
        )
        plt.colorbar(label="Predicted Probability (Mean)")

        plt.imshow(
            grid_uncertainty,
            extent=(x_min, x_max, y_min, y_max),
            origin="lower",
            cmap="binary",
            alpha=0.3,
            aspect="auto",
        )

        plt.contour(xx, yy, grid_mean, levels=[0.5], colors="black", linestyles="--")

        plt.scatter(
            feature1[y == 0],
            feature2[y == 0],
            color="blue",
            label="Class 0",
            edgecolor="k",
            alpha=0.6,
        )
        plt.scatter(
            feature1[y == 1],
            feature2[y == 1],
            color="red",
            label="Class 1",
            edgecolor="k",
            alpha=0.6,
        )

        plt.xlabel(f"Feature {feature1_idx + 1}")
        plt.ylabel(f"Feature {feature2_idx + 1}")
        plt.title("Binary Decision Boundary with Uncertainty")
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()


if __name__ == "__main__":

    ############# Example Usage #############
    from bnn.modules.dense.binary_classification.mcmc import DenseBinaryMCMC
    from bnn.utils.entropy_analysis import EntropyAndMutualInformation
    from bnn.utils.generalization_bound import BayesianAnalysis
    from fake_data import generate_binary_classification_data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss
    import jax.random as jr
    import jax.numpy as jnp

    df = generate_binary_classification_data()
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )
    key = jr.key(0)
    classifier = DenseBinaryMCMC(
        num_warmup=500, num_samples=1000, model_type="deep", hidden_dim=10
    )
    classifier.fit(X_train, y_train, key)
    posteriors_preds = classifier.predict(X_test, key)
    probabilities = jax.nn.sigmoid(posteriors_preds)
    mcmc = classifier.retrieve_results()
    mean_preds = probabilities.mean(axis=0)
    loss = log_loss(np.array(y_test), np.array(mean_preds))
    print(f"loss: {loss}")
    classifier.visualize(X=X_test, y=y_test, feature_indices=(0, 1))
    analysis = BayesianAnalysis(len(X_train), delta=0.05, task_type="binary")
    # Compute PAC-Bayesian bound for MCMC

    bound = analysis.compute_pac_bayesian_bound(
        predictions=probabilities,
        y_true=y_test,
        posterior_samples=mcmc,
        layer_names=[name for name in mcmc.keys() if name != "logits"],
        inference_type="mcmc",
        prior_mean=0,
        prior_std=1,
    )

    print("PAC-Bayesian Bound (MCMC):", bound)
    mi = analysis.compute_mutual_information_bound(
        posterior_samples=mcmc,
        layer_names=[name for name in mcmc if name != "logits"],
        inference_type="mcmc",
    )
    print(f"Mutual Information bound: {mi}")
    uncertainty_quantification = EntropyAndMutualInformation("binary")
    mi, mi1 = uncertainty_quantification.compute_mutual_information(probabilities)
    uncertainty_quantification.visualize(mi, mi1)
