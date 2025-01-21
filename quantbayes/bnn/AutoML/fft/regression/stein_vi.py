from quantbayes.bnn.core.base_task import BaseTask
from quantbayes.bnn.core.base_inference import BaseInference
from quantbayes.bnn.utils.fft_module import fft_matmul
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


class FFTRegressionSteinVI(BaseTask, BaseInference):
    """
    Bayesian Regression Model with FFT-based Circulant Matrix Layer using Stein Variational Inference.
    """

    def __init__(
        self,
        num_steps=1000,
        model_type="shallow",  # New argument
        num_particles=10,
    ):
        super().__init__()
        self.num_steps = num_steps
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
            return lambda X, y=None: self.regression_model(X, y)
        elif self.model_type == "deep":
            return lambda X, y=None: self.deep_regression_model(X, y)

    def regression_model(self, X, y=None):
        """
        Shallow Bayesian Regression Model with FFT-based Circulant Matrix Layer.
        """
        input_size = X.shape[1]
        num_particles = self.num_particles

        first_row = numpyro.sample(
            "first_row", dist.Normal(0, 1).expand([num_particles, input_size])
        )
        bias_circulant = numpyro.sample(
            "bias_circulant", dist.Normal(0, 1).expand([num_particles, input_size])
        )

        hidden_list = []
        for i in range(num_particles):
            hidden = fft_matmul(first_row[i], X) + bias_circulant[i]
            hidden = jax.nn.silu(hidden)
            hidden_list.append(hidden)

        hidden = jnp.stack(hidden_list, axis=0)

        weights_out = numpyro.sample(
            "weights_out", dist.Normal(0, 1).expand([num_particles, hidden.shape[2], 1])
        )
        bias_out = numpyro.sample("bias_out", dist.Normal(0, 1).expand([num_particles]))

        predictions = (
            jnp.einsum("pbi,pio->pbo", hidden, weights_out).squeeze(-1)
            + bias_out[:, None]
        )

        mean_predictions = jnp.mean(predictions, axis=0)
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("obs", dist.Normal(mean_predictions, sigma), obs=y)

    def deep_regression_model(self, X, y=None):
        """
        Deep Bayesian Regression Model with FFT-based Circulant Matrix Layer.
        """
        input_size = X.shape[1]
        num_particles = self.num_particles

        # First FFT layer
        first_row1 = numpyro.sample(
            "first_row1", dist.Normal(0, 1).expand([num_particles, input_size])
        )
        bias_circulant1 = numpyro.sample(
            "bias_circulant1", dist.Normal(0, 1).expand([num_particles, input_size])
        )

        hidden_list1 = []
        for i in range(num_particles):
            hidden1 = fft_matmul(first_row1[i], X) + bias_circulant1[i]
            hidden1 = jax.nn.silu(hidden1)
            hidden_list1.append(hidden1)

        hidden1 = jnp.stack(hidden_list1, axis=0)

        # Second FFT layer
        first_row2 = numpyro.sample(
            "first_row2", dist.Normal(0, 1).expand([num_particles, hidden1.shape[2]])
        )
        bias_circulant2 = numpyro.sample(
            "bias_circulant2",
            dist.Normal(0, 1).expand([num_particles, hidden1.shape[2]]),
        )

        hidden_list2 = []
        for i in range(num_particles):
            hidden2 = fft_matmul(first_row2[i], hidden1[i]) + bias_circulant2[i]
            hidden2 = jax.nn.silu(hidden2)
            hidden_list2.append(hidden2)

        hidden2 = jnp.stack(hidden_list2, axis=0)

        weights_out = numpyro.sample(
            "weights_out",
            dist.Normal(0, 1).expand([num_particles, hidden2.shape[2], 1]),
        )
        bias_out = numpyro.sample("bias_out", dist.Normal(0, 1).expand([num_particles]))

        predictions = (
            jnp.einsum("pbi,pio->pbo", hidden2, weights_out).squeeze(-1)
            + bias_out[:, None]
        )

        mean_predictions = jnp.mean(predictions, axis=0)
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("obs", dist.Normal(mean_predictions, sigma), obs=y)

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
        Perform Stein Variational inference.
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
            rng_key, self.num_steps, X_train, y_train, progress_bar=True
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
        predictions = predictive(rng_key, X_test)["obs"]

        return predictions

    def visualize(self, X_test, y_test, posteriors, feature_index=None):
        """
        Visualize predictions with uncertainty bounds and true targets.
        """
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        mean_preds = np.array(posteriors.mean(axis=0))
        lower_bound = np.percentile(posteriors, 2.5, axis=0)
        upper_bound = np.percentile(posteriors, 97.5, axis=0)

        if (
            X_test.shape[1] == 1
            or feature_index is None
            or not (0 <= feature_index < X_test.shape[1])
        ):
            feature_index = 0

        feature = X_test[:, feature_index]
        sorted_indices = np.argsort(feature)
        feature = feature[sorted_indices]
        y_test = y_test[sorted_indices]
        mean_preds = mean_preds[sorted_indices]
        lower_bound = lower_bound[sorted_indices]
        upper_bound = upper_bound[sorted_indices]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(feature, y_test, color="blue", alpha=0.6, label="True Targets")
        plt.plot(
            feature, mean_preds, color="red", label="Mean Predictions", linestyle="-"
        )
        plt.fill_between(
            feature,
            lower_bound,
            upper_bound,
            color="gray",
            alpha=0.3,
            label="Uncertainty Bounds",
        )
        plt.xlabel(f"Feature {feature_index + 1}")
        plt.ylabel("Target (y_test)")
        plt.title("Model Predictions with Uncertainty and True Targets")
        plt.legend()
        plt.grid(alpha=0.4)
        plt.show()


if __name__ == "__main__":

    ############# Example Usage #############
    from bnn.AutoML.fft.regression.stein_vi import FFTRegressionSteinVI
    from bnn.utils.entropy_analysis import EntropyAndMutualInformation, BayesianAnalysis
    from fake_data import generate_regression_data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import jax.random as jr
    import jax.numpy as jnp

    target_scaler = MinMaxScaler()
    feature_scaler = StandardScaler()
    df = generate_regression_data()
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X = feature_scaler.fit_transform(X)
    y = target_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )
    key = jr.key(0)
    regressor = FFTRegressionSteinVI(num_steps=1000, model_type="deep")
    regressor.fit(X_train, y_train, key)
    posteriors_preds = regressor.predict(X_test, key)
    stein_results = regressor.retrieve_results()
    mean_preds = posteriors_preds.mean(axis=0)
    MAE = mean_absolute_error(np.array(y_test), np.array(mean_preds))
    print(f"Mean Absolute Error: {MAE}")
    regressor.visualize(X_test, y_test, posteriors_preds, 0)
    analysis = BayesianAnalysis(len(X_train), delta=0.05, task_type="regression")
    # Compute PAC-Bayesian bound for MCMC
    from jax.tree_util import tree_flatten

    def extract_layer_names(stein_result):
        param_tree = stein_result.params
        flattened, _ = tree_flatten(param_tree)
        return [name for name in param_tree if not name.startswith(("prec", "sigma"))]

    layer_names = extract_layer_names(stein_results)

    bound = analysis.compute_pac_bayesian_bound(
        predictions=posteriors_preds,
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
    uncertainty_quantification = EntropyAndMutualInformation("regression")
    mi, mi1 = uncertainty_quantification.compute_mutual_information(posteriors_preds)
    uncertainty_quantification.visualize(mi, mi1)
