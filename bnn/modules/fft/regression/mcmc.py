from BNN.core.base_task import BaseTask
from BNN.core.base_inference import BaseInference
from BNN.utils.fft_module import fft_matmul
from numpyro.infer import MCMC, NUTS, Predictive
import jax
import jax.numpy as jnp
import numpyro
import numpy as np
import matplotlib.pyplot as plt
import numpyro.distributions as dist


class FFTRegressionMCMC(BaseTask, BaseInference):
    """
    Bayesian Regression Model with FFT-based Circulant Matrix Layer using MCMC inference.
    """

    def __init__(self, num_warmup=500, num_samples=1000, model_type="shallow"):
        super().__init__()
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.model_type = model_type.lower()
        self.mcmc = None

        # Validate the model_type
        if self.model_type not in ["shallow", "deep"]:
            raise ValueError("model_type must be 'shallow' or 'deep'.")

    def get_default_model(self):
        """
        Return the appropriate model based on the model_type.
        """
        if self.model_type == "shallow":
            return self.regression_model
        elif self.model_type == "deep":
            return self.deep_regression_model

    def regression_model(self, X, y=None):
        """
        Shallow Bayesian Regression Model with FFT-based Circulant Matrix Layer.
        """
        input_size = X.shape[1]

        first_row = numpyro.sample("first_row", dist.Normal(0, 1).expand([input_size]))
        bias_circulant = numpyro.sample(
            "bias_circulant", dist.Normal(0, 1).expand([input_size])
        )

        hidden = fft_matmul(first_row, X) + bias_circulant
        hidden = jax.nn.tanh(hidden)

        weights_out = numpyro.sample(
            "weights_out", dist.Normal(0, 1).expand([hidden.shape[1], 1])
        )
        bias_out = numpyro.sample("bias_out", dist.Normal(0, 1))

        predictions = jnp.matmul(hidden, weights_out).squeeze() + bias_out

        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("mean", dist.Normal(predictions, sigma), obs=y)

    def deep_regression_model(self, X, y=None):
        """
        Deep Bayesian Regression Model with FFT-based Circulant Matrix Layer.
        """
        input_size = X.shape[1]

        # First FFT layer
        first_row1 = numpyro.sample(
            "first_row1", dist.Normal(0, 1).expand([input_size])
        )
        bias_circulant1 = numpyro.sample(
            "bias_circulant1", dist.Normal(0, 1).expand([input_size])
        )

        hidden1 = fft_matmul(first_row1, X) + bias_circulant1
        hidden1 = jax.nn.tanh(hidden1)

        # Second FFT layer
        first_row2 = numpyro.sample(
            "first_row2", dist.Normal(0, 1).expand([hidden1.shape[1]])
        )
        bias_circulant2 = numpyro.sample(
            "bias_circulant2", dist.Normal(0, 1).expand([hidden1.shape[1]])
        )

        hidden2 = fft_matmul(first_row2, hidden1) + bias_circulant2
        hidden2 = jax.nn.tanh(hidden2)

        # Output layer
        weights_out = numpyro.sample(
            "weights_out", dist.Normal(0, 1).expand([hidden2.shape[1], 1])
        )
        bias_out = numpyro.sample("bias_out", dist.Normal(0, 1))

        predictions = jnp.matmul(hidden2, weights_out).squeeze() + bias_out
        predictions = jnp.clip(predictions, -1, 1)
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("mean", dist.Normal(predictions, sigma), obs=y)

    def bayesian_inference(self, X_train, y_train, rng_key):
        """
        Perform MCMC inference.
        """
        kernel = NUTS(self.get_default_model())
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
        if not self.fitted or self.mcmc is None:
            raise RuntimeError("Model must be fitted before making predictions.")
        posterior_samples = self.mcmc.get_samples()
        predictive = Predictive(self.get_default_model(), posterior_samples)
        preds = predictive(rng_key=rng_key, X=X_test)
        return preds["mean"]

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
    from BNN.modules.fft.regression.mcmc import FFTRegressionMCMC
    from BNN.utils.entropy_analysis import EntropyAndMutualInformation
    from BNN.utils.generalization_bound import BayesianAnalysis
    from fake_data import generate_regression_data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import jax.random as jr
    import jax.numpy as jnp

    target_scaler = MinMaxScaler()
    feature_scaler = StandardScaler()
    df = generate_regression_data()
    X, y = df.drop("target", axis=1).values, df["target"].values
    X = feature_scaler.fit_transform(X)
    y = target_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )
    key = jr.key(0)
    regressor = FFTRegressionMCMC(model_type="deep")
    regressor.fit(X_train, y_train, key)
    posteriors_preds = regressor.predict(X_test, key)
    mcmc = regressor.retrieve_results()
    mean_preds = posteriors_preds.mean(axis=0)
    MAE = mean_absolute_error(np.array(y_test), np.array(mean_preds))
    print(f"Mean Absolute Error: {MAE}")
    regressor.visualize(X_test, y_test, posteriors_preds, 0)
    analysis = BayesianAnalysis(len(X_train), delta=0.05, task_type="regression")
    # Compute PAC-Bayesian bound for MCMC
    bound = analysis.compute_pac_bayesian_bound(
        predictions=posteriors_preds,
        y_true=y_test,
        posterior_samples=mcmc,
        layer_names=[name for name in mcmc.keys() if name != "mean"],
        inference_type="mcmc",
        prior_mean=0,
        prior_std=1,
    )
    print("PAC-Bayesian Bound (MCMC):", bound)
    mi = analysis.compute_mutual_information_bound(
        posterior_samples=mcmc,
        layer_names=[name for name in mcmc if name != "mean"],
        inference_type="mcmc",
    )
    print(f"Mutual Information bound: {mi}")
    uncertainty_quantification = EntropyAndMutualInformation("regression")
    mi, mi1 = uncertainty_quantification.compute_mutual_information(posteriors_preds)
    uncertainty_quantification.visualize(mi, mi1)