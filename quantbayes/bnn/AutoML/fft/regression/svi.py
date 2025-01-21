from quantbayes.bnn.core.base_task import BaseTask
from quantbayes.bnn.core.base_inference import BaseInference
from quantbayes.bnn.utils.fft_module import fft_matmul
from typing import Callable
import jax
import jax.numpy as jnp
import numpyro
import numpy as np
import matplotlib.pyplot as plt
from numpyro.optim import Adam
from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide
from jax import random
import numpyro.distributions as dist


class FFTRegressionSVI(BaseTask, BaseInference):
    """
    Bayesian Regression Model with FFT-based Circulant Matrix Layer using SVI inference.
    """

    def __init__(self, num_steps=500, model_type="shallow", track_loss=False):
        super().__init__()
        self.num_steps = num_steps
        self.model_type = model_type.lower()
        self.track_loss = track_loss
        self.svi = None
        self.params = None
        self.loss = None

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

        first_row = numpyro.sample("first_row", dist.Normal(0, 1).expand([input_size]))
        bias_circulant = numpyro.sample(
            "bias_circulant", dist.Normal(0, 1).expand([input_size])
        )

        hidden = fft_matmul(first_row, X) + bias_circulant
        hidden = jax.nn.silu(hidden)

        weights_out = numpyro.sample(
            "weights_out", dist.Normal(0, 1).expand([hidden.shape[1], 1])
        )
        bias_out = numpyro.sample("bias_out", dist.Normal(0, 1))

        predictions = jnp.matmul(hidden, weights_out).squeeze() + bias_out

        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("mean", dist.Normal(predictions, sigma), obs=y)

    def deep_regression_model(self, X, y=None):
        """
        Deep BNN using FFT
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
        hidden1 = jax.nn.silu(hidden1)

        # Second FFT layer
        first_row2 = numpyro.sample(
            "first_row2", dist.Normal(0, 1).expand([hidden1.shape[1]])
        )
        bias_circulant2 = numpyro.sample(
            "bias_circulant2", dist.Normal(0, 1).expand([hidden1.shape[1]])
        )

        hidden2 = fft_matmul(first_row2, hidden1) + bias_circulant2
        hidden2 = jax.nn.silu(hidden2) + hidden1

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
        Perform SVI inference for the Bayesian regression model.
        """
        model = self.get_default_model()
        guide = autoguide.AutoNormal(model)
        optimizer = Adam(0.01)

        self.svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        svi_state = self.svi.init(rng_key, X_train, y_train)

        loss_progression = [] if self.track_loss else None

        for step in range(self.num_steps):
            svi_state, loss = self.svi.update(svi_state, X_train, y_train)
            if self.track_loss:
                loss_progression.append(loss)
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss:.4f}")

        self.params = self.svi.get_params(svi_state)
        if self.track_loss:
            self.loss = loss_progression

    def retrieve_results(self) -> dict:
        """
        Retrieve inference results, optionally including losses.
        """
        if not self.params:
            raise RuntimeError("No inference results available. Fit the model first.")
        if self.track_loss:
            return {"svi": self.svi, "params": self.params, "loss": self.loss}
        return {"svi": self.svi, "params": self.params}

    def fit(self, X_train, y_train, rng_key):
        """
        Fit the model using MCMC.
        """
        self.bayesian_inference(X_train, y_train, rng_key)
        self.fitted = True

    def predict(self, X_test, rng_key, num_samples=100):
        """
        Predict regression values using the posterior.
        """
        if not self.fitted or self.svi is None:
            raise RuntimeError("Model must be fitted before making predictions.")
        predictive = Predictive(
            self.svi.model,
            guide=self.svi.guide,
            params=self.params,
            num_samples=num_samples,
        )
        rng_key = random.PRNGKey(1)
        pred_samples = predictive(rng_key, X=X_test)
        return pred_samples["mean"]

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
    from bnn.AutoML.fft.regression.svi import FFTRegressionSVI
    from bnn.utils.entropy_analysis import EntropyAndMutualInformation, BayesianAnalysis
    from fake_data import generate_regression_data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import mean_absolute_error
    import jax.random as jr
    import jax.numpy as jnp

    df = generate_regression_data()
    X, y = df.drop("target", axis=1), df["target"]
    target_scaler = MinMaxScaler()
    feature_scaler = (
        StandardScaler()
    )  # Using Gaussian priors with mu=0, sigma=1 its important to scale the targets to [0, 1]
    X, y = jnp.array(X), jnp.array(y)
    X = feature_scaler.fit_transform(X)
    y = target_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )
    key = jr.key(0)
    regressor = FFTRegressionSVI(num_steps=1000, model_type="deep", track_loss=True)
    regressor.fit(X_train, y_train, key)
    posteriors_preds = regressor.predict(X_test, key)
    results = regressor.retrieve_results()
    svi = results["svi"]
    params = results["params"]
    loss = results.get("loss")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss) + 1), loss, label="Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss over steps (Convergence)")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    mean_preds = posteriors_preds.mean(axis=0)
    MAE = mean_absolute_error(np.array(y_test), np.array(mean_preds))
    print(f"Mean Absolute Error: {MAE}")
    regressor.visualize(X_test, y_test, posteriors_preds, 0)
    analysis = BayesianAnalysis(len(X_train), delta=0.05, task_type="regression")
    layer_names = sorted(
        set(key.rsplit("_", 2)[0] for key in params.keys() if key.endswith("_auto_loc"))
    )
    # Compute PAC-Bayesian bound for SVI
    bound = analysis.compute_pac_bayesian_bound(
        predictions=posteriors_preds,
        y_true=y_test,
        posterior_samples=params,
        layer_names=layer_names,
        inference_type="svi",
        prior_mean=0,
        prior_std=1,
    )
    print("PAC-Bayesian Bound (SVI):", bound)
    mi = analysis.compute_mutual_information_bound(
        posterior_samples=params,
        layer_names=layer_names,
        inference_type="svi",
    )
    print(f"Mutual Information bound: {mi}")
    uncertainty_quantification = EntropyAndMutualInformation("regression")
    mi, mi1 = uncertainty_quantification.compute_mutual_information(posteriors_preds)
    uncertainty_quantification.visualize(mi, mi1)
