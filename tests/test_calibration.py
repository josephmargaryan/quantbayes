import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from sklearn.model_selection import train_test_split

from quantbayes import bnn
from quantbayes.bnn.utils import (
    plot_calibration_curve,
    expected_calibration_error,
    CalibratedBayesNet,
)
from quantbayes.fake_data import generate_binary_classification_data


# Define your Bayesian neural network model
class BayesNet(bnn.Module):
    def __init__(self):
        super().__init__(task_type="binary", method="nuts")

    def __call__(self, X, y=None):
        N, D = X.shape
        X = bnn.Linear(
            in_features=D,
            out_features=10,
            weight_prior_fn=lambda shape: dist.Cauchy(0, 1)
            .expand(shape)
            .to_event(len(shape)),
            name="first layer",
        )(X)
        X = jax.nn.silu(X)
        X = bnn.Linear(
            in_features=10,
            out_features=1,
            name="out layer",
            weight_prior_fn=lambda shape: dist.Cauchy(0, 1)
            .expand(shape)
            .to_event(len(shape)),
        )(X)
        logits = X.squeeze()
        numpyro.deterministic("logits", logits)
        with numpyro.plate("data", N):
            numpyro.sample("likelihood", dist.Bernoulli(logits=logits), obs=y)


# Example usage
if __name__ == "__main__":
    # Generate synthetic binary classification data
    df = generate_binary_classification_data()
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)

    # Split data into training, calibration, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Further split training data into training and calibration sets
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )

    # Create a random key for JAX
    rng_key = jax.random.PRNGKey(0)

    # Instantiate the Bayesian network and wrap it in the calibrated model using a specified calibration method.
    # Try changing calibration_method to "temp", "isotonic", or "platt"
    bayes_net = BayesNet()
    calibrated_model = CalibratedBayesNet(bayes_net, calibration_method="temp")

    # Fit the model using training and calibration data
    calibrated_model.fit(
        X_train,
        y_train,
        X_calib,
        y_calib,
        rng_key,
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
    )

    # Get calibrated prediction probabilities on the test set
    probs = calibrated_model.predict_proba(X_test, rng_key)
    preds = calibrated_model.predict(X_test, rng_key)

    # Average over the posterior samples to get mean predictions
    mean_probs = probs.mean(axis=0)

    # Plot the calibration curve for the positive class
    plot_calibration_curve(np.array(y_test), mean_probs[:, 1])
    ece_calibrated_model = expected_calibration_error(y_test, mean_probs)
    print(f"ECE of calibrated model: {ece_calibrated_model}")

    # Compare with the uncalibrated model
    normal_model = BayesNet()
    normal_model.compile(num_warmup=500, num_samples=1000, num_chains=1)
    normal_model.fit(X_train, y_train, rng_key)
    preds = normal_model.predict(X_test, rng_key)
    mean_probs_normal = jax.nn.sigmoid(preds).mean(axis=0)
    plot_calibration_curve(np.array(y_test), mean_probs_normal)
    ece_normal_model = expected_calibration_error(y_test, mean_probs_normal)
    print(f"ECE of normal model: {ece_normal_model}")
