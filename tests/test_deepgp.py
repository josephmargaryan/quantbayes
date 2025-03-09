import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from quantbayes import bnn
from quantbayes.bnn.utils import (
    predict_gp,
    sample_gp_prior,
    visualize_gp_kernel,
    visualize_predictions,
)
from quantbayes.fake_data import generate_regression_data

df = generate_regression_data()
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)
print("shape of x train", X_train.shape)


class FeatureExtractor(eqx.Module):
    l1: eqx.Module
    l2: eqx.Module

    def __init__(self, key):
        k1, k2 = jr.split(key, 2)
        self.l1 = eqx.nn.Linear(1, 12, key=k1)
        self.l2 = eqx.nn.Linear(12, 24, key=k2)

    def __call__(self, x):
        x = self.l1(x)
        x = jax.nn.gelu(x)
        x = self.l2(x)
        return x


class GP(bnn.Module):
    def __init__(self):
        super().__init__(method="svi")
        # Instantiate the feature extractor once; its parameters will be trained jointly.
        self.feature_extractor = FeatureExtractor(jr.key(0))
        # You might also instantiate the GP layer here if you prefer.
        self.gp_layer = bnn.GaussianProcessLayer(
            input_dim=24, kernel_type="spectralmixture", name="gp_layer"
        )

    def __call__(self, X, y=None):
        N, in_features = X.shape
        # Use the feature extractor to transform the data.
        X_features = jax.vmap(self.feature_extractor)(X)
        # Compute the kernel matrix from the transformed features.
        kernel_matrix = self.gp_layer(X_features)
        # GP prior over function values.
        f = numpyro.sample(
            "f",
            dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=kernel_matrix),
        )
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("data", N):
            numpyro.sample("obs", dist.Normal(f, sigma), obs=y)
        # Store concrete values for later use.
        self.kernel_matrix = kernel_matrix


tk, vk = jr.split(jr.key(123), 2)
model = GP()
model.compile(num_warmup=500, num_samples=1000)
model.fit(X_train, y_train, tk)


preds = predict_gp(model, X_train, y_train, X_test)
mean_pred, var_pred = preds

# Compute RMSE (make sure y_test is a NumPy array)
y_test_np = np.array(y_test)
rmse = np.sqrt(mean_squared_error(y_test_np, mean_pred))
print("Gaussian Process RMSE:", rmse)

# Visualize the predictions with uncertainty (±2 standard deviations)
fig = visualize_predictions(X_test, mean_pred, var_pred)


visualize_gp_kernel(model.gp_layer, X_test)
sample_gp_prior(model.gp_layer, X_train, num_samples=5)
