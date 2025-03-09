import jax
import jax.numpy as jnp
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
from quantbayes.stochax.utils import analyze_pre_activations, visualize_circulant_layer

df = generate_regression_data()
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)


class GaussianProcess(bnn.Module):
    def __init__(self):
        super().__init__(task_type="gp")

    def __call__(self, X, y=None):
        N, in_features = X.shape
        gp_layer = bnn.GaussianProcessLayer(
            input_dim=in_features, kernel_type="spectralmixture", name="gp_layer"
        )
        kernel_matrix = gp_layer(X)
        f = numpyro.sample(
            "f",
            dist.MultivariateNormal(
                loc=jnp.zeros(X.shape[0]), covariance_matrix=kernel_matrix
            ),
        )
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("data", N):
            numpyro.sample("obs", dist.Normal(f, sigma), obs=y)
        # Immediately extract concrete (non-traced) values:
        self.gp_layer = gp_layer
        self.kernel_matrix = kernel_matrix


tk, vk = jax.random.split(jax.random.key(123), 2)
model = GaussianProcess()
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

################### Fourier eigenvalues ########################


class FFT(bnn.Module):
    def __init__(self, in_features):
        super().__init__(method="nuts", task_type="regression")
        # Instantiate layers once.
        self.out_layer = bnn.Linear(in_features=in_features, out_features=1, name="out")

    def __call__(self, X, y=None):
        N, in_features = X.shape

        fft_layer = bnn.SmoothTruncCirculantLayer(
            in_features=in_features, alpha=1, K=7, name="fft_layer"
        )
        X_pre = fft_layer(X)
        # Apply nonlinearity.
        X_nl = jax.nn.tanh(X_pre)
        # Final linear mapping.
        X_out = self.out_layer(X_nl)
        logits = X_out.squeeze()
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("data", N):
            numpyro.sample("likelihood", dist.Normal(logits, sigma), obs=y)
        self.fft_layer = fft_layer

    def get_preactivations(self, X):
        """
        Compute and return the pre-activations from the FFT layer.
        We use jax.lax.stop_gradient to ensure that no tracer is leaked.
        """
        X_pre = self.fft_layer(X)
        return jax.lax.stop_gradient(X_pre)


model = FFT(1)
model.compile(num_warmup=500, num_samples=1000)
model.fit(X_train, y_train, tk)
fig1, fig2 = visualize_circulant_layer(model, X)
preds = model.predict(X_test, vk, posterior="likelihood")
mean_preds = preds.mean(axis=0)
rmse = np.sqrt(mean_squared_error(np.array(y_test), np.array(mean_pred)))
print(f"FFTCirc RMSE: {rmse}")
conv_matrix = analyze_pre_activations(model, X)

############### Scikit-Learn ##################

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from quantbayes.fake_data import generate_regression_data

# Generate synthetic regression data
X = df.drop("target", axis=1).values  # scikit-learn uses numpy arrays
y = df["target"].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)

# Define a composite kernel: a constant kernel times an RBF kernel plus a white noise kernel.
kernel = C(1.0, (1e-3, 1e3)) * RBF(
    length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))

# Instantiate the GaussianProcessRegressor
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

# Fit the GP model
gp.fit(X_train, y_train)

# Make predictions on the test set
y_pred, sigma = gp.predict(X_test, return_std=True)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Scikit-Learn GP RMSE:", rmse)

# Visualize the covariance matrix (kernel) computed on the training data
K_train = gp.kernel_(X_train)
plt.figure(figsize=(6, 5))
plt.imshow(K_train, cmap="viridis")
plt.title("Scikit-Learn GP Kernel (Covariance Matrix)")
plt.xlabel("Data Point Index")
plt.ylabel("Data Point Index")
plt.colorbar(label="Covariance")
plt.tight_layout()
plt.show()

# Optionally, print the optimized kernel parameters
print("Optimized Kernel:", gp.kernel_)
