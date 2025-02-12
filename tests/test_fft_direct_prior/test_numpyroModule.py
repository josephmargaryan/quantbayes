import jax
import jax.random as jr
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from sklearn.model_selection import train_test_split

from quantbayes import bnn
from quantbayes import fake_data
from quantbayes.bnn.utils import BayesianAnalysis
from quantbayes.stochax.utils import (
    get_fft_full_for_given_params,
    plot_fft_spectrum,
    visualize_circulant_kernel,
)

# Generate synthetic regression data.
df = fake_data.generate_regression_data(n_categorical=10, n_continuous=10)
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define your model.
class MyBayesNetFFTDirect(bnn.Module):
    def __init__(self):
        super().__init__(method="nuts", task_type="regression")
        # Additional attributes if needed

    def __call__(self, X, y=None):
        N, in_features = X.shape
        # 1) Use your direct-Fourier prior layer.
        fft_layer = bnn.FFTDirectPriorLinear(
            in_features=in_features, name="fft_direct_1"
        )
        hidden = fft_layer(X)
        hidden = jax.nn.tanh(hidden)
        # 2) Add another (standard) linear layer.
        w = numpyro.sample("w", dist.Normal(0, 1).expand([in_features, 1]))
        b = numpyro.sample("b", dist.Normal(0, 1))
        out = jnp.dot(hidden, w).squeeze(-1) + b
        numpyro.deterministic("logits", out)

        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("data", N):
            numpyro.sample("likelihood", dist.Normal(out, sigma), obs=y)

        # Store the FFT layer for later visualization.
        self.fft_layer = fft_layer


# Create and compile your model.
train_key, val_key = jr.split(jr.PRNGKey(123), 2)
model = MyBayesNetFFTDirect()
model.compile()
model.fit(X_train, y_train, jr.PRNGKey(34))
# model.visualize(X_test, y_test, posterior="likelihood")

# Generate predictions.
posterior_preds = model.predict(X_test, val_key, posterior="logits")
posterior_samples = model.get_samples

# Perform Bayesian analysis.
bound = BayesianAnalysis(
    num_samples=len(X_train),
    delta=0.05,
    task_type="regression",
    inference_type="mcmc",
    posterior_samples=posterior_samples,
)

bound.compute_pac_bayesian_bound(
    predictions=posterior_preds, y_true=y_test, prior_mean=0.0, prior_std=1.0
)
bound.compute_pac_bayesian_bound(
    predictions=posterior_preds, y_true=y_test, prior_mean=0.0, prior_std=1.0
)

# --- Post-hoc FFT Visualization ---

# (1) Choose a concrete parameter set from the posterior.
# Here we take the first sample for each parameter.
param_dict = {key: value[0] for key, value in posterior_samples.items()}

# (2) Perform a forward pass with a valid RNG key to get a concrete fft_full.
fft_full = get_fft_full_for_given_params(
    model, X_test, param_dict, rng_key=jr.PRNGKey(0)
)

# (3) Plot the Fourier spectrum and circulant kernel.
fig1 = plot_fft_spectrum(fft_full, show=True)
fig2 = visualize_circulant_kernel(fft_full, show=True)
