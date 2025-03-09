import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from sklearn.model_selection import train_test_split

from quantbayes import bnn, fake_data
from quantbayes.bnn.utils import BayesianAnalysis, plot_hdi
from quantbayes.stochax.utils import (
    BlockFFTDirectDet,
    bayesianize,
    plot_fft_spectrum,
    prior_fn,
    visualize_circulant_kernel,
)


# --- Deterministic network using FFTDirectPriorLinear ---
class MyDeterministicNet(eqx.Module):
    layer1: eqx.Module  # FFT-based layer
    layer2: eqx.Module  # Standard linear layer

    def __init__(self, in_features, *, key):
        k1, k2 = jr.split(key, 2)
        self.layer1 = BlockFFTDirectDet(
            in_features=in_features,
            out_features=16,
            block_size=4,
            key=k1,
            init_scale=1.0,
        )
        self.layer2 = eqx.nn.Linear(in_features=in_features, out_features=1, key=k2)

    def __call__(self, x):
        x = self.layer1(x)
        x = jax.nn.tanh(x)
        return self.layer2(x)


def student_t_prior(shape):
    return prior_fn(
        shape,
        mean=0.0,
        std=1.0,
        dist_cls=lambda loc, scale: dist.StudentT(3, loc, scale),
    )


# --- Bayesian network that bayesianizes the deterministic network ---
class MyBayesianNet(bnn.Module):
    def __init__(self, in_features, key):
        super().__init__(method="nuts", task_type="regression")
        # Build the deterministic network once using a concrete key.
        self.deterministic = MyDeterministicNet(in_features, key=key)
        # Keep a reference for visualization.
        self.fft_layer = self.deterministic.layer1

    def __call__(self, X, y=None):
        # Bayesianize the deterministic network inside __call__
        # so that the sample sites are executed in a proper NumPyro context.
        bayesian_net = bayesianize(self.deterministic, student_t_prior)
        pred = jax.vmap(bayesian_net)(X)
        logits = pred.squeeze()
        numpyro.deterministic("logits", logits)
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("likelihood", dist.Normal(logits, sigma), obs=y)


# --- Main script ---
# Generate synthetic data.
df = fake_data.generate_regression_data(n_categorical=0, n_continuous=1)
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and compile the Bayesian model.
in_features = X_train.shape[1]
train_key, val_key = jr.split(jr.PRNGKey(123), 2)
model = MyBayesianNet(in_features, key=jr.PRNGKey(23))
model.compile(num_warmup=10, num_samples=10)
model.fit(X_train, y_train, train_key)

# Visualize predictions.
model.visualize(X_test, y_test, posterior="likelihood", feature_index=0)

# Obtain predictions and compute bounds.
posterior_preds = model.predict(X_test, val_key, posterior="logits")
posterior_samples = model.get_samples
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

# --- Trigger a forward pass on the FFT layer ---
_ = model.fft_layer(
    X_test[:1]
)  # Ensure the FFT layer computes and stores its Fourier coefficients.
fft_full = model.fft_layer.get_fourier_coeffs()  # Now this should be concrete.
fig1 = plot_fft_spectrum(fft_full, show=True)
fig2 = visualize_circulant_kernel(fft_full, show=True)
preds = model.predict(X_test, val_key, posterior="logits")
plot_hdi(preds, X_test)
