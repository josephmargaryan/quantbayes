import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from sklearn.model_selection import train_test_split

from quantbayes import bnn, fake_data
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
model.compile(num_samples=10, num_warmup=10)
model.fit(X_train, y_train, jr.PRNGKey(34))
model.visualize(X_test, y_test, posterior="likelihood")

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


import matplotlib.pyplot as plt
import numpy as np


def plot_fft_spectrum_with_uncertainty(fft_samples, show=True):
    """
    fft_samples: shape (num_samples, n)
    Computes the mean and credible intervals for the magnitude and phase,
    and plots them.
    """
    # Compute statistics across samples
    mag_samples = np.abs(fft_samples)  # shape (num_samples, n)
    phase_samples = np.angle(fft_samples)  # shape (num_samples, n)

    mag_mean = mag_samples.mean(axis=0)
    phase_mean = phase_samples.mean(axis=0)

    # Compute, for example, 95% quantiles
    mag_lower = np.percentile(mag_samples, 2.5, axis=0)
    mag_upper = np.percentile(mag_samples, 97.5, axis=0)
    phase_lower = np.percentile(phase_samples, 2.5, axis=0)
    phase_upper = np.percentile(phase_samples, 97.5, axis=0)

    freq_idx = np.arange(fft_samples.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(freq_idx, mag_mean, "b-", label="Mean Magnitude")
    axes[0].fill_between(
        freq_idx, mag_lower, mag_upper, color="blue", alpha=0.3, label="95% CI"
    )
    axes[0].set_title("FFT Magnitude with Uncertainty")
    axes[0].set_xlabel("Frequency index")
    axes[0].set_ylabel("Magnitude")
    axes[0].legend()

    axes[1].plot(freq_idx, phase_mean, "g-", label="Mean Phase")
    axes[1].fill_between(
        freq_idx, phase_lower, phase_upper, color="green", alpha=0.3, label="95% CI"
    )
    axes[1].set_title("FFT Phase with Uncertainty")
    axes[1].set_xlabel("Frequency index")
    axes[1].set_ylabel("Phase (radians)")
    axes[1].legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def visualize_circulant_kernel_with_uncertainty(
    fft_samples: np.ndarray, show: bool = True
):
    """
    Visualize the uncertainty in the time-domain circulant kernel.

    Parameters:
        fft_samples: np.ndarray of shape (num_samples, n)
            Array of FFT coefficients from multiple posterior samples.
        show: bool, if True, calls plt.show().

    Returns:
        fig: the matplotlib figure.
    """
    num_samples, n = fft_samples.shape

    # Compute the time-domain kernel for each sample using the inverse FFT.
    time_kernels = np.array(
        [np.fft.ifft(fft_sample).real for fft_sample in fft_samples]
    )
    # time_kernels has shape (num_samples, n)

    # Compute summary statistics for the time-domain kernel at each time index.
    kernel_mean = time_kernels.mean(axis=0)
    kernel_lower = np.percentile(time_kernels, 2.5, axis=0)
    kernel_upper = np.percentile(time_kernels, 97.5, axis=0)

    # For the circulant matrix, you could compute the mean circulant matrix
    # by taking the mean kernel and then rolling it.
    C_mean = np.stack([np.roll(kernel_mean, i) for i in range(n)], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the mean kernel with error bars for uncertainty.
    axes[0].errorbar(
        np.arange(n),
        kernel_mean,
        yerr=[kernel_mean - kernel_lower, kernel_upper - kernel_mean],
        fmt="o",
        color="b",
        ecolor="lightgray",
        capsize=3,
    )
    axes[0].set_title("Circulant Kernel (Time Domain) with Uncertainty")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Amplitude")

    # For the circulant matrix, show the mean matrix.
    im = axes[1].imshow(C_mean, cmap="viridis")
    axes[1].set_title("Mean Circulant Matrix")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Index")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


fft_list = []

# Loop over a number of samples from the posterior.
for i in range(100):
    # Build a parameter dictionary using the i-th sample from each posterior parameter.
    sample_param_dict = {key: value[i] for key, value in posterior_samples.items()}

    # Get the fft_full for this sample.
    fft_full = get_fft_full_for_given_params(
        model, X_test, sample_param_dict, rng_key=jr.PRNGKey(i)
    )
    fft_list.append(fft_full)

# Convert list of FFT results to a numpy array of shape (num_samples, n)
fft_samples = np.stack(fft_list, axis=0)

plot_fft_spectrum_with_uncertainty(fft_samples, show=True)
visualize_circulant_kernel_with_uncertainty(fft_samples, show=True)
