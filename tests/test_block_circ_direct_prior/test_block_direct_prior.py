import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from sklearn.model_selection import train_test_split

from quantbayes import bnn, fake_data
from quantbayes.bnn.utils import BayesianAnalysis
from quantbayes.stochax.utils import (
    get_block_fft_full_for_given_params,
    plot_block_fft_spectra,
    visualize_block_circulant_kernels,
)

# Generate synthetic regression data.
df = fake_data.generate_regression_data(n_categorical=0, n_continuous=1)
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Define your model.
class MyCircBlock(bnn.Module):
    def __init__(self):
        super().__init__(task_type="regression", method="nuts")

    def __call__(self, X, y=None):
        N, in_features = X.shape
        block_layer = bnn.BlockFFTDirectPriorLayer(
            in_features=in_features,
            out_features=16,
            block_size=4,
            name="block_fft_layer",
        )
        X = block_layer(X)
        X = jax.nn.tanh(X)
        X = bnn.Linear(in_features=16, out_features=1, name="out")(X)
        logits = X.squeeze()
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        logits = numpyro.deterministic("logits", logits)
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("likelihood", dist.Normal(logits, sigma), obs=y)

        # Store the FFT layer for later visualization.
        self.block_layer = block_layer


# Create and compile your model.
train_key, val_key = jr.split(jr.PRNGKey(123), 2)
model = MyCircBlock()
model.compile(num_warmup=10, num_samples=10)
model.fit(X_train, y_train, jr.PRNGKey(34))
model.visualize(X_test, y_test, posterior="likelihood")

# Generate predictions.
posterior_preds = model.predict(X_test, val_key, posterior="likelihood")
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
print(posterior_samples.keys())

# --- Post-hoc FFT Visualization ---

# (1) Choose a concrete parameter set from the posterior.
# Here we take the first sample for each parameter.
param_dict = {
    key: value[0] for key, value in posterior_samples.items() if key != "logits"
}
# (2) Perform a forward pass with a valid RNG key to get a concrete fft_full.
fft_full = get_block_fft_full_for_given_params(
    model,
    X_test,
    param_dict,
    rng_key=jr.PRNGKey(123),
)

# (3) Plot the Fourier spectrum and circulant kernel.
fig1 = plot_block_fft_spectra(fft_full, show=True)
fig2 = visualize_block_circulant_kernels(fft_full, show=True)


import matplotlib.pyplot as plt
import numpy as np


def plot_block_fft_spectra_with_phase(fft_full_blocks: jnp.ndarray, show: bool = True):
    """
    Plot both the magnitude and phase for each block weight matrix using subplots.
    Expects fft_full_blocks with shape (k_out, k_in, block_size) (complex).
    """
    fft_blocks = np.asarray(fft_full_blocks)
    k_out, k_in, b = fft_blocks.shape
    total = k_out * k_in

    # Create a grid for each block where we have two subplots (mag and phase)
    nrows = int(np.ceil(np.sqrt(total)))
    ncols = int(np.ceil(total / nrows))

    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(nrows, 2, ncols)  # shape (nrows, 2, ncols)
    axes = axes.reshape(-1, ncols)  # for easier iteration over rows

    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            if idx < total:
                # determine block indices
                block_row = idx // k_in
                block_col = idx % k_in
                fft_block = fft_blocks[block_row, block_col]
                mag = np.abs(fft_block)
                phase = np.angle(fft_block)

                # Magnitude subplot (top row for this block)
                ax_mag = axes[i * 2, j]
                ax_mag.stem(mag, linefmt="b-", markerfmt="bo", basefmt="r-")
                ax_mag.set_title(f"Block ({block_row},{block_col}) Mag")
                ax_mag.set_xlabel("Freq index")
                ax_mag.set_ylabel("Magnitude")

                # Phase subplot (bottom row for this block)
                ax_phase = axes[i * 2 + 1, j]
                ax_phase.stem(phase, linefmt="g-", markerfmt="go", basefmt="r-")
                ax_phase.set_title(f"Block ({block_row},{block_col}) Phase")
                ax_phase.set_xlabel("Freq index")
                ax_phase.set_ylabel("Phase (rad)")

                idx += 1
            else:
                # Hide unused subplots
                for k in range(2):
                    axes[i * 2 + k, j].set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_block_fft_spectra_with_uncertainty(
    fft_samples_blocks: np.ndarray, show: bool = True
):
    """
    Plot the mean and 95% credible intervals for the magnitude and phase of each block's FFT.

    Parameters:
      fft_samples_blocks: np.ndarray of shape (num_samples, k_out, k_in, block_size)
      show: whether to call plt.show() at the end.

    Returns:
      A matplotlib figure.
    """
    num_samples, k_out, k_in, b = fft_samples_blocks.shape
    total = k_out * k_in

    # Prepare grids to store statistics for each block.
    # We'll create separate figures for magnitude and phase.
    fig_mag, axes_mag = plt.subplots(
        int(np.ceil(np.sqrt(total))),
        int(np.ceil(total / np.ceil(np.sqrt(total)))),
        figsize=(4 * int(np.ceil(np.sqrt(total))), 3 * int(np.ceil(np.sqrt(total)))),
    )
    fig_phase, axes_phase = plt.subplots(
        int(np.ceil(np.sqrt(total))),
        int(np.ceil(total / np.ceil(np.sqrt(total)))),
        figsize=(4 * int(np.ceil(np.sqrt(total))), 3 * int(np.ceil(np.sqrt(total)))),
    )

    axes_mag = np.array(axes_mag).flatten()
    axes_phase = np.array(axes_phase).flatten()

    for idx in range(total):
        # Determine block indices.
        i = idx // k_in
        j = idx % k_in

        # Extract all samples for block (i,j) => shape (num_samples, b)
        block_samples = fft_samples_blocks[:, i, j, :]  # complex values

        # Compute magnitude and phase samples: shape (num_samples, b)
        mag_samples = np.abs(block_samples)
        phase_samples = np.angle(block_samples)

        # Compute mean and 95% CI for magnitude.
        mag_mean = mag_samples.mean(axis=0)
        mag_lower = np.percentile(mag_samples, 2.5, axis=0)
        mag_upper = np.percentile(mag_samples, 97.5, axis=0)

        # Compute mean and 95% CI for phase.
        phase_mean = phase_samples.mean(axis=0)
        phase_lower = np.percentile(phase_samples, 2.5, axis=0)
        phase_upper = np.percentile(phase_samples, 97.5, axis=0)

        freq_idx = np.arange(b)

        # Plot magnitude uncertainty.
        ax_mag = axes_mag[idx]
        ax_mag.plot(freq_idx, mag_mean, "b-", label="Mean")
        ax_mag.fill_between(
            freq_idx, mag_lower, mag_upper, color="blue", alpha=0.3, label="95% CI"
        )
        ax_mag.set_title(f"Block ({i},{j}) Mag")
        ax_mag.set_xlabel("Freq index")
        ax_mag.set_ylabel("Magnitude")
        ax_mag.legend(fontsize=8)

        # Plot phase uncertainty.
        ax_phase = axes_phase[idx]
        ax_phase.plot(freq_idx, phase_mean, "g-", label="Mean")
        ax_phase.fill_between(
            freq_idx, phase_lower, phase_upper, color="green", alpha=0.3, label="95% CI"
        )
        ax_phase.set_title(f"Block ({i},{j}) Phase")
        ax_phase.set_xlabel("Freq index")
        ax_phase.set_ylabel("Phase (rad)")
        ax_phase.legend(fontsize=8)

    # Hide any extra subplots.
    for ax in axes_mag[total:]:
        ax.set_visible(False)
    for ax in axes_phase[total:]:
        ax.set_visible(False)

    fig_mag.tight_layout()
    fig_phase.tight_layout()
    if show:
        plt.show()
    return fig_mag, fig_phase


def visualize_block_circulant_kernels_with_uncertainty(
    fft_samples_blocks: np.ndarray, show: bool = True
):
    """
    Visualize the uncertainty in the time-domain circulant kernels for each block.

    Parameters:
      fft_samples_blocks: np.ndarray of shape (num_samples, k_out, k_in, block_size)
      show: whether to call plt.show() at the end.

    Returns:
      A matplotlib figure.
    """
    num_samples, k_out, k_in, b = fft_samples_blocks.shape
    total = k_out * k_in

    # For each block, compute the time-domain kernels from each FFT sample.
    # We'll get an array of shape (num_samples, b) per block.
    # Then compute the mean and 95% CI across samples.
    fig, axes = plt.subplots(
        int(np.ceil(np.sqrt(total))),
        int(np.ceil(total / np.ceil(np.sqrt(total)))),
        figsize=(4 * int(np.ceil(np.sqrt(total))), 3 * int(np.ceil(np.sqrt(total)))),
    )
    axes = np.array(axes).flatten()

    for idx in range(total):
        i = idx // k_in
        j = idx % k_in

        # For block (i,j), get FFT samples.
        block_fft_samples = fft_samples_blocks[:, i, j, :]  # shape (num_samples, b)
        # Compute time-domain kernels via IFFT (per sample)
        time_kernels = np.array(
            [np.fft.ifft(sample).real for sample in block_fft_samples]
        )
        # time_kernels: shape (num_samples, b)

        # Compute mean and 95% CI for the kernel.
        # Compute mean and 95% CI for the kernel.
        kernel_mean = time_kernels.mean(axis=0)
        kernel_lower = np.percentile(time_kernels, 2.5, axis=0)
        kernel_upper = np.percentile(time_kernels, 97.5, axis=0)

        # Compute error bars and clip any negative values:
        lower_err = np.clip(kernel_mean - kernel_lower, a_min=0, a_max=None)
        upper_err = np.clip(kernel_upper - kernel_mean, a_min=0, a_max=None)

        # Plot error bars for the kernel using the clipped error values directly:
        ax = axes[idx]
        ax.errorbar(
            np.arange(b),
            kernel_mean,
            yerr=[lower_err, upper_err],
            fmt="o",
            color="b",
            ecolor="lightgray",
            capsize=3,
        )

        ax.set_title(f"Block ({i},{j}) Kernel")
        ax.set_xlabel("Time index")
        ax.set_ylabel("Amplitude")

    # Hide any extra subplots.
    for ax in axes[total:]:
        ax.set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


plot_block_fft_spectra_with_phase(fft_full)

fft_list = []
n_samples = next(iter(posterior_samples.values())).shape[0]
# Loop over a number of samples from the posterior.
for i in range(n_samples):
    # Build a parameter dictionary using the i-th sample for each parameter,
    # but filter out the "logits" key.
    sample_param_dict = {
        key: value[i] for key, value in posterior_samples.items() if key != "logits"
    }

    # Get the fft_full for this sample.
    fft_full = get_block_fft_full_for_given_params(
        model, X_test, sample_param_dict, rng_key=jr.PRNGKey(i)
    )
    fft_list.append(fft_full)

# Convert list of FFT results to a numpy array of shape (num_samples, n)
fft_samples = np.stack(fft_list, axis=0)
visualize_block_circulant_kernels_with_uncertainty(fft_samples)
plot_block_fft_spectra_with_uncertainty(fft_samples)
