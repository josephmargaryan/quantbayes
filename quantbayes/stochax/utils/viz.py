import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from numpyro import handlers

__all__ = [
    "visualize_circulant_layer",
    "visualize_block_circulant_layer",
    "analyze_pre_activations",
    "visualize_deterministic_fft",
    "visualize_deterministic_block_fft",
]


def _get_pre_activation(model, X, seed=123):
    my_rng_key = jr.PRNGKey(seed)
    with handlers.seed(rng_seed=my_rng_key):
        pre_activations = model.get_preactivations(X)
    return pre_activations


def analyze_pre_activations(model, X):
    """
    Computes and visualizes the empirical covariance matrix from pre-activations.

    Parameters:
      X_pre: jnp.ndarray of shape (N, in_features)

    Returns:
      cov_matrix: jnp.ndarray of shape (N, N)

    Example usage:
    Define a function in the class:
        def get_preactivations(self, X):
            X_pre = self.fft_layer(X)
            return jax.lax.stop_gradient(X_pre)
    """
    X_pre = _get_pre_activation(model, X)
    # Compute the empirical covariance matrix (across the data points)
    # Each row is a data point's feature representation.
    X_centered = X_pre - X_pre.mean(axis=0)
    cov_matrix = (X_centered @ X_centered.T) / (X_pre.shape[1] - 1)

    # Visualize the covariance matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(jax.device_get(cov_matrix), cmap="viridis")
    plt.colorbar()
    plt.title("Empirical Covariance of Pre-Activations")
    plt.xlabel("Data Point Index")
    plt.ylabel("Data Point Index")
    plt.tight_layout()
    plt.show()

    return cov_matrix


def get_fft_full_for_given_params(model, X, param_dict, rng_key=jr.PRNGKey(0)):
    """
    Substitute a concrete parameter set into the model and run one forward pass
    with a provided rng_key so that the sample sites receive valid keys.
    This triggers model.fft_layer to store its Fourier coefficients.
    """
    with handlers.seed(rng_seed=rng_key):
        with handlers.substitute(data=param_dict):
            _ = model(X)  # this call now receives a proper PRNG key
    fft_full = model.fft_layer.get_fourier_coeffs()
    return jax.device_get(fft_full)


def get_block_fft_full_for_given_params(model, X, param_dict, rng_key):
    """
    Given a dictionary of parameter draws (e.g. from MCMC) and an rng_key,
    run one forward pass so that the layer sees these values and saves
    its FFT arrays in `.last_fourier_coeffs`.
    """
    with handlers.seed(rng_seed=rng_key):
        with handlers.substitute(data=param_dict):
            _ = model(X)  # triggers the block_layer call
    # Now retrieve the block-layer’s stored FFT
    fft_full = model.block_layer.get_fourier_coeffs()
    return jax.device_get(fft_full)


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


def visualize_block_circulant_matrices_with_uncertainty(
    fft_samples_blocks: np.ndarray, show: bool = True
):
    """
    Visualize the uncertainty in the circulant matrices for each block.
    For each block, the time-domain kernel is computed via IFFT from the posterior
    samples, and the mean circulant matrix is obtained by rolling the mean kernel.

    Parameters:
        fft_samples_blocks: np.ndarray of shape (num_samples, k_out, k_in, block_size)
            Array of FFT coefficients from multiple posterior samples.
        show: bool, if True calls plt.show() at the end.

    Returns:
        A matplotlib figure showing the circulant matrices for each block.
    """
    num_samples, k_out, k_in, b = fft_samples_blocks.shape
    total = k_out * k_in
    nrows = int(np.ceil(np.sqrt(total)))
    ncols = int(np.ceil(total / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    for idx in range(total):
        i = idx // k_in
        j = idx % k_in

        # For block (i,j), compute the time-domain kernel for each sample.
        block_fft_samples = fft_samples_blocks[:, i, j, :]  # shape (num_samples, b)
        time_kernels = np.array(
            [np.fft.ifft(sample).real for sample in block_fft_samples]
        )
        # Compute the mean time-domain kernel.
        kernel_mean = time_kernels.mean(axis=0)
        # Reconstruct the circulant matrix by rolling the mean kernel.
        C_mean = np.stack([np.roll(kernel_mean, shift=k) for k in range(b)], axis=0)

        ax = axes[idx]
        im = ax.imshow(C_mean, cmap="viridis")
        ax.set_title(f"Block ({i},{j}) Circulant Matrix")
        ax.set_xlabel("Index")
        ax.set_ylabel("Index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[total:]:
        ax.set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def _get_fft_samples(model, X):
    posterior_samples = model.get_samples
    for key, value in posterior_samples.items():

        # (3) To visualize uncertainty, loop over multiple posterior samples:
        fft_list = []
        n_samples = 50
        for i in range(n_samples):
            sample_param_dict = {
                key: value[i] for key, value in posterior_samples.items()
            }
            fft_sample = get_fft_full_for_given_params(
                model, X, sample_param_dict, rng_key=jr.PRNGKey(i)
            )
            fft_list.append(fft_sample)
    return np.stack(fft_list, axis=0)


def _get_block_fft_samples(model, X):

    posterior_samples = model.get_samples
    param_dict = {
        key: value[0] for key, value in posterior_samples.items() if key != "logits"
    }
    fft_full_blocks = get_block_fft_full_for_given_params(
        model, X, param_dict, rng_key=jr.PRNGKey(123)
    )
    fft_list_blocks = []
    n_samples = 50
    for i in range(n_samples):
        sample_param_dict = {
            key: value[i] for key, value in posterior_samples.items() if key != "logits"
        }
        fft_sample_block = get_block_fft_full_for_given_params(
            model, X, sample_param_dict, rng_key=jr.PRNGKey(i)
        )
        fft_list_blocks.append(fft_sample_block)

    fft_samples_blocks = np.stack(fft_list_blocks, axis=0)
    return fft_samples_blocks


def visualize_circulant_layer(model, X, show=True):
    """
    Visualizes the FFT spectrum (magnitude and phase) and the time-domain circulant kernel.
    If fft_samples has multiple samples, uncertainty (e.g., 95% CI) is shown.

    """
    fft_samples = _get_fft_samples(model, X)
    # Compute statistics (mean, lower, upper bounds) for FFT spectrum.
    fig1 = plot_fft_spectrum_with_uncertainty(fft_samples, show=False)

    # Compute time-domain kernels from fft_samples.
    fig2 = visualize_circulant_kernel_with_uncertainty(fft_samples, show=False)

    # Optionally combine or display them side by side.
    if show:
        plt.show()
    return fig1, fig2


def visualize_block_circulant_layer(model, X, show=True):
    """
    Visualizes the FFT spectra, phase, time-domain kernels, and full circulant matrices for each block.

    """
    # FFT spectra with uncertainty.
    fft_samples_blocks = _get_block_fft_samples(model, X)
    fig1, fig2 = plot_block_fft_spectra_with_uncertainty(fft_samples_blocks, show=False)
    # Time-domain kernels with uncertainty.
    fig3 = visualize_block_circulant_kernels_with_uncertainty(
        fft_samples_blocks, show=False
    )
    # Full circulant matrices.
    fig4 = visualize_block_circulant_matrices_with_uncertainty(
        fft_samples_blocks, show=False
    )

    if show:
        plt.show()
    return fig1, fig2, fig3, fig4


def visualize_deterministic_fft(layer, X, show=True):
    """
    Runs the deterministic FFT layer on X, extracts the Fourier coefficients,
    and visualizes the magnitude, phase, the 1D reconstructed time-domain kernel,
    and the 2D circulant matrix (mean circulant kernel).

    Parameters:
      layer: the deterministic FFT layer (with a get_fourier_coeffs() method)
      X: input data (jnp.ndarray) to trigger a forward pass
      show: bool, if True, displays the plot

    Returns:
      fig: the matplotlib figure
    """
    # Run a forward pass to update stored Fourier coefficients.
    _ = layer(X)
    fft_coeffs = layer.get_fourier_coeffs()  # deterministic coefficients

    # Compute magnitude and phase
    mag = jnp.abs(fft_coeffs)
    phase = jnp.angle(fft_coeffs)

    # Reconstruct the 1D time-domain kernel via inverse FFT.
    kernel_time = jnp.fft.ifft(fft_coeffs).real

    # For the circulant matrix, roll the 1D kernel across rows.
    n = kernel_time.shape[0]
    C_mean = jnp.stack([jnp.roll(kernel_time, i) for i in range(n)], axis=0)

    # Create a figure with 4 subplots.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Fourier Magnitude
    axes[0, 0].plot(mag)
    axes[0, 0].set_title("Magnitude of Fourier Coefficients")
    axes[0, 0].set_xlabel("Frequency Index")
    axes[0, 0].set_ylabel("Magnitude")

    # Top right: Fourier Phase
    axes[0, 1].plot(phase)
    axes[0, 1].set_title("Phase of Fourier Coefficients")
    axes[0, 1].set_xlabel("Frequency Index")
    axes[0, 1].set_ylabel("Phase (radians)")

    # Bottom left: Reconstructed 1D Time-Domain Kernel
    axes[1, 0].plot(kernel_time)
    axes[1, 0].set_title("Reconstructed 1D Time-Domain Kernel")
    axes[1, 0].set_xlabel("Index")
    axes[1, 0].set_ylabel("Amplitude")

    # Bottom right: Mean Circulant Matrix
    im = axes[1, 1].imshow(jnp.array(C_mean), cmap="viridis")
    axes[1, 1].set_title("Mean Circulant Matrix")
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel("Index")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def visualize_deterministic_block_fft(layer, X, show=True):
    """
    Runs a forward pass of the block circulant FFT layer on X, extracts the full Fourier coefficients,
    and visualizes for each block:
      - Magnitude of Fourier coefficients
      - Phase of Fourier coefficients
      - Reconstructed 1D time-domain kernel (via IFFT)
      - Reconstructed circulant matrix from the 1D kernel.

    Parameters:
      layer: The block circulant FFT layer (with a get_fourier_coeffs() method).
      X: Input data (jnp.ndarray) to trigger a forward pass.
      show: If True, display the plots.

    Returns:
      None.
    """
    # Run a forward pass to update the stored Fourier coefficients.
    _ = layer(X)
    # Retrieve the full Fourier coefficients.
    # Expected shape: (k_out, k_in, block_size)
    fft_coeffs_blocks = layer.get_fourier_coeffs()
    fft_coeffs_blocks = jnp.array(fft_coeffs_blocks)  # Ensure it's a JAX array

    k_out, k_in, b = fft_coeffs_blocks.shape
    total_blocks = k_out * k_in

    # For each block, compute magnitude, phase, 1D kernel (via IFFT), and circulant matrix.
    # We'll create four grids of subplots (one for each visualization type).
    # Determine grid size (rows, cols) for total_blocks.
    grid_rows = int(np.ceil(np.sqrt(total_blocks)))
    grid_cols = int(np.ceil(total_blocks / grid_rows))

    # Prepare figures:
    fig_mag, axes_mag = plt.subplots(
        grid_rows, grid_cols, figsize=(4 * grid_cols, 3 * grid_rows)
    )
    fig_phase, axes_phase = plt.subplots(
        grid_rows, grid_cols, figsize=(4 * grid_cols, 3 * grid_rows)
    )
    fig_kernel_1d, axes_kernel_1d = plt.subplots(
        grid_rows, grid_cols, figsize=(4 * grid_cols, 3 * grid_rows)
    )
    fig_circ, axes_circ = plt.subplots(
        grid_rows, grid_cols, figsize=(4 * grid_cols, 3 * grid_rows)
    )

    # Flatten axes arrays for easy indexing.
    axes_mag = np.array(axes_mag).flatten()
    axes_phase = np.array(axes_phase).flatten()
    axes_kernel_1d = np.array(axes_kernel_1d).flatten()
    axes_circ = np.array(axes_circ).flatten()

    # Loop over blocks.
    block_idx = 0
    for i in range(k_out):
        for j in range(k_in):
            # Extract Fourier coefficients for block (i,j): shape (b,)
            block_fft = fft_coeffs_blocks[i, j, :]

            # Compute magnitude and phase.
            mag = jnp.abs(block_fft)
            phase = jnp.angle(block_fft)

            # Compute the reconstructed 1D time-domain kernel via inverse FFT.
            kernel_1d = jnp.fft.ifft(block_fft).real

            # Compute the circulant matrix: for a circulant matrix,
            # each row is a rolled version of the 1D kernel.
            C = jnp.stack([jnp.roll(kernel_1d, shift=k) for k in range(b)], axis=0)

            # Plot magnitude.
            ax = axes_mag[block_idx]
            ax.plot(mag)
            ax.set_title(f"Block ({i},{j}) Mag")
            ax.set_xlabel("Freq index")
            ax.set_ylabel("Magnitude")

            # Plot phase.
            ax = axes_phase[block_idx]
            ax.plot(phase)
            ax.set_title(f"Block ({i},{j}) Phase")
            ax.set_xlabel("Freq index")
            ax.set_ylabel("Phase (rad)")

            # Plot 1D reconstructed kernel.
            ax = axes_kernel_1d[block_idx]
            ax.plot(kernel_1d)
            ax.set_title(f"Block ({i},{j}) 1D Kernel")
            ax.set_xlabel("Time index")
            ax.set_ylabel("Amplitude")

            # Plot the circulant matrix.
            ax = axes_circ[block_idx]
            im = ax.imshow(jnp.array(C), cmap="viridis")
            ax.set_title(f"Block ({i},{j}) Circulant")
            ax.set_xlabel("Index")
            ax.set_ylabel("Index")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            block_idx += 1

    # Hide any extra subplots if total_blocks doesn't fill the grid.
    for ax in axes_mag[block_idx:]:
        ax.set_visible(False)
    for ax in axes_phase[block_idx:]:
        ax.set_visible(False)
    for ax in axes_kernel_1d[block_idx:]:
        ax.set_visible(False)
    for ax in axes_circ[block_idx:]:
        ax.set_visible(False)

    fig_mag.tight_layout()
    fig_phase.tight_layout()
    fig_kernel_1d.tight_layout()
    fig_circ.tight_layout()

    if show:
        plt.show()

    return fig_mag, fig_phase, fig_kernel_1d, fig_circ
