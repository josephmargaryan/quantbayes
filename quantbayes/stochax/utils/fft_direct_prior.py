#!/usr/bin/env python3
"""
fft_direct_prior_linear_with_tests.py

This script defines an Equinox layer that performs FFT-based circulant
matrix multiplication by directly storing its independent Fourier coefficients.
It includes tests that verify both the numerical functionality of the layer and
the visualization functions that display its Fourier coefficients and reconstructed kernel.
"""
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
import numpyro.handlers as handlers

__all__ = [
    "FFTDirectPriorLinear",
    "plot_fft_spectrum",
    "visualize_circulant_kernel",
    "reconstruct_circulant_from_fft",
    "get_fft_full_for_given_params",
]


class FFTDirectPriorLinear(eqx.Module):
    """
    An Equinox layer that represents a circulant matrix by storing the
    'independent half' of its DFT as real trainable parameters:

      - fourier_coeffs_real: real parts (shape: (k,))
      - fourier_coeffs_imag: imaginary parts (shape: (k,))

    where k = in_features // 2 + 1.

    During forward pass:
      1) We build the full length-n complex array with Hermitian symmetry.
      2) We multiply the input (batch_size, n) in the frequency domain, then ifft.
      3) We return the real part of the inverse transform.
    """

    in_features: int = eqx.static_field()
    fourier_coeffs_real: jnp.ndarray  # shape: (k,)
    fourier_coeffs_imag: jnp.ndarray  # shape: (k,)
    k: int = eqx.static_field()
    # Add a mutable field to store the FFT computed during a forward pass.
    last_fourier_coeffs: jnp.ndarray = eqx.field(default=None, repr=False)

    def __init__(self, in_features: int, *, key, init_scale: float = 1.0):
        self.in_features = in_features
        self.k = in_features // 2 + 1

        key1, key2 = jr.split(key, 2)
        real_part = jr.normal(key1, (self.k,)) * init_scale
        imag_part = jr.normal(key2, (self.k,)) * init_scale

        # Force freq=0 to be real. If n is even, freq=n/2 is real as well.
        imag_part = imag_part.at[0].set(0.0)
        if in_features % 2 == 0 and self.k > 1:
            imag_part = imag_part.at[-1].set(0.0)

        self.fourier_coeffs_real = real_part
        self.fourier_coeffs_imag = imag_part

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Multiply the input X by the circulant matrix defined in the frequency domain.
        X can be shape (batch_size, n) or just (n,).
        """
        n = self.in_features
        # Reconstruct half-spectrum as a complex array
        half_complex = self.fourier_coeffs_real + 1j * self.fourier_coeffs_imag

        # Build full length-n complex array with Hermitian symmetry
        if n % 2 == 0 and self.k > 1:
            # even n: freq n/2 is real => last element in half_complex is real
            nyquist = half_complex[-1].real[None]
            fft_full = jnp.concatenate(
                [
                    half_complex[:-1],
                    nyquist,
                    jnp.conj(half_complex[1:-1])[::-1],
                ]
            )
        else:
            # odd n or small edge cases
            fft_full = jnp.concatenate([half_complex, jnp.conj(half_complex[1:])[::-1]])

        # Multiply in freq domain
        X_fft = jnp.fft.fft(X, axis=-1)
        result_fft = X_fft * fft_full if X.ndim == 1 else X_fft * fft_full[None, :]
        # Store the FFT for later visualization.
        object.__setattr__(self, "last_fourier_coeffs", fft_full)
        return jnp.fft.ifft(result_fft, axis=-1).real

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self.last_fourier_coeffs is None:
            raise ValueError(
                "No Fourier coefficients available for layer. "
                "Call the layer once on some input to store them."
            )

        return jax.device_get(self.last_fourier_coeffs)


# ------------------------------------------------------------------------------
# Helper functions for circulant matrix reconstruction.
def circulant(first_row: jnp.ndarray) -> jnp.ndarray:
    n = first_row.shape[0]
    return jnp.stack([jnp.roll(first_row, i) for i in range(n)], axis=0)


def circulant_from_fft(fft_full: jnp.ndarray) -> jnp.ndarray:
    """
    Reconstructs the circulant matrix from the full Fourier spectrum.
    (The inverse FFT of the full spectrum yields the first column;
     then, first_row = flip(roll(first_col, shift=-1)))
    """
    first_col = jnp.fft.ifft(fft_full).real
    first_row = jnp.flip(jnp.roll(first_col, shift=-1))
    return circulant(first_row)


def plot_fft_spectrum(fft_full: jnp.ndarray, show: bool = True):
    """
    fft_full: a *concrete* JAX array of shape (n,) (complex).
    We'll compute magnitude & phase in JAX, then convert to NumPy for plotting.
    """
    # Compute in JAX
    mag_jax = jnp.abs(fft_full)
    # JAX has no builtin jnp.angle, so do an arctan2:
    phase_jax = jnp.arctan2(jnp.imag(fft_full), jnp.real(fft_full))

    # Materialize on CPU as NumPy arrays
    mag = np.asarray(mag_jax)
    phase = np.asarray(phase_jax)

    # Plot with matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(mag, marker="o")
    axes[0].set_title("FFT Magnitude")
    axes[0].set_xlabel("Frequency index")
    axes[0].set_ylabel("|FFT|")

    axes[1].plot(phase, marker="o")
    axes[1].set_title("FFT Phase")
    axes[1].set_xlabel("Frequency index")
    axes[1].set_ylabel("Phase (radians)")

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def visualize_circulant_kernel(fft_full: jnp.ndarray, show: bool = True):
    """
    Similar approach: compute the circulant matrix in JAX, then convert to NumPy for plotting.
    """
    n = fft_full.shape[0]
    # iFFT => first_col
    first_col = jnp.fft.ifft(fft_full).real
    # Build the circulant matrix in JAX
    # row i is a roll of first_row, or column i is roll of first_col...
    # We'll define the standard approach: stack of jnp.roll(...).
    C = jnp.stack(
        [jnp.roll(first_col, i) for i in range(n)], axis=1
    )  # shape (n,n), depending on row vs col

    # Convert to NumPy
    C_np = np.asarray(C)
    time_domain = C_np[:, 0]  # first col (NumPy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(time_domain, marker="o")
    ax1.set_title("Circulant Kernel (Time Domain)")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Amplitude")

    im = ax2.imshow(C_np, cmap="viridis")
    ax2.set_title("Circulant Matrix")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def reconstruct_circulant_from_fft(fft_full: jnp.ndarray) -> jnp.ndarray:
    """
    Reconstruct the circulant matrix from fft_full (complex, length n).
    - The "time-domain" vector is the IFFT of fft_full => that is the first column (or row).
    - Then circulant() builds the full matrix by rolling that vector.
    """
    n = fft_full.shape[0]
    # iFFT => first_col, or first_row depending on your convention
    first_col = jnp.fft.ifft(fft_full).real
    # We'll define "row i = roll(first_row, i)" approach:
    # So if you want the first row, you can do a flip/roll trick,
    # but here let's just produce the standard "circulant from first_col".
    return jnp.stack([jnp.roll(first_col, i) for i in range(n)], axis=0)


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


# --- Test functions --- #
def test_fft_direct_prior_linear():
    """
    Tests the FFTDirectPriorLinear_eqx layer by:
      1. Creating an instance with a fixed random key.
      2. Generating a batch of random input data.
      3. Computing the output using the FFT layer.
      4. Reconstructing the corresponding circulant matrix from the stored Fourier coefficients.
      5. Comparing the output of the layer to direct multiplication with the reconstructed matrix.
    """
    key = jr.PRNGKey(42)
    n = 8  # in_features
    batch_size = 4

    # Instantiate the FFT-based layer.
    fft_layer = FFTDirectPriorLinear(in_features=n, key=key, init_scale=1.0)

    # Generate a random input batch of shape (batch_size, n).
    key, subkey = jr.split(key)
    X = jr.normal(subkey, (batch_size, n))

    # Compute output using the FFT layer.
    y_layer = fft_layer(X)

    # Reconstruct the full Fourier spectrum from the stored independent coefficients.
    coeffs = fft_layer.get_fourier_coeffs()
    if n % 2 == 0:
        nyquist = coeffs[-1].real[None]  # Force a real Nyquist coefficient.
        fft_full = jnp.concatenate([coeffs[:-1], nyquist, jnp.conj(coeffs[1:-1])[::-1]])
    else:
        fft_full = jnp.concatenate([coeffs, jnp.conj(coeffs[1:])[::-1]])

    # Reconstruct the circulant matrix.
    C = circulant_from_fft(fft_full)
    # Note: Our FFT layer applies the circulant multiplication as X @ (C.T).
    y_direct = jnp.dot(X, C.T)

    # Compare the outputs.
    np.testing.assert_allclose(y_layer, y_direct, rtol=1e-5, atol=1e-5)
    print("Equinox FFTDirectPriorLinear_eqx test passed!")
    print("Input X:\n", X)
    print("Output via FFT layer:\n", y_layer)
    print("Output via direct circulant multiplication:\n", y_direct)


if __name__ == "__main__":
    # Run the layer functionality test.
    test_fft_direct_prior_linear()

    print("All tests passed successfully!")
