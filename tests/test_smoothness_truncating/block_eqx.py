import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.layers import SmoothTruncEquinoxBlockCirculant
from quantbayes.stochax.utils import (
    plot_block_fft_spectra,
    visualize_block_circulant_kernels,
)

if __name__ == "__main__":
    import jax

    # Define a simple deterministic network that uses the FFT layer.
    class DeterministicNet(eqx.Module):
        fft_layer: SmoothTruncEquinoxBlockCirculant
        linear: eqx.nn.Linear

        def __init__(self, in_features: int, out_features: int, *, key):
            key1, key2 = jr.split(key, 2)
            self.fft_layer = SmoothTruncEquinoxBlockCirculant(
                in_features=in_features,
                out_features=16,
                block_size=4,
                alpha=1,
                K=3,
                key=key1,
                init_scale=1.0,
            )
            self.linear = eqx.nn.Linear(
                in_features=16, out_features=out_features, key=key2
            )

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            # Pass the input through the FFT-based layer.
            x = self.fft_layer(x)
            # Apply a nonlinearity.
            x = jax.nn.tanh(x)
            # Then apply a simple linear layer.
            return self.linear(x)

    key = jr.PRNGKey(42)
    in_features = 8
    out_features = 1

    # Instantiate the network.
    net = DeterministicNet(in_features, out_features, key=key)

    # Create a single input vector.
    x = jr.normal(key, (in_features,))

    # Run a forward pass.
    output = net(x)
    print("Output of deterministic network (single input):", output)

    # Trigger the FFT layer's forward pass (which does the block multiplication).
    _ = net.fft_layer(x)

    # Retrieve the Fourier coefficients for each block.
    fft_full_blocks = net.fft_layer.get_fourier_coeffs()

    # Plot block FFT spectra (magnitude).
    fig1 = plot_block_fft_spectra(fft_full_blocks, show=True)
    # Visualize block circulant kernels.
    fig2 = visualize_block_circulant_kernels(fft_full_blocks, show=True)
