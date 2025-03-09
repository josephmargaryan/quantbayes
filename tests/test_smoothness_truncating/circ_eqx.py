import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.layers import SmoothTruncEquinoxCirculant
from quantbayes.stochax.utils import (
    compare_time_domain_truncation,
    plot_fft_spectrum,
    visualize_circulant_kernel,
)

if __name__ == "__main__":
    import jax

    # Define a simple deterministic network that uses the FFT layer.
    class DeterministicNet(eqx.Module):
        fft_layer: SmoothTruncEquinoxCirculant
        linear: eqx.nn.Linear

        def __init__(self, in_features: int, out_features: int, *, key):
            key1, key2 = jr.split(key, 2)
            self.fft_layer = SmoothTruncEquinoxCirculant(
                in_features=in_features,
                alpha=0.6,
                K=3,
                key=key1,
                init_scale=1.0,
            )
            self.linear = eqx.nn.Linear(
                in_features=in_features, out_features=out_features, key=key2
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

    # Trigger the FFT layer's forward pass to update its stored coefficients.
    _ = net.fft_layer(x)

    # Retrieve and plot the Fourier coefficients.
    fft_full = net.fft_layer.get_fourier_coeffs()
    fig1 = plot_fft_spectrum(fft_full, show=True)
    fig2 = visualize_circulant_kernel(fft_full, show=True)
    fig3 = compare_time_domain_truncation(fft_full=fft_full, K=3, title="time comain")
