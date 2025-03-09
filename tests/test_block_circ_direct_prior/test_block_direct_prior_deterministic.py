import equinox as eqx
import jax
import jax.random as jr

# Import your FFT layer and visualization functions.
from quantbayes.stochax.utils import (
    BlockFFTDirectPrior,
    plot_block_fft_spectra,
    visualize_block_circulant_kernels,
)


# Define a simple deterministic network that uses the block FFT layer.
class MyBlockCirculantNet(eqx.Module):
    fft_layer: BlockFFTDirectPrior  # our block-circulant layer
    final_layer: eqx.nn.Linear

    def __init__(self, in_features, hidden_dim, *, key):
        key1, key2, key3 = jr.split(key, 3)
        self.fft_layer = BlockFFTDirectPrior(
            in_features=in_features,
            out_features=hidden_dim,
            block_size=4,
            key=key1,
            init_scale=0.01,
        )
        self.final_layer = eqx.nn.Linear(hidden_dim, 1, key=key2)

    def __call__(self, x):
        h = self.fft_layer(x)
        h = jax.nn.tanh(h)
        return jax.vmap(self.final_layer)(h)


def test_deterministic_net_visualization_single():
    key = jr.PRNGKey(42)
    in_features = 16
    out_features = 1

    # Instantiate the network.
    net = MyBlockCirculantNet(in_features, out_features, key=key)

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


if __name__ == "__main__":
    test_deterministic_net_visualization_single()
