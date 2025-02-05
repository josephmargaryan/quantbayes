import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random
import numpy as np


class CirculantLinear(eqx.Module):
    """
    A custom Equinox layer that implements a linear layer with a circulant matrix.

    The layer stores only the first row `c` (a vector of shape (n,)).
    We define the circulant matrix C so that its first row is c and its i-th row is:

      C[i, :] = jnp.roll(c, i)

    Then, the first column is [c_0, c_{n-1}, c_{n-2}, ..., c_1].

    A well-known fact is that the eigenvalues of C are given by
      λ = fft(first_col)
    and one may compute C @ x via:

      y = real( ifft( fft(x) * fft(first_col) ) )

    Here we compute first_col by flipping c and then rolling by 1.

    Example use casae:
    class MyModel(eqx.Module):

        circulant_layer: CirculantLinear
        final_layer: eqx.nn.Linear

        # in_features and out_features are static (non-trainable) fields.
        in_features: int = eqx.static_field()
        out_features: int = eqx.static_field()

        def __init__(self, in_features: int, out_features: int, *, key):
            # We assume the hidden dimension equals the input dimension.
            self.in_features = in_features
            self.out_features = out_features
            key1, key2 = jax.random.split(key, 2)
            self.circulant_layer = CirculantLinear(in_features, key=key1, init_scale=1.0)
            self.final_layer = eqx.nn.Linear(in_features, out_features, key=key2)

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

            # Our custom layer supports batched inputs because FFT is applied along the last axis.
            h = self.circulant_layer(x)  # shape (batch, in_features)
            h = jax.nn.relu(h)
            # eqx.nn.Linear expects a single vector. We can use jax.vmap to apply it over the batch.
            y = jax.vmap(lambda z: self.final_layer(z))(h)
            return y
    """

    first_row: jnp.ndarray  # shape (n,)
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()

    def __init__(self, in_features: int, *, key, init_scale: float = 1.0):
        # We assume a square weight matrix.
        self.in_features = in_features
        self.out_features = in_features
        self.first_row = jax.random.normal(key, (in_features,)) * init_scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Multiply input x (shape (..., n)) by the circulant matrix.
        """
        # Compute the "first column" of C:
        # first_row = [c0, c1, ..., c_{n-1}]
        # flip(first_row) = [c_{n-1}, c_{n-2}, ..., c0]
        # Rolling that by +1 gives: [c0, c_{n-1}, c_{n-2}, ..., c1]
        first_col = jnp.roll(jnp.flip(self.first_row), shift=1)
        fft_w = jnp.fft.fft(first_col)
        fft_x = jnp.fft.fft(x, axis=-1)
        y = jnp.fft.ifft(fft_x * fft_w, axis=-1)
        return jnp.real(y)


# --- Test of the custom CirculantLinear layer ---


def test_circulant_linear():
    key = jax.random.PRNGKey(42)
    n = 8
    layer = CirculantLinear(n, key=key, init_scale=1.0)

    key, subkey = jax.random.split(key)
    # Create a random input vector of shape (n,)
    x = jax.random.normal(subkey, (n,))

    # Compute output using the custom layer
    y_custom = layer(x)

    # For testing, explicitly build the circulant matrix.
    # According to our definition: the i-th row is jnp.roll(c, i)
    def circulant(first_row):
        return jnp.stack(
            [jnp.roll(first_row, i) for i in range(first_row.shape[0])], axis=0
        )

    C = circulant(layer.first_row)
    y_direct = C @ x

    # Compare the two outputs.
    np.testing.assert_allclose(y_custom, y_direct, rtol=1e-5, atol=1e-5)
    print("Test passed! y_custom and y_direct are equal within tolerance.")
    print("Input x:\n", x)
    print("CirculantLinear output:\n", y_custom)
    print("Direct multiplication output:\n", y_direct)


if __name__ == "__main__":
    test_circulant_linear()
