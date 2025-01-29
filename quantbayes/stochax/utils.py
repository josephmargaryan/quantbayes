import jax.numpy as jnp
import flax.linen as nn
import jax

class FFTDense(nn.Module):
    features: int  # Number of output features (similar to Dense)
    
    @nn.compact
    def __call__(self, x):
        """
        FFT-based Dense Layer
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Transformed output tensor (batch_size, features)
        """
        # Apply FFT on the input (axis=-1 for last dimension)
        fft_x = jnp.fft.fft(x)

        # Learnable weights for modifying frequency components
        fft_weights = self.param("fft_weights", nn.initializers.lecun_normal(), (x.shape[-1], self.features))
        
        # Multiply FFT-transformed input with learnable weights
        fft_transformed = fft_x @ fft_weights  # Matrix multiplication in the frequency domain
        
        # Apply Inverse FFT (convert back to time domain)
        output = jnp.fft.ifft(fft_transformed).real  # Take only the real part
        
        return output


class CirculantFFTDense(nn.Module):
    features: int  # Must be the same as input dimension (in_features)

    @nn.compact
    def __call__(self, x):
        """
        Circulant FFT-Based Dense Layer
        Args:
            x: Input tensor (batch_size, in_features)

        Returns:
            Transformed output tensor (batch_size, features)
        """
        in_features = x.shape[-1]
        assert in_features == self.features, "Strict circulant structure requires in_features == out_features"
        
        # FIX: Use normal initialization explicitly for 1D tensor
        first_row = self.param("first_row", lambda rng, shape: jax.random.normal(rng, shape) * 0.1, (in_features,))

        # Compute the full circulant matrix in Fourier space
        fft_weights = jnp.fft.fft(first_row)  # Fourier transform of the first row

        # Apply FFT to input
        fft_x = jnp.fft.fft(x)

        # Efficient elementwise multiplication in Fourier space (Hadamard product)
        fft_transformed = fft_x * fft_weights  

        # Convert back to time domain using IFFT
        output = jnp.fft.ifft(fft_transformed).real  # Take only the real part

        return output
    

def test_fft_dense():
    """Test function for FFTDense layer"""
    print("Testing FFTDense...")

    # Define model
    fft_dense = FFTDense(features=64)

    # Create dummy input (batch_size=5, in_features=32)
    x = jnp.ones((5, 32))

    # Initialize parameters
    params = fft_dense.init(jax.random.PRNGKey(0), x)

    # Run forward pass
    y = fft_dense.apply(params, x)

    # Check output shape
    assert y.shape == (5, 64), f"Expected output shape (5, 64), got {y.shape}"
    print("✅ FFTDense test passed!")


def test_circulant_fft_dense():
    """Test function for CirculantFFTDense layer"""
    print("Testing CirculantFFTDense...")

    # Define model (in_features must equal out_features)
    circulant_fft_dense = CirculantFFTDense(features=32)

    # Create dummy input (batch_size=5, in_features=32)
    x = jnp.ones((5, 32))

    # Initialize parameters
    params = circulant_fft_dense.init(jax.random.PRNGKey(0), x)

    # Run forward pass
    y = circulant_fft_dense.apply(params, x)

    # Check output shape
    assert y.shape == (5, 32), f"Expected output shape (5, 32), got {y.shape}"
    print("✅ CirculantFFTDense test passed!")


def test_combined_model():
    """Test function for combining CirculantFFTDense with a Dense layer"""
    print("Testing CirculantFFTDense + Dense...")

    class CombinedModel(nn.Module):
        output_dim: int

        @nn.compact
        def __call__(self, x):
            x = CirculantFFTDense(features=x.shape[-1])(x)  # Circulant FFT Dense (in_features == out_features)
            x = nn.relu(x)
            x = nn.Dense(self.output_dim)(x)  # Fully connected layer for flexible output
            return x

    # Define model
    model = CombinedModel(output_dim=64)

    # Create dummy input (batch_size=5, in_features=32)
    x = jnp.ones((5, 32))

    # Initialize parameters
    params = model.init(jax.random.PRNGKey(0), x)

    # Run forward pass
    y = model.apply(params, x)

    # Check output shape
    assert y.shape == (5, 64), f"Expected output shape (5, 64), got {y.shape}"
    print("✅ Combined CirculantFFTDense + Dense test passed!")


if __name__ == "__main__":
    test_fft_dense()
    test_circulant_fft_dense()
    test_combined_model()
