import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class Mamba(nn.Module):
    """
    A simple custom 'state space' model:
      z_{t+1} = tanh(A z_t + B x_t + bias)
      final forecast = W z_{last} + d

    Input shape:  (batch_size, seq_len, input_dim)
    Output shape: (batch_size, 1)
    """

    input_dim: int
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, x, carry: Optional[jnp.ndarray] = None):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            carry: Optional initial hidden state z, shape (batch_size, hidden_dim)

        Returns:
            out: Output tensor of shape (batch_size, 1)
            new_carry: Updated hidden state z, shape (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        if carry is None:
            # Initialize z to zeros
            z = jnp.zeros((batch_size, self.hidden_dim))
        else:
            z = carry

        # Define the state update layer
        state_update = nn.Dense(
            features=self.hidden_dim, name="state_update", use_bias=True
        )

        # Define the output layer
        output_layer = nn.Dense(features=1, name="output_layer", use_bias=True)

        # Iterate over the sequence
        for t in range(seq_len):
            x_t = x[:, t, :]  # Shape: (batch_size, input_dim)
            # Concatenate z and x_t
            zx = jnp.concatenate(
                [z, x_t], axis=-1
            )  # Shape: (batch_size, hidden_dim + input_dim)
            # Update hidden state
            z = jnp.tanh(state_update(zx))  # Shape: (batch_size, hidden_dim)

        # Compute the final output
        out = output_layer(z)  # Shape: (batch_size, 1)

        return out, z


def test_mamba_state_space():
    # Model parameters
    input_dim = 5
    hidden_dim = 16
    batch_size = 8
    seq_len = 20

    # Initialize the model
    model = Mamba(input_dim=input_dim, hidden_dim=hidden_dim)

    # Create a random input tensor
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(
        key, (batch_size, seq_len, input_dim)
    )  # (batch=8, seq_len=20, input_dim=5)

    # Initialize model parameters
    variables = model.init(key, x)

    # Perform a forward pass
    y, new_carry = model.apply(variables, x)

    # Assertions
    assert y.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {y.shape}"
    print("Test passed: Output shape is correct.")
    print("MambaStateSpace output shape:", y.shape)


if __name__ == "__main__":
    test_mamba_state_space()
