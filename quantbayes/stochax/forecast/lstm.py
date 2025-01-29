import jax
import jax.numpy as jnp
from flax import linen as nn


class LSTMModel(nn.Module):
    """
    A simple LSTM-based model for single-step forecasting in Flax:
      - LSTM input: (batch_size, seq_len, input_dim)
      - We take the last hidden state -> project to (batch_size, 1)
    """

    input_dim: int
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    deterministic: bool = True  # For dropout

    @nn.compact
    def __call__(self, x, carry=None):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            carry: Optional initial carry (tuple of (c, h) for each layer)

        Returns:
            y: Output tensor of shape (batch_size, 1)
            new_carry: Updated carry states
        """
        batch_size, seq_len, _ = x.shape

        if carry is None:
            # Initialize carry for all layers
            carry = []
            for _ in range(self.num_layers):
                # Pass hidden_dim to LSTMCell
                lstm_cell = nn.LSTMCell(features=self.hidden_dim, name=f"lstm_cell_{_}")
                carry.append(
                    lstm_cell.initialize_carry(
                        jax.random.PRNGKey(0), (batch_size, self.hidden_dim)
                    )
                )
            carry = tuple(carry)

        for layer in range(self.num_layers):
            # Pass hidden_dim to LSTMCell
            lstm_cell = nn.LSTMCell(
                features=self.hidden_dim, name=f"2lstm_cell_{layer}"
            )
            hidden = carry[layer]
            outputs = []
            for t in range(seq_len):
                hidden, out = lstm_cell(hidden, x[:, t, :])
                outputs.append(out)
            x = jnp.stack(outputs, axis=1)  # (batch_size, seq_len, hidden_dim)
            if self.dropout > 0.0:
                x = nn.Dropout(rate=self.dropout)(x, deterministic=self.deterministic)
            carry = carry[:layer] + (hidden,) + carry[layer + 1 :]

        # Take the last hidden state from the top layer
        last_hidden = carry[-1][1]  # (batch_size, hidden_dim)

        # Final dense layer to produce single scalar
        y = nn.Dense(features=1, name="output_dense")(last_hidden)  # (batch_size, 1)

        return y, carry


def test_lstm_forecast_model():
    # Model parameters
    input_dim = 5
    hidden_dim = 32
    num_layers = 2
    dropout = 0.1
    batch_size = 8
    seq_len = 10

    # Initialize the model
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    # Create a random input tensor
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch_size, seq_len, input_dim))

    # Initialize model parameters
    variables = model.init(key, x)

    # Perform a forward pass
    y, carry = model.apply(variables, x)

    # Assertions
    assert y.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {y.shape}"
    print("Test passed: Output shape is correct.")
    print(f"output shape: {y.shape}")


if __name__ == "__main__":
    test_lstm_forecast_model()
