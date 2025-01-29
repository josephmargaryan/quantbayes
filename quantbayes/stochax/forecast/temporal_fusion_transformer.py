import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Optional


class GatingLayer(nn.Module):
    """
    Gated Linear Unit (GLU) based gating mechanism:
      out = x ⊗ σ(W_g x + b_g)
    """

    input_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, L, input_dim]
        Returns:
            Output tensor of shape [B, L, hidden_dim]
        """
        # Linear transform
        value = nn.Dense(self.hidden_dim)(x)  # [B, L, hidden_dim]
        # Gate
        gate = nn.Dense(self.hidden_dim)(x)  # [B, L, hidden_dim]
        gate = nn.sigmoid(gate)
        return value * gate


class AddNorm(nn.Module):
    """
    Applies skip connection followed by layer normalization.
    """

    normalized_shape: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, sub_layer_out: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Residual connection input
            sub_layer_out: Output of the sub-layer
        Returns:
            Output tensor with applied AddNorm
        """
        return nn.LayerNorm()(x + sub_layer_out)


class MLP(nn.Module):
    """
    A simple feed-forward network: Linear -> ReLU -> Dropout -> Linear
    """

    input_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, L, input_dim]
        Returns:
            Output tensor of shape [B, L, output_dim]
        """
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout)(x, deterministic=not train)
        x = nn.Dense(self.output_dim)(x)
        return x


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN):
      Combines MLP and gating with a residual connection and LayerNorm.
    """

    input_dim: int
    hidden_dim: int
    output_dim: Optional[int] = None
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, L, input_dim]
        Returns:
            Output tensor of shape [B, L, output_dim or input_dim]
        """
        if self.output_dim is None:
            self.output_dim = self.input_dim

        # Feedforward network
        hidden = nn.Dense(self.hidden_dim)(x)
        hidden = nn.relu(hidden)
        hidden = nn.Dropout(self.dropout)(hidden, deterministic=not train)
        hidden = nn.Dense(self.output_dim)(hidden)

        # Gating
        gated = GatingLayer(input_dim=self.output_dim, hidden_dim=self.output_dim)(
            hidden
        )

        # Residual connection and LayerNorm
        out = nn.LayerNorm()(x + gated)
        return out


class MultiHeadAttention(nn.Module):
    embed_dim: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        q: jnp.ndarray,
        kv: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            q: Query tensor of shape [B, T_q, embed_dim]
            kv: Key/Value tensor of shape [B, T_kv, embed_dim]
            mask: Optional attention mask of shape [B, num_heads, T_q, T_kv]
            deterministic: Whether to disable dropout.

        Returns:
            Output tensor of shape [B, T_q, embed_dim]
        """
        mha = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
        )
        return mha(q, kv, mask=mask, deterministic=deterministic)


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) in Flax:
      - Input embedding for continuous and categorical variables.
      - LSTM for local processing.
      - Multi-head attention for long-range dependencies.
      - Gated Residual Networks and AddNorm for skip connections.
      - Final projection for forecasting.
    """

    input_dim: int
    d_model: int = 64
    lstm_hidden_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    num_lstm_layers: int = 1
    output_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, L, input_dim]
            train: Whether the model is in training mode.

        Returns:
            Forecast output of shape [B, output_dim]
        """
        # 1. Input embedding
        x_emb = nn.Dense(self.d_model, name="input_projection")(x)  # [B, L, d_model]

        # 2. LSTM for local processing
        batch_size, seq_len, _ = x.shape
        lstm_cell = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(features=self.lstm_hidden_dim, name="lstm_cell")

        carry = nn.LSTMCell(features=self.lstm_hidden_dim).initialize_carry(
            rng=jax.random.PRNGKey(0),
            input_shape=(batch_size, self.d_model),  # Input shape to the LSTM cell
        )

        carry, lstm_out = lstm_cell(carry, x_emb)  # [B, L, lstm_hidden_dim]
        lstm_out = nn.Dense(self.d_model, name="lstm_projection")(lstm_out)

        # 3. Multi-head attention
        attn_out = MultiHeadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
        )(lstm_out, lstm_out, deterministic=not train)
        attn_out = AddNorm(normalized_shape=self.d_model)(lstm_out, attn_out)

        # 4. Gated Residual Network
        grn_out = GatedResidualNetwork(
            input_dim=self.d_model,
            hidden_dim=self.d_model,
            output_dim=self.d_model,
            dropout=self.dropout,
        )(attn_out, train=train)

        # 5. Final projection to output
        out = nn.Dense(self.output_dim, name="output_layer")(
            grn_out
        )  # [B, L, output_dim]
        return out[:, -1, :]  # Return the last time step's output


def test_temporal_fusion_transformer():
    key = jax.random.PRNGKey(0)
    batch_size = 4
    seq_len = 10
    input_dim = 8
    d_model = 64
    output_dim = 1

    x = jax.random.normal(key, (batch_size, seq_len, input_dim))  # [B, L, input_dim]

    model = TemporalFusionTransformer(
        input_dim=input_dim,
        d_model=d_model,
        lstm_hidden_dim=64,
        num_heads=4,
        dropout=0.1,
        num_lstm_layers=1,
        output_dim=output_dim,
    )
    variables = model.init(jax.random.PRNGKey(1), x, train=True)
    y = model.apply(variables, x, train=False)

    print("TFT output shape:", y.shape)  # Expected: [B, output_dim]


test_temporal_fusion_transformer()
