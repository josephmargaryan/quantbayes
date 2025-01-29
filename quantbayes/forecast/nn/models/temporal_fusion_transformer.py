import torch
import torch.nn as nn
import torch.nn.functional as F
from quantbayes.forecast.nn.base import MonteCarloMixin, BaseModel

# -----------------------------------------------------------------------------
# 1. Basic Building Blocks
# -----------------------------------------------------------------------------


class GatingLayer(nn.Module):
    """
    Gated Linear Unit (GLU) based gating mechanism:
      out = x ⊗ σ(W_g x + b_g)
    where x is a linear transformation of the input.
    """

    def __init__(self, input_dim, hidden_dim):
        super(GatingLayer, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.gate_linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len, input_dim]
        Returns: Tensor of shape [batch_size, seq_len, hidden_dim]
        """
        # Linear transform
        value = self.linear(x)
        # Gate
        gate = self.gate_linear(x)
        gate = torch.sigmoid(gate)
        return value * gate


class AddNorm(nn.Module):
    """
    Applies skip connection followed by layer normalization:
      out = LayerNorm(x + sub_layer(x))
    """

    def __init__(self, normalized_shape):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x, sub_layer_out):
        """
        x: Residual connection input
        sub_layer_out: Output of the sub-layer
        """
        return self.layer_norm(x + sub_layer_out)


class MLP(nn.Module):
    """
    A simple feed-forward network: Linear -> ReLU -> Linear
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len, input_dim]
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -----------------------------------------------------------------------------
# 2. Gated Residual Network
# -----------------------------------------------------------------------------


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN):
      y = LayerNorm(x + GLU( (W1(ReLU(W0(x))) + b1), Wg(...) ))
    See TFT paper for details.
    """

    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        if output_dim is None:
            output_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # GLU gating
        self.gate = GatingLayer(input_dim=output_dim, hidden_dim=output_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len, input_dim]
        """
        # 1) Feedforward
        hidden = self.fc1(x)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)

        # 2) GLU gating (the gating layer also performs a linear transform internally)
        gated = self.gate(hidden)

        # 3) Residual connection + layer norm
        out = self.layer_norm(x + gated)

        return out


# -----------------------------------------------------------------------------
# 3. Multi-Head Attention
# -----------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention mechanism:
      Att(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: [batch_size, seq_len, d_model]
        mask: Optional attention mask
        """
        B, L, _ = q.shape

        # 1) Linear projections
        q = self.W_q(q)  # [B, L, d_model]
        k = self.W_k(k)
        v = self.W_v(v)

        # 2) Reshape for multi-head: [B, L, num_heads, d_k]
        q = q.reshape(B, L, self.num_heads, self.d_k).transpose(1, 2)
        k = k.reshape(B, L, self.num_heads, self.d_k).transpose(1, 2)
        v = v.reshape(B, L, self.num_heads, self.d_k).transpose(1, 2)

        # 3) Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # [B, num_heads, L, d_k]

        # 4) Combine heads
        context = context.transpose(1, 2).contiguous().reshape(B, L, self.d_model)

        # 5) Final linear
        out = self.fc_out(context)
        return out, attn


# -----------------------------------------------------------------------------
# 4. TemporalFusionTransformer Module
# -----------------------------------------------------------------------------


class TemporalFusionTransformer(BaseModel, MonteCarloMixin):
    """
    Simplified TFT implementation. Key steps:
      1) Input embeddings (continuous/categorical -> d_model)
      2) LSTM-based local processing (encoder/decoder)
      3) Multi-head attention for long-range dependencies
      4) Gated Residual Networks (with skip connections)
      5) Final projection to output
    """

    def __init__(
        self,
        input_dim,  # number of input features (continuous + categorical)
        d_model=64,  # hidden dimension
        lstm_hidden_dim=64,  # LSTM hidden dimension
        num_heads=4,  # multi-head attention
        dropout=0.1,
        num_lstm_layers=1,  # depth of LSTM
        output_dim=1,  # dimension of the forecast output
    ):
        super(TemporalFusionTransformer, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.output_dim = output_dim

        # ---------------------------------------------------------------------
        # 4.1. Simple embedding for all inputs (continuous)
        # In practice, you might have separate embeddings for categorical vars.
        # Here, we do a single linear projection from input_dim -> d_model
        # ---------------------------------------------------------------------
        self.input_projection = nn.Linear(input_dim, d_model)

        # ---------------------------------------------------------------------
        # 4.2. LSTM for local processing (Encoder + Decoder).
        # We'll do a single LSTM for simplicity. Typically, TFT uses:
        #  - Past sequence: feed to LSTM encoder
        #  - Future known inputs: LSTM decoder
        # For simplicity, treat them as one LSTM here.
        # ---------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.lstm_proj = nn.Linear(lstm_hidden_dim, d_model)

        # ---------------------------------------------------------------------
        # 4.3. Multi-head Attention layer (to combine local context with global)
        # ---------------------------------------------------------------------
        self.attention = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.attention_add_norm = AddNorm(d_model)

        # ---------------------------------------------------------------------
        # 4.4. Gated Residual Network (applied after MHA)
        # ---------------------------------------------------------------------
        self.post_attn_gated_residual = GatedResidualNetwork(
            input_dim=d_model, hidden_dim=d_model, output_dim=d_model, dropout=dropout
        )

        # ---------------------------------------------------------------------
        # 4.5. Final feedforward (MLP) to produce forecast
        # ---------------------------------------------------------------------
        self.output_layer = nn.Linear(d_model, output_dim)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        """
        x: Input of shape [batch_size, seq_len, input_dim]
        Returns: Forecast of shape [batch_size, seq_len, output_dim]
        """

        # 1) Embed inputs
        #    shape => [B, seq_len, d_model]
        x_emb = self.input_projection(x)

        # 2) LSTM for local processing
        #    shape => [B, seq_len, d_model]
        lstm_out, _ = self.lstm(x_emb)
        # project LSTM output up to d_model dimension
        lstm_out = self.lstm_proj(lstm_out)

        # 3) Multi-head attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        #    Add & Norm
        x_attn = self.attention_add_norm(lstm_out, attn_out)

        # 4) Gated Residual Network on top of the attention output
        grn_out = self.post_attn_gated_residual(x_attn)

        # 5) Final projection to output dimension
        #    shape => [B, seq_len, output_dim]
        out = self.output_layer(grn_out)

        return out[:, -1, :]  # [batch_size, seq_len, output_dim]
