import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """
    A single transformer block with multi-head self-attention and feed-forward network.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: (seq_len, batch_size, d_model)
        attn_output, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src


class TabPFN(nn.Module):
    """
    A simplified TabPFN-like model for tabular data.

    Parameters:
      - input_dim: Number of input features.
      - d_model: Dimensionality of the embedding space.
      - nhead: Number of attention heads.
      - num_layers: Number of transformer blocks.
      - dim_feedforward: Dimensionality of the feed-forward network.
      - dropout: Dropout probability.
      - num_classes: Number of output classes (or 1 for regression).
    """

    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=4,
        num_layers=6,
        dim_feedforward=128,
        dropout=0.1,
        num_classes=2,
    ):
        super().__init__()
        # Embed the input features.
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Optional positional embedding if you plan to treat each feature as a separate token.
        # Here we assume one token per sample (i.e. treat the whole feature vector as one token).
        # If you prefer to treat each feature as a token, adjust the reshape accordingly.
        # self.pos_embedding = nn.Parameter(torch.zeros(input_dim, d_model))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, input_dim)
        """
        # Embed the input
        x = self.input_embedding(x)  # shape: (batch_size, d_model)

        # If treating the entire feature vector as one token, add a dummy sequence dimension:
        x = x.unsqueeze(0)  # shape: (1, batch_size, d_model)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Remove the sequence dimension
        x = x.squeeze(0)  # shape: (batch_size, d_model)

        # Classification/regression head
        logits = self.classifier(x)  # shape: (batch_size, num_classes)
        return logits


# -----------------------------
# Example usage:
# -----------------------------
if __name__ == "__main__":
    # Suppose you have a tabular dataset with 20 features.
    input_dim = 20
    num_classes = 2  # for binary classification

    # Create a dummy input tensor (batch_size, input_dim)
    dummy_input = torch.randn(32, input_dim)

    # Instantiate the model
    model = TabPFN(
        input_dim,
        d_model=64,
        nhead=4,
        num_layers=4,
        dim_feedforward=128,
        dropout=0.1,
        num_classes=num_classes,
    )

    # Forward pass
    outputs = model(dummy_input)
    print("Model outputs shape:", outputs.shape)  # Expected: (32, 2)

    # For a real task, you'd define a loss, optimizer, and training loop.
