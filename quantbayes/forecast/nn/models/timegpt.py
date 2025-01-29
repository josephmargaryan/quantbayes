import torch
import torch.nn as nn
import torch.optim as optim
from quantbayes.forecast.nn.base import MonteCarloMixin, BaseModel
import math
import numpy as np
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings for each position in the sequence.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)

    def forward(self, x):
        """
        x: shape (batch_size, seq_len, d_model)
        Return: x + pos_embed (same shape)
        """
        batch_size, seq_len, d_model = x.shape

        # positions = [0, 1, 2, ... seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(
            0
        )  # shape (1, seq_len)

        # Expand to batch_size if needed (though we just broadcast add).
        pos_emb = self.pos_embedding(positions)  # shape (1, seq_len, d_model)
        return x + pos_emb


class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # (Optional) Add your favorite normalization & dropout
        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, attn_mask=None):
        """
        x: shape (batch_size, seq_len, d_model)
        attn_mask: shape (seq_len, seq_len) or (batch_size*n_heads, seq_len, seq_len)
        returns: same shape as x
        """
        # Self-attention
        # MultiheadAttention in PyTorch expects shape: (batch, seq, d_model) if batch_first=True
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + attn_output  # skip-connection (if we want minimal overhead)

        # Feed-forward
        ff_output = self.linear2(torch.relu(self.linear1(x)))
        x = x + ff_output  # skip-connection

        return x


def generate_subsequent_mask(seq_len):
    """
    Create a (seq_len, seq_len) mask for causal self-attention
    (where entry (i,j) = True means j can't attend to i if j < i).

    Actually we want to mask out future tokens =>
    we create a mask where the upper-triangle (j>i) is True.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask


class TimeGTP(MonteCarloMixin, BaseModel):
    def __init__(
        self,
        input_dim,  # dimension of each time step's features
        d_model=64,  # embedding (hidden) dimension
        nhead=4,
        num_layers=2,
        seq_len=30,
        dim_feedforward=256,
        max_len=5000,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Project input_dim -> d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional embedding
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_len=max_len)

        # Stacked GPT-style decoder blocks
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, nhead, dim_feedforward) for _ in range(num_layers)]
        )

        # Final linear to produce 1 scalar from the last token's hidden state
        self.output_linear = nn.Linear(d_model, 1)

        # (Optional) we can register the causal mask once if seq_len is fixed
        # but to be flexible, we can generate it on the fly in forward()
        # or store for each max seq_len.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        returns: (batch_size, 1)
        """
        bsz, seq_len, inp_dim = x.shape

        # 1) Project input to d_model dimension
        x = self.input_proj(x)  # shape: (bsz, seq_len, d_model)

        # 2) Add positional encodings
        x = self.pos_encoding(x)  # shape: (bsz, seq_len, d_model)

        # 3) Generate causal mask
        #    shape (seq_len, seq_len), with True in upper triangle
        attn_mask = generate_subsequent_mask(seq_len).to(x.device)

        # 4) Pass through each DecoderBlock
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # 5) Extract the hidden state at the LAST time step
        #    shape of last token's representation is (batch_size, d_model)
        x_last = x[:, -1, :]

        # 6) Output linear -> single scalar
        out = self.output_linear(x_last)  # shape: (batch_size, 1)
        return out


def generate_sine_data(batch_size=32, seq_len=10):
    """
    Generate some random sine waves (or a simple pattern) as a toy dataset.
    For each item in the batch:
       - We produce a sine wave segment of length seq_len,
       - We then define the 'label' as the next future value (the sine at seq_len+1).
    """
    freq = torch.rand(batch_size) * 2.0 + 0.5  # random freq in [0.5, 2.5)
    phase = torch.rand(batch_size) * 2 * math.pi

    # time points
    t = torch.linspace(0, 2 * math.pi, seq_len + 1)

    # shape => (batch_size, seq_len+1)
    sequences = []
    labels = []
    for i in range(batch_size):
        x = torch.sin(freq[i] * t + phase[i])
        sequences.append(x[:-1])  # first seq_len points
        labels.append(x[-1].unsqueeze(0))  # the next point
    sequences = torch.stack(sequences, dim=0)  # (batch_size, seq_len)
    labels = torch.stack(labels, dim=0)  # (batch_size, 1)

    # Expand dimension to match (batch_size, seq_len, input_dim=1)
    sequences = sequences.unsqueeze(-1)
    return sequences, labels


if __name__ == "__main__":
    # Quick test
    seq_len = 10
    train_seq, train_label = generate_sine_data(batch_size=32, seq_len=seq_len)
    print(train_seq.shape, train_label.shape)  # e.g. (32, 10, 1), (32, 1)

    # Hyperparams
    d_model = 32
    nhead = 4
    num_layers = 2
    dim_feedforward = 64

    model = TimeGTP(
        input_dim=1,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        seq_len=seq_len,
        dim_feedforward=dim_feedforward,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs = 200

    model.train()
    for epoch in range(num_epochs):
        # Generate fresh synthetic batch each epoch (for demonstration).
        # In a real scenario, you'd iterate over a dataset of multiple batches.
        x_batch, y_batch = generate_sine_data(batch_size=32, seq_len=seq_len)

        optimizer.zero_grad()
        y_pred = model(x_batch)  # shape (batch_size, 1)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        x_test, y_test = generate_sine_data(batch_size=5, seq_len=seq_len)
        y_pred = model(x_test)
        print("Targets:", y_test.view(-1).tolist())
        print("Preds:  ", y_pred.view(-1).tolist())
