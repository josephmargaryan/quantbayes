import torch
import torch.nn as nn
from quantbayes.forecast.nn import BaseModel, MonteCarloMixin


class LSTM(BaseModel, MonteCarloMixin):
    """
    A simple LSTM-based model for single-step forecasting:
      - LSTM input: (B, seq_len, input_dim)
      - We take the last hidden state -> project to (B,1)
    """

    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # LSTM: batch_first = True means input is (B, L, input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Final linear layer to produce single scalar
        self.fc = nn.Linear(hidden_dim, 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        """
        x: shape (B, seq_len, input_dim)
        => (B, 1)
        """
        B, L, C = x.shape

        # Run the LSTM
        # out: (B, L, hidden_dim)
        # (h_n, c_n): each is (num_layers, B, hidden_dim)
        out, (h_n, c_n) = self.lstm(x)

        # The last hidden state of the top LSTM layer is h_n[-1], shape (B, hidden_dim)
        # Alternatively, we can also do out[:, -1, :] which is the last time step's output
        # but typically for forecasting, we use the final hidden state:
        last_hidden = h_n[-1]  # (B, hidden_dim)

        # Project to single step
        y = self.fc(last_hidden)  # (B, 1)
        return y


if __name__ == "__main__":
    # Example usage
    model = LSTM(input_dim=5, hidden_dim=32, num_layers=2, dropout=0.1)
    x = torch.randn(8, 10, 5)  # (batch=8, seq_len=10, input_dim=5)
    y = model(x)  # => (8, 1)
    print("LSTMStyleNet output shape:", y.shape)
