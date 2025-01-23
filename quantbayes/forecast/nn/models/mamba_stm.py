import torch
import torch.nn as nn
from quantbayes.forecast.nn import BaseModel, MonteCarloMixin

class MambaStateSpace(BaseModel, MonteCarloMixin):
    """
    A simple custom 'state space' model:
      z_{t+1} = tanh(A z_t + B x_t + bias)
      final forecast = W z_{last} + d

    Input shape:  (B, seq_len, input_dim)
    Output shape: (B, 1)
    """
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # State update parameters: A, B, plus a bias
        # We'll store them as nn.Linear for convenience: we give it (z_t + x_t) 
        # dimension: hidden_dim + input_dim -> hidden_dim
        self.state_update = nn.Linear(hidden_dim + input_dim, hidden_dim)
        
        # Output layer: from the final hidden state -> (1)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        """
        x: shape (B, seq_len, input_dim)
        Returns: (B, 1)
        """
        B, L, C = x.shape
        # We'll maintain a hidden state z of shape (B, hidden_dim).
        # Initialize z to zeros (you can do a learnable param if desired)
        z = torch.zeros(B, self.hidden_dim, device=x.device)

        # Unroll over time
        for t in range(L):
            x_t = x[:, t, :]  # shape (B, input_dim)
            # concat z_t and x_t -> (B, hidden_dim + input_dim)
            zx = torch.cat([z, x_t], dim=1)
            # update step
            z_next = self.state_update(zx)  # (B, hidden_dim)
            z_next = torch.tanh(z_next)
            z = z_next

        # After the last step, we map z -> a single scalar
        out = self.output_layer(z)  # (B, 1)
        return out

if __name__ == "__main__":
    # Example usage
    model = MambaStateSpace(input_dim=5, hidden_dim=16)
    x = torch.randn(8, 20, 5)  # (batch=8, seq_len=20, input_dim=5)
    y = model(x)               # => (8, 1)
    print("MambaStateSpace output shape:", y.shape)
