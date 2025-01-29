import torch
import torch.nn as nn
from quantbayes.forecast.nn.base import MonteCarloMixin, BaseModel


class NBeatsBlock(MonteCarloMixin, BaseModel):
    """
    A single N-BEATS block in "generic" mode:
      - MLP => two parameter vectors (theta_b, theta_f)
      - backcast = linear_b(theta_b)
      - forecast = linear_f(theta_f)
    """

    def __init__(
        self,
        input_size,  # L * input_dim (flattened size)
        hidden_dim=256,
        n_harmonics=1,  # not used here, but a placeholder if you want expansions
        n_layers=4,
        basis="generic",
    ):
        """
        :param input_size: Flattened input size = L * input_dim
        :param hidden_dim: Width of hidden layers in MLP
        :param n_layers:   Number of fully-connected layers in the block
        :param basis:      'generic', 'trend', 'seasonal' in original code. We'll keep 'generic' here.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.basis = basis

        # MLP for theta
        fc_layers = []
        in_dim = input_size
        for _ in range(n_layers):
            fc_layers.append(nn.Linear(in_dim, hidden_dim))
            fc_layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.fc = nn.Sequential(*fc_layers)

        # The MLP produces "theta" which we then split into backcast & forecast components
        # We'll define the output dimension of the MLP to be 2 * input_size
        # so we can split half for backcast, half for forecast (in the simplest case).
        # Because we want single-step forecast, we can do:
        # backcast size = input_size
        # forecast size = 1
        # => total = input_size + 1
        self.theta_dim = self.input_size + 1
        self.last_fc = nn.Linear(hidden_dim, self.theta_dim)

        # For "generic basis", we map the backcast part of theta directly to length input_size,
        # and forecast part directly to length = 1
        # So effectively:
        # backcast = backcast_proj(theta_b) -> shape (B, input_size)
        # forecast = forecast_proj(theta_f) -> shape (B, 1)
        self.backcast_lin = (
            nn.Identity()
        )  # can skip or do nn.Linear(backcast_size, backcast_size)
        self.forecast_lin = nn.Identity()  # same idea
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        """
        x: shape (B, input_size)
        Returns (backcast, forecast):
          - backcast shape (B, input_size)
          - forecast shape (B, 1)
        """
        # 1) pass input through the MLP
        hidden = self.fc(x)  # (B, hidden_dim)
        theta = self.last_fc(hidden)  # (B, input_size + 1)

        # 2) split into backcast and forecast parts
        theta_b = theta[:, : self.input_size]  # (B, input_size)
        theta_f = theta[:, self.input_size :]  # (B, 1)

        # 3) map them to actual outputs (generic basis => direct identity map)
        backcast = self.backcast_lin(theta_b)  # shape (B, input_size)
        forecast = self.forecast_lin(theta_f)  # shape (B, 1)

        return backcast, forecast


class NBeatsStack(nn.Module):
    """
    A stack of multiple N-BEATS blocks. The final forecast is sum of each block's forecast.
    The residual for block i+1 is x_{i+1} = x_i - backcast_i.
    """

    def __init__(
        self,
        input_size,  # L * input_dim
        num_blocks=3,
        block_hidden=256,
        n_layers=4,
        basis="generic",
    ):
        super().__init__()
        self.input_size = input_size
        self.num_blocks = num_blocks

        blocks = []
        for _ in range(num_blocks):
            block = NBeatsBlock(
                input_size=input_size,
                hidden_dim=block_hidden,
                n_layers=n_layers,
                basis=basis,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """
        x: shape (B, input_size)
        Returns: (B, 1)
        """
        # We'll accumulate forecasts and update residual each block
        forecast_final = torch.zeros(x.shape[0], 1, device=x.device)
        residual = x

        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_final = forecast_final + forecast

        return forecast_final


class NBeats(nn.Module):
    """
    A simplified N-BEATS model for single-step forecasting,
    with "generic basis" blocks, no future covariates.

    Input shape:  (B, L, input_dim)
    Output shape: (B, 1)
    """

    def __init__(
        self,
        seq_len,  # L
        input_dim=1,
        num_blocks=3,
        block_hidden=256,
        n_layers=4,
        basis="generic",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.input_size = seq_len * input_dim  # flatten
        self.n_beats_stack = NBeatsStack(
            input_size=self.input_size,
            num_blocks=num_blocks,
            block_hidden=block_hidden,
            n_layers=n_layers,
            basis=basis,
        )

    def forward(self, x):
        """
        x: shape (B, L, input_dim)
        -> flatten -> pass to NBeatsStack -> get forecast (B, 1)
        """
        B, L, C = x.shape
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got L={L}"
        assert C == self.input_dim, f"Expected input_dim={self.input_dim}, got C={C}"

        # Flatten
        x_flat = x.reshape(B, -1)  # (B, L*C)

        forecast = self.n_beats_stack(x_flat)  # (B, 1)
        return forecast


if __name__ == "__main__":
    batch_size = 16
    seq_len = 24
    input_dim = 2  # e.g. 2 features

    model = NBeats(
        seq_len=seq_len,
        input_dim=input_dim,
        num_blocks=3,
        block_hidden=128,
        n_layers=4,
        basis="generic",
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    y = model(x)  # (B, 1)
    print("Output shape:", y.shape)  # [16, 1]
