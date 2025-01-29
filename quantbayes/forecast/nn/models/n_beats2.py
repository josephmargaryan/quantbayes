import torch
import torch.nn as nn
from quantbayes.forecast.nn import BaseModel, MonteCarloMixin


# --------------------------------------------------------------------
# 1) Helper: Trend & Seasonal Basis Matrices
# --------------------------------------------------------------------
def trend_matrix(L, degree, device=None):
    """
    Create a (L x degree) polynomial matrix, e.g. for i in [0..L-1],
    columns are [1, i, i^2, ..., i^(degree-1)] (scaled).
    """
    t = torch.linspace(0, 1, steps=L, device=device).unsqueeze(1)  # (L, 1)
    powers = torch.cat([t**p for p in range(degree)], dim=1)  # (L, degree)
    return powers


def seasonal_matrix(L, harmonics, device=None):
    """
    Create a (L x 2*harmonics) matrix with columns of sin/cos expansions:
       [cos(2 pi k t), sin(2 pi k t), ...] for k=1..harmonics
    """
    t = torch.linspace(0, 1, steps=L, device=device)  # (L,)
    mats = []
    for k in range(1, harmonics + 1):
        mats.append(torch.cos(2 * torch.pi * k * t))  # (L,)
        mats.append(torch.sin(2 * torch.pi * k * t))
    mat = torch.stack(mats, dim=1)  # (L, 2*harmonics)
    return mat


# --------------------------------------------------------------------
# 2) NBeatsBlock (Generic / Trend / Seasonal)
# --------------------------------------------------------------------
class NBeatsBlock(nn.Module):
    """
    A single N-BEATS block supporting 'generic', 'trend', or 'seasonal' bases
    for univariate or (for 'generic') multi-variate.
    """

    def __init__(
        self,
        input_size,  # Flattened length = L * input_dim
        backcast_length,  # L  (time steps)
        input_dim=1,  # how many features (must be 1 if trend/seasonal)
        basis="generic",
        hidden_dim=256,
        n_layers=4,
        degree_of_polynomial=2,
        n_harmonics=2,
    ):
        super().__init__()
        self.input_size = input_size  # L * input_dim
        self.backcast_length = backcast_length  # L
        self.input_dim = input_dim
        self.basis = basis
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.degree_of_polynomial = degree_of_polynomial
        self.n_harmonics = n_harmonics

        # If user wants 'trend' or 'seasonal' but has input_dim > 1, raise an error
        if self.basis in ["trend", "seasonal"] and input_dim != 1:
            raise ValueError(
                f"Basis '{self.basis}' is only implemented for univariate time series "
                f"(input_dim=1). For multivariate, use basis='generic'."
            )

        # Build MLP
        fc_layers = []
        in_dim = input_size
        for _ in range(n_layers):
            fc_layers.append(nn.Linear(in_dim, hidden_dim))
            fc_layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.fc = nn.Sequential(*fc_layers)

        # Decide how large "theta" is for backcast & forecast
        if basis == "generic":
            # backcast size = input_size
            # forecast size = 1
            self.backcast_size = input_size
            self.forecast_size = 1
        elif basis == "trend":
            # Univariate => input_size = L*1 = L
            # backcast => (L x degree_of_polynomial)
            self.backcast_size = self.backcast_length * self.degree_of_polynomial
            # forecast => (1 x degree_of_polynomial)
            self.forecast_size = self.degree_of_polynomial
        elif basis == "seasonal":
            # Univariate => input_size = L*1 = L
            # backcast => (L x (2*n_harmonics))
            self.backcast_size = self.backcast_length * (2 * self.n_harmonics)
            # forecast => (1 x (2*n_harmonics))
            self.forecast_size = 2 * self.n_harmonics
        else:
            raise ValueError(f"Unknown basis: {basis}")

        total_theta_dim = self.backcast_size + self.forecast_size
        self.last_fc = nn.Linear(hidden_dim, total_theta_dim)

    def forward(self, x):
        """
        x: shape (B, input_size).
        Return (backcast, forecast):
          backcast shape (B, input_size) for 'generic' or flattened time dimension for others
          forecast shape (B, 1) for generic, or (B, 1) for others
        """
        B = x.shape[0]
        device = x.device

        # 1) MLP
        hidden = self.fc(x)  # (B, hidden_dim)
        theta = self.last_fc(hidden)  # (B, backcast_size + forecast_size)
        theta_b = theta[:, : self.backcast_size]
        theta_f = theta[:, self.backcast_size :]

        if self.basis == "generic":
            backcast = theta_b  # (B, input_size)
            forecast = theta_f  # (B, 1)

        elif self.basis == "trend":
            # backcast_poly => (B, L, degree_of_polynomial)
            backcast_poly = theta_b.view(
                B, self.backcast_length, self.degree_of_polynomial
            )
            forecast_poly = theta_f.view(B, 1, self.degree_of_polynomial)

            t_back = trend_matrix(
                self.backcast_length, self.degree_of_polynomial, device=device
            )
            t_fore = trend_matrix(1, self.degree_of_polynomial, device=device)

            # (B, L, deg) * (L, deg)? We want (B, L) after summation over deg
            # We'll do elementwise multiply with broadcasting
            # Actually let's do: for each t, backcast[t] = sum_{m} backcast_poly[t,m] * t_back[t,m]
            # so:
            backcast = (backcast_poly * t_back.unsqueeze(0)).sum(dim=2)  # (B, L)
            # forecast => (B, 1)
            forecast = (forecast_poly * t_fore.unsqueeze(0)).sum(dim=2)  # (B, 1)

            # Flatten backcast to match x shape (B, L) => if x was univariate => x shape is (B, L)
            # We keep it as (B, L). We'll subtract from residual at stack-level.

        elif self.basis == "seasonal":
            # (B, L, 2*n_harmonics)
            backcast_fourier = theta_b.view(
                B, self.backcast_length, 2 * self.n_harmonics
            )
            forecast_fourier = theta_f.view(B, 1, 2 * self.n_harmonics)

            s_back = seasonal_matrix(
                self.backcast_length, self.n_harmonics, device=device
            )
            s_fore = seasonal_matrix(1, self.n_harmonics, device=device)

            backcast = (backcast_fourier * s_back.unsqueeze(0)).sum(dim=2)  # (B, L)
            forecast = (forecast_fourier * s_fore.unsqueeze(0)).sum(dim=2)  # (B, 1)

        return backcast, forecast


# --------------------------------------------------------------------
# 3) NBeatsStack
# --------------------------------------------------------------------
class NBeatsStack(nn.Module):
    """
    Multiple N-BEATS blocks in series (residual approach).
    Summation of forecast from each block => final forecast.
    """

    def __init__(
        self,
        input_size,
        backcast_length,
        input_dim=1,
        num_blocks=3,
        block_hidden=256,
        n_layers=4,
        basis="generic",
        degree_of_polynomial=2,
        n_harmonics=2,
    ):
        super().__init__()
        self.input_size = input_size
        self.backcast_length = backcast_length
        self.input_dim = input_dim
        self.num_blocks = num_blocks

        blocks = []
        for _ in range(num_blocks):
            block = NBeatsBlock(
                input_size=input_size,
                backcast_length=backcast_length,
                input_dim=input_dim,
                basis=basis,
                hidden_dim=block_hidden,
                n_layers=n_layers,
                degree_of_polynomial=degree_of_polynomial,
                n_harmonics=n_harmonics,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """
        x: shape (B, input_size) if basis='generic' or univariate flattened (B, L) if trend/seasonal.

        If 'generic' + multivariate: x => (B, L*input_dim)
        If 'trend'/'seasonal' => univariate => x => (B, L)
        Returns final forecast => (B, 1).
        """
        B = x.shape[0]
        # We'll accumulate forecasts
        forecast_final = torch.zeros(B, 1, device=x.device)

        residual = x
        for block in self.blocks:
            backcast, forecast = block(residual)
            # For generic basis => backcast shape (B, input_size)
            # For trend/seasonal => (B, L)
            # Either way, subtract from residual
            residual = residual - backcast
            forecast_final = forecast_final + forecast

        return forecast_final


# --------------------------------------------------------------------
# 4) Full Model: NBeatsNoFuture
# --------------------------------------------------------------------
class NBeats2(BaseModel, MonteCarloMixin):
    """
    N-BEATS model for single-step forecasting, with selectable basis:
      - 'generic' (can handle input_dim>1)
      - 'trend'/'seasonal' (assumes univariate => input_dim=1)

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
        degree_of_polynomial=2,
        n_harmonics=2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.basis = basis  # <---- ADD THIS LINE
        self.input_size = seq_len * input_dim
        self.backcast_length = seq_len

        if basis in ["trend", "seasonal"] and input_dim != 1:
            raise ValueError(
                f"Basis='{basis}' is only implemented for univariate series (input_dim=1). "
                f"For multivariate, use basis='generic'."
            )

        self.n_beats_stack = NBeatsStack(
            input_size=(
                self.seq_len if basis in ["trend", "seasonal"] else self.input_size
            ),
            backcast_length=self.backcast_length,
            input_dim=(1 if basis in ["trend", "seasonal"] else input_dim),
            num_blocks=num_blocks,
            block_hidden=block_hidden,
            n_layers=n_layers,
            basis=basis,
            degree_of_polynomial=degree_of_polynomial,
            n_harmonics=n_harmonics,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        B, L, C = x.shape
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"
        assert C == self.input_dim, f"Expected input_dim={self.input_dim}, got {C}"

        # Now we can check self.basis safely
        if self.basis == "generic":
            # Flatten to (B, L*C)
            x_flat = x.reshape(B, -1)
            forecast = self.n_beats_stack(x_flat)
        else:
            # 'trend' or 'seasonal' => univariate => input_dim=1
            x_single = x[..., 0]  # (B, L)
            forecast = self.n_beats_stack(x_single)

        return forecast


# --------------------------------------------------------------------
# 5) Quick Test
# --------------------------------------------------------------------
if __name__ == "__main__":
    batch_size = 4
    seq_len = 16

    # ------------------------
    # 5.1 Generic + multi-variate
    # ------------------------
    input_dim = 2
    model_generic = NBeats2(
        seq_len=seq_len,
        input_dim=input_dim,
        num_blocks=2,
        block_hidden=128,
        basis="generic",
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    y = model_generic(x)
    print("Generic =>", y.shape)  # (B, 1)

    # ------------------------
    # 5.2 Trend + univariate
    # ------------------------
    input_dim = 1
    model_trend = NBeats2(
        seq_len=seq_len,
        input_dim=input_dim,
        num_blocks=2,
        block_hidden=128,
        basis="trend",
        degree_of_polynomial=3,
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    y = model_trend(x)
    print("Trend =>", y.shape)  # (B, 1)

    # ------------------------
    # 5.3 Seasonal + univariate
    # ------------------------
    input_dim = 1
    model_seasonal = NBeats2(
        seq_len=seq_len,
        input_dim=input_dim,
        num_blocks=2,
        block_hidden=128,
        basis="seasonal",
        n_harmonics=5,
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    y = model_seasonal(x)
    print("Seasonal =>", y.shape)  # (B, 1)
