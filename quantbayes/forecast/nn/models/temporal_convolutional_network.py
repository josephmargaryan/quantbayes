import torch
import torch.nn as nn
import torch.nn.functional as F
from quantbayes.forecast.nn import BaseModel, MonteCarloMixin


class TemporalConvBlock(nn.Module):
    """
    A single TCN residual block:
      - 2 x (Conv1d -> ReLU -> Dropout)
      - Residual/skip connection
      - Causal convolutions with dilation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        dropout=0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dropout = dropout

        # Calculate padding to ensure "causal" and keep length the same
        # For a kernel_size=3, effective receptive field = dilation*(kernel_size-1)+1
        # We want to pad left = dilation*(kernel_size-1)
        self.padding = (kernel_size - 1) * dilation

        # First conv layer
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second conv layer
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout_layer = nn.Dropout(dropout)

        # If in_channels != out_channels, we use a 1x1 conv to match dimensions for residual
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        """
        x shape: (B, in_channels, L)
        returns shape: (B, out_channels, L)
        """
        # First convolution
        out = self.conv1(x)
        # Remove the "future" part by slicing => out[:, :, :-self.padding]
        #   if we want strictly causal at each time step.
        #   However, typically TCN keeps the same length and relies on the padding at left.
        #   We just trust the "left padding" for causality.
        #   We'll keep length the same for convenience, ignoring the trailing output that might
        #   have partial future leakage. But standard TCN references do so.
        out = out[:, :, : x.size(2)]  # ensure output length is same as x if needed
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout_layer(out)

        # Second convolution
        out = self.conv2(out)
        out = out[:, :, : x.size(2)]  # keep same length
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout_layer(out)

        # Residual
        res = x
        if self.downsample is not None:
            # match channel dimensions
            res = self.downsample(res)
            res = res[:, :, : x.size(2)]  # keep length

        return F.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    A multi-layer TCN:
      - Each layer is a TemporalConvBlock with increasing dilation.
      - Channels might be [in_channels, hidden_channels, ..., out_channels].
    """

    def __init__(
        self,
        in_channels,  # input feature channels
        num_channels_list,  # list of output channels for each layer
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__()
        layers = []
        prev_channels = in_channels
        for i, out_channels in enumerate(num_channels_list):
            dilation = 2**i
            block = TemporalConvBlock(
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            layers.append(block)
            prev_channels = out_channels

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x shape: (B, in_channels, L)
        returns: (B, last_out_channels, L)
        """
        return self.network(x)


class TCNForecaster(BaseModel, MonteCarloMixin):
    """
    A TCN for time-series forecasting:
      - Input shape:  (B, seq_len, input_dim)
      - Output shape: (B, 1) (predicts single step, e.g. last time step)
    """

    def __init__(
        self, input_dim=1, tcn_channels=[32, 32, 64], kernel_size=3, dropout=0.2
    ):
        super().__init__()
        # The TCN expects channels as the "input_features".
        # We'll transform (B, L, input_dim) => (B, input_dim, L).
        self.input_dim = input_dim
        self.tcn_channels = tcn_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Build TCN backbone
        # First layer expects "in_channels=input_dim"
        self.tcn = TemporalConvNet(
            in_channels=input_dim,
            num_channels_list=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # The output of TCN has channels = tcn_channels[-1].
        # We'll reduce that to 1 output with a linear layer on the last time step's embedding.
        self.output_linear = nn.Linear(tcn_channels[-1], 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        """
        x: (B, seq_len, input_dim)
        returns: (B, 1)
        """
        B, L, C = x.shape
        # (B, C, L) for Conv1d
        x_t = x.transpose(1, 2)  # shape (B, input_dim, seq_len)

        # Pass through TCN
        tcn_out = self.tcn(x_t)  # (B, tcn_channels[-1], L)

        # We take the last time step: tcn_out[:, :, -1] => (B, tcn_channels[-1])
        last_step = tcn_out[:, :, -1]  # (B, tcn_channels[-1])

        # Final projection to 1 dimension
        out = self.output_linear(last_step)  # (B, 1)
        return out


if __name__ == "__main__":
    import torch

    batch_size = 8
    seq_len = 20
    input_dim = 3

    model = TCNForecaster(
        input_dim=input_dim, tcn_channels=[16, 16, 32], kernel_size=3, dropout=0.1
    )

    x = torch.randn(batch_size, seq_len, input_dim)  # e.g. 3 features
    y = model(x)  # => (batch_size, 1)
    print("Output shape:", y.shape)  # should be (8, 1)

    # You could then do an MSE loss, for example:
    target = torch.randn(batch_size, 1)
    loss_fn = nn.MSELoss()
    loss = loss_fn(y, target)
    loss.backward()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(x)  # (B, 1)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
