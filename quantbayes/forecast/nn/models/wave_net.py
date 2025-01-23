import torch
import torch.nn as nn
import torch.nn.functional as F
from quantbayes.forecast.nn.base import MonteCarloMixin, BaseModel

class WaveNetResidualBlock(nn.Module):
    """
    A single residual block in WaveNet:
      - Dilated causal conv -> gating (tanh x sigm)
      - 1x1 conv for residual
      - 1x1 conv for skip
    """
    def __init__(self,
                 in_channels,
                 dilation,
                 kernel_size=2,
                 residual_channels=32,
                 skip_channels=32):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size

        # For kernel_size=2 with dilation=d, we pad left = d*(kernel_size-1).
        self.padding = (kernel_size - 1) * dilation

        self.conv_filter = nn.Conv1d(
            in_channels, residual_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        self.conv_gate = nn.Conv1d(
            in_channels, residual_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )

        # 1x1 conv for residual and skip outputs
        self.conv_residual = nn.Conv1d(residual_channels, in_channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        """
        x: (B, in_channels, L)
        Returns:
          out => same shape as x (residual updated)
          skip => (B, skip_channels, L) 
        """
        # 1) Convolution outputs
        filter_out = self.conv_filter(x)   # shape => (B, residual_channels, L + self.padding)
        gate_out   = self.conv_gate(x)     # shape => (B, residual_channels, L + self.padding)

        # 2) Trim the extra frames on the right to keep length = L
        L = x.size(2)  # original length
        filter_out = filter_out[:, :, :L]  # (B, residual_channels, L)
        gate_out   = gate_out[:, :, :L]    # (B, residual_channels, L)

        # 3) Gated activation
        gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)  # (B, residual_channels, L)

        # 4) Skip path
        skip = self.conv_skip(gated)       # (B, skip_channels, L)

        # 5) Residual path
        residual = self.conv_residual(gated)  # (B, in_channels, L)
        out = x + residual  # shape (B, in_channels, L)

        return out, skip


class WaveNet(MonteCarloMixin, BaseModel):
    """
    A WaveNet for single-step forecasting:
      Input shape:  (B, seq_len, input_dim)
      Output shape: (B, 1)
    """
    def __init__(self,
                 input_dim=1,        # #features
                 residual_channels=32,
                 skip_channels=32,
                 dilation_depth=6,   # number of layers
                 kernel_size=2):
        super().__init__()
        self.input_dim = input_dim
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilation_depth = dilation_depth
        self.kernel_size = kernel_size

        # 1) initial 1x1 conv: map input_dim => residual_channels
        self.input_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=residual_channels,
            kernel_size=1
        )

        # 2) stack of residual blocks with dilations [1,2,4,...2^(dilation_depth-1)]
        self.blocks = nn.ModuleList()
        for i in range(dilation_depth):
            dilation = 2 ** i
            block = WaveNetResidualBlock(
                in_channels=residual_channels,
                dilation=dilation,
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                skip_channels=skip_channels
            )
            self.blocks.append(block)

        # 3) Combine skip outputs. 
        # We'll do two 1x1 conv layers on the sum of all skip connections
        self.postprocess1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.postprocess2 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)

        # final linear for single-step forecast
        self.final_fc = nn.Linear(skip_channels, 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        """
        x: (B, L, input_dim)
        => returns: (B, 1)
        """
        B, L, C = x.shape

        # rearrange to (B, C, L)
        x_t = x.transpose(1, 2)  # (B, input_dim, L)

        # initial 1x1 conv -> (B, residual_channels, L)
        current = self.input_conv(x_t)

        skip_total = 0.0

        # pass through each residual block
        for block in self.blocks:
            current, skip_out = block(current)
            # current => (B, residual_channels, L)
            # skip_out => (B, skip_channels, L)
            skip_total = skip_total + skip_out

        # (B, skip_channels, L)
        out = F.relu(skip_total)
        out = F.relu(self.postprocess1(out))
        out = F.relu(self.postprocess2(out))
        # out => (B, skip_channels, L)

        # take last time step => (B, skip_channels)
        out_last = out[:, :, -1]

        # final FC => (B, 1)
        pred = self.final_fc(out_last)
        return pred


if __name__ == "__main__":
    import torch
    batch_size = 8
    seq_len = 30
    input_dim = 2

    model = WaveNet(
        input_dim=input_dim,
        residual_channels=32,
        skip_channels=32,
        dilation_depth=4,  # => dilations = 1,2,4,8
        kernel_size=2
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    y = model(x)  # => (B, 1)
    print("Output shape:", y.shape)

    target = torch.randn(batch_size, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = loss_fn(y, target)
    loss.backward()
    optimizer.step()
    print("Loss:", loss.item())
