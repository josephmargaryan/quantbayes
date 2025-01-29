from quantbayes.forecast.nn import BaseModel, MonteCarloMixin
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# 1. Series Decomposition
#    This is typically done with a moving-average filter to get "trend"
#    and then "seasonality = x - trend".
#    We'll do a simple version with a fixed window.
# ---------------------------------------------------------------------
class SeriesDecomposition(nn.Module):
    """Decompose a time series into trend + seasonal using simple moving average."""

    def __init__(self, kernel_size=25):
        super().__init__()
        # Here, we create a 1D depthwise-conv that acts like a moving average
        # for each feature dimension, if we want. For simplicity, do it channel-by-channel.
        self.kernel_size = kernel_size
        # We fix the weights to average = 1 / kernel_size
        # For multi-feature: groups=in_channels (depthwise).
        # But let's do a simpler approach: just do the average ourselves in forward.
        # The code below is a placeholder if you want a learnable conv:
        # self.avg_pool = nn.Conv1d(
        #     in_channels=1, out_channels=1,
        #     kernel_size=kernel_size, stride=1,
        #     padding=kernel_size//2, groups=1, bias=False
        # )
        # self.avg_pool.weight.data = torch.ones_like(self.avg_pool.weight.data) / kernel_size
        # self.avg_pool.weight.requires_grad = False

    def forward(self, x):
        """
        x: shape (B, L, C). We'll treat each feature dimension independently.
        Return: trend, seasonal
        """
        B, L, C = x.shape
        # For each feature in range(C), compute a moving average over time dimension L.
        # We'll do it in a naive way for clarity (can optimize with F.conv1d).
        trend = []
        pad = self.kernel_size // 2
        # We can pad the time dimension so the convolution is "same" sized
        x_pad = F.pad(x.transpose(1, 2), (pad, pad), mode="reflect")  # (B, C, L+2*pad)
        # Now x_pad shape is [B, C, L + kernel_size - 1]
        # We'll do a moving average by convolving with 1/kernel_size
        # shape => (B, C, L)
        for i in range(C):
            # conv1d for that channel
            # But let's do direct
            channel_i = x_pad[:, i : i + 1, :]  # shape [B, 1, L+2pad]
            weight = (
                torch.ones(1, 1, self.kernel_size, device=x.device) / self.kernel_size
            )
            # conv1d
            trend_i = F.conv1d(channel_i, weight, padding=0)  # [B, 1, L]
            trend.append(trend_i)
        trend = torch.cat(trend, dim=1)  # [B, C, L]
        trend = trend.transpose(1, 2)  # [B, L, C]
        seasonal = x - trend
        return trend, seasonal


# ---------------------------------------------------------------------
# 2. Frequency (“Fourier”) Block
#    Instead of typical dot-product attention, we do:
#    - transform input to frequency domain (e.g., FFT)
#    - apply some random or top-K frequency selection
#    - inverse transform
# ---------------------------------------------------------------------
class FourierBlock(nn.Module):
    """A simple frequency block that uses 1D FFT -> processing -> iFFT."""

    def __init__(self, top_k_freq=16):
        super().__init__()
        self.top_k_freq = top_k_freq
        # Potentially, we can do something more elaborate,
        # but let's keep it extremely simple for demonstration.

    def forward(self, x):
        """
        x: [B, L, C], real-valued
        We'll do FFT along time dimension => [B, L, C] -> [B, L, C_complex]
        Actually, PyTorch's rfft -> shape [B, L, C, 2] or so...
        We'll keep it simple with torch.fft.fft
        """
        B, L, C = x.shape

        # 1) FFT
        freq_x = torch.fft.rfft(x, dim=1)  # shape [B, L//2+1, C], complex

        # 2) Possibly select top-K largest frequencies? Let's do magnitude-based
        #    This is a typical approach in Fedformer to reduce complexity.
        mag = freq_x.abs()  # [B, L//2+1, C]
        # Flatten B*C dimension to pick top-K globally or do local? We'll do a naive approach:
        # We'll pick top_k_freq frequencies for each (B, C) slice.
        # We skip the complicated aggregator from the official code to keep it short.
        # That means for each batch and channel, we pick top_k_freq freq indices.
        # This is a simplistic approach. In practice, there's an aggregator across layers, etc.
        freq_x_filtered = torch.zeros_like(freq_x)
        for b in range(B):
            for c_ in range(C):
                # magnitude for that b,c => shape [L//2+1]
                mag_bc = mag[b, :, c_]
                # top-k indices
                top_k = torch.topk(mag_bc, k=min(self.top_k_freq, len(mag_bc)))[1]
                # keep only those frequencies
                freq_x_filtered[b, top_k, c_] = freq_x[b, top_k, c_]
        # freq_x_filtered is basically freq_x but zeroed except top-k freqs

        # 3) iFFT
        out = torch.fft.irfft(freq_x_filtered, n=L, dim=1)  # shape [B, L, C], real
        return out


# ---------------------------------------------------------------------
# 3. The Fedformer Encoder (single block version)
#    We can stack multiple frequency blocks, or combine wavelet/fourier.
# ---------------------------------------------------------------------
class FedformerEncoderLayer(nn.Module):
    """
    One layer that:
      - Possibly does a frequency block on "seasonal" component
      - Then has a feed-forward path
      - Optionally merges 'trend' back or passes it onward
    """

    def __init__(self, d_model=64, top_k_freq=16, dropout=0.1):
        super().__init__()
        self.fourier_block = FourierBlock(top_k_freq=top_k_freq)
        self.linear = nn.Linear(d_model, d_model)  # basic feed-forward
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, seasonal, trend):
        """
        seasonal, trend: [B, L, d_model]
        We'll process the 'seasonal' with the FourierBlock,
        then do a small feed-forward, then combine with skip-connection, etc.
        """
        # 1) frequency modeling of seasonal
        freq_seasonal = self.fourier_block(seasonal)  # [B, L, d_model]

        # 2) simple feed-forward on freq_seasonal
        out = self.linear(freq_seasonal)
        out = self.dropout(out)
        out = self.layernorm(seasonal + out)  # skip connection on seasonal

        # We can pass 'trend' forward unmodified or do some manipulation
        # For simplicity: just keep trend as is
        return out, trend


class FedformerEncoder(nn.Module):
    """Stack multiple FedformerEncoderLayer with series decomposition each time."""

    def __init__(
        self, d_model=64, layer_num=2, decomp_ks=25, top_k_freq=16, dropout=0.1
    ):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size=decomp_ks)
        self.layers = nn.ModuleList(
            [
                FedformerEncoderLayer(
                    d_model=d_model, top_k_freq=top_k_freq, dropout=dropout
                )
                for _ in range(layer_num)
            ]
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, L, d_model]
        We do repeated decomposition -> frequency block -> ...
        Return final seasonal+trend combined, or separate them.
        """
        seasonal_init, trend_init = self.decomp(x)  # [B, L, d_model] each

        seasonal, trend = seasonal_init, trend_init
        for layer in self.layers:
            seasonal, trend = layer(seasonal, trend)
            # optional further decomposition each layer as in the official code
            # but let's keep it simpler for demonstration.

        # combine them for final output, or keep separate if you'd like
        out = seasonal + trend
        out = self.layernorm(out)
        return out  # [B, L, d_model]


# ---------------------------------------------------------------------
# 4. Full Fedformer (Simplified) for Single-Step
# ---------------------------------------------------------------------
class Fedformer(BaseModel, MonteCarloMixin):
    """
    A simplified Fedformer-like model with:
      - Input linear embedding
      - FedformerEncoder
      - Output projection (single-step forecast -> we take last step's output)

    Input shape: (B, L, input_dim)
    Output shape: (B, 1)
    """

    def __init__(
        self,
        input_dim,
        d_model=64,
        layer_num=2,
        top_k_freq=16,
        dropout=0.1,
        kernel_size=25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        # 1) Input embedding
        self.embedding = nn.Linear(input_dim, d_model)
        # 2) Fedformer encoder
        self.encoder = FedformerEncoder(
            d_model=d_model,
            layer_num=layer_num,
            decomp_ks=kernel_size,
            top_k_freq=top_k_freq,
            dropout=dropout,
        )
        # 3) Final projection
        self.projection = nn.Linear(d_model, 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        """
        x: (B, L, input_dim)
        returns: (B, 1)
        """
        # 1) embed
        x = self.embedding(x)  # [B, L, d_model]
        # 2) fedformer encoder
        enc_out = self.encoder(x)  # [B, L, d_model]
        # 3) take the last time step
        last_token = enc_out[:, -1, :]  # [B, d_model]
        # 4) project to 1
        out = self.projection(last_token)  # [B, 1]
        return out


# ---------------------------------------------------------------------
# 5. Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    batch_size = 4
    seq_len = 32
    input_dim = 8

    model = Fedformer(
        input_dim=input_dim,
        d_model=32,
        layer_num=2,
        top_k_freq=10,
        dropout=0.1,
        kernel_size=15,
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    y = model(x)
    print("Output shape:", y.shape)  # should be [4, 1]
