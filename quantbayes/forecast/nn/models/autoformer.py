from quantbayes.forecast.nn import BaseModel, MonteCarloMixin
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1) Series Decomposition (same as in the Fedformer example, or simplified)
# -----------------------------------------------------------------------------
class SeriesDecomposition(nn.Module):
    """Decompose a time series into trend + seasonal using a simple moving average."""
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        x: shape (B, L, C)
        Returns: (trend, seasonal) each shape (B, L, C)
        """
        B, L, C = x.shape
        pad = self.kernel_size // 2
        # Pad in time dimension
        x_pad = F.pad(x.transpose(1, 2), (pad, pad), mode='reflect')  # (B, C, L + 2*pad)

        # Compute moving average (trend) for each channel
        trend = []
        for c_ in range(C):
            channel_data = x_pad[:, c_:c_+1, :]  # [B, 1, L+2*pad]
            weight = torch.ones(1, 1, self.kernel_size, device=x.device) / self.kernel_size
            trend_c = F.conv1d(channel_data, weight, padding=0)  # [B, 1, L]
            trend.append(trend_c)
        trend = torch.cat(trend, dim=1)          # [B, C, L]
        trend = trend.transpose(1,2)            # [B, L, C]
        seasonal = x - trend
        return trend, seasonal

# -----------------------------------------------------------------------------
# 2) AutoCorrelation Block
#    Instead of standard "dot-product attention," Autoformer uses an
#    "AutoCorrelation" mechanism that tries to find "lags" with the strongest
#    correlation, primarily done in the frequency domain.
# -----------------------------------------------------------------------------
class AutoCorrelationBlock(nn.Module):
    """
    A simplified version of the "Auto-Correlation" mechanism:
      - Transform to frequency domain via FFT
      - Multiply Q * conj(K) => correlation in freq domain
      - iFFT -> get correlation in time domain
      - Possibly pick top-k correlated time-lags
      - Reconstruct an output

    For simplicity, we'll skip some advanced features and do a naive approach
    similar to the "FourierBlock" used in the simplified Fedformer code.
    """
    def __init__(self, top_k=16):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        """
        x: shape [B, L, C] (we treat x as both Q and K for "self-correlation")
        Returns: shape [B, L, C]
        """
        B, L, C = x.shape

        # 1) FFT
        X_freq = torch.fft.rfft(x, dim=1)  # shape [B, L//2+1, C], complex

        # 2) "AutoCorrelation" => X_freq * conj(X_freq) = power spectrum
        #    Actually in the official code, Q and K might be different. We'll treat them as the same for self-corr.
        #    In practice: Corr = iFFT( FFT(Q)*conj(FFT(K)) ). But Q=K => we do iFFT(|X_freq|^2).
        AC_freq = X_freq * torch.conj(X_freq)  # [B, L//2+1, C]

        # 3) iFFT -> get correlation in time domain
        corr_time = torch.fft.irfft(AC_freq, n=L, dim=1)  # [B, L, C], real

        # 4) Optionally pick top-k lags in time dimension (largest absolute correlation)
        #    We'll do a naive approach: zero all except top_k largest correlations for each (B, C).
        #    This is conceptual; real code might do more advanced "Time Delay Aggregation."
        out = torch.zeros_like(corr_time)
        mag = corr_time.abs()  # [B, L, C]
        for b in range(B):
            for c_ in range(C):
                # top-k in magnitude
                topk_indices = torch.topk(mag[b, :, c_], k=min(self.top_k, L))[1]
                out[b, topk_indices, c_] = corr_time[b, topk_indices, c_]

        # 5) (optional) We might want to combine correlation with x. We'll keep it simple and return out.
        return out

# -----------------------------------------------------------------------------
# 3) Autoformer Encoder Layer
# -----------------------------------------------------------------------------
class AutoformerEncoderLayer(nn.Module):
    """
    - Takes seasonal + trend
    - Applies AutoCorrelationBlock to 'seasonal'
    - Then feed-forward, skip-connection
    """
    def __init__(self, d_model=64, top_k=16, dropout=0.1):
        super().__init__()
        self.autocorr = AutoCorrelationBlock(top_k=top_k)
        self.ff = nn.Linear(d_model, d_model)  # simplistic feed-forward
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, seasonal, trend):
        """
        seasonal, trend: [B, L, d_model]
        We'll process the 'seasonal' part via auto-correlation
        """
        # 1) auto-correlation
        auto_seasonal = self.autocorr(seasonal)  # [B, L, d_model]

        # 2) feed-forward + skip
        out = self.ff(auto_seasonal)
        out = self.dropout(out)
        out = self.layernorm(seasonal + out)

        # trend is passed through (or you can do advanced manipulations)
        return out, trend

# -----------------------------------------------------------------------------
# 4) AutoformerEncoder: stacked layers + repeated decomposition if desired
# -----------------------------------------------------------------------------
class AutoformerEncoder(nn.Module):
    """
    Repeated decomposition and auto-correlation across multiple layers.
    """
    def __init__(self, d_model=64, layer_num=2, kernel_size=25, top_k=16, dropout=0.1):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size=kernel_size)
        self.layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model=d_model, top_k=top_k, dropout=dropout)
            for _ in range(layer_num)
        ])
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, L, d_model]
        1) Decompose into trend + seasonal
        2) Repeatedly pass them through AutoformerEncoderLayer
        3) Return final sum (trend + seasonal) or just seasonal
        """
        trend, seasonal = self.decomp(x)  # [B, L, d_model]

        # In official code, each layer might re-decompose the residual. We'll keep it simpler.
        for layer in self.layers:
            seasonal, trend = layer(seasonal, trend)

        out = seasonal + trend
        out = self.layernorm(out)
        return out

# -----------------------------------------------------------------------------
# 5) Full "AutoformerNoFuture"
# -----------------------------------------------------------------------------
class Autoformer(BaseModel, MonteCarloMixin):
    """
    A simplified Autoformer for single-step forecasting:
      - Input linear embedding
      - AutoformerEncoder (with auto-correlation blocks)
      - Final projection to 1 dimension
      - Return last time step's forecast
    """
    def __init__(
        self,
        input_dim,
        d_model=64,
        layer_num=2,
        top_k=16,
        dropout=0.1,
        kernel_size=25
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        # 1) input embedding
        self.embedding = nn.Linear(input_dim, d_model)
        # 2) Autoformer encoder
        self.encoder = AutoformerEncoder(d_model=d_model,
                                         layer_num=layer_num,
                                         kernel_size=kernel_size,
                                         top_k=top_k,
                                         dropout=dropout)
        # 3) projection
        self.projection = nn.Linear(d_model, 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        """
        x: (B, L, input_dim)
        => out: (B, 1)
        """
        # embed
        x = self.embedding(x)    # [B, L, d_model]
        # encoder
        enc_out = self.encoder(x)  # [B, L, d_model]
        # take last step
        last_token = enc_out[:, -1, :]  # [B, d_model]
        # project
        out = self.projection(last_token)  # [B, 1]
        return out

# -----------------------------------------------------------------------------
# 6) Quick Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    batch_size = 4
    seq_len = 16
    input_dim = 8

    model = Autoformer(
        input_dim=input_dim,
        d_model=32,
        layer_num=2,
        top_k=8,
        dropout=0.1,
        kernel_size=7
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    y = model(x)
    print("Output shape:", y.shape)  # (4, 1)
