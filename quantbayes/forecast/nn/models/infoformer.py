import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from quantbayes.forecast.nn.base import MonteCarloMixin, BaseModel

# --------------------------------------------------------------------
# 1) Attention Masks
#    (Copied or adapted from the official Informer code)
# --------------------------------------------------------------------
class TriangularCausalMask:
    """Mask out subsequent positions (for causal attention)."""
    def __init__(self, B, L, device="cpu"):
        mask_shape = (B, 1, L, L)
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    """Mask used in ProbAttention to handle causality."""
    def __init__(self, B, H, L, index, scores, device="cpu"):
        # for each batch, head, and query index, block attention to future positions
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None], 
                             torch.arange(H)[None, :, None],
                             index, :]
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


# --------------------------------------------------------------------
# 2) Attention Mechanisms: FullAttention & ProbAttention
# --------------------------------------------------------------------
class FullAttention(nn.Module):
    """Classic full self-attention with an optional causal mask."""
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries/keys/values: [B, L, H, D]
        """
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape

        scale = self.scale or 1./math.sqrt(D)
        # [B, H, L, D] x [B, H, D, S] => [B, H, L, S]
        scores = torch.einsum("blhd,bshd->bhls", queries, keys)
        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask.mask, -np.inf)
        attn = F.softmax(scores * scale, dim=-1)
        attn = self.dropout(attn)
        # [B, H, L, S] x [B, H, S, D] => [B, H, L, D]
        out = torch.einsum("bhls,bshd->blhd", attn, values)

        if self.output_attention:
            return out.contiguous(), attn
        else:
            return out.contiguous(), None


class ProbAttention(nn.Module):
    """
    ProbSparse self-attention from the Informer paper.
    For brevity, we include it directly. If you'd rather use FullAttention,
    just swap it in the AttentionLayer below.
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Q: [B, H, L_Q, D], K: [B, H, L_K, D]
        sample_k = factor * ln(L_k)
        n_top    = factor * ln(L_q)
        """
        B, H, L_K, D = K.shape
        _, _, L_Q, _ = Q.shape

        # 1) sample K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # random sampling
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        # Q * K_sample: [B, H, L_Q, sample_k]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        # 2) find the Top_n queries with highest average attention
        M = Q_K_sample.max(dim=-1)[0] - (Q_K_sample.mean(dim=-1))
        M_top = M.topk(n_top, sorted=False)[1]
        # 3) Use the reduced Q for final QK
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # simple average (or sum)
            context = V.mean(dim=-2, keepdim=True).expand(-1, -1, L_Q, -1).clone()
        else:
            # for causal mask, we can't just average over all V
            # (must do cumsum). Usually requires L_Q == L_V
            context = V.cumsum(dim=-2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V)

        if self.output_attention:
            # Build a full attn map for analysis
            full_attn = torch.zeros((B, H, context_in.shape[2], L_V), device=V.device)
            full_attn[torch.arange(B)[:, None, None],
                      torch.arange(H)[None, :, None],
                      index, :] = attn
            return context_in, full_attn
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask):
        """
        queries, keys, values: [B, L, H, D] after projection
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # transpose to [B, H, L, D] for convenience
        queries = queries.transpose(2, 1)  # [B, H, L_Q, D]
        keys    = keys.transpose(2, 1)     # [B, H, L_K, D]
        values  = values.transpose(2, 1)   # [B, H, L_K, D]

        U_part = self.factor * int(np.ceil(np.log(L_K)))  # c*ln(L_K)
        U_part = U_part if U_part < L_K else L_K
        u = self.factor * int(np.ceil(np.log(L_Q)))       # c*ln(L_Q)
        u = u if u < L_Q else L_Q

        # 1) get indices of top queries
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        # 2) scale
        scale = self.scale or 1./math.sqrt(D)
        scores_top = scores_top * scale
        # 3) get initial context
        context = self._get_initial_context(values, L_Q)
        # 4) update context
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        # revert shape to [B, L_Q, H, D]
        context = context.transpose(2, 1).contiguous()  # [B, L_Q, H, D]
        return context, attn


# --------------------------------------------------------------------
# 3) Attention Layer
# --------------------------------------------------------------------
class AttentionLayer(nn.Module):
    """
    Projects queries/keys/values, runs them through an attention mechanism,
    and projects output back to d_model.
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection   = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection   = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries, keys, values: [B, L, d_model]
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Linear projections
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, S, H, -1)
        values  = self.value_projection(values).view(B, S, H, -1)

        # Forward through attention
        out, attn = self.inner_attention(queries, keys, values, attn_mask=attn_mask)
        # out shape = [B, L, H, Dv]
        if self.mix:
            # optional: mix heads back on the time dimension
            out = out.transpose(2, 1).contiguous()  # [B, L, H, Dv] -> [B, H, L, Dv] -> ...
            out = out.view(B, L, -1)
        else:
            out = out.view(B, L, -1)

        # Final projection
        out = self.out_projection(out)  # [B, L, d_model]
        return out, attn


# --------------------------------------------------------------------
# 4) Encoder Components
# --------------------------------------------------------------------
class ConvLayer(nn.Module):
    """
    Distilling Convolution Layer used in Informer to halve the sequence length.
    """
    def __init__(self, c_in, dropout=0.05):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=1,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.downConv(x)   # [B, D, L]
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)       # [B, D, L/2]
        x = x.transpose(1, 2)  # [B, L/2, D]
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    """Single encoder layer = self-attention + FFN + residual + layernorm."""
    def __init__(self, self_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = self_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x: [B, L, d_model]
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        
        y2 = y.transpose(-1, 1)
        y2 = self.conv1(y2)
        y2 = self.activation(y2)
        y2 = self.conv2(y2)
        y2 = y2.transpose(1, -1)
        y = self.dropout(y2)
        y = self.norm2(x + y)

        return y, attn


class Encoder(nn.Module):
    """
    The full encoder = N stacked EncoderLayers
    Optionally with ConvLayer-based "distilling" in between layers (reducing seq_len).
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)      # list of EncoderLayer
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x: [B, L, d_model]
        attns = []
        if self.conv_layers is not None:
            # "distilling" mode
            for attn_layer, conv_layer in zip(self.attn_layers[:-1], self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
                x = conv_layer(x)  # halve sequence length
            # last layer
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            # no distilling
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


# --------------------------------------------------------------------
# 5) Embeddings
#    For simplicity, we do:
#    - ValueEmbedding => either a conv1d or a simple linear
#    - PositionalEmbedding => standard sinusoidal
#    Combined => data_embedding
# --------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    """Classic Transformer sinusoidal positional embedding."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(-(math.log(10000.0) / d_model) * torch.arange(0, d_model, 2))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, L, d_model] -> we only need L
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """1D-conv-based embedding (from original Informer)."""
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in,
                                   out_channels=d_model,
                                   kernel_size=3,
                                   padding=1,
                                   padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        x: [B, L, c_in]
        We'll permute to [B, c_in, L], apply conv, then permute back.
        """
        x = x.transpose(1, 2)                      # [B, c_in, L]
        x = self.tokenConv(x)                      # [B, d_model, L]
        x = x.transpose(1, 2).contiguous()         # [B, L, d_model]
        return x


class DataEmbedding(nn.Module):
    """Combine TokenEmbedding + PositionalEmbedding (and optional dropout)."""
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        x: [B, L, c_in]
        Returns: [B, L, d_model]
        """
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


# --------------------------------------------------------------------
# 6) Final: The Simplified Informer Model (Encoder-Only, 1-step output)
# --------------------------------------------------------------------
class Informer(BaseModel, MonteCarloMixin):
    """
    A simplified Informer-like model for *one-step* time series forecasting
    from a single input sequence (no future covariates).
    
    Input shape:  (batch_size, seq_len, input_dim)
    Output shape: (batch_size, 1)  -> the prediction for the *last time step*
    """
    def __init__(
        self,
        input_dim,    # c_in
        d_model=64,
        n_heads=4,
        e_layers=3,   # number of encoder layers
        d_ff=256,
        dropout=0.1,
        attn='prob',  # 'prob' or 'full'
        distil=True,
        activation='gelu',
    ):
        super(Informer, self).__init__()
        self.seq_len = None   # not strictly needed, you can store if helpful
        self.d_model = d_model
        self.n_heads = n_heads

        # Embedding
        self.enc_embedding = DataEmbedding(input_dim, d_model, dropout)

        # Choose attention
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Build the encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                self_attention = AttentionLayer(
                    Attn(False, factor=5, attention_dropout=dropout, output_attention=False),
                    d_model, n_heads, mix=False
                ),
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            )
            for _ in range(e_layers)
        ])

        # Optional conv-layers for distillation
        self.conv_layers = None
        if distil and e_layers > 1:
            # We create e_layers-1 ConvLayers, each halves the seq len
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, dropout=dropout)
                for _ in range(e_layers-1)
            ])

        self.encoder_norm = nn.LayerNorm(d_model)
        self.encoder = Encoder(
            attn_layers=self.encoder_layers,
            conv_layers=self.conv_layers,
            norm_layer=self.encoder_norm
        )

        # Final projection from the encoder output to 1 dimension
        self.projection = nn.Linear(d_model, 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        Return: (batch, 1)
        """
        # 1) Embedding
        enc_out = self.enc_embedding(x)  # [B, L, d_model]

        # 2) Encoder forward
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # [B, L', d_model]
        #    If distilling is on, L' might be smaller than L (e.g. L/2^N).
        #    Typically, if we keep e_layers=3, we might end with L' = L/(2^(e_layers-1)).

        # 3) Take the final time step from enc_out
        #    If distil has reduced the length, the "last time step" is effectively enc_out[:, -1, :].
        last_hidden = enc_out[:, -1, :]  # shape: [B, d_model]

        # 4) Project to a single value
        out = self.projection(last_hidden)  # [B, 1]
        return out

if __name__ == "__main__":
    import torch

    batch_size = 16
    seq_len = 32
    input_dim = 8

    model = Informer(
        input_dim=input_dim,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
        attn='prob',   # or 'full'
        distil=True,
        activation='gelu'
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    y = model(x)
    print("Output shape:", y.shape)  # [16, 1]
