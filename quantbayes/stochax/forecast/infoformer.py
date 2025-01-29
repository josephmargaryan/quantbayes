import math
from typing import Optional, Tuple, Any

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


# --------------------------------------------------------------------
# 1) Attention Masks
# --------------------------------------------------------------------
class TriangularCausalMask:
    """Mask out subsequent positions (for causal attention)."""

    def __init__(self, B: int, L: int):
        """Initializes the mask.

        Args:
            B (int): Batch size.
            L (int): Sequence length.
        """
        self._mask = jnp.triu(jnp.ones((B, 1, L, L), dtype=bool), k=1)

    @property
    def mask(self) -> jnp.ndarray:
        """Returns the triangular causal mask."""
        return self._mask


class ProbMask:
    """Mask used in ProbAttention to handle causality."""

    def __init__(self, B: int, H: int, L: int, index: jnp.ndarray, scores: jnp.ndarray):
        """
        Args:
            B (int): Batch size.
            H (int): Number of attention heads.
            L (int): Sequence length.
            index (jnp.ndarray): Indices for masking.
            scores (jnp.ndarray): Attention scores.
        """
        # Create upper triangular mask
        _mask = jnp.triu(jnp.ones((L, scores.shape[-1]), dtype=bool), k=1)
        # Expand mask to match dimensions
        _mask_ex = jnp.broadcast_to(_mask[None, None, :, :], (B, H, L, scores.shape[-1]))
        # Gather relevant mask slices based on the index
        indicator = jnp.take_along_axis(_mask_ex, index[:, :, :, None], axis=2)
        self._mask = jnp.reshape(indicator, scores.shape)

    @property
    def mask(self) -> jnp.ndarray:
        """Returns the probabilistic mask."""
        return self._mask


# --------------------------------------------------------------------
# 2) Attention Mechanisms: FullAttention & ProbAttention
# --------------------------------------------------------------------
class FullAttention(nn.Module):
    """Classic full self-attention with an optional causal mask."""
    mask_flag: bool = True
    scale: Optional[float] = None
    attention_dropout: float = 0.1
    output_attention: bool = False

    @nn.compact
    def __call__(
        self,
        queries: jnp.ndarray,  # [B, H, L_Q, D]
        keys: jnp.ndarray,     # [B, H, L_K, D]
        values: jnp.ndarray,   # [B, H, L_K, D]
        attn_mask: Optional[jnp.ndarray] = None,
        *,
        train: bool = True,
        rngs: Optional[dict] = None
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Args:
            queries: [B, H, L_Q, D].
            keys: [B, H, L_K, D].
            values: [B, H, L_K, D].
            attn_mask: Optional attention mask [B, H, L_Q, L_K].
            train: Boolean flag for training mode (affects dropout).
            rngs: Dictionary of PRNG keys for dropout.

        Returns:
            Tuple of:
                - Output tensor [B, H, L_Q, D].
                - Attention weights [B, H, L_Q, L_K] if output_attention=True.
        """
        B, H, L_Q, D = queries.shape
        _, _, L_K, _ = keys.shape

        scale = self.scale or (1.0 / jnp.sqrt(D))
        # Compute scaled dot-product attention scores
        scores = jnp.einsum("bhqd,bhkd->bhqk", queries, keys) * scale  # [B, H, L_Q, L_K]

        if self.mask_flag and attn_mask is not None:
            # Ensure attn_mask has the same shape as scores
            scores = jnp.where(attn_mask, -jnp.inf, scores)

        attn = nn.softmax(scores, axis=-1)  # [B, H, L_Q, L_K]
        attn = nn.Dropout(rate=self.attention_dropout)(
            attn,
            deterministic=not train,
            rng=rngs['dropout'] if rngs and 'dropout' in rngs else None
        )

        out = jnp.einsum('bhqk,bhkd->bhqd', attn, values)  # [B, H, L_Q, D]

        if self.output_attention:
            return out, attn
        else:
            return out, None


class ProbAttention(nn.Module):
    """
    ProbSparse self-attention from the Informer paper.
    """
    mask_flag: bool = True
    factor: int = 5
    scale: Optional[float] = None
    attention_dropout: float = 0.1
    output_attention: bool = False

    @nn.compact
    def __call__(
        self,
        queries: jnp.ndarray,  # [B, H, L_Q, D]
        keys: jnp.ndarray,     # [B, H, L_K, D]
        values: jnp.ndarray,   # [B, H, L_K, D]
        attn_mask: Optional[jnp.ndarray] = None,
        *,
        train: bool = True,
        rngs: Optional[dict] = None
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Args:
            queries: [B, H, L_Q, D].
            keys: [B, H, L_K, D].
            values: [B, H, L_K, D].
            attn_mask: Optional attention mask [B, H, L_Q, L_K].

        Returns:
            Tuple of context and optional attention weights.
        """
        B, H, L_Q, D = queries.shape
        _, _, L_K, _ = keys.shape

        U_part = self.factor * int(math.ceil(math.log(L_K)))  # c*ln(L_K)
        U_part = min(U_part, L_K)
        u = self.factor * int(math.ceil(math.log(L_Q)))      # c*ln(L_Q)
        u = min(u, L_Q)

        # 1) Get indices of top queries
        scores_top, index = self._prob_QK(queries, keys, U_part, u)

        scale = self.scale or (1.0 / math.sqrt(D))
        scores_top = scores_top * scale

        # 2) Get initial context
        context = self._get_initial_context(values, L_Q)

        # 3) Update context
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask, B, H, rngs, train)

        # Revert shape to [B, L_Q, H, D]
        context = jnp.transpose(context, (0, 2, 1, 3))  # [B, L_Q, H, D]
        return context, attn

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, D = K.shape
        _, _, L_Q, _ = Q.shape

        # 1) Sample K
        rng = self.make_rng('dropout')  # Reuse rng for deterministic behavior
        index_sample = jax.random.randint(rng, (L_Q, sample_k), 0, L_K)
        K_sample = jnp.take(K, index_sample, axis=2)  # [B, H, L_Q, sample_k, D]

        # Compute Q * K_sample
        Q_K_sample = jnp.einsum("bhqd,bhqkd->bhqk", Q, K_sample)  # [B, H, L_Q, sample_k]

        # 2) Find top n queries with highest average attention
        M = jnp.max(Q_K_sample, axis=-1) - jnp.mean(Q_K_sample, axis=-1)  # [B, H, L_Q]
        M_top = jnp.argsort(M, axis=-1)[:, :, -n_top:]  # [B, H, n_top]

        # 3) Use the reduced Q for final QK
        Q_reduce = jnp.take_along_axis(Q, M_top[:, :, :, None], axis=2)  # [B, H, n_top, D]
        Q_K = jnp.einsum("bhqd,bhkd->bhqk", Q_reduce, K)  # [B, H, n_top, L_K]
        return Q_K, M_top  # [B, H, n_top, L_K], [B, H, n_top]

    def _get_initial_context(self, V, L_Q):
        if not self.mask_flag:
            context = jnp.mean(V, axis=2, keepdims=True).repeat(L_Q, axis=2)  # [B, H, L_Q, D]
        else:
            context = jnp.cumsum(V, axis=2)  # [B, H, L_Q, D]
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask, B, H, rngs, train):
        if self.mask_flag and attn_mask is not None:
            # Create probabilistic mask
            mask = ProbMask(B, H, L_Q, index, scores).mask
            scores = jnp.where(mask, -jnp.inf, scores)

        attn = nn.softmax(scores, axis=-1)  # [B, H, L_Q, L_K]
        attn = nn.Dropout(rate=self.attention_dropout)(
            attn,
            deterministic=not train,
            rng=rngs['dropout'] if rngs and 'dropout' in rngs else None
        )
        updated_context = context_in.at[:, :, index, :].set(jnp.einsum("bhqk,bhkd->bhqd", attn, V))  # [B, H, L_Q, D]

        if self.output_attention:
            # Build a full attn map for analysis
            full_attn = jnp.zeros((B, H, context_in.shape[2], V.shape[2]), dtype=attn.dtype)
            full_attn = full_attn.at[:, :, index, :].set(attn)
            return updated_context, full_attn
        else:
            return updated_context, None




# --------------------------------------------------------------------
# 3) Attention Layer
# --------------------------------------------------------------------
class AttentionLayer(nn.Module):
    """
    Projects queries/keys/values, runs them through an attention mechanism,
    and projects output back to d_model.
    """
    attention: Any  # The attention module, e.g., FullAttention or ProbAttention
    d_model: int
    n_heads: int
    d_keys: Optional[int] = None
    d_values: Optional[int] = None
    mix: bool = False

    @nn.compact
    def __call__(
        self,
        queries: jnp.ndarray,  # [B, L, d_model]
        keys: jnp.ndarray,     # [B, S, d_model]
        values: jnp.ndarray,   # [B, S, d_model]
        attn_mask: Optional[jnp.ndarray] = None,
        *,
        train: bool = True,
        rngs: Optional[dict] = None
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Args:
            queries: [B, L, d_model]
            keys: [B, S, d_model]
            values: [B, S, d_model]
            attn_mask: Optional attention mask [B, H, L, S]
            train: Boolean flag for training mode (affects dropout)
            rngs: Dictionary of PRNG keys for dropout

        Returns:
            Tuple of:
                - Output tensor of shape [B, L, d_model]
                - Attention weights (if output_attention=True in the attention module)
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Set default dimensions if not provided
        d_keys = self.d_keys or (self.d_model // self.n_heads)
        d_values = self.d_values or (self.d_model // self.n_heads)

        # Define and instantiate projection layers as submodules
        query_proj = nn.Dense(features=H * d_keys, use_bias=False, name='query_projection')
        key_proj = nn.Dense(features=H * d_keys, use_bias=False, name='key_projection')
        value_proj = nn.Dense(features=H * d_values, use_bias=False, name='value_projection')
        out_proj = nn.Dense(features=self.d_model, use_bias=False, name='out_projection')

        # Apply projections
        queries_proj = query_proj(queries)  # [B, L, H * d_keys]
        keys_proj = key_proj(keys)          # [B, S, H * d_keys]
        values_proj = value_proj(values)    # [B, S, H * d_values]

        # Reshape to [B, H, L, d_keys]
        queries_proj = queries_proj.reshape(B, L, H, d_keys)
        keys_proj = keys_proj.reshape(B, S, H, d_keys)
        values_proj = values_proj.reshape(B, S, H, d_values)

        # Transpose to [B, H, L, d_keys]
        queries_proj = jnp.transpose(queries_proj, (0, 2, 1, 3))  # [B, H, L, d_keys]
        keys_proj = jnp.transpose(keys_proj, (0, 2, 1, 3))        # [B, H, S, d_keys]
        values_proj = jnp.transpose(values_proj, (0, 2, 1, 3))    # [B, H, S, d_values]

        # Forward through attention
        attention_output, attn = self.attention(
            queries_proj,
            keys_proj,
            values_proj,
            attn_mask=attn_mask,
            train=train,
            rngs=rngs
        )
        # attention_output shape: [B, H, L, D]

        if self.mix:
            # Optional: mix heads back on the time dimension
            # Transpose to [B, L, H, D]
            attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))  # [B, L, H, D]
            # Reshape to [B, L, H * D]
            attention_output = attention_output.reshape(B, L, H * d_values)
        else:
            # Concatenate heads: [B, L, H * D]
            # First, transpose to [B, L, H, D]
            attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))  # [B, H, L, D] -> [B, L, H, D]
            # Then, reshape to [B, L, H * D]
            attention_output = attention_output.reshape(B, L, H * d_values)

        # Final projection back to d_model
        out = out_proj(attention_output)  # [B, L, d_model]

        return out, attn


# --------------------------------------------------------------------
# 4) ConvLayer
# --------------------------------------------------------------------
class ConvLayer(nn.Module):
    """
    Distilling Convolution Layer used in Informer to halve the sequence length.
    """
    c_in: int
    dropout: float = 0.05

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,  # [B, L, D]
        *,
        train: bool = True,
        rngs: Optional[dict] = None
    ) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, L, D], where
               B - Batch size,
               L - Sequence length,
               D - Feature dimension.
            train: Boolean flag indicating training mode (affects BatchNorm and Dropout).
            rngs: Dictionary of PRNG keys for dropout.

        Returns:
            Output tensor of shape [B, L/2, D].
        """
        # 1. Circular Padding
        x_padded = jnp.pad(x, pad_width=((0, 0), (1, 1), (0, 0)), mode='wrap')  # [B, L+2, D]

        # 2. Convolution
        x_conv = nn.Conv(
            features=self.c_in,
            kernel_size=(3,),
            strides=(1,),
            padding='VALID',  # No additional padding, since we manually padded
            use_bias=True,
            name='downConv'
        )(x_padded)  # [B, L, D]

        # 3. Batch Normalization
        x_norm = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            axis=-1,  # Normalize over the feature axis
            name='norm'
        )(x_conv)  # [B, L, D]

        # 4. Activation
        x_act = nn.elu(x_norm)  # [B, L, D]

        # 5. Max Pooling to halve sequence length
        B, L, D = x_act.shape
        L_new = (L // 2) * 2  # Ensure even sequence length
        x_trimmed = x_act[:, :L_new, :]  # [B, L_new, D]
        x_pooled = x_trimmed.reshape(B, L_new // 2, 2, D).max(axis=2)  # [B, L_new//2, D]

        # 6. Dropout
        x_drop = nn.Dropout(
            rate=self.dropout
        )(x_pooled, deterministic=not train, rng=rngs['dropout'] if rngs and 'dropout' in rngs else None)  # [B, L_new//2, D]

        return x_drop


# --------------------------------------------------------------------
# 5) Embeddings
# --------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    def setup(self):
        pe = np.zeros((self.max_len, self.d_model), dtype=np.float32)
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = jax.device_put(pe[None, :, :])  # [1, max_len, d_model]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, L, d_model].

        Returns:
            Tensor with positional encodings added, of shape [B, L, d_model].
        """
        # Ensure the positional encodings match the sequence length
        seq_len = x.shape[1]
        pos_encoding = self.pe[:, :seq_len, :]  # [1, L, d_model]

        # Debugging: Check shapes
        assert pos_encoding.shape == (1, seq_len, self.d_model), f"Positional encoding shape mismatch: {pos_encoding.shape}"
        assert x.shape[2] == self.d_model, f"Feature dimension mismatch: {x.shape[2]} vs {self.d_model}"

        return x + pos_encoding


class TokenEmbedding(nn.Module):
    """1D-conv-based embedding (from original Informer)."""
    c_in: int  # Number of input features (input_dim)
    d_model: int  # Desired output embedding dimension

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, L, c_in]

        Returns:
            Tensor of shape [B, L, d_model]
        """
        # Apply Conv1D directly without transposing
        x = nn.Conv(
            features=self.d_model,  # Number of output channels (embedding size)
            kernel_size=(3,),
            strides=(1,),
            padding='SAME',
            use_bias=True,
            name='tokenConv'
        )(x)  # [B, L, d_model]

        # Debugging: Check output feature dimensions
        assert x.shape[-1] == self.d_model, (
            f"Conv1D output feature dimension mismatch: {x.shape[-1]} vs {self.d_model}"
        )

        return x

# Corrected DataEmbedding
class DataEmbedding(nn.Module):
    c_in: int
    d_model: int
    dropout: float = 0.1

    def setup(self):
        self.value_embedding = TokenEmbedding(c_in=self.c_in, d_model=self.d_model)
        self.position_embedding = PositionalEncoding(d_model=self.d_model, max_len=5000)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, x: jnp.ndarray, *, train: bool = True, rngs: Optional[dict] = None) -> jnp.ndarray:
        """
        Args:
            x: [B, L, c_in]

        Returns:
            [B, L, d_model]
        """
        # Generate embeddings
        value_emb = self.value_embedding(x)  # [B, L, d_model]
        assert value_emb.shape[2] == self.d_model, (
            f"Value embedding feature dimension mismatch: {value_emb.shape[2]} vs {self.d_model}"
        )

        # Add positional encodings
        x = self.position_embedding(value_emb)

        # Apply dropout
        x = self.dropout_layer(x, deterministic=not train, rng=rngs['dropout'] if rngs and 'dropout' in rngs else None)
        return x


# --------------------------------------------------------------------
# 6) Encoder Components
# --------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """Single encoder layer = self-attention + FFN + residual + layernorm."""
    attention: Any  # The attention module, e.g., AttentionLayer
    d_model: int
    d_ff: Optional[int] = None
    dropout: float = 0.1
    activation: str = "relu"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,  # [B, L, d_model]
        attn_mask: Optional[jnp.ndarray] = None,
        *,
        train: bool = True,
        rngs: Optional[dict] = None
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Forward pass for the EncoderLayer.

        Args:
            x: Input tensor of shape [B, L, d_model]
            attn_mask: Optional attention mask [B, H, L, S]
            train: Boolean flag for training mode (affects dropout)
            rngs: Dictionary of PRNG keys for dropout

        Returns:
            Tuple of:
                - Output tensor of shape [B, L, d_model]
                - Attention weights (if output_attention=True in the attention module)
        """
        d_ff = self.d_ff or 4 * self.d_model

        # Self-attention
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, train=train, rngs=rngs)

        # Dropout and residual connection
        new_x = nn.Dropout(rate=self.dropout)(
            new_x,
            deterministic=not train,
            rng=rngs['dropout'] if rngs and 'dropout' in rngs else None
        )
        # Debugging: Check shapes before addition
        assert new_x.shape == x.shape, f"Shape mismatch before residual addition: {new_x.shape} vs {x.shape}"
        x = x + new_x  # [B, L, d_model]

        # LayerNorm1
        y = nn.LayerNorm(name='norm1')(x)  # [B, L, d_model]

        # Feed-Forward Network (FFN)
        y_ffn = nn.Dense(features=d_ff, use_bias=True, name='conv1')(y)  # [B, L, d_ff]

        # Activation
        if self.activation == "relu":
            y_ffn = nn.relu(y_ffn)
        else:
            y_ffn = nn.gelu(y_ffn)

        # Second Dense layer to project back to d_model
        y_ffn = nn.Dense(features=self.d_model, use_bias=True, name='conv2')(y_ffn)  # [B, L, d_model]

        # Dropout
        y_ffn = nn.Dropout(
            rate=self.dropout
        )(y_ffn, deterministic=not train, rng=rngs['dropout'] if rngs and 'dropout' in rngs else None)  # [B, L, d_model]

        # Residual connection and LayerNorm2
        # Debugging: Check shapes before addition
        assert y_ffn.shape == x.shape, f"Shape mismatch before second residual addition: {y_ffn.shape} vs {x.shape}"
        y = x + y_ffn  # [B, L, d_model]
        y = nn.LayerNorm(name='norm2')(y)  # [B, L, d_model]

        return y, attn


class Encoder(nn.Module):
    """
    The full encoder = N stacked EncoderLayers
    Optionally with ConvLayer-based "distilling" in between layers (reducing seq_len).
    """
    attn_layers: Any  # List of EncoderLayer
    conv_layers: Optional[Any] = None  # List of ConvLayer
    norm_layer: Optional[Any] = None  # LayerNorm

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,  # [B, L, d_model]
        attn_mask: Optional[jnp.ndarray] = None,
        *,
        train: bool = True,
        rngs: Optional[dict] = None
    ) -> Tuple[jnp.ndarray, list]:
        """
        Forward pass for the Encoder.

        Args:
            x: [B, L, d_model]
            attn_mask: Optional attention mask [B, H, L, S]
            train: Boolean flag for training mode (affects dropout)
            rngs: Dictionary of PRNG keys for dropout

        Returns:
            Tuple of:
                - Output tensor [B, L', d_model]
                - List of attention weights from each layer
        """
        attns = []
        if self.conv_layers is not None:
            # "Distilling" mode
            for attn_layer, conv_layer in zip(self.attn_layers[:-1], self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask, train=train, rngs=rngs)
                attns.append(attn)
                x = conv_layer(x, train=train, rngs=rngs)  # Halve sequence length
                # Debugging: Check shapes after conv
                assert x.ndim == 3, f"ConvLayer output has incorrect number of dimensions: {x.shape}"
                assert x.shape[2] == self.attn_layers[0].d_model, f"ConvLayer feature dimension mismatch: {x.shape[2]} vs {self.attn_layers[0].d_model}"
            # Last layer
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask, train=train, rngs=rngs)
            attns.append(attn)
        else:
            # No distilling
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, train=train, rngs=rngs)
                attns.append(attn)

        if self.norm_layer is not None:
            x = self.norm_layer(x)
            # Debugging: Check shape after normalization
            assert x.ndim == 3, f"Normalization output has incorrect number of dimensions: {x.shape}"


        return x, attns


# --------------------------------------------------------------------
# 6) Informer Model
# --------------------------------------------------------------------
class Informer(nn.Module):
    """
    A simplified Informer-like model for one-step time series forecasting
    from a single input sequence (no future covariates).

    Input shape:  (batch_size, seq_len, input_dim)
    Output shape: (batch_size, 1)  -> the prediction for the last time step
    """
    input_dim: int
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 3  # number of encoder layers
    d_ff: int = 256
    dropout: float = 0.1
    attn: str = 'prob'  # 'prob' or 'full'
    distil: bool = True
    activation: str = "gelu"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,  # [B, L, input_dim]
        *,
        train: bool = True,
        rngs: Optional[dict] = None
    ) -> jnp.ndarray:
        """
        Forward pass of the Informer model.

        Args:
            x: [B, L, input_dim]
            train: Boolean flag for training mode (affects dropout)
            rngs: Dictionary of PRNG keys for dropout

        Returns:
            [B, 1]
        """
        # 1. Embedding
        enc_out = DataEmbedding(c_in=self.input_dim, d_model=self.d_model, dropout=self.dropout)(
            x, train=train, rngs=rngs
        )  # [B, L, d_model]

        # Debugging: Check shape after embedding
        assert enc_out.ndim == 3, f"Embedding output has incorrect number of dimensions: {enc_out.shape}"
        assert enc_out.shape[2] == self.d_model, f"Embedding feature dimension mismatch: {enc_out.shape[2]} vs {self.d_model}"

        # 2. Define Attention mechanism
        if self.attn == 'prob':
            Attn = ProbAttention
        else:
            Attn = FullAttention

        # 3. Build the encoder layers and optional conv layers for distilling
        attn_layers = []
        conv_layers = []
        for i in range(self.e_layers):
            attn_layer = EncoderLayer(
                attention=AttentionLayer(
                    attention=Attn(
                        mask_flag=False,  # Usually False within AttentionLayer
                        factor=5,
                        attention_dropout=self.dropout,
                        output_attention=False
                    ),
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    mix=False
                ),
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout=self.dropout,
                activation=self.activation
            )
            attn_layers.append(attn_layer)
            if self.distil and i < self.e_layers - 1:
                conv_layer = ConvLayer(c_in=self.d_model, dropout=self.dropout)
                conv_layers.append(conv_layer)

        # Instantiate Encoder
        encoder = Encoder(
            attn_layers=attn_layers,
            conv_layers=conv_layers if self.distil and self.e_layers > 1 else None,
            norm_layer=nn.LayerNorm(name='encoder_norm') if self.e_layers > 0 else None
        )

        # 4. Encoder forward
        enc_out, attns = encoder(enc_out, attn_mask=None, train=train, rngs=rngs)  # [B, L', d_model]

        # Debugging: Check shape after encoder
        assert enc_out.ndim == 3, f"Encoder output has incorrect number of dimensions: {enc_out.shape}"
        assert enc_out.shape[2] == self.d_model, f"Encoder feature dimension mismatch: {enc_out.shape[2]} vs {self.d_model}"

        # 5. Take the final time step from enc_out
        last_hidden = enc_out[:, -1, :]  # [B, d_model]

        # Debugging: Check shape of last_hidden
        assert last_hidden.ndim == 2, f"Last hidden state has incorrect number of dimensions: {last_hidden.shape}"
        assert last_hidden.shape[1] == self.d_model, f"Last hidden state feature dimension mismatch: {last_hidden.shape[1]} vs {self.d_model}"

        # 6. Project to a single value
        projection = nn.Dense(features=1, name='projection')(last_hidden)  # [B,1]

        return projection


# --------------------------------------------------------------------
# 7) Test Function
# --------------------------------------------------------------------
def test_informer():
    """
    Test the Informer model with a random input.
    """
    # Define parameters
    batch_size = 16
    seq_len = 32
    input_dim = 8
    d_model = 64
    n_heads = 4
    e_layers = 2
    d_ff = 256
    dropout = 0.1
    attn = 'prob'  # or 'full'
    distil = True
    activation = 'gelu'

    # Initialize random keys
    rng = jax.random.PRNGKey(0)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    # Create random input tensor [B, L, input_dim]
    x = jax.random.normal(init_rng, (batch_size, seq_len, input_dim))
    print(f"Input shape {x.shape}")

    # Instantiate the Informer model
    informer = Informer(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=dropout,
        attn=attn,
        distil=distil,
        activation=activation
    )

    # Initialize parameters
    variables = informer.init(
        {'params': init_rng, 'dropout': dropout_rng},
        x,
        train=True
    )

    # Apply the Informer model with mutable batch_stats
    y, updated_state = informer.apply(
        variables,
        x,
        train=True,
        rngs={'dropout': dropout_rng},
        mutable=['batch_stats']  # Allow updates to batch_stats
    )

    # Print output shape
    print("Output shape:", y.shape)  # Expected: [16, 1]
    # print("Updated state:", updated_state)  # Inspect updated batch_stats



# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
if __name__ == "__main__":
    test_informer()
