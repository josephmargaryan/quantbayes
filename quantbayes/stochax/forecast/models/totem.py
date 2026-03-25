import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# -------------------------------------------------------------------
# 1. Codebook and Quantization
# -------------------------------------------------------------------
class Codebook(eqx.Module):
    embeddings: jnp.ndarray  # shape (num_embeddings, embedding_dim)

    def __init__(self, num_embeddings: int, embedding_dim: int, *, key):
        self.embeddings = jr.normal(key, (num_embeddings, embedding_dim))

    def __call__(self, indices: jnp.ndarray) -> jnp.ndarray:
        return self.embeddings[indices]

    def quantize(self, z_e: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # z_e is assumed to have shape (C, L) with C == embedding_dim.
        emb = self.embeddings  # shape (E, C)
        # Transpose z_e to (L, C) for pairwise distance computation.
        z_e_t = z_e.T  # shape (L, C)

        def _dist(z):
            return jnp.sum((emb - z) ** 2, axis=-1)

        dist = jax.vmap(_dist)(z_e_t)  # shape (L, E)
        indices = jnp.argmin(dist, axis=-1)  # shape (L,)
        z_q_flat = self.embeddings[indices]  # shape (L, C)
        z_q = z_q_flat.T  # shape (C, L)
        return z_q, indices


# -------------------------------------------------------------------
# 2. VQ-VAE: Encoder / Decoder + Codebook
# -------------------------------------------------------------------
class TOTEMEncoder(eqx.Module):
    conv: eqx.nn.Conv1d

    def __init__(
        self, in_channels: int, latent_channels: int, kernel_size=4, stride=2, *, key
    ):
        self.conv = eqx.nn.Conv1d(
            in_channels,
            latent_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="SAME",
            key=key,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: shape (in_channels, L)
        return self.conv(x)


class TOTEMDecoder(eqx.Module):
    tconv: eqx.nn.ConvTranspose1d

    def __init__(
        self, latent_channels: int, out_channels: int, kernel_size=4, stride=2, *, key
    ):
        self.tconv = eqx.nn.ConvTranspose1d(
            in_channels=latent_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="SAME",
            key=key,
        )

    def __call__(self, z_q: jnp.ndarray) -> jnp.ndarray:
        # z_q: shape (latent_channels, L)
        return self.tconv(z_q)


class TOTEMVQVAE(eqx.Module):
    encoder: TOTEMEncoder
    codebook: Codebook
    decoder: TOTEMDecoder

    def __init__(
        self, in_channels: int, latent_channels: int, num_embeddings: int, *, key
    ):
        k_enc, k_cbook, k_dec = jr.split(key, 3)
        self.encoder = TOTEMEncoder(in_channels, latent_channels, key=k_enc)
        self.codebook = Codebook(num_embeddings, latent_channels, key=k_cbook)
        self.decoder = TOTEMDecoder(latent_channels, in_channels, key=k_dec)

    def __call__(
        self, x: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # x: shape (in_channels, L)
        z_e = self.encoder(x)
        z_q, indices = self.codebook.quantize(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, z_e, z_q, indices


# -------------------------------------------------------------------
# 3. TOTEM Transformer (for discrete tokens)
# -------------------------------------------------------------------
# A minimal TransformerBlock and MLP (similar to previous models).
class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    activation: callable

    def __init__(self, embed_dim: int, hidden_dim: int, dropout_p: float, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.fc1 = eqx.nn.Linear(embed_dim, hidden_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_dim, embed_dim, key=k2)
        self.dropout = eqx.nn.Dropout(dropout_p, inference=False)
        self.activation = jax.nn.relu

    def __call__(self, x, *, key=None):
        y = self.fc1(x)
        y = self.activation(y)
        if key is not None:
            key1, key2 = jr.split(key)
        else:
            key1 = key2 = None
        y = self.dropout(y, key=key1)
        y = self.fc2(y)
        y = self.dropout(y, key=key2)
        return y


class TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    dropout1: eqx.nn.Dropout
    norm2: eqx.nn.LayerNorm
    mlp: MLP
    dropout2: eqx.nn.Dropout

    def __init__(
        self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout_p: float, *, key
    ):
        k1, k2, k3, k4, k5 = jr.split(key, 5)
        self.norm1 = eqx.nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embed_dim,
            key_size=embed_dim,
            value_size=embed_dim,
            output_size=embed_dim,
            dropout_p=dropout_p,
            inference=False,
            use_query_bias=False,
            use_key_bias=False,
            use_value_bias=False,
            use_output_bias=False,
            key=k2,
        )
        self.dropout1 = eqx.nn.Dropout(dropout_p, inference=False)
        self.norm2 = eqx.nn.LayerNorm(embed_dim, eps=1e-6)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_dim, dropout_p, key=k4)
        self.dropout2 = eqx.nn.Dropout(dropout_p, inference=False)

    def __call__(self, x: jnp.ndarray, *, key=None):
        if key is not None:
            key_attn, key_mlp = jr.split(key)
        else:
            key_attn = key_mlp = None
        y = jax.vmap(self.norm1)(x)
        attn_out = self.attn(y, y, y, key=key_attn)
        attn_out = self.dropout1(attn_out, key=key_attn)
        x = x + attn_out
        y = jax.vmap(self.norm2)(x)
        if key_mlp is not None:
            keys_mlp = jr.split(key_mlp, y.shape[0])
        else:
            keys_mlp = [None] * y.shape[0]
        mlp_out = jax.vmap(self.mlp)(y, key=keys_mlp)
        mlp_out = self.dropout2(mlp_out, key=key_attn)
        x = x + mlp_out
        return x


class TOTEMTransformer(eqx.Module):
    layers: list[TransformerBlock]
    final_linear: eqx.nn.Linear

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_p: float,
        *,
        key,
    ):
        ks = jr.split(key, num_layers + 1)
        self.layers = [
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_p, key=ks[i])
            for i in range(num_layers)
        ]
        self.final_linear = eqx.nn.Linear(embed_dim, 1, key=ks[-1])

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # x: shape [seq_len, embed_dim]
        if key is not None:
            block_keys = jr.split(key, len(self.layers))
        else:
            block_keys = [None] * len(self.layers)
        y = x
        for layer, k in zip(self.layers, block_keys):
            y = layer(y, key=k)
        last = y[-1]  # shape (embed_dim,)
        return self.final_linear(last)


# -------------------------------------------------------------------
# 4. TOTEM: End-to-End Model with VQ-VAE and Transformer
# -------------------------------------------------------------------
class TOTEM(eqx.Module):
    vqvae: TOTEMVQVAE
    transformer: TOTEMTransformer

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_embeddings: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_p: float,
        *,
        key,
    ):
        k_vq, k_tx = jr.split(key)
        self.vqvae = TOTEMVQVAE(in_channels, latent_channels, num_embeddings, key=k_vq)
        self.transformer = TOTEMTransformer(
            num_layers, latent_channels, num_heads, mlp_ratio, dropout_p, key=k_tx
        )

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Args:
          x: Raw input of shape [seq_len, in_channels] for one sample.
        Returns:
          A scalar prediction (shape (1,)).
        """
        if x.ndim == 1:
            x = x[:, None]
        # Transpose for the CNN encoder: [in_channels, seq_len]
        x_t = jnp.transpose(x, (1, 0))
        _, _, z_q, _ = self.vqvae(x_t)
        # z_q: shape [latent_channels, L_enc]. Transpose to [L_enc, latent_channels]
        z_q_t = jnp.transpose(z_q, (1, 0))
        return self.transformer(z_q_t, key=key)


# -------------------------------------------------------------------
# 5. Batch Wrapper: TOTEMForecast
# -------------------------------------------------------------------
class TOTEMForecast(eqx.Module):
    model: TOTEM

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_embeddings: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_p: float,
        *,
        key,
    ):
        self.model = TOTEM(
            in_channels,
            latent_channels,
            num_embeddings,
            num_layers,
            num_heads,
            mlp_ratio,
            dropout_p,
            key=key,
        )

    def __call__(self, x: jnp.ndarray, key, state) -> tuple[jnp.ndarray, eqx.nn.State]:
        """
        Args:
          x: Input tensor of shape [N, seq_len, in_channels]
        Returns:
          (predictions, state) with predictions shape [N, 1]
        """
        # Here, we assume the training framework vmaps over samples;
        # so x is a single sample.
        return self.model(x, key=key), state


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    import jax.random as jr

    from quantbayes.fake_data import create_synthetic_time_series
    from quantbayes.stochax.forecast import ForecastingModel

    # Create synthetic data.
    X_train, X_val, y_train, y_val = create_synthetic_time_series()
    # Reshape raw input to [N, seq_len, in_channels] with in_channels == 1.
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_val = y_val.reshape(y_val.shape[0], -1)
    print(f"X train shape: {X_train.shape}")
    print(f"y train shape: {y_train.shape}")

    key = jr.PRNGKey(0)
    model, state = eqx.nn.make_with_state(TOTEMForecast)(
        in_channels=1,
        latent_channels=24,
        num_embeddings=52,
        num_layers=2,  #
        num_heads=2,
        mlp_ratio=4,
        dropout_p=0.1,
        key=key,
    )
    trainer = ForecastingModel(lr=1e-3)
    model, state = trainer.fit(
        model,
        state,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=500,
        patience=100,
        key=jr.PRNGKey(42),
    )
    preds = trainer.predict(model, state, X_val, key=jr.PRNGKey(123))
    print(f"preds shape {preds.shape}")
    trainer.visualize(y_val, preds, title="Forecast vs. Ground Truth")
