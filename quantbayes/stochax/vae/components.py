# components.py
from __future__ import annotations
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from quantbayes.stochax.vae.base import BaseDecoder, BaseEncoder, BaseVAE

__all__ = [
    "MLP_VAE",
    "ConvVAE",
    "MultiHeadAttentionVAE",
    "ViT_VAE",
    # Enc/Dec exported if you want to compose custom VAEs:
    "MLPEncoder",
    "MLPDecoder",
    "CNNEncoder",
    "CNNDecoder",
    "AttentionEncoder",
    "AttentionDecoder",
    "ViTEncoder",
    "ViTDecoder",
]

# ===========
#  MLP blocks
# ===========


class MLPEncoder(eqx.Module, BaseEncoder):
    net: eqx.nn.MLP
    latent_dim: int

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, *, key):
        self.latent_dim = latent_dim
        self.net = eqx.nn.MLP(
            in_size=input_dim,
            out_size=2 * latent_dim,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.gelu,
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, x: jnp.ndarray, *, rng=None, train: bool = False):
        out = jax.vmap(self.net)(x)
        mu, logvar = jnp.split(out, 2, axis=-1)
        return mu, logvar


class MLPDecoder(eqx.Module, BaseDecoder):
    net: eqx.nn.MLP

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, *, key):
        self.net = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=output_dim,  # logits for Bernoulli, mean for Gaussian
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.gelu,
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, z: jnp.ndarray, *, rng=None, train: bool = False):
        return jax.vmap(self.net)(z)


# ===========
#  CNN blocks (NCHW)
# ===========


class CNNEncoder(eqx.Module, BaseEncoder):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    mlp: eqx.nn.MLP
    latent_dim: int
    image_size: int

    def __init__(
        self,
        input_channels: int,
        image_size: int,
        hidden_channels: int,
        latent_dim: int,
        *,
        key,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=3,
            stride=2,
            padding="SAME",
            key=k1,
        )
        self.conv2 = eqx.nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=2,
            padding="SAME",
            key=k2,
        )
        flat_dim = (image_size // 4) * (image_size // 4) * hidden_channels
        self.mlp = eqx.nn.MLP(
            in_size=flat_dim,
            out_size=2 * latent_dim,
            width_size=hidden_channels,
            depth=1,
            activation=jax.nn.gelu,
            final_activation=lambda x: x,
            key=k3,
        )
        self.latent_dim = latent_dim
        self.image_size = image_size

    def __call__(self, x: jnp.ndarray, *, rng=None, train: bool = False):
        x = jax.vmap(self.conv1)(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(self.conv2)(x)
        x = jax.nn.gelu(x)
        x = x.reshape(x.shape[0], -1)
        out = jax.vmap(self.mlp)(x)
        mu, logvar = jnp.split(out, 2, axis=-1)
        return mu, logvar


class CNNDecoder(eqx.Module, BaseDecoder):
    mlp: eqx.nn.MLP
    deconv1: eqx.nn.ConvTranspose2d
    deconv2: eqx.nn.ConvTranspose2d
    image_size: int
    hidden_channels: int
    out_channels: int

    def __init__(
        self,
        latent_dim: int,
        hidden_channels: int,
        output_channels: int,
        image_size: int,
        *,
        key,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        flat_dim = (image_size // 4) * (image_size // 4) * hidden_channels
        self.mlp = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=flat_dim,
            width_size=hidden_channels,
            depth=1,
            activation=jax.nn.gelu,
            final_activation=lambda x: x,
            key=k1,
        )
        self.deconv1 = eqx.nn.ConvTranspose2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=2,
            padding="SAME",
            key=k2,
        )
        self.deconv2 = eqx.nn.ConvTranspose2d(
            hidden_channels,
            output_channels,
            kernel_size=3,
            stride=2,
            padding="SAME",
            key=k3,
        )
        self.image_size = image_size
        self.hidden_channels = hidden_channels
        self.out_channels = output_channels

    def __call__(self, z: jnp.ndarray, *, rng=None, train: bool = False):
        x = jax.vmap(self.mlp)(z)
        b = x.shape[0]
        H = W = self.image_size // 4
        hc = self.hidden_channels
        x = x.reshape(b, hc, H, W)
        x = jax.vmap(self.deconv1)(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(self.deconv2)(x)
        return x  # logits (Bernoulli) or mean (Gaussian)


# =================
#  Attention blocks
# =================


class AttentionEncoder(eqx.Module, BaseEncoder):
    attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.Linear
    latent_dim: int
    input_dim: int
    num_heads: int
    seq_length: int
    pos_embedding: jnp.ndarray  # (seq_len, input_dim)

    def __init__(
        self, input_dim: int, latent_dim: int, num_heads: int, seq_length: int, *, key
    ):
        k0, k1, k2 = jax.random.split(key, 3)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=input_dim, key=k0
        )
        self.mlp = eqx.nn.Linear(
            in_features=input_dim, out_features=2 * latent_dim, key=k1
        )
        self.pos_embedding = jax.random.normal(k2, (seq_length, input_dim))
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.seq_length = seq_length

    def __call__(self, x: jnp.ndarray, *, rng=None, train: bool = False):
        # x: (B, T, D)
        pe = self.pos_embedding[None, :, :]

        def attn_seq(seq):
            return self.attn(seq, seq, seq)

        x_pe = x + pe
        out = jax.vmap(attn_seq)(x_pe)
        pooled = jnp.mean(out, axis=1)
        o2 = jax.vmap(self.mlp)(pooled)
        mu, logvar = jnp.split(o2, 2, axis=-1)
        return mu, logvar


class AttentionDecoder(eqx.Module, BaseDecoder):
    attn: eqx.nn.MultiheadAttention
    proj_in: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    seq_length: int
    hidden_dim: int
    output_dim: int
    num_heads: int
    pos_embedding: jnp.ndarray  # (seq_len, hidden_dim)

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_length: int,
        num_heads: int,
        *,
        key,
    ):
        k0, k1, k2, k3 = jax.random.split(key, 4)
        self.proj_in = eqx.nn.Linear(latent_dim, hidden_dim, key=k0)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=hidden_dim, key=k1
        )
        self.proj_out = eqx.nn.Linear(hidden_dim, output_dim, key=k2)
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.pos_embedding = jax.random.normal(k3, (seq_length, hidden_dim))

    def __call__(self, z: jnp.ndarray, *, rng=None, train: bool = False):
        # z: (B, Z)
        h = jax.vmap(self.proj_in)(z)  # (B, H)
        seq = (
            jnp.repeat(h[:, None, :], self.seq_length, axis=1)
            + self.pos_embedding[None, :, :]
        )

        def attn_seq(tokens):
            return self.attn(tokens, tokens, tokens)

        attn_out = jax.vmap(attn_seq)(seq)  # (B, T, H)
        y = jax.vmap(lambda t: jax.vmap(self.proj_out)(t))(attn_out)  # (B, T, D_out)
        return y  # logits or mean per step


# ============
#  ViT blocks (NCHW images)
# ============


class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Linear
    patch_size: int

    def __init__(
        self, input_channels: int, output_shape: int, patch_size: int, key: PRNGKeyArray
    ):
        self.patch_size = patch_size
        self.linear = eqx.nn.Linear(
            self.patch_size**2 * input_channels, output_shape, key=key
        )

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "num_patches d"]:
        # NCHW single image -> sequence of patches
        x = einops.rearrange(
            x,
            "c (h ph) (w pw) -> (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return jax.vmap(self.linear)(x)


class AttentionBlock(eqx.Module):
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        *,
        key,
    ):
        k0, k1, k2 = jax.random.split(key, 3)
        self.layer_norm1 = eqx.nn.LayerNorm(input_dim)
        self.layer_norm2 = eqx.nn.LayerNorm(input_dim)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=input_dim, key=k0
        )
        self.linear1 = eqx.nn.Linear(input_dim, hidden_dim, key=k1)
        self.linear2 = eqx.nn.Linear(hidden_dim, input_dim, key=k2)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self, x: jnp.ndarray, enable_dropout: bool, *, key: PRNGKeyArray
    ) -> jnp.ndarray:
        q = jax.vmap(self.layer_norm1)(x)
        attn = self.attention(q, q, q)
        x = x + attn
        y = jax.vmap(self.layer_norm2)(x)
        y = jax.vmap(self.linear1)(y)
        y = jax.nn.gelu(y)
        k1, k2 = jax.random.split(key)
        y = self.dropout1(y, inference=not enable_dropout, key=k1)
        y = jax.vmap(self.linear2)(y)
        y = self.dropout2(y, inference=not enable_dropout, key=k2)
        return x + y


class ViTEncoder(eqx.Module, BaseEncoder):
    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray  # (num_patches+1, d)
    cls_token: jnp.ndarray  # (1, d)
    transformer_blocks: list[AttentionBlock]
    mlp: eqx.nn.Linear
    latent_dim: int
    patch_size: int
    num_patches: int
    embedding_dim: int

    def __init__(
        self,
        image_size: int,
        channels: int,
        patch_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        latent_dim: int,
        dropout_rate: float,
        *,
        key,
    ):
        num_patches = (image_size // patch_size) ** 2
        keys = jax.random.split(key, 3 + num_layers + 1)
        self.patch_embedding = PatchEmbedding(
            channels, embedding_dim, patch_size, keys[0]
        )
        self.positional_embedding = jax.random.normal(
            keys[1], (num_patches + 1, embedding_dim)
        )
        self.cls_token = jax.random.normal(keys[2], (1, embedding_dim))
        self.transformer_blocks = [
            AttentionBlock(
                embedding_dim,
                embedding_dim * 4,
                num_heads,
                dropout_rate,
                key=keys[3 + i],
            )
            for i in range(num_layers)
        ]
        self.mlp = eqx.nn.Linear(embedding_dim, 2 * latent_dim, key=keys[-1])
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim

    def __call__(self, x: jnp.ndarray, *, rng=None, train: bool = False):
        # x: (B, C, H, W) NCHW
        def encode_one(img, k):
            patches = self.patch_embedding(img)  # (P, d)
            tokens = jnp.concatenate([self.cls_token, patches], axis=0)
            tokens = tokens + self.positional_embedding[: tokens.shape[0]]
            # run blocks
            for blk in self.transformer_blocks:
                k, sub = jax.random.split(k)
                tokens = blk(tokens, enable_dropout=train, key=sub)
            return tokens[0]  # CLS

        rngs = jax.random.split(
            rng if rng is not None else jax.random.PRNGKey(0), x.shape[0]
        )
        cls = jax.vmap(encode_one)(x, rngs)
        o = jax.vmap(self.mlp)(cls)
        mu, logvar = jnp.split(o, 2, axis=-1)
        return mu, logvar


class ViTDecoder(eqx.Module, BaseDecoder):
    latent_projection: eqx.nn.Linear
    positional_embedding: jnp.ndarray  # (P, d)
    transformer_blocks: list[AttentionBlock]
    mlp: eqx.nn.Linear  # token -> patch pixels
    patch_size: int
    num_patches: int
    embedding_dim: int
    image_size: int
    channels: int

    def __init__(
        self,
        image_size: int,
        channels: int,
        patch_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        latent_dim: int,
        dropout_rate: float,
        *,
        key,
    ):
        num_patches = (image_size // patch_size) ** 2
        keys = jax.random.split(key, 2 + num_layers + 1)
        self.latent_projection = eqx.nn.Linear(latent_dim, embedding_dim, key=keys[0])
        self.positional_embedding = jax.random.normal(
            keys[1], (num_patches, embedding_dim)
        )
        self.transformer_blocks = [
            AttentionBlock(
                embedding_dim,
                embedding_dim * 4,
                num_heads,
                dropout_rate,
                key=keys[2 + i],
            )
            for i in range(num_layers)
        ]
        patch_dim = patch_size * patch_size * channels
        self.mlp = eqx.nn.Linear(embedding_dim, patch_dim, key=keys[-1])
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.channels = channels

    def __call__(self, z: jnp.ndarray, *, rng=None, train: bool = False):
        def decode_one(zi, k):
            token = self.latent_projection(zi)  # (d,)
            seq = jnp.repeat(token[None, :], self.num_patches, axis=0)
            seq = seq + self.positional_embedding
            for blk in self.transformer_blocks:
                k, sub = jax.random.split(k)
                seq = blk(seq, enable_dropout=train, key=sub)
            patch_vecs = jax.vmap(self.mlp)(seq)  # (P, patch_dim)
            # stitch back to NCHW
            Pside = self.image_size // self.patch_size
            patches = patch_vecs.reshape(
                Pside, Pside, self.patch_size, self.patch_size, self.channels
            )
            img_hwc = jnp.transpose(patches, (0, 2, 1, 3, 4)).reshape(
                self.image_size, self.image_size, self.channels
            )
            img_nchw = jnp.transpose(img_hwc, (2, 0, 1))
            return img_nchw

        rngs = jax.random.split(
            rng if rng is not None else jax.random.PRNGKey(0), z.shape[0]
        )
        return jax.vmap(decode_one)(z, rngs)


# ============
#   Full VAEs
# ============


class MLP_VAE(eqx.Module, BaseVAE):
    encoder: MLPEncoder
    decoder: MLPDecoder
    latent_dim: int
    # Optional scalar logvar for Gaussian likelihood (broadcast in trainer if enabled)
    gauss_logvar_param: jnp.ndarray

    def __init__(
        self, input_dim: int, hidden_dim: int, latent_dim: int, output_dim: int, *, key
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.encoder = MLPEncoder(input_dim, hidden_dim, latent_dim, key=k1)
        self.decoder = MLPDecoder(latent_dim, hidden_dim, output_dim, key=k2)
        self.latent_dim = latent_dim
        self.gauss_logvar_param = jnp.array(
            0.0
        )  # scalar; trainer can broadcast if gaussian_learn_logvar=True

    def sample_z(self, rng, mu, logvar):
        eps = jax.random.normal(rng, shape=mu.shape)
        return mu + jnp.exp(0.5 * logvar) * eps

    def __call__(self, x: jnp.ndarray, rng, *, train: bool = False):
        rng_e, rng_s, rng_d = jax.random.split(rng, 3)
        mu, logvar = self.encoder(x, rng=rng_e, train=train)
        logvar = jnp.clip(logvar, -10.0, 10.0)
        z = self.sample_z(rng_s, mu, logvar)
        dec = self.decoder(z, rng=rng_d, train=train)
        return dec, mu, logvar


class ConvVAE(eqx.Module, BaseVAE):
    encoder: CNNEncoder
    decoder: CNNDecoder
    latent_dim: int
    gauss_logvar_param: jnp.ndarray

    def __init__(
        self,
        image_size: int,
        channels: int,
        hidden_channels: int,
        latent_dim: int,
        *,
        key,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.encoder = CNNEncoder(
            channels, image_size, hidden_channels, latent_dim, key=k1
        )
        self.decoder = CNNDecoder(
            latent_dim, hidden_channels, channels, image_size, key=k2
        )
        self.latent_dim = latent_dim
        self.gauss_logvar_param = jnp.array(0.0)  # scalar

    def sample_z(self, rng, mu, logvar):
        eps = jax.random.normal(rng, shape=mu.shape)
        return mu + jnp.exp(0.5 * logvar) * eps

    def __call__(self, x: jnp.ndarray, rng, *, train: bool = False):
        rng_e, rng_s, rng_d = jax.random.split(rng, 3)
        mu, logvar = self.encoder(x, rng=rng_e, train=train)
        logvar = jnp.clip(logvar, -10.0, 10.0)
        z = self.sample_z(rng_s, mu, logvar)
        dec = self.decoder(z, rng=rng_d, train=train)
        return dec, mu, logvar


class MultiHeadAttentionVAE(eqx.Module, BaseVAE):
    encoder: AttentionEncoder
    decoder: AttentionDecoder
    latent_dim: int
    gauss_logvar_param: jnp.ndarray

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_length: int,
        num_heads: int,
        *,
        key,
    ):
        k1, k2 = jax.random.split(key)
        self.encoder = AttentionEncoder(
            input_dim, latent_dim, num_heads, seq_length, key=k1
        )
        self.decoder = AttentionDecoder(
            latent_dim, hidden_dim, output_dim, seq_length, num_heads, key=k2
        )
        self.latent_dim = latent_dim
        self.gauss_logvar_param = jnp.array(0.0)

    def sample_z(self, rng, mu, logvar):
        eps = jax.random.normal(rng, shape=mu.shape)
        return mu + jnp.exp(0.5 * logvar) * eps

    def __call__(self, x: jnp.ndarray, rng, *, train: bool = False):
        rng_e, rng_s, rng_d = jax.random.split(rng, 3)
        mu, logvar = self.encoder(x, rng=rng_e, train=train)
        logvar = jnp.clip(logvar, -10.0, 10.0)
        z = self.sample_z(rng_s, mu, logvar)
        dec = self.decoder(z, rng=rng_d, train=train)
        return dec, mu, logvar


class ViT_VAE(eqx.Module, BaseVAE):
    encoder: ViTEncoder
    decoder: ViTDecoder
    latent_dim: int
    gauss_logvar_param: jnp.ndarray

    def __init__(
        self,
        image_size: int,
        channels: int,
        patch_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        latent_dim: int,
        dropout_rate: float,
        *,
        key,
    ):
        k1, k2 = jax.random.split(key)
        self.encoder = ViTEncoder(
            image_size,
            channels,
            patch_size,
            embedding_dim,
            num_layers,
            num_heads,
            latent_dim,
            dropout_rate,
            key=k1,
        )
        self.decoder = ViTDecoder(
            image_size,
            channels,
            patch_size,
            embedding_dim,
            num_layers,
            num_heads,
            latent_dim,
            dropout_rate,
            key=k2,
        )
        self.latent_dim = latent_dim
        self.gauss_logvar_param = jnp.array(0.0)

    def sample_z(self, rng, mu, logvar):
        eps = jax.random.normal(rng, shape=mu.shape)
        return mu + jnp.exp(0.5 * logvar) * eps

    def __call__(self, x: jnp.ndarray, rng, *, train: bool = False):
        rng_e, rng_s, rng_d = jax.random.split(rng, 3)
        mu, logvar = self.encoder(x, rng=rng_e, train=train)
        logvar = jnp.clip(logvar, -10.0, 10.0)
        z = self.sample_z(rng_s, mu, logvar)
        dec = self.decoder(z, rng=rng_d, train=train)
        return dec, mu, logvar
