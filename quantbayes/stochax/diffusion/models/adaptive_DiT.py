import math
from typing import List, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange


# -----------------------------------------------------
# Helper: Sinusoidal Time Embedding
# -----------------------------------------------------
class SinusoidalTimeEmb(eqx.Module):
    emb: jnp.ndarray

    def __init__(self, dim: int):
        half_dim = dim // 2
        # Compute frequencies in log space.
        emb = math.log(10000.0) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: scalar time (or 0D array)
        emb = x * self.emb
        out = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return out


# -----------------------------------------------------
# Patch Embedding and Un-Embedding
# -----------------------------------------------------
class PatchEmbed(eqx.Module):
    patch_size: int
    proj: eqx.nn.Linear
    num_patches: int
    embed_dim: int
    channels: int

    def __init__(
        self, channels: int, embed_dim: int, patch_size: int, img_size: tuple, *, key
    ):
        self.channels = channels
        self.patch_size = patch_size
        c, h, w = img_size
        assert c == channels
        assert h % patch_size == 0 and w % patch_size == 0
        num_h = h // patch_size
        num_w = w // patch_size
        self.num_patches = num_h * num_w
        self.embed_dim = embed_dim

        in_features = patch_size * patch_size * channels
        self.proj = eqx.nn.Linear(in_features, embed_dim, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: shape (C, H, W)
        c, h, w = x.shape
        ph = pw = self.patch_size
        # Rearrange into patches: (num_patches, patch_size*patch_size*channels)
        patches = rearrange(x, "c (nh ph) (nw pw) -> (nh nw) (c ph pw)", ph=ph, pw=pw)
        tokens = jax.vmap(self.proj)(patches)
        return tokens


# --- Modified PatchUnembed ---
# This version does not apply a linear projection but simply rearranges tokens.
class PatchUnembed(eqx.Module):
    patch_size: int
    num_patches: int
    channels: int  # effective output channels (e.g. in_channels*2 if learn_sigma True)
    h: int
    w: int

    def __init__(self, channels: int, patch_size: int, img_size: tuple):
        self.channels = channels
        self.patch_size = patch_size
        _, h, w = img_size
        assert h % patch_size == 0 and w % patch_size == 0
        num_h = h // patch_size
        num_w = w // patch_size
        self.num_patches = num_h * num_w
        self.h = h
        self.w = w

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        # tokens: shape (num_patches, patch_size*patch_size*channels)
        ph = pw = self.patch_size
        num_h = self.h // ph
        num_w = self.w // pw
        x = rearrange(
            tokens,
            "(nh nw) (c ph pw) -> c (nh ph) (nw pw)",
            nh=num_h,
            nw=num_w,
            ph=ph,
            pw=pw,
        )
        return x


class LearnablePositionalEmb(eqx.Module):
    pos_emb: jnp.ndarray  # shape (num_patches, embed_dim)

    def __init__(self, num_patches: int, embed_dim: int, *, key):
        # Initialize with small random values.
        self.pos_emb = jr.normal(key, (num_patches, embed_dim)) * 0.02

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: shape (num_patches, embed_dim)
        n, _ = x.shape
        out = x + self.pos_emb[:n]
        return out


# -----------------------------------------------------
# Label Embedder (for classifier-free guidance)
# -----------------------------------------------------
class LabelEmbedder(eqx.Module):
    embedding: eqx.nn.Embedding
    dropout_prob: float

    def __init__(self, num_classes: int, embed_dim: int, dropout_prob: float, *, key):
        # Allocate an extra index for "dropped" labels.
        self.embedding = eqx.nn.Embedding(num_classes + 1, embed_dim, key=key)
        self.dropout_prob = dropout_prob

    def __call__(
        self,
        labels: jnp.ndarray,
        train: bool,
        *,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        if train and self.dropout_prob > 0 and key is not None:
            drop = jr.uniform(key, labels.shape) < self.dropout_prob
            labels = jnp.where(drop, self.embedding.num_embeddings - 1, labels)
        out = self.embedding(labels)
        return out


# -----------------------------------------------------
# Adaptive modulation helper function
# -----------------------------------------------------
def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    out = x * (1.0 + scale) + shift
    return out


# -----------------------------------------------------
# DiT Block with Adaptive LN (adaLN) modulation
# -----------------------------------------------------
class DiTBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    adaLN_modulation: eqx.nn.MLP  # Outputs 6*embed_dim

    dropout_rate: float

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float,
        dropout_rate: float,
        *,
        key,
    ):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.norm1 = eqx.nn.LayerNorm((embed_dim,))
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=embed_dim,
            key_size=embed_dim,
            value_size=embed_dim,
            output_size=embed_dim,
            dropout_p=dropout_rate,
            inference=False,
            key=k1,
        )
        self.norm2 = eqx.nn.LayerNorm((embed_dim,))
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = eqx.nn.MLP(
            in_size=embed_dim,
            out_size=embed_dim,
            width_size=hidden_dim,
            depth=2,
            key=k2,
        )
        self.adaLN_modulation = eqx.nn.MLP(
            in_size=embed_dim,
            out_size=6 * embed_dim,
            width_size=embed_dim * 2,
            depth=1,
            key=k3,
        )
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray, *, key=None) -> jnp.ndarray:
        mod_params = self.adaLN_modulation(cond)  # shape (6*embed_dim,)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            mod_params, 6, axis=-1
        )
        # Attention branch
        x_norm = jax.vmap(self.norm1)(x)
        x_mod = modulate(x_norm, shift_attn, scale_attn)
        attn_out = self.attn(x_mod, x_mod, x_mod, key=key)
        x = x + gate_attn * attn_out
        # MLP branch
        x_norm2 = jax.vmap(self.norm2)(x)
        x_mod2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_out = jax.vmap(self.mlp)(x_mod2)
        x = x + gate_mlp * mlp_out
        return x


# -----------------------------------------------------
# Final Layer with Adaptive LN modulation
# -----------------------------------------------------
class DiTFinalLayer(eqx.Module):
    norm: eqx.nn.LayerNorm
    linear: eqx.nn.Linear
    adaLN_modulation: eqx.nn.MLP

    patch_size: int
    out_channels: int

    def __init__(self, embed_dim: int, patch_size: int, out_channels: int, *, key):
        k1, k2 = jr.split(key, 2)
        self.norm = eqx.nn.LayerNorm((embed_dim,))
        # Map from embed_dim to patch_size*patch_size*out_channels.
        self.linear = eqx.nn.Linear(
            embed_dim, patch_size * patch_size * out_channels, key=k1
        )
        self.adaLN_modulation = eqx.nn.MLP(
            in_size=embed_dim,
            out_size=2 * embed_dim,
            width_size=embed_dim,
            depth=1,
            key=k2,
        )
        self.patch_size = patch_size
        self.out_channels = out_channels

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        mod_params = self.adaLN_modulation(cond)  # shape (2*embed_dim,)
        shift, scale = jnp.split(mod_params, 2, axis=-1)
        x_norm = jax.vmap(self.norm)(x)
        x_mod = modulate(x_norm, shift, scale)
        # Apply the linear layer tokenwise.
        out = jax.vmap(self.linear)(x_mod)
        return out


# -----------------------------------------------------
# DiT (Diffusion Transformer) Module in Equinox
# -----------------------------------------------------
class DiT(eqx.Module):
    patch_embed: PatchEmbed
    pos_embed: LearnablePositionalEmb
    patch_unembed: PatchUnembed
    time_emb: SinusoidalTimeEmb
    time_proj: eqx.nn.MLP
    label_embed: LabelEmbedder

    blocks: List[DiTBlock]
    final_layer: DiTFinalLayer

    embed_dim: int
    num_patches: int
    patch_size: int
    out_channels: int

    def __init__(
        self,
        img_size: tuple,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: float,
        dropout_rate: float,
        time_emb_dim: int,
        num_classes: int,
        learn_sigma: bool,
        *,
        key,
    ):
        # img_size: (C, H, W)
        c, h, w = img_size
        total_keys = (
            depth + 6
        )  # keys for time_proj, patch_embed, pos_embed, patch_unembed, label_embed, final_layer, plus one per block.
        keys = jr.split(key, total_keys)
        (
            k_time,
            k_patch_embed,
            k_pos,
            k_patch_unembed,
            k_label,
            k_final,
            *k_blocks,
        ) = keys

        self.patch_embed = PatchEmbed(
            channels=c,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            key=k_patch_embed,
        )
        # When learn_sigma is True, effective out_channels = in_channels * 2.
        effective_channels = in_channels * 2 if learn_sigma else in_channels
        # Use the modified PatchUnembed (which simply rearranges tokens) with effective channels.
        self.patch_unembed = PatchUnembed(
            channels=effective_channels, patch_size=patch_size, img_size=img_size
        )
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = LearnablePositionalEmb(self.num_patches, embed_dim, key=k_pos)

        self.time_emb = SinusoidalTimeEmb(time_emb_dim)
        self.time_proj = eqx.nn.MLP(
            in_size=time_emb_dim,
            out_size=embed_dim,
            width_size=2 * embed_dim,
            depth=2,
            key=k_time,
        )

        self.label_embed = LabelEmbedder(
            num_classes, embed_dim, dropout_rate, key=k_label
        )

        # Build transformer blocks.
        self.blocks = [
            DiTBlock(embed_dim, n_heads, mlp_ratio, dropout_rate, key=k_blocks[i])
            for i in range(depth)
        ]
        self.final_layer = DiTFinalLayer(
            embed_dim, patch_size, effective_channels, key=k_final
        )

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.out_channels = effective_channels

    def _forward(
        self, t: float, x: jnp.ndarray, label: jnp.ndarray, train: bool, *, key=None
    ) -> jnp.ndarray:
        tokens = self.patch_embed(x)  # (num_patches, embed_dim)
        tokens = self.pos_embed(tokens)  # add positional information

        # Time embedding:
        t_emb = self.time_emb(jnp.array(t))
        t_emb = self.time_proj(t_emb)  # (embed_dim,)

        # Label embedding:
        # (For simplicity, we use a fixed key here; in practice, pass a PRNGKey)
        lab_emb = self.label_embed(label, train, key=jr.PRNGKey(0))
        cond = t_emb + lab_emb  # Conditioning vector (embed_dim,)

        # Broadcast conditioning to all tokens:
        tokens = tokens + jnp.broadcast_to(cond, tokens.shape)

        # Pass tokens through transformer blocks:
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, cond, key=key)

        # Final layer:
        out_tokens = self.final_layer(tokens, cond)
        return out_tokens

    def __call__(
        self, t: float, x: jnp.ndarray, label: jnp.ndarray, train: bool, *, key=None
    ) -> jnp.ndarray:
        """
        If x is batched (shape (B, C, H, W)), vectorize over the batch dimension.
        """
        if x.ndim == 4:

            def single_forward(sample, lab, key):
                return self._forward(t, sample, lab, train, key=key)

            if key is not None:
                keys = jr.split(key, x.shape[0])
                out_tokens = jax.vmap(single_forward)(x, label, keys)
            else:
                out_tokens = jax.vmap(single_forward)(x, label)
        else:
            out_tokens = self._forward(t, x, label, train, key=key)
        out = self.patch_unembed(out_tokens)
        return out


# -----------------------------------------------------
# Test function for DiT in Equinox
# -----------------------------------------------------
def test_dit_forward():
    key = jr.PRNGKey(0)
    # Hyperparameters for testing
    img_size = (3, 32, 32)  # Colored image: (C, H, W)
    patch_size = 4
    in_channels = 3
    embed_dim = 384
    depth = 2  # Small depth for testing
    n_heads = 6
    mlp_ratio = 4.0
    dropout_rate = 0.1
    time_emb_dim = 256
    num_classes = 10
    learn_sigma = True

    model = DiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        dropout_rate=dropout_rate,
        time_emb_dim=time_emb_dim,
        num_classes=num_classes,
        learn_sigma=learn_sigma,
        key=key,
    )

    # Create a dummy input image (C, H, W)
    dummy_img = jnp.ones(img_size)
    # Dummy label (scalar)
    dummy_label = jnp.array(1)
    t = 500.0  # Example timestep

    out = model(t, dummy_img, dummy_label, train=False, key=jr.PRNGKey(42))
    expected_channels = in_channels * 2 if learn_sigma else in_channels
    expected_shape = (expected_channels, img_size[1], img_size[2])
    assert (
        out.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {out.shape}"


if __name__ == "__main__":
    test_dit_forward()
