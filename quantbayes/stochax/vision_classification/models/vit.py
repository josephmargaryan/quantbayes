import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

# -------------------------------------------------------------------
# Patch Embedding Module
#
# Splits an image into non-overlapping patches and projects each patch
# to an embedding vector via a linear layer applied on flattened patches.
# Assumes input images of shape [channels, height, width].
# -------------------------------------------------------------------


class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Linear
    patch_size: int

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        patch_size: int,
        key: PRNGKeyArray,
    ):
        self.patch_size = patch_size
        self.linear = eqx.nn.Linear(
            patch_size**2 * input_channels,
            output_dim,
            key=key,
        )

    def __call__(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "num_patches embedding_dim"]:
        x = einops.rearrange(
            x,
            "c (h ph) (w pw) -> (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        # Apply the same linear layer to each patch.
        x = jax.vmap(self.linear)(x)
        return x


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
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        key: PRNGKeyArray,
    ):
        key1, key2, key3 = jr.split(key, 3)
        self.layer_norm1 = eqx.nn.LayerNorm(embed_dim)
        self.layer_norm2 = eqx.nn.LayerNorm(embed_dim)
        self.attention = eqx.nn.MultiheadAttention(num_heads, embed_dim, key=key1)
        self.linear1 = eqx.nn.Linear(embed_dim, hidden_dim, key=key2)
        self.linear2 = eqx.nn.Linear(hidden_dim, embed_dim, key=key3)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self, x: Float[Array, "num_patches embedding_dim"], key: PRNGKeyArray
    ) -> Float[Array, "num_patches embedding_dim"]:
        # First residual branch with attention.
        x_norm = jax.vmap(self.layer_norm1)(x)
        attn_out = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Second residual branch with MLP.
        x_norm = jax.vmap(self.layer_norm2)(x)
        mlp_hidden = jax.vmap(self.linear1)(x_norm)
        mlp_hidden = jax.nn.gelu(mlp_hidden)

        key1, key2 = jr.split(key, num=2)
        mlp_hidden = self.dropout1(mlp_hidden, key=key1)
        mlp_out = jax.vmap(self.linear2)(mlp_hidden)
        mlp_out = self.dropout2(mlp_out, key=key2)
        x = x + mlp_out

        return x


class VisionTransformer(eqx.Module):
    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray
    cls_token: jnp.ndarray
    attention_blocks: list[AttentionBlock]
    dropout: eqx.nn.Dropout
    mlp: eqx.nn.Sequential
    num_layers: int

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        patch_size: int,
        num_patches: int,
        num_classes: int,
        key: PRNGKeyArray,
        channels: int,
    ):
        key1, key2, key3, key4, key5 = jr.split(key, 5)
        self.patch_embedding = PatchEmbedding(channels, embedding_dim, patch_size, key1)

        # +1 for the CLS token.
        self.positional_embedding = jr.normal(key2, (num_patches + 1, embedding_dim))
        self.cls_token = jr.normal(key3, (1, embedding_dim))
        self.num_layers = num_layers

        # Create a list of transformer blocks.
        self.attention_blocks = [
            AttentionBlock(embedding_dim, hidden_dim, num_heads, dropout_rate, key4)
            for _ in range(num_layers)
        ]

        self.dropout = eqx.nn.Dropout(dropout_rate)

        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(embedding_dim),
                eqx.nn.Linear(embedding_dim, num_classes, key=key5),
            ]
        )

    def __call__(
        self, x: Float[Array, "channels height width"], key: PRNGKeyArray, state
    ) -> tuple[Float[Array, "num_classes"], any]:
        # Embed patches.
        x = self.patch_embedding(x)

        # Prepend the CLS token.
        x = jnp.concatenate((self.cls_token, x), axis=0)
        x += self.positional_embedding[: x.shape[0]]

        # Split keys: one for the initial dropout and one per transformer block.
        keys = jr.split(key, self.num_layers + 1)
        x = self.dropout(x, key=keys[0])
        for block, block_key in zip(self.attention_blocks, keys[1:]):
            x = block(x, key=block_key)
        # Use only the CLS token for classification.
        x = x[0]
        x = self.mlp(x)
        return x, state


# -------------------------------------------------------------------
# Test function with MNIST configuration.
#
# MNIST images: 28x28 grayscale (1 channel). For patching, we use a
# patch_size that divides 28 evenly (e.g., 7, so 4x4=16 patches).
# The classifier outputs 10 classes.
# -------------------------------------------------------------------
def test_vit_mnist_output_shape():
    # MNIST configuration.
    embedding_dim = 64
    hidden_dim = embedding_dim * 4  # MLP expansion factor.
    num_heads = 4
    num_layers = 6
    dropout_rate = 0.1
    patch_size = 7
    img_height = 28
    img_width = 28
    channels = 1
    num_patches = (img_height // patch_size) * (img_width // patch_size)  # 4*4 = 16
    num_classes = 10

    # Create PRNG keys.
    key = jr.PRNGKey(42)
    key, model_key, run_key, input_key = jr.split(key, 4)

    vit = VisionTransformer(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        patch_size=patch_size,
        num_patches=num_patches,
        num_classes=num_classes,
        key=model_key,
        channels=channels,
    )

    # Create a random input image of shape [channels, height, width].
    x = jr.normal(input_key, (channels, img_height, img_width))

    # Run the model. (Dropout is applied using keys.)
    logits, _ = vit(x, key=run_key, state=None)

    # Check that the output has shape (num_classes,).
    assert logits.shape == (
        num_classes,
    ), f"Expected output shape ({num_classes},), got {logits.shape}"
    print("MNIST ViT test passed!")


if __name__ == "__main__":
    test_vit_mnist_output_shape()
