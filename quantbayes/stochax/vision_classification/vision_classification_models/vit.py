import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray
import einops


# -------------------------------------------------------------------
# Patch Embedding Module
#
# Splits an image into non-overlapping patches and projects each patch
# to an embedding vector via a 2D convolution.
# Assumes input images of shape [N, C, H, W] (PyTorch style).
# -------------------------------------------------------------------
class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Embedding
    patch_size: int

    def __init__(
        self,
        input_channels: int,
        output_shape: int,
        patch_size: int,
        key: PRNGKeyArray,
    ):
        self.patch_size = patch_size

        self.linear = eqx.nn.Linear(
            self.patch_size**2 * input_channels,
            output_shape,
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
        input_shape: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        key: PRNGKeyArray,
    ):
        key1, key2, key3 = jr.split(key, 3)

        self.layer_norm1 = eqx.nn.LayerNorm(input_shape)
        self.layer_norm2 = eqx.nn.LayerNorm(input_shape)
        self.attention = eqx.nn.MultiheadAttention(num_heads, input_shape, key=key1)

        self.linear1 = eqx.nn.Linear(input_shape, hidden_dim, key=key2)
        self.linear2 = eqx.nn.Linear(hidden_dim, input_shape, key=key3)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        x: Float[Array, "num_patches embedding_dim"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "num_patches embedding_dim"]:
        input_x = jax.vmap(self.layer_norm1)(x)
        x = x + self.attention(input_x, input_x, input_x)

        input_x = jax.vmap(self.layer_norm2)(x)
        input_x = jax.vmap(self.linear1)(input_x)
        input_x = jax.nn.gelu(input_x)

        key1, key2 = jr.split(key, num=2)

        input_x = self.dropout1(input_x, inference=not enable_dropout, key=key1)
        input_x = jax.vmap(self.linear2)(input_x)
        input_x = self.dropout2(input_x, inference=not enable_dropout, key=key2)

        x = x + input_x

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

        self.positional_embedding = jr.normal(key2, (num_patches + 1, embedding_dim))

        self.cls_token = jr.normal(key3, (1, embedding_dim))

        self.num_layers = num_layers

        self.attention_blocks = [
            AttentionBlock(embedding_dim, hidden_dim, num_heads, dropout_rate, key4)
            for _ in range(self.num_layers)
        ]

        self.dropout = eqx.nn.Dropout(dropout_rate)

        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(embedding_dim),
                eqx.nn.Linear(embedding_dim, num_classes, key=key5),
            ]
        )

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "num_classes"]:
        x = self.patch_embedding(x)

        x = jnp.concatenate((self.cls_token, x), axis=0)

        x += self.positional_embedding[
            : x.shape[0]
        ]  # Slice to the same length as x, as the positional embedding may be longer.

        dropout_key, *attention_keys = jr.split(key, num=self.num_layers + 1)

        x = self.dropout(x, inference=not enable_dropout, key=dropout_key)

        for block, attention_key in zip(self.attention_blocks, attention_keys):
            x = block(x, enable_dropout, key=attention_key)

        x = x[0]  # Select the CLS token.
        x = self.mlp(x)

        return x


def test_vision_transformer_output_shape():
    # Configuration:
    embedding_dim = 768
    hidden_dim = int(embedding_dim * 4)  # using an MLP ratio of 4.0
    num_heads = 12
    num_layers = 12
    dropout_rate = 0.1
    patch_size = 16
    img_height = 224
    img_width = 224
    channels = 3
    num_patches = (img_height // patch_size) * (
        img_width // patch_size
    )  # 14 * 14 = 196
    num_classes = 1000

    # Create PRNG keys.
    key = jr.PRNGKey(0)
    key, model_key, run_key, input_key = jr.split(key, 4)

    # Create an instance of VisionTransformer.
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

    # Run the model. Note: We disable dropout (inference mode).
    logits = vit(x, enable_dropout=False, key=run_key)

    # Check that the output has shape (num_classes,).
    assert logits.shape == (
        num_classes,
    ), f"Expected output shape ({num_classes},), got {logits.shape}"


if __name__ == "__main__":
    test_vision_transformer_output_shape()
    print("Test passed!")
