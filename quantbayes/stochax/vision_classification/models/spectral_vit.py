import jax.random as jr
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from quantbayes.stochax.layers import BlockCirculantProcess
from quantbayes.stochax.vision_classification.models.vit import (
    PatchEmbedding,
    AttentionBlock,
)


class CircViT(eqx.Module):
    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray
    cls_token: jnp.ndarray
    attention_blocks: list[AttentionBlock]
    dropout: eqx.nn.Dropout
    circ_head: "BlockCirculantProcess"  # Use your custom circulant layer.
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
        circ_block_size: int,
        key: PRNGKeyArray,
        channels: int,
    ):
        key1, key2, key3, key4, key5, key6 = jr.split(key, 6)
        self.patch_embedding = PatchEmbedding(channels, embedding_dim, patch_size, key1)

        # +1 for the CLS token.
        self.positional_embedding = jr.normal(key2, (num_patches + 1, embedding_dim))
        self.cls_token = jr.normal(key3, (1, embedding_dim))
        self.num_layers = num_layers

        # Create transformer (attention) blocks.
        self.attention_blocks = [
            AttentionBlock(embedding_dim, hidden_dim, num_heads, dropout_rate, key4)
            for _ in range(num_layers)
        ]

        self.dropout = eqx.nn.Dropout(dropout_rate)

        # Final classification head using the circulant layer.
        # Here, in_features is embedding_dim (from the CLS token),
        # out_features is num_classes, and we set block_size to circ_block_size.
        self.circ_head = BlockCirculantProcess(
            in_features=embedding_dim,
            out_features=num_classes,
            block_size=circ_block_size,
            key=key5,
            init_scale=0.1,
            use_diag=True,
            use_bias=True,
        )
        # Create transformer (attention) blocks.
        self.attention_blocks = [
            AttentionBlock(embedding_dim, hidden_dim, num_heads, dropout_rate, key4)
            for _ in range(num_layers)
        ]

        self.dropout = eqx.nn.Dropout(dropout_rate)

        # Final classification head using the circulant layer.
        # Here, in_features is embedding_dim (from the CLS token),
        # out_features is num_classes, and we set block_size to circ_block_size.
        self.circ_head = BlockCirculantProcess(
            in_features=embedding_dim,
            out_features=num_classes,
            block_size=circ_block_size,
            key=key5,
            init_scale=0.1,
            use_diag=True,
            use_bias=True,
        )

    def __call__(self, x: Float[Array, "channels height width"], key: PRNGKeyArray, state) -> tuple[Float[Array, "num_classes"], any]:
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
        cls_embedding = x[0]
        # Apply the circulant classification head.
        logits = self.circ_head(cls_embedding)
        return logits, state
    
# Example test function for CircViT with MNIST-like configuration.
def test_circvit_mnist_output_shape():
    # Configuration parameters.
    embedding_dim = 64
    hidden_dim = embedding_dim * 4
    num_heads = 4
    num_layers = 6
    dropout_rate = 0.1
    patch_size = 7
    img_height = 28
    img_width = 28
    channels = 1
    num_patches = (img_height // patch_size) * (img_width // patch_size)  # 16 patches.
    num_classes = 10
    circ_block_size = 8  # For example; must be chosen appropriately relative to embedding_dim.

    # Create PRNG keys.
    key = jr.PRNGKey(42)
    key, model_key, run_key, input_key = jr.split(key, 4)

    circvit = CircViT(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        patch_size=patch_size,
        num_patches=num_patches,
        num_classes=num_classes,
        circ_block_size=circ_block_size,
        key=model_key,
        channels=channels,
    )

    # Create a random input image of shape [channels, height, width].
    x = jr.normal(input_key, (channels, img_height, img_width))
    logits, _ = circvit(x, key=run_key, state=None)

    # Check that the output has shape (num_classes,).
    assert logits.shape == (num_classes,), f"Expected output shape ({num_classes},), got {logits.shape}"
    print("CircViT MNIST test passed!")

if __name__ == "__main__":
    test_circvit_mnist_output_shape()