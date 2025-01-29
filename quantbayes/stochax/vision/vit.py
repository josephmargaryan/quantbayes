import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any

class PatchEmbedding(nn.Module):
    patch_size: int
    emb_dim: int
    in_channels: int

    @nn.compact
    def __call__(self, x):
        batch_size, height, width, channels = x.shape
        assert channels == self.in_channels, f"Expected {self.in_channels} channels, got {channels}"
        assert height == width, "Only square images are supported"
        assert height % self.patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (height // self.patch_size) ** 2

        # Conv layer to extract patches and project to embedding dimension
        x = nn.Conv(
            features=self.emb_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            name="conv"
        )(x)
        x = x.reshape(batch_size, num_patches, -1)  # Flatten patches
        return x


class TransformerEncoderBlock(nn.Module):
    emb_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # Layer normalization
        x_norm = nn.LayerNorm(name="ln_1")(x)
        # Multi-head self-attention
        x = x + nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=jnp.float32,
            name="mha"
        )(x_norm, x_norm)
        # Layer normalization
        x_norm = nn.LayerNorm(name="ln_2")(x)
        # MLP block
        x_mlp = nn.Dense(self.mlp_dim, name="fc1")(x_norm)
        x_mlp = nn.gelu(x_mlp)
        x_mlp = nn.Dropout(rate=self.dropout_rate)(x_mlp, deterministic=deterministic)
        x_mlp = nn.Dense(self.emb_dim, name="fc2")(x_mlp)
        x_mlp = nn.Dropout(rate=self.dropout_rate)(x_mlp, deterministic=deterministic)
        x = x + x_mlp
        return x

class VisionTransformer(nn.Module):
    img_size: int
    patch_size: int
    emb_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    num_classes: int
    in_channels: int = 3  # New argument for input channels
    dropout_rate: float = 0.1

    def setup(self):
        self.patch_embedding = PatchEmbedding(
            patch_size=self.patch_size,
            emb_dim=self.emb_dim,
            in_channels=self.in_channels  # Pass the input channels here
        )
        self.cls_token = self.param(
            'cls_token',
            nn.initializers.zeros,
            (1, 1, self.emb_dim)
        )
        self.pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.zeros,
            (1, (self.img_size // self.patch_size) ** 2 + 1, self.emb_dim)
        )
        self.transformer_encoder = [
            TransformerEncoderBlock(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate
            ) for _ in range(self.num_layers)
        ]
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.mlp_head = nn.Dense(self.num_classes, name="mlp_head")

    def __call__(self, x, train: bool = True):
        # Extract patch embeddings
        x = self.patch_embedding(x)
        batch_size = x.shape[0]

        # Add class token
        cls_tokens = jnp.tile(self.cls_token, (batch_size, 1, 1))
        x = jnp.concatenate([cls_tokens, x], axis=1)

        # Add positional embedding
        x = x + self.pos_embedding

        # Apply dropout
        x = self.dropout(x, deterministic=not train)

        # Transformer encoder
        for block in self.transformer_encoder:
            x = block(x, deterministic=not train)

        # Classification head
        cls_output = x[:, 0]  # Extract the class token output
        logits = self.mlp_head(cls_output)
        return logits

# Example usage
if __name__ == "__main__":
    # Model configuration
    model_config = {
        'in_channels': 3,
        'img_size': 224,
        'patch_size': 16,
        'emb_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'mlp_dim': 3072,
        'num_classes': 1000,
        'dropout_rate': 0.1
    }

    # Initialize the model
    model = VisionTransformer(**model_config)

    # Create random input tensor
    rng = jax.random.PRNGKey(0)
    rngs = {'params': jax.random.PRNGKey(1), 'dropout': jax.random.PRNGKey(2)}
    dummy_input = jax.random.normal(rng, (1, 224, 224, 3))

    # Initialize model parameters
    params = model.init(rngs, dummy_input)

    # Perform a forward pass
    logits = model.apply(params, dummy_input, rngs={'dropout': jax.random.PRNGKey(3)})
    print("Logits shape:", logits.shape)  # Should be (1, 1000)
