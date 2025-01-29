import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from quantbayes.bnn import Module
from typing import Optional
from quantbayes.bnn import Module, PositionalEncoding, TransformerEncoder


class ViT(Module):
    """
    A minimal Vision Transformer for classification using your custom Transformer layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        method: str = "nuts",
        task_type: str = "multiclass",
    ):
        """
        :param in_channels: Number of channels in input images.
        :param image_size: Height/width of the (square) input images.
        :param patch_size: Size of each patch (square).
        :param embed_dim: Dimensionality of patch embeddings.
        :param num_heads: Number of heads in MultiHeadSelfAttention.
        :param hidden_dim: Hidden dim for the feedforward inside the Transformer.
        :param num_layers: Number of TransformerEncoder blocks.
        :param num_classes: Number of output classes for classification.
        :param method: Bayesian inference method.
        :param task_type: "classification" or "binary", etc.
        """
        super().__init__(method=method, task_type=task_type)
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Number of patches
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)

        # If we add a CLS token, we have seq_len = num_patches + 1
        # For simplicity, let's do it. Otherwise we can do average pooling.
        self.seq_len = self.num_patches + 1

    def patchify(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Convert (batch, in_channels, H, W) -> (batch, num_patches, patch_dim).
        where patch_dim = patch_size * patch_size * in_channels
        """
        B, C, H, W = X.shape
        ph = self.patch_size
        # Reshape into patches
        X = X.reshape(
            B,
            C,
            H // ph,
            ph,
            W // ph,
            ph,
        )
        # Rearrange axes: (B, H//ph, W//ph, C, ph, ph)
        X = X.transpose(0, 2, 4, 1, 3, 5)
        # Flatten each patch
        patch_dim = ph * ph * C
        X = X.reshape(B, (H // ph) * (W // ph), patch_dim)
        return X

    def linear_embedding(self, patches: jnp.ndarray):
        """
        Project patch vectors to `embed_dim`.
        patches shape: (batch, num_patches, patch_dim)
        return shape: (batch, num_patches, embed_dim)
        """

        B, NP, PD = patches.shape
        # Flatten for a single Linear call
        patches_flat = patches.reshape(-1, PD)

        # Shared linear projection for all patches
        embed_w = numpyro.sample(
            "patch_embed_w", dist.Normal(0, 1).expand([PD, self.embed_dim])
        )
        embed_b = numpyro.sample(
            "patch_embed_b", dist.Normal(0, 1).expand([self.embed_dim])
        )
        embedded = jnp.dot(patches_flat, embed_w) + embed_b
        # Reshape back
        embedded = embedded.reshape(B, NP, self.embed_dim)
        return embedded

    def __call__(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None):
        """
        Forward pass of a minimal ViT:
          1) Patchify
          2) Linear embedding of patches
          3) Add [CLS] token
          4) PositionalEncoding
          5) N x TransformerEncoder
          6) Take the CLS token -> final Linear -> classification
        """

        # 2) Embed patches
        embedded_patches = self.linear_embedding(X)  # (B, num_patches, embed_dim)

        B = embedded_patches.shape[0]

        # 3) [CLS] token
        # We'll treat the CLS token as a learnable parameter (like in real ViTs).
        cls_token = numpyro.param(
            "cls_token", jnp.zeros((1, 1, self.embed_dim))
        )  # shape (1,1,embed_dim)
        # Expand to batch size
        cls_tokens = jnp.tile(cls_token, [B, 1, 1])  # (B,1,embed_dim)

        # Concat along sequence dimension
        tokens = jnp.concatenate([cls_tokens, embedded_patches], axis=1)
        # tokens shape: (B, seq_len, embed_dim)  where seq_len = num_patches + 1

        # 4) Positional encoding
        # Our seq_len = self.num_patches + 1
        # We'll create a PositionalEncoding that matches that length exactly.
        pe = PositionalEncoding(
            seq_len=self.seq_len,
            embed_dim=self.embed_dim,
            name="vit_positional_encoding",
        )
        tokens = pe(tokens)  # shape (B, seq_len, embed_dim)

        # 5) Pass through TransformerEncoder blocks
        x = tokens
        for i in range(self.num_layers):
            block = TransformerEncoder(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                name=f"encoder_block_{i}",
            )
            x = block(x)  # (B, seq_len, embed_dim)

        # x[:, 0] is the CLS token representation
        cls_out = x[:, 0]  # shape (B, embed_dim)

        # Final classification head
        # We'll do a single linear (embed_dim -> num_classes)
        w_head = numpyro.sample(
            "head_w", dist.Normal(0, 1).expand([self.embed_dim, self.num_classes])
        )
        b_head = numpyro.sample("head_b", dist.Normal(0, 1).expand([self.num_classes]))
        logits = jnp.dot(cls_out, w_head) + b_head

        numpyro.deterministic("logits", logits)

        # Classification
        # If "task_type" is classification with 'num_classes', do:
        numpyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits
