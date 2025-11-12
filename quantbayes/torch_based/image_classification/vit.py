"""
vision_transformer.py

A self-contained script that implements:
1. A Vision Transformer from scratch in PyTorch.
2. Extraction of multi-head attention maps (shape [B, num_heads, N, N]).
3. A test routine (forward pass on random data).
4. An optional visualization function for attention (requires matplotlib).
"""

import torch
import torch.nn as nn

# For optional visualization
try:
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


class PatchEmbedding(nn.Module):
    """
    Splits the image into patches and embeds them using a Conv layer.
    By default:
        - patch_size=16
        - stride=16
    So for an image of size (H, W), you get (H/16)*(W/16) patches, each of dimension embed_dim.
    """

    def __init__(self, in_channels=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        x: [B, in_channels, H, W]
        return: [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape

        # Ensure H, W are divisible by patch_size for simplicity:
        assert H % self.patch_size == 0, "Input height not divisible by patch_size."
        assert W % self.patch_size == 0, "Input width not divisible by patch_size."

        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]

        return x


class LearnablePositionalEncoding(nn.Module):
    """
    Simple learnable 1D positional embeddings for each patch index.
    """

    def __init__(self, num_positions=196, embed_dim=768):
        """
        num_positions should be equal to number of patches (+1 if using a class token).
        """
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))

    def forward(self, x):
        """
        x: [B, N, embed_dim]
        """
        N = x.size(1)  # number of tokens
        return x + self.pos_embed[:, :N, :]


class MultiHeadSelfAttention(nn.Module):
    """
    A wrapper around nn.MultiheadAttention that:
     - Operates in [B, N, D] format (batch_first=True).
     - Stores the attention maps for later inspection.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        # IMPORTANT: We pass 'average_attn_weights=False' so we get shape [B, num_heads, N, N].
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        # Will store the attention weights here after each forward
        self.last_attn_map = None

    def forward(self, x):
        """
        x: [B, N, D]
        returns: output of attention (same shape as x)
        """
        # Self-attention: query=key=value=x
        attn_out, attn_weights = self.mha(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        # attn_out: [B, N, D]
        # attn_weights: [B, num_heads, N, N]

        # store for later
        self.last_attn_map = attn_weights  # shape [B, num_heads, N, N]
        return self.attn_dropout(attn_out)


class TransformerBlock(nn.Module):
    """
    A single Transformer block: MSA + MLP + skip/res connections + layer norms.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, N, D]
        # Attention + Residual
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        x = x + attn_out

        # FFN + Residual
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x


class VisionTransformer(nn.Module):
    """
    A full Vision Transformer for classification (or representation) with:
      - Patch embedding
      - (Optional) class token
      - Positional encoding
      - Stacked Transformer blocks
      - A final classification head
      - The ability to extract and store attention maps
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        use_cls_token=True,
    ):
        """
        img_size: input image dimension (assume square for simplicity).
        patch_size: size of each patch (assume square).
        in_channels: input channels (3 for RGB).
        num_classes: for classification head.
        embed_dim: dimension of patch embeddings.
        depth: number of Transformer blocks.
        num_heads: number of attention heads.
        mlp_ratio: ratio for hidden dimension in MLP w.r.t. embed_dim.
        dropout: dropout rate.
        use_cls_token: if True, use a [CLS] token for classification.
        """
        super().__init__()
        self.use_cls_token = use_cls_token

        # Calculate how many patches
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size."
        num_patches = (img_size // patch_size) * (img_size // patch_size)

        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)

        # Class token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # +1 for the class token
            num_positions = num_patches + 1
        else:
            self.cls_token = None
            num_positions = num_patches

        # Learnable 1D positional encoding
        self.pos_encoding = LearnablePositionalEncoding(num_positions, embed_dim)

        # Dropout after adding positional embedding
        self.pos_drop = nn.Dropout(dropout)

        # Stacked transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Where we store attention maps from each block
        self.attn_maps = []

    def forward(self, x):
        """
        Forward pass:
        1) Patchify + embed
        2) (Optional) prepend class token
        3) Add positional encodings
        4) Pass through each Transformer block (collect attention maps)
        5) Final norm + classification head (using the CLS token if present)
        """
        B = x.shape[0]
        self.attn_maps = []  # reset

        # [B, num_patches, embed_dim]
        x = self.patch_embed(x)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]

        # Add learnable positional embedding
        x = self.pos_encoding(x)
        x = self.pos_drop(x)

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)
            # Save the attention map from the block's MSA
            attn_map = block.attn.last_attn_map  # [B, num_heads, seq_len, seq_len]
            self.attn_maps.append(attn_map)

        # Final norm
        x = self.norm(x)

        # Classification using the first token if we have CLS, otherwise average
        if self.use_cls_token:
            cls_logits = x[:, 0]  # [B, embed_dim]
        else:
            cls_logits = x.mean(dim=1)

        # Final head
        logits = self.head(cls_logits)  # [B, num_classes]
        return logits

    def get_attention_maps(self):
        """
        Returns a list of attention maps from each block.
        Each element is shape [B, num_heads, seq_len, seq_len].
        """
        return self.attn_maps


def visualize_attention(attn_map, layer_idx=0, head_idx=0):
    """
    A helper function to visualize a single attention map using matplotlib.
    attn_map: Tensor of shape [B, num_heads, N, N]
              (e.g. from model.attn_maps[layer_idx])
    layer_idx: which layer's attention to visualize
    head_idx: which head to visualize
    """
    if not HAVE_MPL:
        print("Matplotlib not installed. Cannot visualize.")
        return

    # attn_map is 4D: [B, num_heads, N, N]
    # We'll pick the first batch element for display
    attn_for_display = attn_map[0, head_idx].detach().cpu().numpy()  # shape [N, N]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 5))
    plt.imshow(attn_for_display, cmap="viridis")
    plt.colorbar()
    plt.title(f"Layer {layer_idx}, Head {head_idx}")
    plt.show()


if __name__ == "__main__":
    # Quick test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionTransformer(
        img_size=128,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=64,  # smaller embed dim for a quicker test
        depth=4,  # fewer layers
        num_heads=4,  # 4 heads
        mlp_ratio=2.0,
        dropout=0.1,
        use_cls_token=True,
    ).to(device)

    x = torch.randn(2, 3, 128, 128).to(device)
    logits = model(x)
    print("Output shape:", logits.shape)  # [2, 10]

    attn_maps = model.get_attention_maps()
    print("Number of layers:", len(attn_maps))
    for i, amap in enumerate(attn_maps):
        print(f"Layer {i} attention map shape:", amap.shape)  # [B, num_heads, N, N]

    # Optional visualization for the first layer, first head
    if HAVE_MPL and len(attn_maps) > 0:
        visualize_attention(attn_maps[0], layer_idx=0, head_idx=0)
