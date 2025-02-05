import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    Splits the image into patches and embeds them.
    """

    def __init__(self, in_channels=3, embed_dim=256, patch_size=16):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        x: [B, in_channels, H, W]
        output: [B, num_patches, embed_dim]
        """
        x = self.conv(x)  # shape => [B, embed_dim, H/patch_size, W/patch_size]
        B, E, H, W = x.shape
        x = x.flatten(2)  # => [B, E, H*W]
        x = x.transpose(1, 2)  # => [B, H*W, E]
        return x, (H, W)


class PositionalEncoding(nn.Module):
    """
    Standard learnable 1D positional encoding for patches.
    """

    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))

    def forward(self, x):
        """
        x: [B, N, E]
        """
        N = x.size(1)  # number of patches
        return x + self.pos_embed[:, :N, :]


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, dim_feedforward=512, num_layers=4):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        """
        x: [B, N, E]
        """
        x = self.transformer_encoder(x)  # => [B, N, E]
        return x


class ViTSegmentation(nn.Module):
    """
    A very simplified Vision Transformer-based segmentation model:
    1. Patch embedding.
    2. Optional positional encoding.
    3. Transformer encoder.
    4. Reshape to 2D and apply a head for segmentation.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=2,
        patch_size=16,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        hidden_dim=512,
    ):
        super(ViTSegmentation, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.pos_embed = PositionalEncoding(embed_dim)
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dim_feedforward=hidden_dim,
            num_layers=num_layers,
        )
        # segmentation head: project from embed_dim to num_classes
        self.seg_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        # Keep track of features
        self.features = {}

    def forward(self, x):
        self.features = {}
        # 1) Patch embedding
        patches, (H, W) = self.patch_embed(x)  # [B, N, E], N = H*W
        self.features["patches"] = patches

        # 2) Positional encoding
        patches = self.pos_embed(patches)
        self.features["pos_encoded_patches"] = patches

        # 3) Transformer encoder
        encoded = self.encoder(patches)  # [B, N, E]
        self.features["transformer_output"] = encoded

        # 4) Reshape tokens back to 2D
        B = x.size(0)
        encoded_2d = encoded.transpose(1, 2).view(B, -1, H, W)  # => [B, E, H, W]
        self.features["encoded_2d"] = encoded_2d

        # 5) Segmentation head
        logits = self.seg_head(encoded_2d)  # => [B, num_classes, H, W]
        self.features["logits"] = logits

        # Optional upsampling to the original size if needed:
        # logits = nn.functional.interpolate(logits, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return logits


if __name__ == "__main__":
    # Simple test
    model = ViTSegmentation(
        in_channels=3,
        num_classes=2,
        patch_size=16,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        hidden_dim=512,
    )

    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    print("Output shape:", y.shape)
    for k, v in model.features.items():
        print(k, v.shape if isinstance(v, torch.Tensor) else v)
