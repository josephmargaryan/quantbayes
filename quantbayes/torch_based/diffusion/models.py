# diffusion_lib/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


########################################
# 2) Vision Transformer for Image Diffusion
########################################


class VisionTransformerDiffusion(nn.Module):
    """
    A Vision Transformer for diffusion on images (inspired by ViT).
    """

    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, patch_size * patch_size * in_channels),
        )

    def forward(self, x, t):
        B = x.size(0)

        # Patchify
        x = self.patch_embedding(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add learnable position embeddings
        x = x + self.pos_embedding

        # Encode
        x = self.transformer(x)  # (B, num_patches, embed_dim)

        # MLP head to reconstruct patches
        x = self.mlp_head(x)  # (B, num_patches, patch_size^2 * in_channels)

        # Unpatchify
        h_patches = w_patches = int(math.sqrt(self.num_patches))
        x = x.view(
            B, h_patches, w_patches, self.patch_size, self.patch_size, self.in_channels
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.in_channels, self.img_size, self.img_size)

        return x


########################################
# 3) Transformers for Time Series
########################################


class TimeSeriesTransformer(nn.Module):
    """
    A transformer-based model for time-series diffusion.
    Input shape: (B, seq_len, input_dim)
    Output shape: same as input
    """

    def __init__(
        self,
        seq_len=100,
        input_dim=1,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.input_projection = nn.Linear(input_dim, embed_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(embed_dim, input_dim)

    def forward(self, x, t):
        # x: (B, seq_len, input_dim)
        x_embed = self.input_projection(x)  # (B, seq_len, embed_dim)
        x_enc = self.encoder(x_embed)  # (B, seq_len, embed_dim)
        out = self.output_projection(x_enc)  # (B, seq_len, input_dim)
        return out


########################################
# 4) Simple MLP or 1D CNN for Tabular Data
########################################


class TabularDiffusionModel(nn.Module):
    """
    If you have continuous columns + some categorical columns,
    you'll pass them in as separate Tensors. We'll embed the cat columns if available.
    """

    def __init__(
        self,
        continuous_dim,
        categorical_dims=None,  # e.g. [10, 20, 5] if 3 cat features
        hidden_dim=128,
        time_emb_dim=64,
        num_layers=4,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.categorical_dims = categorical_dims

        # Build embeddings if categorical
        self.cat_embeddings = nn.ModuleList()
        if categorical_dims is not None:
            for cat_size in categorical_dims:
                # You can choose an embed dimension or do some rule-of-thumb
                embed_dim = min(16, cat_size // 2 + 1)
                self.cat_embeddings.append(nn.Embedding(cat_size, embed_dim))
            cat_embed_dim = sum(e.embedding_dim for e in self.cat_embeddings)
        else:
            cat_embed_dim = 0

        # MLP in/out dimension = continuous_dim + cat_embed_dim
        input_dim = continuous_dim + cat_embed_dim

        # Create your main MLP
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(
            nn.Linear(hidden_dim, input_dim)
        )  # final output matches input dimension
        self.mlp = nn.Sequential(*layers)

        # Time embedding
        self.time_emb = nn.Linear(time_emb_dim, hidden_dim)

    def forward(self, cont_x, cat_x, t):
        """
        cont_x: (B, continuous_dim)
        cat_x: (B, num_cat_features) [integers], or None
        t: (B,) discrete steps or an already embedded vector
        """
        # If we have cat data, embed each column
        if self.categorical_dims is not None and cat_x is not None:
            cat_emb_list = []
            for i, emb_layer in enumerate(self.cat_embeddings):
                # cat_x[:, i] is the i-th categorical column
                cat_emb = emb_layer(cat_x[:, i])
                cat_emb_list.append(cat_emb)
            # Concatenate
            cat_emb_concat = torch.cat(
                cat_emb_list, dim=-1
            )  # shape (B, sum_of_embed_dims)
            x = torch.cat(
                [cont_x, cat_emb_concat], dim=-1
            )  # final shape (B, input_dim)
        else:
            # Only continuous
            x = cont_x

        # Time embedding
        if t.dim() == 1:
            t_emb = self.time_emb(self.sinusoidal_embedding(t, self.time_emb_dim))
        else:
            t_emb = self.time_emb(t)

        # Condition by simply adding or concatenating
        # For example, let's do a naive addition:
        x = x + t_emb

        # Pass through MLP
        out = self.mlp(x)
        return out

    def sinusoidal_embedding(self, t, dim):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(0, half_dim, device=t.device).float()
            / half_dim
        )
        freqs = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb
