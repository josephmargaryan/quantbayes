import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------
# 1. Strided CNN Encoder / Decoder
# -------------------------------------------------------------------
class TOTEMEncoder(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size=4, stride=2):
        super().__init__()
        # Example: a single conv layer. Adjust as needed.
        self.conv = nn.Conv1d(
            in_channels,
            latent_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        # x: shape (N, in_channels, L)
        return self.conv(x)  # (N, latent_channels, L//stride approx)


class TOTEMDecoder(nn.Module):
    def __init__(self, latent_channels, out_channels, kernel_size=4, stride=2):
        super().__init__()
        self.tconv = nn.ConvTranspose1d(
            latent_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )

    def forward(self, z_q):
        # z_q: shape (N, latent_channels, L_enc)
        return self.tconv(z_q)  # (N, out_channels, L_dec approx)


# -------------------------------------------------------------------
# 2. VQ-VAE with codebook
# -------------------------------------------------------------------
class TOTEMVQVAE(nn.Module):
    def __init__(self, in_channels, latent_channels, num_embeddings, embedding_dim):
        """
        Args:
          in_channels: dimension of the raw time series
          latent_channels: dimension of the latent space
          num_embeddings: number of codebook entries
          embedding_dim: dimension of each codebook entry (should match latent_channels typically)
        """
        super().__init__()
        self.encoder = TOTEMEncoder(in_channels, latent_channels)
        # We'll rely on embedding_dim == latent_channels for simplicity
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.decoder = TOTEMDecoder(latent_channels, in_channels)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # If you prefer a simpler random init, you can do:
        # nn.init.normal_(self.codebook.weight, mean=0, std=1)

    def forward(self, x):
        """
        x: (N, in_channels, L)
        returns:
          x_recon, z_e, z_q, indices
        """
        # 1) encode
        z_e = self.encoder(x)  # (N, latent_channels, L_enc)

        # 2) quantize
        z_q, indices = self.quantize(z_e)  # same shape as z_e, plus indices

        # 3) decode
        x_recon = self.decoder(z_q)

        return x_recon, z_e, z_q, indices

    def quantize(self, z_e):
        """
        z_e: (N, latent_channels, L_enc)
        We'll flatten across (latent_channels, L_enc) as if each position is a 'token.'
        """
        N, C, L = z_e.shape
        # Flatten => shape [N*L, C]
        z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, C)  # shape (N*L, C)

        # compute distances to codebook entries
        # codebook.weight: shape (num_embeddings, embedding_dim = C)
        # => we want pairwise distance
        # dist[i, e] = || z_e_flat[i] - codebook[e] ||^2
        # We'll do a typical approach: expand or use torch.cdist / manual
        # For large codebooks, cdist can be expensive. For demonstration:
        codebook_weight = self.codebook.weight  # (E, C)
        # squared norms:
        z_e_sq = (z_e_flat**2).sum(dim=1, keepdim=True)  # (N*L, 1)
        cbook_sq = (codebook_weight**2).sum(dim=1)  # (E,)
        # z_e_flat @ codebook_weight.T => shape (N*L, E)
        # dist(i,e) = z_e_sq[i] + cbook_sq[e] - 2 * z_e_flat[i] dot codebook_weight[e]
        dist = (
            z_e_sq
            + cbook_sq.unsqueeze(0)
            - 2 * torch.matmul(z_e_flat, codebook_weight.t())
        )

        # Argmin
        indices = dist.argmin(dim=-1)  # (N*L,)
        # Gather embeddings
        z_q_flat = self.codebook(indices)  # (N*L, C)

        # Reshape back
        z_q = z_q_flat.view(N, L, C).permute(0, 2, 1).contiguous()  # (N, C, L)

        return z_q, indices.view(N, L)


# -------------------------------------------------------------------
# 3. Transformer for tokens
# -------------------------------------------------------------------
class TOTEMTransformer(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, dropout_p=0.1):
        super().__init__()
        # A simple TransformerEncoder with PyTorch's built-in modules
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout_p,
            activation="relu",
            batch_first=True,  # we pass (N, seq_len, embed_dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_linear = nn.Linear(embed_dim, 1)

    def forward(self, z_q):
        """
        z_q: shape (N, embed_dim, seq_len) or (N, C, L)
        We want to pass (N, seq_len, embed_dim) to the transformer with batch_first=True.
        So we transpose to get (N, L, C).
        Returns: shape (N, 1)
        """
        x = z_q.transpose(1, 2)  # (N, L, C)
        y = self.transformer(x)  # (N, L, C)
        # take last token
        last = y[:, -1, :]  # (N, C)
        out = self.final_linear(last)  # (N, 1)
        return out


# -------------------------------------------------------------------
# 4. TOTEM: end-to-end for forecasting
# -------------------------------------------------------------------
class TOTEM(nn.Module):
    def __init__(
        self,
        in_channels,
        latent_channels,
        num_embeddings,
        num_layers,
        num_heads,
        dropout_p=0.1,
    ):
        super().__init__()
        # VQ-VAE
        self.vqvae = TOTEMVQVAE(
            in_channels=in_channels,
            latent_channels=latent_channels,
            num_embeddings=num_embeddings,
            embedding_dim=latent_channels,
        )
        # Transformer
        self.transformer = TOTEMTransformer(
            embed_dim=latent_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_p=dropout_p,
        )

    def forward(self, x):
        """
        x: shape (N, in_channels, L) = e.g. (batch, 1, seq_len)
        returns: (N, 1)
        """
        # 1) run VQ-VAE
        x_recon, z_e, z_q, indices = self.vqvae(x)
        # 2) run Transformer -> final forecast
        out = self.transformer(z_q)  # shape (N,1)
        return out


# Example usage:
if __name__ == "__main__":
    batch_size = 8
    seq_len = 32
    in_channels = 1

    model = TOTEM(
        in_channels=in_channels,
        latent_channels=16,
        num_embeddings=128,
        num_layers=2,
        num_heads=2,
        dropout_p=0.1,
    )

    # Suppose an input of shape (N, in_channels, L)
    x = torch.randn(batch_size, in_channels, seq_len)
    preds = model(x)
    print(preds.shape)  # => [8, 1]
