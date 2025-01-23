import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# -----------------------
# 1) Define VAE networks
# -----------------------
class VAEEncoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        returns: mu, logvar each shape (batch_size, latent_dim)
        """
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=20, output_dim=784):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 400)
        self.fc_out = nn.Linear(400, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        """
        z: (batch_size, latent_dim)
        returns: reconstructed x of shape (batch_size, output_dim)
        """
        h = self.relu(self.fc(z))
        x_recon = self.sigmoid(self.fc_out(h))
        return x_recon


class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        """
        z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        """
        Returns: recon_x, mu, logvar
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


# -----------------------
# 2) Define VAE loss
# -----------------------
def vae_loss(x, x_recon, mu, logvar):
    """
    -ELBO = reconstruction_loss + KL
    Here, we'll do a simple MSE or BCE for reconstruction. 
    We'll use BCE if data is in [0,1].
    """
    bce = nn.functional.binary_cross_entropy(
        x_recon, x, reduction='sum'
    )
    # KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kl


# -----------------------
# 3) Example Training Loop
# -----------------------
def train_vae(model, data_loader, optimizer, epochs=5, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, in data_loader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch_x)
            loss = vae_loss(batch_x, x_recon, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader.dataset):.4f}")


# -----------------------
# 4) Demo on random data
# -----------------------
if __name__ == "__main__":
    # Generate random data (like MNIST flattened, in [0,1])
    # For real usage, load actual MNIST or another dataset.
    X = torch.rand(1000, 784)
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = VAE(input_dim=784, latent_dim=20)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_vae(model, dataloader, optimizer, epochs=5)
