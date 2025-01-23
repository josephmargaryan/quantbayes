import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# 1) Define Generator & Discriminator
# -----------------------
class Generator(nn.Module):
    def __init__(self, noise_dim=16, hidden_dim=32, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# -----------------------
# 2) Training Loop
# -----------------------
def train_gan(
    generator, discriminator,
    data_loader,
    g_optimizer, d_optimizer,
    noise_dim=16, epochs=5,
    device='cpu'
):
    criterion = nn.BCELoss()

    generator.to(device)
    discriminator.to(device)

    for epoch in range(epochs):
        for real_data, in data_loader:
            real_data = real_data.to(device)

            batch_size = real_data.size(0)
            # ============ Train Discriminator ============
            # Step 1: real data -> label=1
            labels_real = torch.ones(batch_size, 1, device=device)
            preds_real = discriminator(real_data)
            loss_real = criterion(preds_real, labels_real)

            # Step 2: fake data -> label=0
            z = torch.randn(batch_size, noise_dim, device=device)
            fake_data = generator(z)
            labels_fake = torch.zeros(batch_size, 1, device=device)
            preds_fake = discriminator(fake_data.detach())
            loss_fake = criterion(preds_fake, labels_fake)

            d_loss = loss_real + loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ============ Train Generator ============
            z = torch.randn(batch_size, noise_dim, device=device)
            fake_data = generator(z)
            preds_fake = discriminator(fake_data)
            labels_gen = torch.ones(batch_size, 1, device=device)  # generator wants disc to say "1"
            g_loss = criterion(preds_fake, labels_gen)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")


# -----------------------
# 3) Demo on 1D random data
# -----------------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    # Generate a simple 1D dataset (e.g. Gaussian)
    real_distribution = torch.randn(1000, 1)

    dataset = TensorDataset(real_distribution)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize models
    G = Generator(noise_dim=16, hidden_dim=32, output_dim=1)
    D = Discriminator(input_dim=1, hidden_dim=32)

    # Optimizers
    g_optimizer = optim.Adam(G.parameters(), lr=1e-3)
    d_optimizer = optim.Adam(D.parameters(), lr=1e-3)

    # Train
    train_gan(G, D, dataloader, g_optimizer, d_optimizer, noise_dim=16, epochs=5)
