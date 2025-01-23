import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -----------------------
# 1) Define Beta schedule
# -----------------------
def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    """
    Linear schedule for betas (noise intensities).
    """
    return torch.linspace(start, end, timesteps)

# -----------------------
# 2) Diffusion Utilities
# -----------------------
class Diffusion:
    """
    Implements forward noising process coefficients and inverse sampling.
    We'll store alpha, alpha_cumprod, etc.
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]], dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def noisy_sample(self, x0, t, device='cpu'):
        """
        q(x_t | x_0) = N(x_t; sqrt_alphas_cumprod[t]*x_0, (1 - a_cumprod[t])*I).
        Sample x_t given x_0 and a time step t.
        """
        batch_size = x0.size(0)
        noise = torch.randn_like(x0).to(device)
        sqrt_ac = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1,1)
        # broadcasting if needed
        return sqrt_ac * x0 + sqrt_1m_ac * noise

# -----------------------
# 3) The Model: predicts noise given (x_t, t)
# -----------------------
class SimpleDiffusionModel(nn.Module):
    """
    Very minimal MLP that tries to predict the noise in x_t for 1D data.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # (x_t, t_emb) -> hidden
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)   # predict the noise dimension = 1
        )

    def forward(self, x, t):
        """
        x: (batch_size, 1)
        t: (batch_size,) integer time steps
        We'll embed t in a trivial way (like t / T).
        """
        t_emb = t.float().unsqueeze(-1) / 1000.0  # naive embedding
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)


# -----------------------
# 4) Training Loop
# -----------------------
def train_diffusion(
    model, diffusion, data_loader, optimizer,
    timesteps=1000, device='cpu', epochs=5
):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for real_data, in data_loader:
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # sample a random t for each sample in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            # x_t from x_0
            x_noisy = []
            for i in range(batch_size):
                x_noisy.append(diffusion.noisy_sample(real_data[i:i+1], t[i], device=device))
            x_noisy = torch.cat(x_noisy, dim=0)  # shape (batch_size, 1)

            # model predicts the noise
            epsilon_pred = model(x_noisy, t)

            # ground truth noise = x_noisy - sqrt(a_cumprod[t]* x_0 ) ... 
            # but simpler is to sample the reference noise we used
            # Actually, let's do the standard L2: we know q(x_t|x_0),
            # so x_t = sqrt_ac[t]*x_0 + sqrt(1 - ac[t]) * eps
            # => eps = (x_t - sqrt_ac[t]*x_0) / sqrt(1 - ac[t])

            sqrt_ac = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)
            sqrt_1m_ac = diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            epsilon_target = (x_noisy - sqrt_ac * real_data) / sqrt_1m_ac

            loss = nn.functional.mse_loss(epsilon_pred, epsilon_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data_loader):.4f}")


# -----------------------
# 5) Demo on 1D data
# -----------------------
if __name__ == "__main__":
    # let's produce random 1D data from e.g. Gaussian(3, 1)
    import math

    N = 1000
    data = 3.0 + 1.0*torch.randn(N,1)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    diffusion = Diffusion(timesteps=1000, beta_start=1e-4, beta_end=0.02)
    model = SimpleDiffusionModel(hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_diffusion(model, diffusion, dataloader, optimizer, timesteps=1000, epochs=5)
