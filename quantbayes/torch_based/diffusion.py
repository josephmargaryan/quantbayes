import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# -----------------------
# 1) Define Beta Schedule and Diffusion Class
# -----------------------
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion:
    """
    Diffusion process for images. The forward (q) process gradually adds Gaussian noise,
    and the reverse (p) process denoises the image.
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]], dim=0
        )

        # For images, we will later reshape these scalars to (batch, 1, 1, 1)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1 - self.alphas_cumprod + 1e-8)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior (reverse) process parameters
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat(
                [self.posterior_variance[1:2], self.posterior_variance[1:]], dim=0
            )
            + 1e-8
        )
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x0, t, noise=None):
        """
        Sample x_t ~ q(x_t|x0) for image x0 given timestep t.
        Here, t is a tensor of indices (shape: (batch,)).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        device = x0.device
        # Reshape coefficients to broadcast over images:
        sqrt_ac = self.sqrt_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_1m_ac * noise

    def q_posterior_mean_variance(self, x0, x_t, t):
        """
        Compute q(x_{t-1} | x_t, x0) parameters.
        """
        device = x0.device
        coef1 = self.posterior_mean_coef1[t].to(device).view(-1, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].to(device).view(-1, 1, 1, 1)
        posterior_mean = coef1 * x0 + coef2 * x_t
        variance = self.posterior_variance[t].to(device).view(-1, 1, 1, 1)
        log_variance = (
            self.posterior_log_variance_clipped[t].to(device).view(-1, 1, 1, 1)
        )
        return posterior_mean, variance, log_variance

    def p_mean_variance(self, model, x, t, clip_denoised=True, model_kwargs=None):
        """
        Given x_t and timestep t, use the model to predict noise and compute
        the reverse process mean and variance.
        """
        if model_kwargs is None:
            model_kwargs = {}
        # The model predicts the noise
        eps = model(x, t, **model_kwargs)
        device = x.device
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        x0_pred = sqrt_recip * x - sqrt_recipm1 * eps
        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        posterior_mean, posterior_variance, posterior_log_variance = (
            self.q_posterior_mean_variance(x0_pred, x, t)
        )
        return {
            "mean": posterior_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_x0": x0_pred,
        }

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None):
        """
        Sample x_{t-1} from p(x_{t-1}|x_t) using the reverse process.
        """
        out = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)  # no noise when t == 0
        # Ensure log_variance has same dimensions as x
        log_var = out["log_variance"]
        if log_var.ndim == 3:
            log_var = log_var.unsqueeze(1)
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * log_var) * noise
        return {"sample": sample, "pred_x0": out["pred_x0"]}

    def p_sample_loop(self, model, shape, device):
        """
        Iteratively sample from the reverse process starting from noise.
        """
        x = torch.randn(shape, device=device)
        for t_step in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t_step, dtype=torch.long, device=device)
            out = self.p_sample(model, x, t_tensor)
            x = out["sample"].view(shape)
        return x

    def training_losses(self, model, x0, t, model_kwargs=None, noise=None):
        """
        Compute MSE loss between predicted noise and the actual noise used to
        produce x_t from x0.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)
        model_output = model(x_t, t, **(model_kwargs or {}))
        return F.mse_loss(model_output, noise)


# -----------------------
# 2) Define a Simple Convolutional Diffusion Model for MNIST
# -----------------------
class SimpleCNN(nn.Module):
    """
    A small CNN that takes an image and a naive time embedding and predicts noise.
    The input image shape is (batch, 1, 28, 28) and the model outputs noise of the same shape.
    """

    def __init__(self, hidden_channels=32):
        super(SimpleCNN, self).__init__()
        # We'll embed t as an extra channel (after scaling)
        # The input will then have 2 channels (1 for the image and 1 for the time embedding)
        self.conv1 = nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.conv_out = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        x: (batch, 1, 28, 28) noisy images
        t: (batch,) timesteps (integers)
        We create a naive time embedding by normalizing t and replicating it to match spatial dimensions.
        """
        # Normalize t (assume timesteps=1000)
        t_emb = (
            t.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )  # shape (batch, 1, 1, 1)
        t_emb = t_emb / 1000.0
        t_emb = t_emb.expand(
            x.size(0), 1, x.size(2), x.size(3)
        )  # shape (batch, 1, 28, 28)
        # Concatenate the image and time embedding along the channel dimension.
        inp = torch.cat([x, t_emb], dim=1)  # shape (batch, 2, 28, 28)
        h = F.relu(self.conv1(inp))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        out = self.conv_out(h)  # shape (batch, 1, 28, 28)
        return out


# -----------------------
# 3) Training Loop for MNIST
# -----------------------
def train_diffusion(
    model, diffusion, data_loader, optimizer, timesteps=1000, device="cpu", epochs=5
):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for real_data, _ in data_loader:
            # real_data: (batch, 1, 28, 28), in range [0,1]
            # Scale to [-1,1]
            real_data = real_data.to(device) * 2 - 1
            batch_size = real_data.size(0)
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            loss = diffusion.training_losses(model, real_data, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data_loader):.4f}")


# -----------------------
# 4) Main: Load MNIST, Train, and Sample
# -----------------------
if __name__ == "__main__":
    # Load MNIST dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # images in [0,1]
        ]
    )
    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    timesteps = 200  # using a smaller number for speed
    diffusion = GaussianDiffusion(timesteps=timesteps, beta_start=1e-4, beta_end=0.02)
    model = SimpleCNN(hidden_channels=32)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training diffusion model on MNIST...")
    train_diffusion(
        model,
        diffusion,
        train_loader,
        optimizer,
        timesteps=timesteps,
        device=device,
        epochs=5,
    )

    print("Generating samples...")
    samples = diffusion.p_sample_loop(model, shape=(64, 1, 28, 28), device=device)
    # Scale samples back from [-1,1] to [0,1]
    samples = (samples + 1) / 2.0
    samples = samples.clamp(0, 1).detach().cpu()

    # Plot a grid of generated samples
    grid = np.transpose(
        torchvision.utils.make_grid(samples, nrow=8, padding=2, normalize=True).numpy(),
        (1, 2, 0),
    )
    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.axis("off")
    plt.title("Generated MNIST Samples")
    plt.show()
