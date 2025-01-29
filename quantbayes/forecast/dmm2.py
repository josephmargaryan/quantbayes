import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For reproducibility
torch.manual_seed(0)
np.random.seed(0)


def generate_synthetic_data(T=1000, num_sequences=10):
    """Generate a synthetic dataset of time-series of length T."""
    all_x = []
    all_z = []

    for _ in range(num_sequences):
        z = torch.zeros(T)
        x = torch.zeros(T)

        # Initialize z_0
        z[0] = torch.randn(1) * 0.1
        x[0] = z[0] + 0.05 * torch.randn(1)

        for t in range(1, T):
            z[t] = 0.9 * z[t - 1] + 0.1 * torch.randn(1)  # linear transition with noise
            x[t] = z[t] + 0.05 * torch.randn(1)  # emission with noise

        all_x.append(x)
        all_z.append(z)

    # Shape: (num_sequences, T)
    x_data = torch.stack(all_x, dim=0)
    z_data = torch.stack(all_z, dim=0)

    return x_data, z_data


T = 100  # Length of each sequence
N = 10  # Number of sequences
x_data, z_data = generate_synthetic_data(T=T, num_sequences=N)

# We will use x_data as our observations
print(f"x_data shape = {x_data.shape}, z_data shape = {z_data.shape}")


class TransitionModel(nn.Module):
    def __init__(self, z_dim=1, hidden_dim=16):
        super().__init__()
        self.rnn = nn.LSTM(z_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2 * z_dim)

    def forward(self, z_prev):
        out, _ = self.rnn(z_prev.unsqueeze(1))  # (B, 1, hidden_dim)
        mu, log_var = self.fc(out.squeeze(1)).chunk(2, dim=-1)
        return mu, log_var


class EmissionModel(nn.Module):
    """
    p(x_t | z_t): outputs mean and log-variance of x_t given z_t.
    """

    def __init__(self, z_dim=1, hidden_dim=16, x_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * x_dim),  # output: [mean, log_var] of x_t
        )

    def forward(self, z_t):
        """Compute mean and log-variance for x_t given z_t."""
        out = self.net(z_t)
        mu, log_var = out.chunk(2, dim=-1)
        return mu, log_var


class InferenceModel(nn.Module):
    """
    q(z_t | x_t, z_{t-1}): outputs mean and log-variance of z_t given x_t and z_{t-1}.
    """

    def __init__(self, z_dim=1, x_dim=1, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * z_dim),  # output: [mean, log_var]
        )

    def forward(self, x_t, z_prev):
        """Compute mean and log-variance for z_t given (x_t, z_{t-1})."""
        inp = torch.cat([x_t, z_prev], dim=-1)
        out = self.net(inp)
        mu, log_var = out.chunk(2, dim=-1)
        return mu, log_var


class DeepMarkovModel(nn.Module):
    def __init__(self, x_dim=1, z_dim=1, hidden_dim=16):
        super().__init__()
        self.z_dim = z_dim

        self.transition_model = TransitionModel(z_dim=z_dim, hidden_dim=hidden_dim)
        self.emission_model = EmissionModel(
            z_dim=z_dim, hidden_dim=hidden_dim, x_dim=x_dim
        )
        self.inference_model = InferenceModel(
            z_dim=z_dim, x_dim=x_dim, hidden_dim=hidden_dim
        )

        # We will assume the prior for z_1 ~ N(0, I).
        # Alternatively, you could learn an initial state distribution.

    def reparameterize(self, mu, log_var):
        """Reparameterization trick: z = mu + sigma * eps."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        """
        Perform a forward pass over a single sequence x of shape (T, x_dim).
        Returns the negative ELBO for that sequence.

        x: (T, x_dim)
        """
        T = x.size(0)

        # We'll store kl_divergences and log_likelihoods for each time step
        kld_list = []
        log_likelihood_list = []

        # Initialize z_prev ~ N(0, I)
        # We'll treat z_0 as a fixed zero. Another approach might be to learn it or sample from N(0,I).
        z_prev = torch.zeros((1, self.z_dim), device=x.device)

        for t in range(T):
            # =========== Inference (q) ===============
            # q(z_t | x_t, z_{t-1})
            mu_q, log_var_q = self.inference_model(x[t].unsqueeze(0), z_prev)
            z_t = self.reparameterize(mu_q, log_var_q)  # sample from q

            # =========== Prior / Transition ================
            if t == 0:
                # prior for z_1 is standard normal
                mu_p = torch.zeros_like(mu_q)
                log_var_p = torch.zeros_like(log_var_q)
            else:
                mu_p, log_var_p = self.transition_model(z_prev)

            # =========== Emission ===============
            # p(x_t | z_t)
            mu_x, log_var_x = self.emission_model(z_t)

            # =========== Compute Terms for ELBO ================
            # KL[q(z_t) || p(z_t|z_{t-1})] = KL(N(mu_q, var_q) || N(mu_p, var_p))
            #   = 0.5 * sum( log_var_p - log_var_q + (var_q + (mu_q - mu_p)^2)/var_p - 1 )

            # For stable computation, we often do kl_div_elementwise and sum over z-dim.
            var_q = torch.exp(log_var_q)
            var_p = torch.exp(log_var_p)

            kld = 0.5 * torch.sum(
                log_var_p - log_var_q + var_q / var_p + (mu_q - mu_p) ** 2 / var_p - 1,
                dim=-1,
            )

            # log p(x_t | z_t) = -0.5 * [ log(2*pi) + log_var_x + (x_t - mu_x)^2 / var_x ]
            var_x = torch.exp(log_var_x)
            log_likelihood = -0.5 * torch.sum(
                np.log(2 * np.pi) + log_var_x + (x[t].unsqueeze(0) - mu_x) ** 2 / var_x,
                dim=-1,
            )

            kld_list.append(kld)
            log_likelihood_list.append(log_likelihood)

            # Prepare for next time step
            z_prev = z_t

        # Sum over time
        kld_total = torch.sum(torch.cat(kld_list))
        log_likelihood_total = torch.sum(torch.cat(log_likelihood_list))

        # Negative ELBO = KL - E[ log p(x|z) ]
        nelbo = kld_total - log_likelihood_total

        return nelbo, kld_total, -log_likelihood_total


# Hyperparameters
x_dim = 1
z_dim = 1
hidden_dim = 32
lr = 1e-3
num_epochs = 200

model = DeepMarkovModel(x_dim=x_dim, z_dim=z_dim, hidden_dim=hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

x_data = x_data.to(device)  # shape (N, T)
# We won't use a separate val/test set for this demo, but in practice, you should!

losses = []

for epoch in range(num_epochs):
    epoch_nelbo = 0.0

    # We'll do a simple full-batch approach (all sequences at once)
    # For each sequence, we compute the loss and sum.
    # Or you can sum across all sequences in a single pass if you handle
    # them in parallel with batch dimension. We'll do it in a loop for clarity.

    optimizer.zero_grad()
    total_nelbo = 0.0

    for n in range(N):
        x_seq = x_data[n]  # shape (T,)
        # Make sure it is (T, x_dim)
        x_seq = x_seq.unsqueeze(-1)

        nelbo, kld, nll = model(x_seq)
        total_nelbo += nelbo

    # Average or sum
    total_nelbo = total_nelbo / N

    total_nelbo.backward()
    optimizer.step()

    losses.append(total_nelbo.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, NELBO = {total_nelbo.item():.4f}")

# Plot training loss curve
plt.figure(figsize=(6, 4))
plt.plot(losses, label="Training NELBO")
plt.xlabel("Epoch")
plt.ylabel("NELBO")
plt.title("Deep Markov Model Training")
plt.legend()
plt.show()

# Let's pick one sequence
seq_id = 0
x_seq = x_data[seq_id].unsqueeze(-1)  # shape (T, 1)

model.eval()
with torch.no_grad():
    T_ = x_seq.size(0)
    z_prev = torch.zeros((1, z_dim), device=device)

    recon = []
    for t in range(T_):
        mu_q, log_var_q = model.inference_model(x_seq[t].unsqueeze(0), z_prev)
        z_t = mu_q  # use the mean for reconstruction
        mu_x, log_var_x = model.emission_model(z_t)
        recon.append(mu_x.squeeze(0).cpu().item())
        z_prev = z_t

recon = np.array(recon)

# Plot original vs reconstruction
plt.figure(figsize=(6, 4))
plt.plot(x_seq.cpu().numpy(), label="Observed x")
plt.plot(recon, label="Reconstructed x", linestyle="--")
plt.legend()
plt.title("DMM Reconstruction on One Sequence")
plt.show()
