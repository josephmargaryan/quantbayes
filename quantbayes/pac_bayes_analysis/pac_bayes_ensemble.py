import torch
import torch.nn.functional as F
import torch.optim as optim

# Assume we have M pre-trained models (here, for simplicity, we simulate their empirical losses)
M = 10
# Example: random empirical losses for each model on a validation set
empirical_losses = torch.rand(M)  # In practice, compute these on your validation data

# Use a uniform prior over models
prior = torch.ones(M) / M  # π(h_i) = 1/M

# We will optimize unnormalized log-weights 'z' so that ρ = softmax(z)
z = torch.randn(M, requires_grad=True)
optimizer = optim.Adam([z], lr=0.01)
lam = 0.1  # Trade-off parameter


def kl_divergence(softmax_z, prior):
    # KL divergence between discrete distributions ρ and π
    return (softmax_z * (torch.log(softmax_z + 1e-12) - torch.log(prior + 1e-12))).sum()


num_iterations = 1000
for iteration in range(num_iterations):
    optimizer.zero_grad()
    # Compute ensemble weights using softmax: ρ = softmax(z)
    rho = F.softmax(z, dim=0)
    # Ensemble empirical loss: weighted sum over empirical losses
    ens_loss = (rho * empirical_losses).sum()
    # KL divergence between ρ and the uniform prior
    kl = kl_divergence(rho, prior)
    # Overall objective: empirical loss + λ * KL divergence
    loss = ens_loss + lam * kl
    loss.backward()
    optimizer.step()

    if (iteration + 1) % 100 == 0:
        print(
            f"Iteration {iteration + 1}, Ensemble Loss: {ens_loss.item():.4f}, KL: {kl.item():.4f}, Total: {loss.item():.4f}"
        )

# Final ensemble weights:
rho_final = F.softmax(z, dim=0).detach()
print("Final ensemble weights:", rho_final)
