import numpy as np
import matplotlib.pyplot as plt


# Target distribution: Mixture of two Gaussians
def p(x):
    return 0.3 * np.exp(-0.5 * ((x + 2) / 0.5) ** 2) / (
        0.5 * np.sqrt(2 * np.pi)
    ) + 0.7 * np.exp(-0.5 * ((x - 3) / 1.0) ** 2) / (1.0 * np.sqrt(2 * np.pi))


# Metropolis-Hastings MCMC
def metropolis_hastings(p, n_samples=10000, proposal_std=1.0):
    samples = []
    x = 0  # Start at x = 0
    for _ in range(n_samples):
        x_new = x + np.random.normal(0, proposal_std)  # Propose a new state
        acceptance_ratio = p(x_new) / p(x)  # Acceptance ratio
        if np.random.rand() < acceptance_ratio:
            x = x_new  # Accept the proposal
        samples.append(x)
    return np.array(samples)


# Run MCMC
samples = metropolis_hastings(p, n_samples=10000, proposal_std=1.0)

# Plot results
x = np.linspace(-5, 7, 1000)
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.6, label="MCMC Samples")
plt.plot(x, p(x), label="Target Distribution (p(x))", linewidth=2)
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.title("Sampling from a Bimodal Distribution using MCMC")
plt.show()
