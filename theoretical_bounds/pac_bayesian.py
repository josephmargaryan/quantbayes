import numpy as np
import matplotlib.pyplot as plt


def pac_bayesian_bound(empirical_loss, kl_divergence, n, delta):
    """
    Compute the PAC-Bayesian bound.

    Parameters:
    - empirical_loss: Empirical loss of the model.
    - kl_divergence: KL divergence between posterior and prior.
    - n: Number of samples.
    - delta: Probability of failure.

    Returns:
    - bound: Generalization error bound.
    """
    term = (kl_divergence + np.log(2 * np.sqrt(n) / delta)) / n
    bound = empirical_loss + np.sqrt(term / 2)
    return bound


# Visualization
if __name__ == "__main__":
    kl_values = np.linspace(0, 1, 100)
    n_samples = 100
    delta = 0.05
    empirical_loss = 0.1
    bounds = [
        pac_bayesian_bound(empirical_loss, kl, n_samples, delta) for kl in kl_values
    ]

    plt.figure(figsize=(8, 6))
    plt.plot(kl_values, bounds, label="PAC-Bayesian Bound")
    plt.xlabel("KL Divergence")
    plt.ylabel("Generalization Error Bound")
    plt.title("PAC-Bayesian Bound Visualization")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualization with varying empirical loss
    kl_values = np.linspace(0, 1, 100)
    empirical_losses = [0.05, 0.1, 0.2]  # Varying empirical loss
    n_samples = 100
    delta = 0.05

    plt.figure(figsize=(8, 6))
    for empirical_loss in empirical_losses:
        bounds = [
            pac_bayesian_bound(empirical_loss, kl, n_samples, delta) for kl in kl_values
        ]
        plt.plot(kl_values, bounds, label=f"Empirical Loss = {empirical_loss}")

    plt.xlabel("KL Divergence")
    plt.ylabel("Generalization Error Bound")
    plt.title("PAC-Bayesian Bound with Varying Empirical Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
