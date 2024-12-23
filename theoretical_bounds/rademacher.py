import numpy as np
import matplotlib.pyplot as plt


def rademacher_bound(empirical_loss, rademacher_complexity, n, delta):
    """
    Compute the Rademacher complexity bound.

    Parameters:
    - empirical_loss: Empirical loss of the model.
    - rademacher_complexity: Complexity of the hypothesis class.
    - n: Number of samples.
    - delta: Probability of failure.

    Returns:
    - bound: Generalization error bound.
    """
    term = rademacher_complexity + np.sqrt(np.log(1 / delta) / (2 * n))
    bound = empirical_loss + term
    return bound


# Visualization
if __name__ == "__main__":
    rademacher_values = np.linspace(0.01, 0.1, 100)
    empirical_loss = 0.15
    n_samples = 200
    delta = 0.05
    bounds = [
        rademacher_bound(empirical_loss, rc, n_samples, delta)
        for rc in rademacher_values
    ]

    plt.figure(figsize=(8, 6))
    plt.plot(rademacher_values, bounds, label="Rademacher Bound")
    plt.xlabel("Rademacher Complexity")
    plt.ylabel("Generalization Error Bound")
    plt.title("Rademacher Complexity Bound Visualization")
    plt.legend()
    plt.grid(True)
    plt.show()
