import numpy as np
import matplotlib.pyplot as plt


def hoeffding_bound(empirical_mean, n, epsilon):
    """
    Compute the Hoeffding bound for a given empirical mean.

    Parameters:
    - empirical_mean: Observed mean.
    - n: Number of samples.
    - epsilon: Deviation from the mean.

    Returns:
    - bound: Probability of deviation.
    """
    return 2 * np.exp(-2 * n * epsilon**2)


# Visualization
if __name__ == "__main__":
    sample_sizes = np.arange(10, 1000, 10)
    epsilon = 0.1
    bounds = [
        hoeffding_bound(empirical_mean=0.5, n=n, epsilon=epsilon) for n in sample_sizes
    ]

    plt.figure(figsize=(8, 6))
    plt.plot(sample_sizes, bounds, label=f"Epsilon = {epsilon}")
    plt.xlabel("Number of Samples (n)")
    plt.ylabel("Hoeffding Bound")
    plt.title("Hoeffding's Inequality Visualization")
    plt.legend()
    plt.grid(True)
    plt.show()
