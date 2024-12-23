import numpy as np
import matplotlib.pyplot as plt


def vc_bound(n, d_vc, delta):
    """
    Compute the VC-dimension generalization bound.

    Parameters:
    - n: Number of samples.
    - d_vc: VC-dimension of the hypothesis class.
    - delta: Probability of failure.

    Returns:
    - bound: Generalization error bound.
    """
    return np.sqrt((8 / n) * (d_vc * np.log(2 * n / d_vc) + np.log(4 / delta)))


# Visualization
if __name__ == "__main__":
    sample_sizes = np.arange(10, 1000, 10)
    d_vc = 10
    delta = 0.05
    bounds = [vc_bound(n, d_vc, delta) for n in sample_sizes]

    plt.figure(figsize=(8, 6))
    plt.plot(sample_sizes, bounds, label=f"VC-Dimension = {d_vc}")
    plt.xlabel("Number of Samples (n)")
    plt.ylabel("VC Bound")
    plt.title("VC-Dimension Generalization Bound Visualization")
    plt.legend()
    plt.grid(True)
    plt.show()
