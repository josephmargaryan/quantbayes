from hoeffding import hoeffding_bound
from vc import vc_bound
from pac_bayesian import pac_bayesian_bound
from rademacher import rademacher_bound
import numpy as np
import matplotlib.pyplot as plt


def test_stock_price_bounds():
    """
    Test bounds on a stock price prediction example.
    """
    y_true = np.random.choice([0, 1], size=1000)
    y_pred = np.random.choice([0, 1], size=1000)
    n_samples = len(y_true)
    empirical_loss = np.mean(y_true != y_pred)

    # Hoeffding's inequality
    epsilon = 0.05
    hoeffding = hoeffding_bound(empirical_loss, n_samples, epsilon)

    # VC-dimension bound
    d_vc = 50
    delta = 0.05
    vc = vc_bound(n_samples, d_vc, delta)

    # PAC-Bayesian bound
    kl_divergence = 0.1
    pac_bayes = pac_bayesian_bound(empirical_loss, kl_divergence, n_samples, delta)

    # Rademacher complexity bound
    rademacher_complexity = 0.01
    rademacher = rademacher_bound(
        empirical_loss, rademacher_complexity, n_samples, delta
    )

    results = {
        "Hoeffding": hoeffding,
        "VC": vc,
        "PAC-Bayesian": pac_bayes,
        "Rademacher": rademacher,
    }

    # Combined Visualization
    plt.figure(figsize=(12, 8))

    # Plot Hoeffding's bound
    bounds_hoeffding = [hoeffding_bound(0.5, n, 0.1) for n in range(10, 1000, 10)]
    plt.plot(range(10, 1000, 10), bounds_hoeffding, label="Hoeffding")

    # Plot VC-Dimension bound
    bounds_vc = [vc_bound(n, d_vc=10, delta=0.05) for n in range(10, 1000, 10)]
    plt.plot(range(10, 1000, 10), bounds_vc, label="VC-Dimension")

    # Add labels and legend
    plt.xlabel("Number of Samples (n)")
    plt.ylabel("Bound Value")
    plt.title("Comparison of Theoretical Bounds")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Results:", results)
    return results


if __name__ == "__main__":
    test_stock_price_bounds()
