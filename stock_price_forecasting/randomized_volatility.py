import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.stats import lognorm
from scipy.integrate import quad

# Configure logging
logging.basicConfig(level=logging.INFO)


class RandomizedVolatilityModel:
    def __init__(
        self,
        initial_price,
        time_horizon,
        steps,
        mu=0.05,
        kappa=2.0,
        theta=0.04,
        sigma=0.2,
    ):
        self.initial_price = initial_price
        self.time_horizon = time_horizon
        self.steps = steps
        self.mu = mu  # Drift rate
        self.kappa = kappa  # Mean-reversion rate
        self.theta = theta  # Long-term variance
        self.sigma = sigma  # Volatility of variance
        self.dt = time_horizon / steps

    def simulate_price(self, variance_distribution):
        """
        Simulate the price evolution with stochastic volatility and randomized initial variance.
        """
        logging.info("Sampling initial variance...")
        initial_variance = variance_distribution.rvs()

        logging.info(f"Sampled variance: {initial_variance}")
        prices = [self.initial_price]
        variances = [initial_variance]

        for _ in range(self.steps):
            z1 = np.random.normal()
            z2 = np.random.normal()

            # Correlate the two Brownian motions (if needed, introduce a correlation parameter)
            z2 = z1 + z2 * np.sqrt(1 - 0**2)

            variance = variances[-1]

            # Update variance using a mean-reverting process (e.g., Heston-style dynamics)
            d_variance = (
                self.kappa * (self.theta - variance) * self.dt
                + self.sigma * np.sqrt(variance) * np.sqrt(self.dt) * z2
            )
            variance = max(
                variance + d_variance, 0
            )  # Ensure variance remains non-negative

            # Update price
            dS = (
                self.mu * prices[-1] * self.dt
                + np.sqrt(variance) * prices[-1] * np.sqrt(self.dt) * z1
            )
            prices.append(prices[-1] + dS)

            variances.append(variance)

        self.prices = prices
        self.variances = variances
        return prices

    def plot_simulation(self):
        """Visualize the simulated price path and volatility."""
        time = np.linspace(0, self.time_horizon, self.steps + 1)

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time, self.prices, label="Simulated Price")
        plt.title("Simulated Stock Price")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.grid()
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(time, self.variances, label="Variance", color="orange")
        plt.title("Simulated Variance Dynamics")
        plt.xlabel("Time")
        plt.ylabel("Variance")
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    logging.info("Setting up the randomized volatility model...")

    # Define the lognormal distribution for variance
    mean_variance = 0.04
    std_dev = 0.01
    shape = np.sqrt(np.log(1 + (std_dev / mean_variance) ** 2))
    scale = mean_variance / np.sqrt(1 + (std_dev / mean_variance) ** 2)
    variance_distribution = lognorm(s=shape, scale=scale)

    # Initialize the model
    model = RandomizedVolatilityModel(
        initial_price=100, time_horizon=1, steps=252, mu=0.05
    )

    # Simulate and visualize the price path
    prices = model.simulate_price(variance_distribution)
    model.plot_simulation()
