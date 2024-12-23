import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


class HestonModel:
    def __init__(self, data, v0=0.04, theta=0.04, kappa=2.0, sigma=0.1, rho=-0.7):
        self.data = data
        self.last_price = data["Close"].iloc[-1]
        self.v0 = v0  # Initial variance
        self.theta = theta  # Long-term variance
        self.kappa = kappa  # Speed of mean reversion
        self.sigma = sigma  # Volatility of variance
        self.rho = rho  # Correlation between price and variance

    def simulate(self, steps=10, simulations=1000):
        dt = 1 / 252  # Daily time step
        prices = np.zeros((steps + 1, simulations))
        variances = np.zeros((steps + 1, simulations))
        prices[0] = self.last_price
        variances[0] = self.v0

        for t in range(1, steps + 1):
            # Correlated random shocks
            z1 = np.random.normal(size=simulations)
            z2 = np.random.normal(size=simulations)
            z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

            # Variance process
            variances[t] = np.abs(
                variances[t - 1]
                + self.kappa * (self.theta - variances[t - 1]) * dt
                + self.sigma * np.sqrt(variances[t - 1] * dt) * z2
            )

            # Price process
            prices[t] = prices[t - 1] * np.exp(
                -0.5 * variances[t - 1] * dt + np.sqrt(variances[t - 1] * dt) * z1
            )

        return prices, variances

    def plot_simulated_prices(self, prices, steps):
        plt.figure(figsize=(10, 6))
        plt.plot(prices[:, :50])  # Plot the first 50 paths
        plt.title(f"Heston Model: Simulated Stock Price Paths ({steps} steps)")
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    import yfinance as yf

    ticker = "NVO"
    stock = yf.Ticker(ticker)
    data = stock.history(start="2020-01-01", interval="1d")[["Close"]]
    data = data.asfreq("B").dropna()  # Ensure frequency is set

    # Heston Model
    logging.info("Initializing Heston Model...")
    heston_model = HestonModel(data)
    prices, variances = heston_model.simulate(steps=30, simulations=1000)

    # Plotting Results
    heston_model.plot_simulated_prices(prices, steps=30)
