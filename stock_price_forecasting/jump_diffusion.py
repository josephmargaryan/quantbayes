import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


class JumpDiffusionModel:
    def __init__(
        self,
        data,
        drift=0.05,
        volatility=0.2,
        jump_intensity=0.1,
        jump_mean=0.01,
        jump_std=0.02,
    ):
        self.data = data
        self.last_price = data["Close"].iloc[-1]
        self.drift = drift
        self.volatility = volatility
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def simulate(self, steps=10, simulations=1000):
        dt = 1 / 252
        prices = np.zeros((steps + 1, simulations))
        prices[0] = self.last_price

        for t in range(1, steps + 1):
            # Standard GBM component
            z = np.random.normal(size=simulations)
            gbm = self.drift * dt + self.volatility * np.sqrt(dt) * z

            # Jump component
            jumps = np.random.poisson(self.jump_intensity * dt, simulations)
            jump_sizes = np.random.normal(self.jump_mean, self.jump_std, simulations)
            jump_effect = jumps * jump_sizes

            # Combine GBM and jumps
            prices[t] = prices[t - 1] * np.exp(gbm + jump_effect)

        return prices


def plot_simulated_prices(prices, steps):
    plt.figure(figsize=(10, 6))
    plt.plot(prices[:, :50])  # Plot the first 50 paths
    plt.title(f"Jump Diffusion Model: Simulated Stock Price Paths ({steps} steps)")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")

    # Add LaTeX formula to the plot
    formula = r"$dS_t = S_t (\mu dt + \sigma dW_t + J_t), \ J_t = \sum_{i=1}^{N_t} Y_i$"
    plt.text(
        0.02,
        0.95,
        formula,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.1),
    )

    plt.grid()
    plt.show()


if __name__ == "__main__":
    import yfinance as yf

    ticker = "NVO"
    stock = yf.Ticker(ticker)
    data = stock.history(start="2020-01-01", interval="1d")[["Close"]]
    data = data.asfreq("B").dropna()  # Ensure frequency is set

    # Jump Diffusion Model
    logging.info("Initializing Jump Diffusion Model...")
    jump_model = JumpDiffusionModel(data)
    prices = jump_model.simulate(steps=30, simulations=1000)

    # Plotting Results with Formula
    plot_simulated_prices(prices, steps=30)
