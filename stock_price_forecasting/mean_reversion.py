import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


class MeanReversionModel:
    def __init__(self, data, mean_price=None, speed=0.1, volatility=0.2):
        self.data = data
        self.last_price = data["Close"].iloc[-1]
        self.mean_price = mean_price if mean_price else data["Close"].mean()
        self.speed = speed  # Speed of mean reversion
        self.volatility = volatility

    def simulate(self, steps=10, simulations=1000):
        dt = 1 / 252
        prices = np.zeros((steps + 1, simulations))
        prices[0] = self.last_price

        for t in range(1, steps + 1):
            z = np.random.normal(size=simulations)
            prices[t] = (
                prices[t - 1]
                + self.speed * (self.mean_price - prices[t - 1]) * dt
                + self.volatility * np.sqrt(dt) * z
            )

        return prices

    def plot_simulated_prices(self, prices, steps):
        plt.figure(figsize=(10, 6))
        plt.plot(prices[:, :50])  # Plot the first 50 paths
        plt.title(f"Mean Reversion Model: Simulated Stock Price Paths ({steps} steps)")
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    import yfinance as yf

    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    data = stock.history(start="2020-01-01", interval="1d")[["Close"]]
    data = data.asfreq("B").dropna()  # Ensure frequency is set

    # Mean Reversion Model
    logging.info("Initializing Mean Reversion Model...")
    mean_reversion_model = MeanReversionModel(data)
    prices = mean_reversion_model.simulate(steps=30, simulations=1000)

    # Plotting Results
    mean_reversion_model.plot_simulated_prices(prices, steps=30)
