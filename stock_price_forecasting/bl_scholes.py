import numpy as np
import logging
import matplotlib.pyplot as plt

#######################################
####Geometric Brownian Motion (GBM)####
#######################################


logging.basicConfig(level=logging.INFO)


class BlackScholesModel:
    def __init__(self, data):
        self.data = data
        self.last_price = data["Close"].iloc[-1]
        self.log_returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()

    def calculate_parameters(self):
        """Calculate drift and volatility based on historical data."""
        logging.info("Calculating drift and volatility...")
        self.drift = self.log_returns.mean() - (0.5 * self.log_returns.var())
        self.volatility = self.log_returns.std()

    def simulate_future_prices(self, steps=10, simulations=1000):
        """Simulate future stock prices using Geometric Brownian Motion."""
        logging.info("Simulating future stock prices...")
        dt = 1  # Assuming daily steps
        random_shocks = np.random.normal(0, 1, (steps, simulations))
        price_paths = np.zeros((steps + 1, simulations))
        price_paths[0] = self.last_price

        for t in range(1, steps + 1):
            price_paths[t] = price_paths[t - 1] * np.exp(
                self.drift * dt + self.volatility * np.sqrt(dt) * random_shocks[t - 1]
            )

        self.price_paths = price_paths
        return price_paths

    def plot_simulated_prices(self, steps=10):
        """Visualize simulated price paths."""
        logging.info("Plotting simulated price paths...")
        plt.figure(figsize=(10, 6))
        plt.plot(self.price_paths[:, :50])
        plt.title(f"Simulated Stock Price Paths ({steps} steps)")
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.grid()
        plt.show()

    def plot_probability_distribution(self, step):
        """Visualize the probability distribution of prices at a given step."""
        logging.info(f"Plotting price distribution at step {step}...")
        plt.figure(figsize=(10, 6))
        plt.hist(self.price_paths[step], bins=50, alpha=0.7, color="blue")
        plt.title(f"Probability Distribution of Prices at Step {step}")
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    import yfinance as yf

    ticker = "NVO"
    stock = yf.Ticker(ticker)
    data = stock.history(start="2020-01-01", interval="1d")[["Close"]]
    data = data.asfreq("B").dropna()

    bs_model = BlackScholesModel(data)
    bs_model.calculate_parameters()
    simulated_prices = bs_model.simulate_future_prices(steps=30, simulations=1000)

    bs_model.plot_simulated_prices(steps=30)
    bs_model.plot_probability_distribution(step=30)
