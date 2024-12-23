import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# Configure logging
logging.basicConfig(level=logging.INFO)


class StockTradingEnv(gym.Env):
    """Custom Environment for Stock Trading with Gymnasium."""

    def __init__(self, data, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_assets = initial_balance

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # [0: Sell, 1: Hold, 2: Buy]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(2,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(
            seed=seed
        )  # Call the parent class's reset method for seed handling
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_assets = self.initial_balance
        return self._get_observation(), {}

    def step(self, action):
        current_price = self.data["Close"].iloc[self.current_step]

        if action == 0:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0
        elif action == 2:  # Buy
            shares_bought = self.balance // current_price
            self.shares_held += shares_bought
            self.balance -= shares_bought * current_price

        self.current_step += 1
        self.total_assets = self.balance + self.shares_held * current_price
        reward = self.total_assets - self.initial_balance

        # Check termination conditions
        terminated = (
            self.current_step >= len(self.data) - 1
        )  # Episode ends when data is exhausted
        truncated = False  # No truncation logic in this implementation

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self):
        current_price = self.data["Close"].iloc[self.current_step]
        return np.array([self.balance, current_price])

    def render(self, mode="human"):
        print(
            f"Step: {self.current_step}, Balance: {self.balance}, Shares Held: {self.shares_held}, Total Assets: {self.total_assets}"
        )


class Visualizer:
    """Handles visualizations for the trading environment."""

    @staticmethod
    def plot_portfolio_value(portfolio_values):
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_values, label="Portfolio Value", color="blue")
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Time Steps")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def plot_stock_prices_with_actions(stock_prices, actions):
        plt.figure(figsize=(10, 6))
        plt.plot(stock_prices, label="Stock Price", color="black")

        # Highlight actions
        buy_steps = [i for i, a in enumerate(actions) if a == 2]  # Buy
        sell_steps = [i for i, a in enumerate(actions) if a == 0]  # Sell
        hold_steps = [i for i, a in enumerate(actions) if a == 1]  # Hold

        plt.scatter(
            buy_steps,
            [stock_prices[i] for i in buy_steps],
            color="green",
            label="Buy",
            marker="^",
        )
        plt.scatter(
            sell_steps,
            [stock_prices[i] for i in sell_steps],
            color="red",
            label="Sell",
            marker="v",
        )
        plt.title("Stock Prices and Actions")
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def plot_rewards_per_step(rewards_per_step):
        plt.figure(figsize=(10, 6))
        plt.plot(rewards_per_step, label="Reward Per Step", color="purple")
        plt.title("Reward Per Step")
        plt.xlabel("Time Steps")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def plot_cumulative_rewards(rewards_per_step):
        cumulative_rewards = np.cumsum(rewards_per_step)
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_rewards, label="Cumulative Reward", color="orange")
        plt.title("Cumulative Reward Over Time")
        plt.xlabel("Time Steps")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    import yfinance as yf

    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    data = stock.history(start="2020-01-01", interval="1d")[["Close"]]
    data = data.asfreq("B").dropna()  # Ensure frequency is set

    # Initialize Environment
    logging.info("Initializing Stock Trading Environment...")
    env = StockTradingEnv(data)

    # Train PPO Agent
    logging.info("Training PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Test the Agent
    logging.info("Testing the Agent...")

    # Initialize lists to store data for visualization
    portfolio_values = []
    stock_prices = []
    actions = []
    rewards_per_step = []

    obs, _ = env.reset()
    for _ in range(len(data) - 1):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # Track portfolio value and actions
        portfolio_values.append(env.total_assets)
        stock_prices.append(env.data["Close"].iloc[env.current_step])
        actions.append(action)
        rewards_per_step.append(reward)

        if terminated or truncated:
            break
        env.render()

    logging.info("Simulation Complete.")

    # Visualizations
    Visualizer.plot_portfolio_value(portfolio_values)
    Visualizer.plot_stock_prices_with_actions(stock_prices, actions)
    Visualizer.plot_rewards_per_step(rewards_per_step)
    Visualizer.plot_cumulative_rewards(rewards_per_step)
