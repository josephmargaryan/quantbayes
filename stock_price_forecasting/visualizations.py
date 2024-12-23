import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class StockPlotter:
    def __init__(self, X, y, ticker):
        """
        Initialize the StockPlotter class.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable (stock prices).
            ticker (str): Stock ticker symbol (e.g., 'AAPL').
        """
        self.X = X
        self.y = y
        self.ticker = ticker

        if not isinstance(self.X.index, pd.DatetimeIndex) or not isinstance(
            self.y.index, pd.DatetimeIndex
        ):
            raise ValueError("The index of X and y must be a DatetimeIndex.")

        self.timezone = self.y.index.tz  # Capture timezone of the data
        print(f"StockPlotter initialized for {self.ticker}.")

    def filter_data(self, days):
        """
        Filter the data for the last 'days' days.

        Args:
            days (int): Number of days to filter.

        Returns:
            pd.Series: Filtered target (closing prices).
        """
        cutoff_date = datetime.now(self.timezone) - timedelta(
            days=days
        )  # Ensure timezone-aware
        filtered_y = self.y[self.y.index >= cutoff_date]
        return filtered_y

    def plot_data(self, data, title, xlabel="Date", ylabel="Price", figsize=(14, 7)):
        """
        Plot the stock prices.

        Args:
            data (pd.Series): Data to plot.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            figsize (tuple): Figure size for the plot.
        """
        if data.empty:
            print(f"No data available for {title}.")
            return

        plt.figure(figsize=figsize)
        plt.plot(
            data.index,
            data.values,
            label=f"{self.ticker} Stock Price",
            color="blue",
            linewidth=2,
        )
        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.show()

    def plot_last_week(self):
        """Plot stock prices for the last week."""
        data = self.filter_data(7)
        self.plot_data(data, title=f"{self.ticker} - Stock Price (Last Week)")

    def plot_last_month(self):
        """Plot stock prices for the last month."""
        data = self.filter_data(30)
        self.plot_data(data, title=f"{self.ticker} - Stock Price (Last Month)")

    def plot_last_six_months(self):
        """Plot stock prices for the last six months."""
        data = self.filter_data(180)
        self.plot_data(data, title=f"{self.ticker} - Stock Price (Last Six Months)")


if __name__ == "__main__":

    from get_data import StockData

    stock_data = StockData(ticker="NVO", start_date="2023-01-01", interval="1h")

    X, y = stock_data.get_ml_ready_data()
    stock_plotter = StockPlotter(X=X, y=y, ticker="NVO")

    stock_plotter.plot_last_week()
    stock_plotter.plot_last_month()
    stock_plotter.plot_last_six_months()
