import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class StockDataProcessor:
    def __init__(self, ticker, start_date, interval="1d"):
        """
        Initialize the StockDataProcessor class.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL').
            start_date (str): Start date for fetching data (e.g., '2023-01-01').
            interval (str): Data interval (e.g., '1h', '1d').
        """
        self.ticker = ticker
        self.start_date = start_date
        self.interval = interval
        self.data = None
        self.scaler = MinMaxScaler()

        # Fetch and preprocess the data
        self.fetch_data()
        self.preprocess_pipeline()

    def fetch_data(self):
        """Fetch historical stock data from Yahoo Finance."""
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, interval=self.interval)

            if self.data.empty:
                raise ValueError(
                    f"No data found for ticker '{self.ticker}' with the given parameters."
                )

            self.data = self.data[["Close"]]
            self.data.index = pd.to_datetime(self.data.index)
            print(f"Fetched data for {self.ticker}: {len(self.data)} rows")
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            self.data = None

    def preprocess_pipeline(self):
        """Execute the full preprocessing pipeline."""
        if self.data is not None:
            self.add_returns()
            self.add_features()
            self.add_rsi()
            self.add_macd()
            self.add_bollinger_bands()
            self.handle_missing_values()
            self.handle_outliers(column="Close")
            self.normalize_data()
            print(f"Preprocessing completed for {self.ticker}.")
        else:
            print("No data available to preprocess.")

    def add_returns(self):
        """Add percentage returns as a feature."""
        self.data["Return"] = self.data["Close"].pct_change()
        self.data.dropna(inplace=True)

    def add_features(self):
        """Add basic moving averages and volatility features."""
        self.data["MA_10"] = self.data["Close"].rolling(window=10).mean()
        self.data["MA_50"] = self.data["Close"].rolling(window=50).mean()
        self.data["Volatility"] = self.data["Close"].rolling(window=10).std()

    def add_rsi(self, period=14):
        """Add Relative Strength Index (RSI) feature."""
        delta = self.data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        self.data["RSI"] = 100 - (100 / (1 + rs))

    def add_macd(self, short_window=12, long_window=26, signal_window=9):
        """Add Moving Average Convergence Divergence (MACD) feature."""
        self.data["EMA_12"] = (
            self.data["Close"].ewm(span=short_window, adjust=False).mean()
        )
        self.data["EMA_26"] = (
            self.data["Close"].ewm(span=long_window, adjust=False).mean()
        )
        self.data["MACD"] = self.data["EMA_12"] - self.data["EMA_26"]
        self.data["Signal_Line"] = (
            self.data["MACD"].ewm(span=signal_window, adjust=False).mean()
        )

    def add_bollinger_bands(self, window=20):
        """Add Bollinger Bands features."""
        self.data["BB_Middle"] = self.data["Close"].rolling(window=window).mean()
        self.data["BB_Upper"] = (
            self.data["BB_Middle"] + 2 * self.data["Close"].rolling(window=window).std()
        )
        self.data["BB_Lower"] = (
            self.data["BB_Middle"] - 2 * self.data["Close"].rolling(window=window).std()
        )

    def handle_missing_values(self):
        """Handle missing values by dropping them."""
        self.data.dropna(inplace=True)

    def handle_outliers(self, column, lower_quantile=0.01, upper_quantile=0.99):
        """Clip extreme values to handle outliers."""
        q_low = self.data[column].quantile(lower_quantile)
        q_high = self.data[column].quantile(upper_quantile)
        self.data[column] = np.clip(self.data[column], q_low, q_high)

    def normalize_data(self):
        """Normalize data using Min-Max scaling."""
        scaled_data = self.scaler.fit_transform(self.data)
        self.data = pd.DataFrame(
            scaled_data, index=self.data.index, columns=self.data.columns
        )

    def get_ml_ready_data(self, seq_len=30):
        """
        Get ML-ready data: features (X) and target (y) with sequence data.

        Args:
            seq_len (int): Number of past time steps (sequence length) for each sample.

        Returns:
            tuple: (X, y) where X is the feature matrix with shape (num_samples, seq_len, input_dim)
                and y is the target variable with shape (num_samples,).
        """
        features = [
            "MA_10",
            "MA_50",
            "Volatility",
            "RSI",
            "MACD",
            "Signal_Line",
            "BB_Upper",
            "BB_Lower",
        ]

        # Check if all required features are available
        if all(f in self.data.columns for f in features):
            # Extract features and target
            feature_data = self.data[features].values
            target_data = self.data["Close"].values

            # Create sequences for X and corresponding y
            X = []
            y = []
            for i in range(seq_len, len(feature_data)):
                X.append(feature_data[i - seq_len : i])  # Past `seq_len` data points
                y.append(target_data[i])  # Target is the value at the current time step

            X = np.array(X)  # Shape: (num_samples, seq_len, input_dim)
            y = np.array(y)  # Shape: (num_samples,)

            print(f"Prepared ML data: Features shape {X.shape}, Target shape {y.shape}")
            return X, y
        else:
            print("Some features are missing. Returning None.")
            return None, None

    def plot(self, days=30, title="Stock Prices", figsize=(14, 7)):
        """
        Plot the stock price for the last 'days' days.

        Args:
            days (int): Number of days to plot.
            title (str): Plot title.
            figsize (tuple): Figure size.
        """
        filtered_data = self.data.tail(days)
        plt.figure(figsize=figsize)
        plt.plot(
            filtered_data.index,
            filtered_data["Close"],
            label=f"{self.ticker} - Close",
            color="blue",
        )
        plt.title(title, fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Initialize the processor
    processor = StockDataProcessor(
        ticker="AAPL", start_date="2022-01-01", interval="1d"
    )

    # Fetch ML-ready data
    X, y = processor.get_ml_ready_data()

    # Plot stock prices for the last 30 days
    processor.plot(days=30, title="AAPL Stock Prices - Last 30 Days")
