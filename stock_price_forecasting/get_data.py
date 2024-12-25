import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class StockData:
    def __init__(self, ticker, start_date, interval="1h"):
        """
        Initialize the StockData class and preprocess the data.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL').
            start_date (str): Start date for fetching data (e.g., '2023-01-01').
            interval (str): Data interval (e.g., '1h', '1d').
        """
        self.ticker = ticker
        self.start_date = start_date
        self.interval = interval
        self.data = None
        self.fetch_data()
        self.preprocess_pipeline()

    def fetch_data(self):
        """Fetch historical data for the stock ticker."""
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, interval=self.interval)

            if self.data.empty:
                raise ValueError(
                    f"No data found for ticker '{self.ticker}' with the given parameters."
                )

            self.data = self.data[["Close"]]
            print(f"Fetched data for {self.ticker}: {len(self.data)} rows")
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            self.data = None

    def preprocess_pipeline(self):
        """Execute the complete preprocessing pipeline automatically."""
        if self.data is not None:
            self.add_returns()
            self.add_features()
            self.add_rsi()
            self.add_macd()
            self.add_bollinger_bands()
            self.handle_missing_values()
            self.handle_outliers(column="Close")
            self.normalize_data()
            print(f"Preprocessing pipeline completed for {self.ticker}.")
        else:
            print("No data available to preprocess.")

    def add_returns(self):
        """Add returns to the dataset."""
        self.data["Return"] = self.data["Close"].pct_change()
        self.data.dropna(inplace=True)

    def add_features(self):
        """Add basic features for ML models."""
        self.data["MA_10"] = self.data["Close"].rolling(window=10).mean()
        self.data["MA_50"] = self.data["Close"].rolling(window=50).mean()
        self.data["Volatility"] = self.data["Close"].rolling(window=10).std()

    def add_rsi(self, period=14):
        """Add the Relative Strength Index (RSI) feature."""
        delta = self.data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        self.data["RSI"] = 100 - (100 / (1 + rs))

    def add_macd(self, short_window=12, long_window=26, signal_window=9):
        """Add the Moving Average Convergence Divergence (MACD) feature."""
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
        """Add Bollinger Bands feature."""
        self.data["BB_Middle"] = self.data["Close"].rolling(window=window).mean()
        self.data["BB_Upper"] = (
            self.data["BB_Middle"] + 2 * self.data["Close"].rolling(window=window).std()
        )
        self.data["BB_Lower"] = (
            self.data["BB_Middle"] - 2 * self.data["Close"].rolling(window=window).std()
        )

    def handle_missing_values(self):
        """Drop rows with missing values."""
        self.data.dropna(inplace=True)
        print("Missing values handled.")

    def handle_outliers(self, column, lower_quantile=0.01, upper_quantile=0.99):
        """Handle outliers by clipping extreme values."""
        q_low = self.data[column].quantile(lower_quantile)
        q_high = self.data[column].quantile(upper_quantile)
        self.data[column] = np.clip(self.data[column], q_low, q_high)
        print(f"Outliers handled for {column}.")

    def normalize_data(self):
        """Normalize data using Min-Max scaling."""
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.data)
        self.data = pd.DataFrame(
            scaled_data, index=self.data.index, columns=self.data.columns
        )
        print("Data normalized.")

    def get_ml_ready_data(self):
        """
        Get features (X) and target (y) for ML models.

        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target variable.
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
        if all(f in self.data.columns for f in features):
            X = self.data[features]
            y = self.data["Close"]
            print(
                f"Data prepared for ML models. Features shape: {X.shape}, Target shape: {y.shape}"
            )
            return X, y
        else:
            print("Required features are missing.")
            return None, None


if __name__ == "__main__":
    # Initialize StockData
    stock_data = StockData(ticker="AAPL", start_date="2024-11-01", interval="1h")

    # Get ML-ready data
    X, y = stock_data.get_ml_ready_data()

    # Print sample
    print(X.head())
    print(y.head())
