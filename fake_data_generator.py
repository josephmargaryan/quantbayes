import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def geometric_brownian_motion(
    n_steps, mu=0.001, sigma=0.02, start_price=100, random_seed=None
):
    """
    Generate a geometric Brownian motion time series.

    Args:
        n_steps (int): Number of time steps.
        mu (float): Drift coefficient (mean return).
        sigma (float): Volatility coefficient (standard deviation of returns).
        start_price (float): Initial price.
        random_seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Simulated prices.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = 1  # Time step size (can adjust for finer granularity)
    returns = np.random.normal(
        loc=(mu - 0.5 * sigma**2) * dt, scale=sigma * np.sqrt(dt), size=n_steps
    )
    prices = start_price * np.exp(np.cumsum(returns))  # GBM formula
    return prices


def create_gbm_dataframe(n_samples, n_features, n_steps, random_seed=None):
    """
    Create a DataFrame with synthetic GBM-based features (X) and closing prices (y).

    Args:
        n_samples (int): Number of samples (rows) in the dataset.
        n_features (int): Number of features for X.
        n_steps (int): Number of time steps for y.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing features (X) and closing prices (y).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate features (X) with independent GBM processes
    X = []
    for _ in range(n_features):
        feature_prices = geometric_brownian_motion(
            n_steps=n_samples, mu=0.001, sigma=0.02, start_price=100
        )
        X.append(feature_prices)

    # Generate closing prices (y) as a GBM process
    y = geometric_brownian_motion(
        n_steps=n_samples, mu=0.0015, sigma=0.025, start_price=150
    )

    # Combine into a DataFrame
    data = {f"feature_{i+1}": X[i] for i in range(n_features)}
    data["Close"] = y
    df = pd.DataFrame(data)
    return df


# Visualize the generated data
def visualize_gbm_data(df):
    """
    Visualize GBM data for features and closing prices.

    Args:
        df (pd.DataFrame): DataFrame containing GBM features (X) and closing prices (y).
    """
    # Plot all features
    plt.figure(figsize=(12, 6))
    for col in df.columns[:-1]:  # Exclude 'Close'
        plt.plot(df[col], label=col, alpha=0.7)
    plt.title("Geometric Brownian Motion: Features")
    plt.xlabel("Time Steps")
    plt.ylabel("Feature Values")
    plt.legend(loc="upper left")
    plt.show()

    # Plot the closing price
    plt.figure(figsize=(12, 6))
    plt.plot(df["Close"], label="Close", color="red")
    plt.title("Geometric Brownian Motion: Closing Price")
    plt.xlabel("Time Steps")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.show()


def generate_simple_regression_data(n_samples, n_features, random_seed=None):
    """
    Generate simple, predictable synthetic data with features and target.

    Args:
        n_samples (int): Number of samples (rows).
        n_features (int): Number of features for X.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing features (X) and closing prices (y).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Time index
    t = np.arange(n_samples)

    # Generate features (X) as simple trends + noise
    X = {}
    for i in range(n_features):
        trend = 0.05 * (i + 1) * t  # Linear trend
        seasonal = 5 * np.sin(2 * np.pi * t / 50)  # Sinusoidal pattern
        noise = np.random.normal(scale=0.5, size=n_samples)  # Small noise
        X[f"feature_{i+1}"] = trend + seasonal + noise

    # Generate target (y) as a weighted combination of features
    weights = np.linspace(
        1, n_features, n_features
    )  # Higher weights for later features
    target = sum(weights[i] * X[f"feature_{i+1}"] for i in range(n_features))
    target = target / n_features + np.random.normal(
        scale=2.0, size=n_samples
    )  # Add small noise

    # Combine features and target into a DataFrame
    df = pd.DataFrame(X)
    df["target"] = target

    return df


if __name__ == "__main__":
    n_samples = 500
    n_features = 8
    n_steps = n_samples
    random_seed = 42

    gbm_data = create_gbm_dataframe(
        n_samples, n_features, n_steps, random_seed=random_seed
    )
    simple_data = generate_simple_regression_data(
        n_samples, n_features, random_seed=random_seed
    )
