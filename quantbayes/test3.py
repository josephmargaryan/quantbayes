import numpy as np
import pandas as pd
import yfinance as yf
from test2 import run_sde


def create_multiple_trajectories(
    df,
    date_col="date",
    value_col="y",
    freq="Y",
    normalize=False,
    log_transform=False,
    rolling_window=None,
    add_noise=False,
    noise_std=0.1,
):
    """
    Creates multiple trajectories from a time series DataFrame by slicing, normalizing, log-transforming,
    and applying rolling windows or noise if specified.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a date column and a value column.
        date_col (str): Name of the date column.
        value_col (str): Name of the value column.
        freq (str): Frequency for slicing ('Y' for yearly, 'M' for monthly, etc.).
        normalize (bool): If True, normalize each trajectory to [0, 1].
        log_transform (bool): If True, apply log transformation to the values.
        rolling_window (int): Size of rolling window for smoothing. If None, no smoothing is applied.
        add_noise (bool): If True, add Gaussian noise to the values.
        noise_std (float): Standard deviation of the Gaussian noise to add.

    Returns:
        tuple: A tuple (time_steps, observations) formatted as:
            - time_steps: Shape (num_trajectories, time_steps)
            - observations: Shape (num_trajectories, time_steps, 1)
    """
    df = df.copy()

    # Ensure the date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort by date
    df = df.sort_values(date_col)
    df.set_index(date_col, inplace=True)

    trajectories = []
    for period, group in df.groupby(pd.Grouper(freq=freq)):
        if len(group) < 2:  # Skip if not enough data
            continue
        time_steps = np.arange(len(group))
        values = group[value_col].values

        # Apply log transformation if required
        if log_transform:
            values = np.log1p(values)

        # Apply rolling window smoothing if specified
        if rolling_window:
            values = (
                pd.Series(values)
                .rolling(window=rolling_window, min_periods=1)
                .mean()
                .values
            )

        # Add Gaussian noise if specified
        if add_noise:
            values += np.random.normal(0, noise_std, size=len(values))

        # Normalize values to [0, 1] if required
        if normalize:
            min_val = values.min()
            max_val = values.max()
            values = (values - min_val) / (max_val - min_val + 1e-10)

        trajectories.append((time_steps, values))

    # Format for Neural SDE
    time_steps = np.array([t[0] for t in trajectories])
    observations = np.array([t[1] for t in trajectories])[:, :, None]

    return time_steps, observations


def slice_time_series(df, date_col, value_col, interval="yearly", n_days=None):
    """
    Slice a DataFrame with a date column into multiple trajectories.

    Parameters:
        df (pd.DataFrame): The input DataFrame with a date column.
        date_col (str): The name of the date column.
        value_col (str): The column containing the values to slice.
        interval (str): The slicing interval. Options: "yearly", "monthly", "daily", "n_days".
        n_days (int): Number of days for "n_days" slicing (e.g., every 27th day).

    Returns:
        tuple: (time_steps, observations), formatted for Neural SDE.
    """
    df[date_col] = pd.to_datetime(df[date_col])  # Ensure the date column is datetime
    trajectories = []

    if interval == "yearly":
        for year, group in df.groupby(df[date_col].dt.year):
            trajectories.append(group[value_col].values)

    elif interval == "monthly":
        for (year, month), group in df.groupby(
            [df[date_col].dt.year, df[date_col].dt.month]
        ):
            trajectories.append(group[value_col].values)

    elif interval == "daily":
        for day, group in df.groupby(df[date_col].dt.date):
            trajectories.append(group[value_col].values)

    elif interval == "n_days":
        if n_days is None:
            raise ValueError("For 'n_days' interval, n_days must be specified.")
        for i in range(0, len(df), n_days):
            sliced = df.iloc[i : i + n_days]
            if len(sliced) == n_days:  # Ensure full slices
                trajectories.append(sliced[value_col].values)
    else:
        raise ValueError(
            f"Unsupported interval: {interval}. Choose 'yearly', 'monthly', 'daily', or 'n_days'."
        )

    # Convert trajectories to Neural SDE format
    max_length = max(len(traj) for traj in trajectories)
    trajectories_padded = [
        np.pad(traj, (0, max_length - len(traj)), constant_values=np.nan)
        for traj in trajectories
    ]

    time_steps = np.arange(max_length)  # Uniform time steps
    observations = np.array(trajectories_padded)[
        :, :, None
    ]  # Shape (num_trajectories, time_steps, 1)

    real_data = (
        np.tile(
            time_steps[None, :], (observations.shape[0], 1)
        ),  # Duplicate time steps
        observations,  # Shape: (num_trajectories, time_steps, 1)
    )
    return real_data


def normalize_data(real_data):
    """
    Normalize real data to the range [0, 1].

    Parameters:
        real_data (tuple): (time_steps, observations), where:
            - time_steps: Shape (num_trajectories, time_steps)
            - observations: Shape (num_trajectories, time_steps, 1)

    Returns:
        tuple: Normalized real_data and normalization parameters (min_val, max_val).
    """
    time_steps, observations = real_data
    min_val = observations.min(axis=(1, 2), keepdims=True)
    max_val = observations.max(axis=(1, 2), keepdims=True)
    normalized_observations = (observations - min_val) / (max_val - min_val)
    return (time_steps, normalized_observations), min_val, max_val


def rescale_data(generated_observations, min_val, max_val):
    """
    Rescale generated data back to the original scale.

    Parameters:
        generated_observations (numpy.ndarray): Generated data in normalized range [0, 1].
        min_val (numpy.ndarray): Minimum values used for normalization.
        max_val (numpy.ndarray): Maximum values used for normalization.

    Returns:
        numpy.ndarray: Rescaled data to the original scale.
    """
    return generated_observations * (max_val - min_val) + min_val


def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch daily stock data for a list of tickers and return it as a DataFrame.

    Parameters:
        tickers (list of str): List of stock tickers to fetch.
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: DataFrame with columns [date, stock1, stock2, ...].
    """
    # Download data using yfinance
    data = yf.download(tickers, start=start_date, end=end_date, interval="1d")

    # Ensure data is in the expected format
    if "Adj Close" in data.columns:  # Multi-index DataFrame (e.g., multiple tickers)
        data = data["Adj Close"]

    # Handle missing data by forward-filling
    data = data.ffill().bfill()

    # Reset index and rename date column in a single operation
    data = data.reset_index().rename(columns={"Date": "date"})

    return data


tickers = [
    "AAPL",
    "MSFT",
    "GOOG",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "NFLX",
    "INTC",
    "AMD",
    "BA",
    "GE",
    "JPM",
    "GS",
    "BAC",
    "WFC",
    "V",
    "MA",
    "PYPL",
    "ADBE",
    "CRM",
    "ORCL",
    "IBM",
    "CSCO",
    "QCOM",
    "TXN",
    "AVGO",
    "SPGI",
    "COST",
    "WMT",
    "TGT",
    "HD",
    "LOW",
    "DIS",
    "NKE",
    "MCD",
    "SBUX",
    "KO",
    "PEP",
    "MO",
    "UNH",
    "PFE",
    "MRK",
    "JNJ",
    "ABBV",
    "LLY",
    "BMY",
    "CVS",
    "AMGN",
    "GILD",
]

df = fetch_stock_data(tickers, "2023-06-24", "2024-06-24")


def prepare_real_data_from_stocks(df):
    """
    Prepare stock data for use in the Neural SDE format.

    Parameters:
        df (pd.DataFrame): DataFrame with columns [date, stock1, stock2, ...].

    Returns:
        tuple: A tuple (time_steps, observations) formatted as:
            - time_steps: Shape (num_trajectories, time_steps)
            - observations: Shape (num_trajectories, time_steps, 1)
    """
    # Convert dates to normalized time steps
    time_steps = (df["date"] - df["date"].min()).dt.days.values

    # Extract stock observations (exclude the date column)
    observations = df.iloc[:, 1:].values.T  # Shape: (num_stocks, time_steps)
    observations = observations[
        :, :, None
    ]  # Add a singleton dimension (Shape: (num_stocks, time_steps, 1))

    # Tile time_steps for each trajectory
    real_data = (
        np.tile(
            time_steps[None, :], (observations.shape[0], 1)
        ),  # Shape: (num_stocks, time_steps)
        observations,  # Shape: (num_stocks, time_steps, 1)
    )
    return real_data


real_stock_data = prepare_real_data_from_stocks(df)

normalized_real_data, min_val, max_val = normalize_data(real_stock_data)


def fake_data_test():

    # Simulated real-world data
    time_steps = np.linspace(0, 63, 64)
    num_trajectories = 100  # Number of independent trajectories

    # Generate 10 different sine waves with added noise
    observations = np.sin(time_steps[:, None]) + np.random.normal(  # Base sine wave
        0, 0.1, (64, num_trajectories)
    )  # Add noise

    # Format `real_data` for multiple trajectories
    real_data = (
        np.tile(time_steps[None, :], (num_trajectories, 1)),  # Duplicate time steps
        observations.T[:, :, None],  # Shape: (10, 64, 1)
    )
    return real_data


fake_data = fake_data_test()


df = run_sde(
    real_data=normalized_real_data,  # Required parameter: Tuple of (ts, ys)
    initial_noise_size=5,
    noise_size=3,
    hidden_size=16,
    width_size=16,
    depth=1,
    generator_lr=2e-5,
    discriminator_lr=1e-4,
    batch_size=10,  # Adjust to match dataset size
    steps=50,
    steps_per_print=1,
    seed=5678,
    max_plot_trajectories=30,
)

import matplotlib.pyplot as plt


def plot_trajectories(df, num_trajectories=30):
    """
    Visualize original and generated trajectories.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Time', 'Original', and 'Generated' columns.
        num_trajectories (int): Max number of trajectories to plot.
    """
    # Reshape 'Original' and 'Generated' to get individual trajectories
    time_steps = df["Time"].unique()
    num_time_steps = len(time_steps)
    total_trajectories = len(df) // num_time_steps

    # Reshape data into (trajectories, time_steps)
    original_trajectories = df["Original"].values.reshape(
        total_trajectories, num_time_steps
    )
    generated_trajectories = df["Generated"].values.reshape(
        total_trajectories, num_time_steps
    )

    # Plot a subset of trajectories
    plt.figure(figsize=(10, 6))
    for i in range(min(num_trajectories, total_trajectories)):
        plt.plot(
            time_steps,
            original_trajectories[i],
            label="Original" if i == 0 else "",
            color="blue",
            alpha=0.5,
        )
        plt.plot(
            time_steps,
            generated_trajectories[i],
            label="Generated" if i == 0 else "",
            color="red",
            alpha=0.5,
        )

    plt.title(f"Original vs. Generated Trajectories (max {num_trajectories})")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.show()


# Call the function
plot_trajectories(df)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, wasserstein_distance


def analyze_trajectories(df):
    """
    Analyze and compare the original and generated trajectories in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['Time', 'Original', 'Generated'].
    """
    # Reshape trajectories
    time_steps = df["Time"].unique()
    num_time_steps = len(time_steps)
    total_trajectories = len(df) // num_time_steps

    original_trajectories = df["Original"].values.reshape(
        total_trajectories, num_time_steps
    )
    generated_trajectories = df["Generated"].values.reshape(
        total_trajectories, num_time_steps
    )

    # Compute mean and std for each trajectory
    original_means = original_trajectories.mean(axis=1)
    original_stds = original_trajectories.std(axis=1)
    generated_means = generated_trajectories.mean(axis=1)
    generated_stds = generated_trajectories.std(axis=1)

    print(
        f"Original Mean (all trajectories): {original_means.mean():.4f}, Std: {original_stds.mean():.4f}"
    )
    print(
        f"Generated Mean (all trajectories): {generated_means.mean():.4f}, Std: {generated_stds.mean():.4f}"
    )

    # Plot distributions of means
    sns.histplot(
        original_means, kde=True, label="Original Mean", color="blue", alpha=0.6
    )
    sns.histplot(
        generated_means, kde=True, label="Generated Mean", color="red", alpha=0.6
    )
    plt.legend()
    plt.title("Distribution of Trajectory Means")
    plt.show()

    # Plot distributions of standard deviations
    sns.histplot(original_stds, kde=True, label="Original Std", color="blue", alpha=0.6)
    sns.histplot(
        generated_stds, kde=True, label="Generated Std", color="red", alpha=0.6
    )
    plt.legend()
    plt.title("Distribution of Trajectory Standard Deviations")
    plt.show()

    # Correlation between original and generated trajectories (per time step)
    correlation = np.corrcoef(
        original_trajectories.flatten(), generated_trajectories.flatten()
    )[0, 1]
    print(f"Correlation between Original and Generated: {correlation:.4f}")

    # Compute moving averages
    original_ma = (
        pd.DataFrame(original_trajectories.T).rolling(window=10).mean().T.values
    )
    generated_ma = (
        pd.DataFrame(generated_trajectories.T).rolling(window=10).mean().T.values
    )

    plt.figure(figsize=(10, 6))
    for i in range(min(10, total_trajectories)):  # Plot max 10 trajectories
        plt.plot(
            time_steps,
            original_ma[i],
            label="Original MA" if i == 0 else "",
            color="blue",
            alpha=0.6,
        )
        plt.plot(
            time_steps,
            generated_ma[i],
            label="Generated MA" if i == 0 else "",
            color="red",
            alpha=0.6,
        )
    plt.legend()
    plt.title("Moving Averages of Trajectories")
    plt.show()

    # Compute volatility
    original_volatility = (
        pd.DataFrame(original_trajectories.T).rolling(window=10).std().T.values
    )
    generated_volatility = (
        pd.DataFrame(generated_trajectories.T).rolling(window=10).std().T.values
    )

    plt.figure(figsize=(10, 6))
    for i in range(min(10, total_trajectories)):  # Plot max 10 trajectories
        plt.plot(
            time_steps,
            original_volatility[i],
            label="Original Volatility" if i == 0 else "",
            color="blue",
            alpha=0.6,
        )
        plt.plot(
            time_steps,
            generated_volatility[i],
            label="Generated Volatility" if i == 0 else "",
            color="red",
            alpha=0.6,
        )
    plt.legend()
    plt.title("Volatility of Trajectories")
    plt.show()

    # KL Divergence and Wasserstein Distance
    hist_original, _ = np.histogram(
        original_trajectories.flatten(), bins=30, density=True
    )
    hist_generated, _ = np.histogram(
        generated_trajectories.flatten(), bins=30, density=True
    )

    kl_divergence = entropy(hist_original + 1e-10, hist_generated + 1e-10)
    print(f"KL Divergence: {kl_divergence:.4f}")

    wasserstein = wasserstein_distance(
        original_trajectories.flatten(), generated_trajectories.flatten()
    )
    print(f"Wasserstein Distance: {wasserstein:.4f}")


# Call the analysis function
analyze_trajectories(df)
