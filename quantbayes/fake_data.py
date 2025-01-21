import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
import jax
import jax.numpy as jnp


def generate_regression_data(n_samples=1000, n_features=1, random_seed=None):
    """
    Generate synthetic sinusoidal data with slight stochasticity.

    Args:
        n_samples (int): Number of samples (rows).
        n_features (int): Number of features for X.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing features (X) and target (y).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Time index
    t = np.linspace(0, 2 * np.pi * 10, n_samples)  # Covers multiple cycles

    # Generate features (X) as sinusoidal waves with slight noise
    X = {}
    for i in range(n_features):
        freq = 0.1 * (i + 1)  # Slight frequency variation for each feature
        phase = np.random.uniform(0, 2 * np.pi)  # Random phase shift
        amplitude = 1 + 0.5 * i  # Gradual increase in amplitude
        noise = np.random.normal(scale=0.1, size=n_samples)  # Slight noise

        X[f"feature_{i+1}"] = amplitude * np.sin(freq * t + phase) + noise

    # Generate target (y) as a weighted combination of features
    target = sum((i + 1) * X[f"feature_{i+1}"] for i in range(n_features))
    target = target / n_features + np.random.normal(
        scale=0.2, size=n_samples
    )  # Slight noise added to target

    # Combine features and target into a DataFrame
    df = pd.DataFrame(X)
    df["target"] = target

    return df


def generate_binary_classification_data(
    n_samples=500, n_features=5, class_sep=1.0, random_seed=42
):
    """
    Generate synthetic binary classification data with noise and feature overlap.

    Args:
        n_samples (int): Number of samples (rows).
        n_features (int): Number of features (columns).
        class_sep (float): Separation between the two classes (lower values make it harder).
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing features (X) and target (y).
    """
    np.random.seed(random_seed)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,  # Only half of the features are informative
        n_redundant=n_features // 4,  # Some features are linear combinations of others
        n_clusters_per_class=2,  # Multiple clusters per class for overlap
        class_sep=class_sep,  # Control difficulty of classification
        flip_y=0.1,  # Introduce label noise (10%)
        random_state=random_seed,
    )

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(X.shape[1])])
    df["target"] = y

    return df


def generate_multiclass_classification_data(
    n_samples=500, n_features=2, n_classes=3, cluster_std=2.0, random_seed=42
):
    """
    Generate synthetic multiclass classification data with overlapping clusters.

    Args:
        n_samples (int): Number of samples (rows).
        n_features (int): Number of features (columns).
        n_classes (int): Number of target classes.
        cluster_std (float): Standard deviation of clusters (higher values make it harder).
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing features (X) and target (y).
    """
    np.random.seed(random_seed)
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        cluster_std=cluster_std,  # Control difficulty of classification
        random_state=random_seed,
    )

    # Add noise to make it more challenging
    noise = np.random.normal(0, 0.5, size=X.shape)
    X += noise

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(X.shape[1])])
    df["target"] = y

    return df


def create_synthetic_time_series(
    n_samples=1000, seq_len=10, num_features=1, train_split=0.8
):
    """
    Create synthetic time series data.

    Parameters:
    n_samples: Total number of time points
    seq_len: Number of timesteps in a sequence
    num_features: Number of features
    train_split: Fraction of data used for training

    Returns:
    X_train, X_val, y_train, y_val
    """
    # Time variable
    t = jnp.linspace(0, 10, n_samples)

    # Random key
    key = jax.random.PRNGKey(0)

    # Create a simple sine wave with added noise as target
    key, noise_key = jax.random.split(key)
    y = jnp.sin(t) + 0.1 * jax.random.normal(noise_key, t.shape)

    # Features: Add some noise to the sine wave for feature inputs
    sequences = []
    for _ in range(num_features):
        key, feature_key = jax.random.split(key)
        feature = y + 0.1 * jax.random.normal(feature_key, y.shape)
        sequences.append(feature)
    X = jnp.stack(sequences, axis=-1)

    # Reshape into sequences
    sequences_X = []
    sequences_y = []
    for i in range(n_samples - seq_len):
        sequences_X.append(X[i : i + seq_len])
        sequences_y.append(y[i + seq_len])

    sequences_X = jnp.array(sequences_X)
    sequences_y = jnp.array(sequences_y)

    # Split into train and validation sets
    split_idx = int(train_split * len(sequences_X))
    X_train, X_val = sequences_X[:split_idx], sequences_X[split_idx:]
    y_train, y_val = sequences_y[:split_idx], sequences_y[split_idx:]

    return X_train, X_val, y_train, y_val
