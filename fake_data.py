import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs


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
