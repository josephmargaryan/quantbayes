import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
import jax
import jax.numpy as jnp


def generate_regression_data(
    n_samples=1000, n_continuous=1, n_categorical=0, random_seed=None
):
    """
    Generate synthetic regression data with a mix of continuous and categorical features.

    Parameters
    ----------
    n_samples : int
        Number of samples (rows).
    n_continuous : int
        Number of continuous features.
    n_categorical : int
        Number of categorical features.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame containing continuous and categorical features and target variable.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    df = pd.DataFrame()

    # Generate continuous features (e.g., sinusoidal waves with noise)
    t = np.linspace(0, 2 * np.pi * 10, n_samples)
    for i in range(n_continuous):
        freq = 0.1 * (i + 1)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = 1 + 0.5 * i
        noise = np.random.normal(scale=0.1, size=n_samples)
        df[f"cont_feature_{i+1}"] = amplitude * np.sin(freq * t + phase) + noise

    # Generate categorical features (random discrete values)
    for i in range(n_categorical):
        # For simplicity, create categorical features with 3 levels (0, 1, 2)
        df[f"cat_feature_{i+1}"] = np.random.choice([0, 1, 2], size=n_samples)

    # Generate target as a combination of continuous features (weighted sum) plus noise.
    if n_continuous > 0:
        target = sum(df[f"cont_feature_{i+1}"] for i in range(n_continuous))
        target = target / n_continuous
    else:
        target = 0
    # Add some noise
    target += np.random.normal(scale=0.2, size=n_samples)
    df["target"] = target

    return df


def generate_binary_classification_data(
    n_samples=500,
    n_continuous=3,
    n_categorical=2,
    class_sep=1.0,
    random_seed=42,
    n_categories=3,
    n_clusters_per_class=None,  # allow user to override if desired
):
    """
    Generate synthetic binary classification data with both continuous and categorical features.

    Args:
        n_samples (int): Number of samples.
        n_continuous (int): Number of continuous features.
        n_categorical (int): Number of categorical features.
        class_sep (float): Class separation parameter.
        random_seed (int): Random seed for reproducibility.
        n_categories (int): Number of categories for each categorical feature.
        n_clusters_per_class (int, optional): Number of clusters per class for continuous features.
                                              If not provided, it will be set based on n_continuous.

    Returns:
        pd.DataFrame: DataFrame with continuous features, categorical features, and target.
    """
    np.random.seed(random_seed)

    # Adjust n_informative: if n_continuous is very low, use all features as informative.
    if n_continuous < 4:
        n_informative = n_continuous
    else:
        n_informative = n_continuous // 2

    # Automatically adjust n_clusters_per_class if not provided.
    if n_clusters_per_class is None:
        n_clusters_per_class = 1 if n_continuous == 1 else 2

    # Generate continuous features.
    X_cont, y = make_classification(
        n_samples=n_samples,
        n_features=n_continuous,
        n_informative=n_informative,
        n_redundant=max(0, n_continuous // 4),
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        flip_y=0.1,
        random_state=random_seed,
    )

    # Generate categorical features.
    X_cat = np.random.randint(0, n_categories, size=(n_samples, n_categorical))

    # Create DataFrames.
    df_cont = pd.DataFrame(
        X_cont, columns=[f"cont_feature_{i+1}" for i in range(n_continuous)]
    )
    df_cat = pd.DataFrame(
        X_cat, columns=[f"cat_feature_{i+1}" for i in range(n_categorical)]
    )

    # Convert categorical columns to type 'category'.
    for col in df_cat.columns:
        df_cat[col] = df_cat[col].astype("category")

    # Combine the features and add the target column.
    df = pd.concat([df_cont, df_cat], axis=1)
    df["target"] = y

    return df


def generate_multiclass_classification_data(
    n_samples=500,
    n_continuous=2,
    n_categorical=2,
    n_classes=3,
    cluster_std=2.0,
    random_seed=42,
    n_categories=3,
):
    """
    Generate synthetic multiclass classification data with both continuous and categorical features.

    Args:
        n_samples (int): Number of samples (rows).
        n_continuous (int): Number of continuous features.
        n_categorical (int): Number of categorical features.
        n_classes (int): Number of target classes.
        cluster_std (float): Standard deviation of clusters (controls difficulty).
        random_seed (int): Random seed for reproducibility.
        n_categories (int): Number of distinct categories for each categorical feature.

    Returns:
        pd.DataFrame: DataFrame containing continuous features, categorical features, and target (y).
    """
    np.random.seed(random_seed)

    # Generate continuous features using make_blobs.
    X_cont, y = make_blobs(
        n_samples=n_samples,
        n_features=n_continuous,
        centers=n_classes,
        cluster_std=cluster_std,
        random_state=random_seed,
    )

    # Add noise to the continuous features.
    noise = np.random.normal(0, 0.5, size=X_cont.shape)
    X_cont = X_cont + noise

    # Generate categorical features by sampling random integers.
    X_cat = np.random.randint(0, n_categories, size=(n_samples, n_categorical))

    # Create DataFrames.
    df_cont = pd.DataFrame(
        X_cont, columns=[f"cont_feature_{i+1}" for i in range(n_continuous)]
    )
    df_cat = pd.DataFrame(
        X_cat, columns=[f"cat_feature_{i+1}" for i in range(n_categorical)]
    )
    for col in df_cat.columns:
        df_cat[col] = df_cat[col].astype("category")

    # Combine continuous and categorical features and add the target.
    df = pd.concat([df_cont, df_cat], axis=1)
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
