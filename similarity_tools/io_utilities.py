import pandas as pd
import pyarrow.parquet as pq


def load_csv(file_path):
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(file_path)


def load_parquet(file_path):
    """Load a Parquet file into a DataFrame."""
    return pd.read_parquet(file_path)


def save_parquet(df, file_path):
    """Save a DataFrame to a Parquet file."""
    pq.write_table(df.to_arrow(), file_path)
