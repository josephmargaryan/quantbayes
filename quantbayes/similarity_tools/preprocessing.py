import pandas as pd
import re


def clean_text(text):
    """Basic text cleaning: lowercasing, removing special characters."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_dataframe(df, columns):
    """Apply text cleaning to specified columns in a DataFrame."""
    for col in columns:
        df[col] = df[col].apply(clean_text)
    return df
