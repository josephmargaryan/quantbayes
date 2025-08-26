import pickle
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def clean_text(text):
    """
    Clean input text by removing unwanted characters, links, and emojis.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)
    return text


def load_and_preprocess_data(
    path, label_col=None, text_col="x", save_label_encoder=False
):
    """
    Load and preprocess dataset for training, validation, or inference.

    Parameters:
        path (str): Path to the CSV file.
        label_col (str): Column name for the labels. Set to None if no labels are present.
        text_col (str): Column name for the text data.
        save_label_encoder (bool): Whether to save the label encoder as a pickle file.

    Returns:
        train_df (DataFrame): Training dataset.
        val_df (DataFrame): Validation dataset.
        num_classes (int): Number of unique classes (if labels are provided).
    """
    df = pd.read_csv(path)

    df["x"] = df[text_col].apply(clean_text)

    num_classes = None
    if label_col:
        le = LabelEncoder()
        df["y"] = le.fit_transform(df[label_col])
        num_classes = len(le.classes_)

        if save_label_encoder:
            with open("label_encoder.pkl", "wb") as f:
                pickle.dump(le, f)

    if label_col:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        return train_df, val_df, num_classes
    else:
        return df, num_classes


if __name__ == "__main__":
    dataset_path = "/content/news_data.csv"
    label_column = "Company"  # Column containing labels
    text_column = "News"  # Column containing text data

    train_df, val_df, num_classes = load_and_preprocess_data(
        path=dataset_path,
        label_col=label_column,
        text_col=text_column,
        save_label_encoder=True,
    )

    train_df.to_csv("train_data.csv", index=False)
    val_df.to_csv("val_data.csv", index=False)

    print("Data preprocessing complete!")
    print(f"Number of classes: {num_classes}")
    print("Training data saved to 'train_data.csv'.")
    print("Validation data saved to 'val_data.csv'.")


#### For inference
"""
if __name__ == "__main__":
    dataset_path = "/content/new_data.csv"  
    text_column = "News"  

    # Load and preprocess the data
    new_data, _ = load_and_preprocess_data(
        path=dataset_path,
        label_col=None,  # No labels in new data
        text_col=text_column,
        save_label_encoder=False,
    )

    new_data.to_csv("processed_new_data.csv", index=False)
    print(f"New data preprocessing complete!")
"""
