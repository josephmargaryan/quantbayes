import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class HeadTailDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, head_tokens, tail_tokens):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head_tokens = head_tokens
        self.tail_tokens = tail_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data.iloc[index]["Sentence"]
        sentiment = self.data.iloc[index]["Sentiment"]

        tokens = self.tokenizer(sentence, add_special_tokens=False, truncation=False)[
            "input_ids"
        ]

        if len(tokens) > self.max_length - 2:
            tokens = tokens[: self.head_tokens] + tokens[-self.tail_tokens :]

        tokens = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]

        attention_mask = [1] * len(tokens)
        while len(tokens) < self.max_length:
            tokens.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(sentiment, dtype=torch.long),
        }


class HeadTailDatasetInference(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, head_tokens, tail_tokens):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head_tokens = head_tokens
        self.tail_tokens = tail_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data.iloc[index]["Sentence"]

        tokens = self.tokenizer(sentence, add_special_tokens=False, truncation=False)[
            "input_ids"
        ]

        if len(tokens) > self.max_length - 2:
            tokens = tokens[: self.head_tokens] + tokens[-self.tail_tokens :]

        tokens = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]

        attention_mask = [1] * len(tokens)
        while len(tokens) < self.max_length:
            tokens.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def extract_hidden_states(model, dataloader, device, has_labels=True):
    """
    Extract hidden states, logits, and predictions from a model.

    Args:
        model: The model from which to extract hidden states, logits, and predictions.
        dataloader: DataLoader providing input data.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
        has_labels: Whether the DataLoader includes labels (default: True).

    Returns:
        embeddings (numpy.ndarray): Extracted hidden states (CLS token embeddings).
        logits (numpy.ndarray): Raw logits from the model.
        predictions (numpy.ndarray): Predicted class indices.
        labels (numpy.ndarray or None): Corresponding labels if available, else None.
    """
    model.eval()
    all_embeddings = []
    all_logits = []
    all_predictions = []
    all_labels = [] if has_labels else None

    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc="Extracting embeddings, logits, and predictions"
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states[-1][:, 0, :]
            all_embeddings.append(hidden_states.cpu())

            logits = outputs.logits
            all_logits.append(logits.cpu())

            predictions = torch.argmax(logits, dim=1)
            all_predictions.append(predictions.cpu())

            if has_labels:
                labels = batch["labels"].to(device)
                all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    logits = torch.cat(all_logits, dim=0).numpy()
    predictions = torch.cat(all_predictions, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy() if has_labels else None
    return embeddings, logits, predictions, labels


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    df = pd.read_csv("path/to/data.csv")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(label_encoder.classes_)
    )
    model.to(device)
    dataset = HeadTailDataset(
        df, tokenizer, max_length=510, head_tokens=127, tail_tokens=387
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    embeddings, logits, predictions, labels = extract_hidden_states(
        model, dataloader, device
    )

    ### For inference without labels
    """    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.to(device)
    dataset = HeadTailDatasetInference(df, tokenizer, max_length=510, head_tokens=127, tail_tokens=387)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    embeddings, logits, predictions, labels = extract_hidden_states(model, dataloader, device, has_labels=False)
    """
