import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        """
        Args:
            sequences (list of lists): List of variable-length time series sequences.
            targets (list of values): Corresponding targets for each sequence.
        """
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(
            self.targets[idx], dtype=torch.float32
        )


def pad_collate(batch):
    """Collate function to pad sequences to the same length and create masks."""
    sequences, targets = zip(*batch)

    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # Create mask (1 for valid time steps, 0 for padding)
    masks = torch.zeros_like(padded_sequences, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        masks[i, : seq.size(0)] = 1

    targets = torch.stack(targets)
    return padded_sequences, masks, targets


def create_loaders(data, target_column, sequence_length, batch_size, val_split=0.2):
    """
    Create train and validation data loaders for time series data.

    Args:
        data (pd.DataFrame): DataFrame containing time series data.
        target_column (str): Name of the column containing the target variable.
        sequence_length (int): Length of each input sequence.
        batch_size (int): Batch size for DataLoader.
        val_split (float): Fraction of data to use for validation.

    Returns:
        train_loader, val_loader: DataLoaders for training and validation.
    """
    # Split data into features and targets
    features = data.drop(columns=[target_column]).values
    targets = data[target_column].values

    # Create sequences and corresponding targets
    sequences, sequence_targets = [], []
    for i in range(len(features) - sequence_length):
        sequences.append(features[i : i + sequence_length])
        sequence_targets.append(targets[i + sequence_length])

    # Convert to numpy arrays
    sequences = np.array(sequences, dtype=np.float32)
    sequence_targets = np.array(sequence_targets, dtype=np.float32)

    # Split into training and validation sets
    split_idx = int(len(sequences) * (1 - val_split))
    train_sequences, val_sequences = sequences[:split_idx], sequences[split_idx:]
    train_targets, val_targets = (
        sequence_targets[:split_idx],
        sequence_targets[split_idx:],
    )

    # Create datasets
    train_dataset = TimeSeriesDataset(train_sequences, train_targets)
    val_dataset = TimeSeriesDataset(val_sequences, val_targets)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate
    )

    return train_loader, val_loader
