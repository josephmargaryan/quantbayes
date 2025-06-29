"""
Common data‑loading utilities.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms


def load_nonlinear_svm_csv(path: str | Path):
    """
    Return
      X: np.ndarray of shape (n,2),
      y: np.ndarray of shape (n,) with labels in {-1,+1}.
    Automatically handles a header row like 'x,y,label'.
    """
    df = pd.read_csv(path)  # read header if present
    # if columns are named x, y, label, use them; else fall back to first 3 columns:
    if set(df.columns[:3]) >= {"x", "y", "label"}:
        X = df[["x", "y"]].values.astype(float)
        y = df["label"].values.astype(int)
    else:
        # no header: assume first two are features, third is label
        X = df.iloc[:, :2].values.astype(float)
        y = df.iloc[:, 2].values.astype(int)
    # ensure labels are ±1
    unique = np.unique(y)
    if set(unique) <= {0, 1}:
        # map {0,1} -> {-1,+1}
        y = np.where(y == 1, +1, -1)
    return X, y


def load_mnist_digits(digits=(3, 8)):
    """
    Returns (X_train, y_train), (X_test, y_test) with labels in {0,1}
    where 0 ≡ digits[0] and 1 ≡ digits[1].
    """
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(
        root=".", train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        root=".", train=False, download=True, transform=transform
    )

    def _filter(ds):
        idx = [(label in digits) for _, label in ds]
        images = torch.stack([ds[i][0] for i, keep in enumerate(idx) if keep])
        labels = torch.tensor([ds[i][1] for i, keep in enumerate(idx) if keep])
        labels = (labels == digits[1]).long()  # map to {0,1}
        X = images.view(len(images), -1).numpy() / 255.0
        y = labels.numpy()
        return X, y

    return _filter(mnist_train), _filter(mnist_test)
