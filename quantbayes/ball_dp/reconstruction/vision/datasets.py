# quantbayes/ball_dp/reconstruction/vision/datasets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_mnist_numpy(
    *,
    root: str,
    train: bool = True,
    download: bool = True,
    flatten: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: float32 in [0,1], shape (N,1,28,28) or (N,784) if flatten=True
      y: int64 (N,)
    """
    from torchvision.datasets import MNIST  # lazy import

    ds = MNIST(root=root, train=train, download=download)
    X = ds.data.numpy().astype(np.float32) / 255.0  # (N,28,28)
    y = ds.targets.numpy().astype(np.int64)
    X = X[:, None, :, :]  # (N,1,28,28)
    if flatten:
        X = X.reshape(X.shape[0], -1)
    return X, y


def load_cifar10_numpy(
    *,
    root: str,
    train: bool = True,
    download: bool = True,
    flatten: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: float32 in [0,1], shape (N,3,32,32) or (N,3072) if flatten=True
      y: int64 (N,)
    """
    from torchvision.datasets import CIFAR10  # lazy import

    ds = CIFAR10(root=root, train=train, download=download)
    X = np.asarray(ds.data, dtype=np.float32) / 255.0  # (N,32,32,3)
    y = np.asarray(ds.targets, dtype=np.int64)
    X = np.transpose(X, (0, 3, 1, 2))  # (N,3,32,32)
    if flatten:
        X = X.reshape(X.shape[0], -1)
    return X, y


# ---------------- paired dataset for encoder/decoder pipeline ----------------


@dataclass
class PairedTransformDataset:
    """
    Wrap a torchvision dataset (returning PIL image, label) into two transforms:
      x_enc: input to public encoder (e.g. resize+normalize)
      x_tgt: target image to reconstruct (e.g. ToTensor only)
    """

    base: any
    transform_enc: any
    transform_tgt: any

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, y = self.base[idx]  # PIL + int
        x_enc = self.transform_enc(img)
        x_tgt = self.transform_tgt(img)
        return x_enc, x_tgt, int(y)


def make_mnist_paired_loader_for_resnet(
    *,
    root: str,
    train: bool,
    batch_size: int,
    num_workers: int = 2,
    download: bool = True,
    shuffle: bool = False,
):
    """
    MNIST -> ResNet pipeline:
      - encoder input: grayscale->3ch, resize 224, ToTensor, normalize ImageNet
      - target: ToTensor (1x28x28) in [0,1]
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms

    base = MNIST(root=root, train=train, download=download, transform=None)

    tf_enc = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    tf_tgt = transforms.Compose(
        [
            transforms.ToTensor(),  # (1,28,28) in [0,1]
        ]
    )

    ds = PairedTransformDataset(base=base, transform_enc=tf_enc, transform_tgt=tf_tgt)
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_cifar10_paired_loader_for_resnet(
    *,
    root: str,
    train: bool,
    batch_size: int,
    num_workers: int = 2,
    download: bool = True,
    shuffle: bool = False,
):
    """
    CIFAR10 -> ResNet pipeline:
      - encoder input: resize 224, ToTensor, normalize ImageNet
      - target: ToTensor (3x32x32) in [0,1]
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10
    from torchvision import transforms

    base = CIFAR10(root=root, train=train, download=download, transform=None)

    tf_enc = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    tf_tgt = transforms.Compose(
        [
            transforms.ToTensor(),  # (3,32,32) in [0,1]
        ]
    )

    ds = PairedTransformDataset(base=base, transform_enc=tf_enc, transform_tgt=tf_tgt)
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def extract_embeddings_and_targets(
    *,
    loader,
    encoder,
    device: str = "cuda",
    l2_normalize_embeddings: bool = False,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    loader yields (x_enc, x_tgt, y).
    encoder produces embedding vectors.

    Returns:
      E: (N,d) float32
      X_tgt: (N,C,H,W) float32 in [0,1]
      y: (N,) int64
    """
    import torch

    dev = torch.device(
        device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    encoder = encoder.to(dev).eval()

    Es, Xs, Ys = [], [], []
    with torch.no_grad():
        for bi, (x_enc, x_tgt, y) in enumerate(loader):
            if (max_batches is not None) and (bi >= int(max_batches)):
                break
            x_enc = x_enc.to(dev)
            z = encoder(x_enc)
            if isinstance(z, (tuple, list)):
                z = z[0]
            if l2_normalize_embeddings:
                z = torch.nn.functional.normalize(z, p=2, dim=1)

            Es.append(z.detach().cpu().numpy().astype(np.float32))
            Xs.append(x_tgt.detach().cpu().numpy().astype(np.float32))
            Ys.append(np.asarray(y, dtype=np.int64))

    E = np.concatenate(Es, axis=0)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return E, X, Y


if __name__ == "__main__":
    # quick smoke: load small MNIST and CIFAR arrays
    X, y = load_mnist_numpy(root="./data", train=True, download=True, flatten=False)
    print("MNIST:", X.shape, y.shape, X.min(), X.max())
    Xc, yc = load_cifar10_numpy(root="./data", train=True, download=True, flatten=False)
    print("CIFAR10:", Xc.shape, yc.shape, Xc.min(), Xc.max())
    print("[OK] datasets smoke.")
