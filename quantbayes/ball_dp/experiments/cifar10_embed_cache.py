# quantbayes/ball_dp/experiments/cifar10_embed_cache.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from quantbayes.ball_dp.utils.seeding import set_global_seed
from quantbayes.ball_dp.utils.io import ensure_dir


@dataclass
class CIFAR10EmbedConfig:
    data_dir: str = "./data"
    cache_npz: str = "./cache/cifar10_resnet18_embeds.npz"
    batch_size: int = 256
    device: str = "cuda"
    weights: str = "DEFAULT"  # "DEFAULT" or "NONE"
    l2_normalize: bool = False
    seed: int = 0
    num_workers: int = 2


class ResNet18Embedder(nn.Module):
    """
    ResNet18 penultimate embeddings: (B,512)
    """

    def __init__(self, weights: str = "DEFAULT"):
        super().__init__()
        w = None
        if weights and weights.upper() != "NONE":
            try:
                w = (
                    models.ResNet18_Weights.DEFAULT
                    if weights.upper() == "DEFAULT"
                    else None
                )
            except Exception:
                w = None
        m = models.resnet18(weights=w)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.out_dim = 512

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return h.flatten(1)


def _cifar10_loaders(
    data_dir: str, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


@torch.no_grad()
def _extract_embeddings(
    loader: DataLoader, model: nn.Module, device: torch.device, l2_normalize: bool
) -> Tuple[np.ndarray, np.ndarray]:
    zs, ys = [], []
    for x, y in loader:
        x = x.to(device)
        z = model(x).detach().cpu().numpy().astype(np.float32)
        if l2_normalize:
            n = np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
            z = z / n
        zs.append(z)
        ys.append(y.numpy().astype(np.int64))
    Z = np.concatenate(zs, axis=0)
    Y = np.concatenate(ys, axis=0)
    return Z, Y


def get_or_compute_cifar10_embeddings(
    cfg: CIFAR10EmbedConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Ztr, ytr, Zte, yte
    """
    set_global_seed(cfg.seed)

    cache_path = Path(cfg.cache_npz)
    ensure_dir(cache_path.parent)

    if cache_path.exists():
        d = np.load(cache_path)
        return (
            d["Ztr"].astype(np.float32),
            d["ytr"].astype(np.int64),
            d["Zte"].astype(np.float32),
            d["yte"].astype(np.int64),
        )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = _cifar10_loaders(
        cfg.data_dir, cfg.batch_size, cfg.num_workers
    )
    enc = ResNet18Embedder(weights=cfg.weights).to(device).eval()

    Ztr, ytr = _extract_embeddings(train_loader, enc, device, cfg.l2_normalize)
    Zte, yte = _extract_embeddings(test_loader, enc, device, cfg.l2_normalize)

    np.savez_compressed(cache_path, Ztr=Ztr, ytr=ytr, Zte=Zte, yte=yte)
    return Ztr, ytr, Zte, yte
