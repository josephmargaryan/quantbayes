#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from quantbayes.ball_dp.experiments.embedding_io import (
    default_cache_dir,
    get_jax_device,
    load_embedding_npz,
    save_embedding_npz,
)


def make_loader(
    dataset, batch_size: int, num_workers: int, pin_memory: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, a_min=eps, a_max=None)


def build_feature_extractor(torch_device: torch.device):
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Identity()  # [B, 512]
    model.eval()
    model.to(torch_device)

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            weights.transforms(),
        ]
    )
    return model, transform


def extract_embeddings(
    loader: DataLoader,
    model: nn.Module,
    torch_device: torch.device,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    all_embeddings = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(torch_device, non_blocking=(torch_device.type == "cuda"))
            features = model(images)

            all_embeddings.append(features.cpu().numpy().astype(np.float32, copy=False))
            all_labels.append(labels.numpy().astype(np.int32, copy=False))

    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)
    X = l2_normalize_rows(X)
    return X, y


def default_output_path(data_root: str = "./data") -> Path:
    return default_cache_dir(data_root) / "mnist_resnet18_embeddings.npz"


def load_mnist_resnet18_embeddings(
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    require_jax_gpu: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Compute embeddings from scratch and return JAX arrays.
    """
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jax_device = get_jax_device(require_jax_gpu)

    model, transform = build_feature_extractor(torch_device)

    train_dataset = torchvision.datasets.MNIST(
        root=data_root,
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_root,
        train=False,
        transform=transform,
        download=True,
    )

    pin_memory = torch_device.type == "cuda"
    train_loader = make_loader(train_dataset, batch_size, num_workers, pin_memory)
    test_loader = make_loader(test_dataset, batch_size, num_workers, pin_memory)

    X_train_np, y_train_np = extract_embeddings(
        train_loader, model, torch_device, desc="MNIST train"
    )
    X_test_np, y_test_np = extract_embeddings(
        test_loader, model, torch_device, desc="MNIST test"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    X_train = jax.device_put(X_train_np, device=jax_device)
    y_train = jax.device_put(y_train_np, device=jax_device)
    X_test = jax.device_put(X_test_np, device=jax_device)
    y_test = jax.device_put(y_test_np, device=jax_device)
    return X_train, y_train, X_test, y_test


def load_or_create_mnist_resnet18_embeddings(
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    require_jax_gpu: bool = True,
    cache_path: str | Path | None = None,
    force_recompute: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if cache_path is None:
        cache_path = default_output_path(data_root)
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_recompute:
        return load_embedding_npz(cache_path, require_jax_gpu=require_jax_gpu)

    X_train, y_train, X_test, y_test = load_mnist_resnet18_embeddings(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        require_jax_gpu=require_jax_gpu,
    )
    save_embedding_npz(cache_path, X_train, y_train, X_test, y_test)
    return X_train, y_train, X_test, y_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute/cache canonical-split MNIST embeddings with pretrained ResNet18 "
            "and return/save JAX arrays."
        )
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional .npz output path. Default: ./data/embeddings/mnist_resnet18_embeddings.npz",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing cached .npz and recompute embeddings.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Use CPU if JAX GPU is unavailable instead of raising an error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = (
        Path(args.output) if args.output else default_output_path(args.data_root)
    )

    X_train, y_train, X_test, y_test = load_or_create_mnist_resnet18_embeddings(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        require_jax_gpu=not args.allow_cpu_fallback,
        cache_path=output_path,
        force_recompute=args.force_recompute,
    )

    print(f"JAX default backend: {jax.default_backend()}")
    print(f"type(X_train): {type(X_train)}")
    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"X_test shape:  {X_test.shape}, dtype: {X_test.dtype}")
    print(f"y_test shape:  {y_test.shape}, dtype: {y_test.dtype}")
    print(f"Cache path: {output_path}")
    print()
    print("Notebook usage:")
    print("from quantbayes.ball_dp.experiments.embedding_io import load_embedding_npz")
    print(
        f"X_train, y_train, X_test, y_test = "
        f'load_embedding_npz(r"{output_path}", require_jax_gpu=False)'
    )


if __name__ == "__main__":
    main()
