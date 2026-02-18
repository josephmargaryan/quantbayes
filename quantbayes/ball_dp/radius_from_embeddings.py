#!/usr/bin/env python3
"""
radius_from_embeddings.py

Compute within-label kNN distances in embedding space, pick a policy radius r
as a percentile of those distances, plot a histogram, and optionally visualize
neighbor pairs near the chosen threshold.

Includes a runnable demo with CIFAR10 + ImageNet-pretrained ResNet18 embeddings.

Deps:
  pip install torch torchvision numpy scikit-learn matplotlib tqdm
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(x, **kwargs):
        return x


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class IndexedDataset(Dataset):
    """Wrap a torchvision dataset to also return the original index."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, y, idx


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denormalize_img(x: torch.Tensor) -> torch.Tensor:
    """
    x: (3,H,W) tensor normalized with ImageNet stats.
    returns tensor in [0,1] (approximately) for plotting.
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(3, 1, 1)
    y = x * std + mean
    return y.clamp(0.0, 1.0)


def build_resnet18_embedder(device: torch.device) -> torch.nn.Module:
    """
    Returns a feature extractor that outputs the penultimate ResNet18 embedding (512-d).
    """
    # torchvision >= 0.13 uses weights enums; older versions may differ.
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:  # fallback
        model = models.resnet18(pretrained=True)

    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


@torch.inference_mode()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    l2_normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      E: (N,d) float32 embeddings
      y: (N,) int labels
      idx: (N,) int dataset indices (original indices)
    """
    embs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    idxs: List[np.ndarray] = []

    for x, y, idx in tqdm(loader, desc="Extracting embeddings"):
        x = x.to(device, non_blocking=True)
        z = model(x)  # (B,d)
        if l2_normalize:
            z = torch.nn.functional.normalize(z, p=2, dim=1)
        embs.append(z.detach().cpu().numpy().astype(np.float32))
        ys.append(y.numpy().astype(np.int64))
        idxs.append(idx.numpy().astype(np.int64))

    E = np.concatenate(embs, axis=0)
    Y = np.concatenate(ys, axis=0)
    I = np.concatenate(idxs, axis=0)
    return E, Y, I


@dataclass
class WithinLabelKNNResult:
    distances: np.ndarray  # (M,)
    pairs: List[Tuple[int, int, float, int]]  # (idx, nn_idx, dist, label)


def within_label_knn_distances(
    E: np.ndarray,
    Y: np.ndarray,
    I: np.ndarray,
    *,
    k: int = 1,
    percentile_on: str = "kth",  # "nn" or "kth"
    max_per_class: Optional[int] = 2000,
    metric: str = "euclidean",
    seed: int = 0,
) -> WithinLabelKNNResult:
    """
    Computes within-label nearest neighbor distances (or k-th neighbor distances).

    k:
      - if k=1 -> nearest neighbor (excluding self)
      - if k>1 -> k-th neighbor (excluding self)
    max_per_class:
      - to keep it practical, we can subsample each class before building kNN.
        Set None to use all points (may be slow on CIFAR10).
    percentile_on:
      - "nn": always use nearest neighbor distance (dist[:,1])
      - "kth": use dist[:,k] (with dist[:,0]=self)
    """
    assert E.ndim == 2
    assert Y.shape[0] == E.shape[0] == I.shape[0]
    assert k >= 1

    rng = np.random.default_rng(seed)

    all_dists: List[float] = []
    all_pairs: List[Tuple[int, int, float, int]] = []

    classes = np.unique(Y)
    for c in classes:
        idxs_c = np.where(Y == c)[0]
        if idxs_c.size < 2:
            continue

        # Subsample per class for tractability
        if max_per_class is not None and idxs_c.size > max_per_class:
            idxs_c = rng.choice(idxs_c, size=max_per_class, replace=False)

        X = E[idxs_c]  # (Nc,d)

        n_neighbors = min(k + 1, X.shape[0])  # +1 because self is included
        if n_neighbors < 2:
            continue

        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="auto")
        nn.fit(X)
        dist, ind = nn.kneighbors(X, return_distance=True)

        # dist[:,0] should be 0 (self). Choose which column defines your distance stat.
        col = 1 if percentile_on == "nn" else min(k, dist.shape[1] - 1)

        for j in range(X.shape[0]):
            anchor_row = idxs_c[j]
            neigh_local = ind[j, col]
            neigh_row = idxs_c[neigh_local]

            d = float(dist[j, col])
            all_dists.append(d)

            anchor_idx = int(I[anchor_row])
            neigh_idx = int(I[neigh_row])
            all_pairs.append((anchor_idx, neigh_idx, d, int(c)))

    return WithinLabelKNNResult(
        distances=np.asarray(all_dists, dtype=np.float32),
        pairs=all_pairs,
    )


def plot_distance_histogram(
    distances: np.ndarray,
    *,
    r: float,
    bins: int = 60,
    title: str = "Within-label kNN distances",
    out_path: Optional[str] = None,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=bins)
    plt.axvline(r, linestyle="--", linewidth=2)
    plt.xlabel("distance")
    plt.ylabel("count")
    plt.title(title + f"  |  r = {r:.4f}")
    plt.grid(True, alpha=0.3)
    if out_path is not None:
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
    plt.show()


def save_neighbor_examples(
    base_dataset: Dataset,
    pairs: Sequence[Tuple[int, int, float, int]],
    *,
    out_path: str,
    max_rows: int = 8,
    title: str = "Neighbor examples near r",
) -> None:
    """
    Saves a 2-column grid: anchor | within-label neighbor
    base_dataset should be indexable by the original idx (e.g., CIFAR10 train set).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pairs = list(pairs)[:max_rows]
    n = len(pairs)
    if n == 0:
        print("No pairs provided for visualization.")
        return

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(6, 3 * n))
    if n == 1:
        axes = np.array([axes])  # make it (1,2)

    for row, (i, j, d, c) in enumerate(pairs):
        xi, yi = base_dataset[i]
        xj, yj = base_dataset[j]

        # xi/xj are already transformed (normalized); undo for display if tensor
        if torch.is_tensor(xi):
            xi_disp = denormalize_img(xi).permute(1, 2, 0).cpu().numpy()
            xj_disp = denormalize_img(xj).permute(1, 2, 0).cpu().numpy()
        else:
            # If dataset returns PIL, just use it directly
            xi_disp = xi
            xj_disp = xj

        axes[row, 0].imshow(xi_disp)
        axes[row, 0].axis("off")
        axes[row, 0].set_title(f"anchor idx={i}  y={yi}")

        axes[row, 1].imshow(xj_disp)
        axes[row, 1].axis("off")
        axes[row, 1].set_title(f"nn idx={j}  y={yj}  dist={d:.4f}")

    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved neighbor grid to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--subset",
        type=int,
        default=20000,
        help="Use a subset of CIFAR10 train set for speed. Set 0 to use full.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--k", type=int, default=1, help="Within-label kNN (excluding self)."
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=2000,
        help="Subsample per class before kNN. Set 0 to use all.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Pick r as this percentile of within-label distances.",
    )
    parser.add_argument(
        "--l2_normalize",
        action="store_true",
        help="If set, L2-normalize embeddings before distances (then euclidean ~ angular).",
    )

    parser.add_argument("--out_dir", type=str, default="./radius_out")
    parser.add_argument("--n_examples", type=int, default=8)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    os.makedirs(args.out_dir, exist_ok=True)

    # CIFAR10 -> resize to 224 for ImageNet-pretrained ResNet
    try:
        interp = transforms.InterpolationMode.BICUBIC
    except Exception:
        interp = transforms.InterpolationMode.BILINEAR

    tfm = transforms.Compose(
        [
            transforms.Resize(224, interpolation=interp),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    base_train = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=tfm
    )
    ds = IndexedDataset(base_train)

    if args.subset and args.subset > 0 and args.subset < len(ds):
        subset_idx = np.random.default_rng(args.seed).choice(
            len(ds), size=args.subset, replace=False
        )
        ds = Subset(ds, indices=subset_idx.tolist())

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_resnet18_embedder(device)
    E, Y, I = extract_embeddings(model, loader, device, l2_normalize=args.l2_normalize)

    max_per_class = None if (args.max_per_class == 0) else int(args.max_per_class)
    res = within_label_knn_distances(
        E,
        Y,
        I,
        k=int(args.k),
        max_per_class=max_per_class,
        metric="euclidean",
        seed=args.seed,
        percentile_on="nn" if args.k == 1 else "kth",
    )

    if res.distances.size == 0:
        raise RuntimeError("No within-label distances computed (too small subset?)")

    r = float(np.percentile(res.distances, args.percentile))
    print(f"Computed r at p{args.percentile:.1f}: r = {r:.6f}")
    print(
        f"Distance stats: mean={res.distances.mean():.6f}, std={res.distances.std():.6f}, "
        f"min={res.distances.min():.6f}, max={res.distances.max():.6f}"
    )

    hist_path = os.path.join(args.out_dir, "within_label_knn_hist.png")
    plot_distance_histogram(
        res.distances,
        r=r,
        out_path=hist_path,
        title=f"Within-label kNN distances (CIFAR10, ResNet18, subset={args.subset})",
    )

    # Pick examples “near r” for qualitative inspection
    pairs_sorted = sorted(res.pairs, key=lambda t: abs(t[2] - r))
    chosen = pairs_sorted[: max(1, int(args.n_examples))]
    grid_path = os.path.join(args.out_dir, "neighbor_examples_near_r.png")
    save_neighbor_examples(
        base_train, chosen, out_path=grid_path, max_rows=args.n_examples
    )


if __name__ == "__main__":
    main()
