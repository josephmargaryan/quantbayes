# quantbayes/ball_dp/reconstruction/reporting.py
from __future__ import annotations

import csv
import json
import pathlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | pathlib.Path, obj: Any) -> None:
    path = pathlib.Path(path)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def save_csv(path: str | pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    path = pathlib.Path(path)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def psnr_from_mse(mse_val: float, *, max_val: float = 1.0) -> float:
    mse_val = float(mse_val)
    if mse_val <= 0:
        return float("inf")
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(mse_val))


def plot_hist(
    values: Sequence[float],
    *,
    title: str,
    xlabel: str,
    save_path: Optional[str | pathlib.Path] = None,
    bins: int = 30,
    show: bool = True,
) -> None:
    v = np.asarray(values, dtype=np.float64)
    plt.figure()
    plt.hist(v, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_curve(
    ys: Sequence[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Optional[str | pathlib.Path] = None,
    show: bool = True,
) -> None:
    y = np.asarray(ys, dtype=np.float64)
    plt.figure()
    plt.plot(y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_image_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    nrows: int,
    ncols: int,
    save_path: Optional[str | pathlib.Path] = None,
    show: bool = True,
) -> None:
    assert len(images) == len(titles)
    plt.figure(figsize=(2.3 * ncols, 2.3 * nrows))
    for i, (img, ttl) in enumerate(zip(images, titles)):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.set_title(ttl, fontsize=8)
        ax.axis("off")
        im = np.asarray(img)
        if im.ndim == 2:
            ax.imshow(im, cmap="gray", vmin=0.0, vmax=1.0)
        elif im.ndim == 3 and im.shape[0] in (1, 3):  # CHW -> HWC
            ax.imshow(np.transpose(im, (1, 2, 0)))
        else:
            ax.imshow(im)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=200)
    if show:
        plt.show()
    plt.close()
