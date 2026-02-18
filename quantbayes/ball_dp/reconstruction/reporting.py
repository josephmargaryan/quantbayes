# quantbayes/ball_dp/reconstruction/reporting.py
from __future__ import annotations

import csv
import json
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
    plt.figure(figsize=(2.2 * ncols, 2.2 * nrows))
    for i, (img, ttl) in enumerate(zip(images, titles)):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.set_title(ttl, fontsize=9)
        ax.axis("off")
        im = np.asarray(img)
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        elif im.ndim == 3 and im.shape[0] in (1, 3):  # CHW -> HWC
            ax.imshow(np.transpose(im, (1, 2, 0)))
        elif im.ndim == 3:
            ax.imshow(im)
        else:
            ax.imshow(im.reshape(28, 28), cmap="gray")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=200)
    if show:
        plt.show()
    plt.close()
