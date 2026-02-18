# quantbayes/ball_dp/reconstruction/vision/plotting.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _to_hwc(img_chw: np.ndarray) -> np.ndarray:
    img_chw = np.asarray(img_chw, dtype=np.float32)
    if img_chw.ndim != 3:
        raise ValueError("Expected (C,H,W)")
    C, H, W = img_chw.shape
    if C == 1:
        return img_chw[0]
    if C == 3:
        return np.transpose(img_chw, (1, 2, 0))
    raise ValueError(f"Unsupported C={C}")


def save_recon_grid(
    orig: np.ndarray,
    recon: np.ndarray,
    out_path: str | Path,
    *,
    n: int = 16,
    title: Optional[str] = None,
) -> None:
    """
    Save a 2-column grid: original | reconstruction.
    orig, recon: (N,C,H,W) in [0,1]
    """
    import matplotlib.pyplot as plt

    orig = np.asarray(orig, dtype=np.float32)
    recon = np.asarray(recon, dtype=np.float32)
    if orig.shape != recon.shape:
        raise ValueError("orig and recon must have same shape (N,C,H,W)")
    if orig.ndim != 4:
        raise ValueError("Expected (N,C,H,W)")

    N = min(int(n), orig.shape[0])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows=N, ncols=2, figsize=(5, 2.2 * N))
    if N == 1:
        axes = np.array([axes])

    for i in range(N):
        ax0, ax1 = axes[i, 0], axes[i, 1]
        ax0.imshow(_to_hwc(orig[i]), cmap="gray" if orig.shape[1] == 1 else None)
        ax0.axis("off")
        ax0.set_title("orig")

        ax1.imshow(_to_hwc(recon[i]), cmap="gray" if recon.shape[1] == 1 else None)
        ax1.axis("off")
        ax1.set_title("recon")

    if title:
        fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_single_grid(
    imgs: np.ndarray,
    out_path: str | Path,
    *,
    n: int = 16,
    title: Optional[str] = None,
) -> None:
    """
    Save a 1-column grid of images (N,C,H,W).
    """
    import matplotlib.pyplot as plt

    imgs = np.asarray(imgs, dtype=np.float32)
    if imgs.ndim != 4:
        raise ValueError("Expected (N,C,H,W)")

    N = min(int(n), imgs.shape[0])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(3, 2.2 * N))
    if N == 1:
        axes = np.array([axes])

    for i in range(N):
        ax = axes[i]
        ax.imshow(_to_hwc(imgs[i]), cmap="gray" if imgs.shape[1] == 1 else None)
        ax.axis("off")

    if title:
        fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # quick smoke: save random MNIST-like images
    rng = np.random.default_rng(0)
    orig = rng.random((4, 1, 28, 28), dtype=np.float32)
    recon = rng.random((4, 1, 28, 28), dtype=np.float32)
    save_recon_grid(orig, recon, "tmp_recon_grid.png", n=4, title="smoke")
    print("[OK] plotting smoke wrote tmp_recon_grid.png")
