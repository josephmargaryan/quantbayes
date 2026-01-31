# quantbayes/ball_dp/utils/plotting.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def save_errorbar_plot(
    x: List[float],
    curves: Dict[str, Tuple[List[float], List[float]]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str | Path,
    xscale_log: bool = True,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for label, (means, stds) in curves.items():
        plt.errorbar(x, means, yerr=stds, marker="o", linewidth=1, label=label)
    if xscale_log:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_line_plot(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(x, y, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
