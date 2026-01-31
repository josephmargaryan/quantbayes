from __future__ import annotations
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


def _read_csv_grouped(
    path: str, group_keys: List[str], y_key: str
) -> Dict[tuple, List[float]]:
    groups: Dict[tuple, List[float]] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            key = tuple(float(row[k]) if k != "tag" else row[k] for k in group_keys)
            y = float(row[y_key])
            groups.setdefault(key, []).append(y)
    return groups


def plot_excess_vs_lambda(
    csv_path: str, tag: str, eps_values: List[float], out_png: str
) -> None:
    data = _read_csv_grouped(csv_path, ["tag", "epsilon", "lambda"], "value")
    plt.figure(figsize=(6.0, 4.0))
    for eps in eps_values:
        xs, ys = [], []
        for (t, e, lam), vals in sorted(data.items(), key=lambda kv: kv[0][2]):
            if t == tag and abs(e - eps) < 1e-12:
                xs.append(lam)
                ys.append(np.mean(vals))
        if xs:
            plt.semilogx(xs, ys, marker="o", linewidth=2, label=f"ε={eps}")
    plt.xlabel("λ (strong convexity, log scale)")
    plt.ylabel("Excess log-loss (train)")
    plt.title(f"{tag}: excess vs λ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_excess_vs_dim(
    csv_path: str, tag: str, eps_values: List[float], out_png: str
) -> None:
    data = _read_csv_grouped(csv_path, ["tag", "epsilon", "d"], "value")
    plt.figure(figsize=(6.0, 4.0))
    for eps in eps_values:
        xs, ys = [], []
        for (t, e, d), vals in sorted(data.items(), key=lambda kv: kv[0][2]):
            if t == tag and abs(e - eps) < 1e-12:
                xs.append(d)
                ys.append(np.mean(vals))
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=2, label=f"ε={eps}")
    plt.xlabel("dimension d")
    plt.ylabel("Excess log-loss (train)")
    plt.title(f"{tag}: excess vs d")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
