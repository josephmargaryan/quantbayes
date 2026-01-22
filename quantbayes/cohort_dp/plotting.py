# cohort_dp/plotting.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_eps_sweep(
    rows: List[Dict[str, Any]], out_dir: str, title_suffix: str = ""
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def label(r):
        return f"{r['mechanism']}(k={r['k_service']})"

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(label(r), []).append(r)
    for k, v in groups.items():
        v.sort(key=lambda x: float(x["eps_total"]))

    # dist_ratio vs eps
    plt.figure()
    for k, v in groups.items():
        xs = [float(x["eps_total"]) for x in v]
        ys = [float(x["dist_ratio"]) for x in v]
        plt.plot(xs, ys, marker="o", label=k)
    plt.xlabel("eps_total (per API call)")
    plt.ylabel("dist_ratio (retrieved_mean / true_knn_mean)")
    plt.title(f"Utility vs Privacy (dist_ratio){title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(out_dir, "dist_ratio_vs_eps.png"), dpi=160, bbox_inches="tight"
    )
    plt.close()

    # attacker exact vs eps
    plt.figure()
    for k, v in groups.items():
        xs = [float(x["eps_total"]) for x in v]
        ys = [float(x["attacker_exact"]) for x in v]
        plt.plot(xs, ys, marker="o", label=k)
    plt.xlabel("eps_total (per API call)")
    plt.ylabel("attacker exact success")
    plt.title(f"Attack risk vs Privacy (exact ID){title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(out_dir, "attacker_exact_vs_eps.png"), dpi=160, bbox_inches="tight"
    )
    plt.close()

    # attacker within-r vs eps
    plt.figure()
    for k, v in groups.items():
        xs = [float(x["eps_total"]) for x in v]
        ys = [float(x["attacker_within_r"]) for x in v]
        plt.plot(xs, ys, marker="o", label=k)
    plt.xlabel("eps_total (per API call)")
    plt.ylabel("attacker within-r success")
    plt.title(f"Attack localization vs Privacy (within r){title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(out_dir, "attacker_within_r_vs_eps.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()


def plot_q_sweep(
    rows: List[Dict[str, Any]], out_dir: str, title_suffix: str = ""
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    groups: Dict[Tuple[str, int, float], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (str(r["mechanism"]), int(r["k_service"]), float(r["eps_total"]))
        groups.setdefault(key, []).append(r)

    for (mech, kserv, eps), v in groups.items():
        v.sort(key=lambda x: int(x["Q"]))
        xs = [int(x["Q"]) for x in v]
        ys = [float(x["attacker_exact"]) for x in v]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Q (number of queries)")
        plt.ylabel("attacker exact success")
        plt.title(f"Attacker exact vs Q: {mech}(k={kserv}), eps={eps}{title_suffix}")
        plt.grid(True, alpha=0.3)
        fname = f"attacker_exact_vs_Q__{mech}_k{kserv}_eps{eps}.png".replace(".", "p")
        plt.savefig(os.path.join(out_dir, fname), dpi=160, bbox_inches="tight")
        plt.close()
