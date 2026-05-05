#!/usr/bin/env python3
"""Create an improved figure set for a completed nonconvex thesis run.

Usage
-----
python make_nonconvex_figures2_updated.py \
  --run-dir ~/Desktop/quantbayes_cluster_results/nonconvex_thesis_4625

The script expects the CSV outputs produced by
`run_nonconvex_thesis_experiment.py` and writes a new `figures2/` directory
inside the run directory.

Compared with the first figures2 script, this version:
  * moves attack legends outside the plotting area;
  * keeps missing ERM unknown-inclusion entries visible as N/A;
  * adds an explicit Gamma_ball-vs-Gamma_std finite-prior bound-tightness figure;
  * clarifies the ReRo curve titles so they are not mistaken for a direct
    Ball-bound-vs-standard-bound comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

MODEL_ORDER = ["ERM", "Ball-DP", "Std-DP"]
PRIVATE_MODEL_ORDER = ["Ball-DP", "Std-DP"]
MODEL_COLORS = {
    "ERM": "tab:gray",
    "Ball-DP": "tab:blue",
    "Std-DP": "tab:orange",
}
MODE_LABELS = {
    "known_inclusion": r"known / revealed",
    "unknown_inclusion": r"unknown / hidden",
    "rdp": r"RDP",
}
MODE_TITLES = {
    "known_inclusion": r"Known-inclusion / revealed-transcript",
    "unknown_inclusion": r"Unknown-inclusion / hidden-transcript",
    "rdp": r"RDP conversion",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 260,
            "figure.figsize": (7.2, 4.8),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.0,
            "axes.titlesize": 12.5,
            "legend.frameon": False,
            "legend.fontsize": 9.6,
            "font.size": 10.5,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def ordered_models(values: Iterable[str]) -> list[str]:
    uniq = list(dict.fromkeys(values))
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(uniq, key=lambda x: rank.get(x, 999))


def load_csv(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def finite_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def mean_ci(series: pd.Series, z: float = 1.96) -> tuple[float, float, float]:
    x = pd.Series(series).dropna().astype(float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    mu = float(np.mean(x))
    if len(x) == 1:
        return (mu, mu, mu)
    half = float(z * np.std(x, ddof=1) / np.sqrt(len(x)))
    return (mu, mu - half, mu + half)


def write_notes(
    fig_dir: Path,
    data_summary: pd.DataFrame,
    feasibility: pd.DataFrame,
    theorem: pd.DataFrame,
) -> None:
    notes: list[str] = []
    notes.append("Nonconvex figures2 notes\n")
    if not data_summary.empty:
        notes.append("\nData summary:\n")
        notes.append(data_summary.to_string(index=False) + "\n")
    if not feasibility.empty:
        row = feasibility.iloc[0]
        notes.append(
            "\nFeasibility interpretation:\n"
            f"At radius r={row['radius']:.2f} and support size m={int(row['support_size_m'])}, "
            f"there are {int(row['anchors_with_at_least_m_candidates'])} private anchors with at least m same-label public candidates. "
            f"The maximum candidate-bank size is {int(row['max_candidate_count'])}, the median is {row['median_candidate_count']:.2f}, "
            f"and the mean is {row['mean_candidate_count']:.2f}.\n"
        )
    if not theorem.empty:
        row = theorem.iloc[0]
        notes.append(
            "\nTheorem constants:\n"
            f"certified L_z = {row['certified_lz']:.6g}, "
            f"delta_ball = {row['delta_ball']:.6g}, delta_standard = {row['delta_standard']:.6g}, "
            f"delta_ball / delta_standard = {row['ball_to_standard_ratio']:.4f}.\n"
        )
    notes.append(
        "\nFigure guide:\n"
        "- fig01_feasibility: selected-support feasibility for the synthetic data.\n"
        "- fig02_utility: aggregate mean public accuracy with confidence intervals.\n"
        "- fig03_rero_known and fig04_rero_unknown: representative mechanism-level ReRo curves. These compare trained mechanisms, not Ball-vs-standard certification of the same release.\n"
        "- fig05_attack_known and fig06_attack_unknown: empirical exact-ID success with chance baseline and theorem-side bounds. Missing ERM/unknown is marked N/A because zero-noise ERM makes the hidden-inclusion Gaussian mixture likelihood singular.\n"
        "- fig07_noise_vs_utility: utility versus calibrated noise, with average empirical exact-ID success annotated.\n"
        "- fig09_bound_tightness: direct finite-prior comparison of Gamma_ball versus Gamma_std for the same private release and mode.\n"
    )
    (fig_dir / "README_figures2.txt").write_text("".join(notes))


def plot_feasibility(
    fig_dir: Path, support_feasibility: pd.DataFrame, supports: pd.DataFrame
) -> None:
    if support_feasibility.empty or supports.empty:
        return
    row = support_feasibility.iloc[0]
    m = int(row["support_size_m"])
    radius = float(row["radius"])

    support_level = supports.groupby(
        ["support_id", "center_source_id", "support_hash"], as_index=False
    ).agg(bank_size=("bank_size", "first"))
    bank_sizes = finite_float(support_level["bank_size"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.1), constrained_layout=True)

    bins = min(12, max(4, len(np.unique(bank_sizes))))
    axes[0].hist(bank_sizes, bins=bins, alpha=0.88)
    axes[0].axvline(
        m,
        linestyle="--",
        linewidth=1.6,
        color="black",
        label=rf"support size threshold $m={m}$",
    )
    axes[0].set_xlabel(r"feasible candidate-bank size")
    axes[0].set_ylabel(r"count of selected support anchors")
    axes[0].set_title(rf"Selected finite supports at radius $r={radius:.2f}$")
    axes[0].legend(loc="upper left")

    sorted_sizes = np.sort(bank_sizes)
    axes[1].scatter(np.arange(len(sorted_sizes)), sorted_sizes, s=28, alpha=0.9)
    axes[1].axhline(
        m,
        linestyle="--",
        linewidth=1.6,
        color="black",
        label=rf"minimum support size $m={m}$",
    )
    axes[1].set_xlabel(r"selected support index")
    axes[1].set_ylabel(r"candidate-bank size")
    axes[1].set_title(r"Bank sizes for the selected supports")
    axes[1].legend(loc="upper left")

    note = (
        f"Summary: {int(row['anchors_with_at_least_m_candidates'])} private anchors "
        f"have at least m={m} same-label public candidates."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=9.6)
    savefig(fig, fig_dir / "fig01_feasibility")


def plot_utility(fig_dir: Path, utility_summary: pd.DataFrame) -> None:
    if utility_summary.empty:
        return
    models = ordered_models(utility_summary["model"])
    sub = utility_summary.set_index("model").loc[models]

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    x = np.arange(len(sub))
    mean_acc = finite_float(sub["mean_accuracy"])
    err_low = mean_acc - finite_float(sub["accuracy_ci_low"])
    err_high = finite_float(sub["accuracy_ci_high"]) - mean_acc
    colors = [MODEL_COLORS.get(m, "tab:gray") for m in sub.index]

    ax.bar(
        x,
        mean_acc,
        yerr=np.vstack([err_low, err_high]),
        capsize=4,
        alpha=0.88,
        color=colors,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sub.index)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(r"mean public accuracy")
    ax.set_title(r"Aggregate utility across replacement trials")

    for i, row in enumerate(sub.itertuples()):
        eps_text = (
            r"$\varepsilon=\infty$"
            if not np.isfinite(row.mean_epsilon_primary)
            else rf"$\bar{{\varepsilon}}={row.mean_epsilon_primary:.2f}$"
        )
        noise_text = rf"$\bar{{\sigma}}={row.mean_noise_multiplier:.2f}$"
        n_text = rf"$n={int(row.n_trials)}$"
        ax.text(
            i,
            min(1.03, row.mean_accuracy + 0.05),
            eps_text + "\n" + noise_text + "\n" + n_text,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    savefig(fig, fig_dir / "fig02_utility")


def plot_rero_split(
    fig_dir: Path,
    curves: pd.DataFrame,
    include_mode: str,
    fname: str,
    title: str,
) -> None:
    if curves.empty:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.9), constrained_layout=True)
    kappa_vals = np.sort(finite_float(curves["kappa"]))
    kappa_vals = np.unique(kappa_vals[np.isfinite(kappa_vals)])
    ax.plot(
        kappa_vals,
        kappa_vals,
        linestyle="--",
        linewidth=1.25,
        color="black",
        label=r"baseline $\kappa$",
    )

    # These curves compare mechanisms under the plotted bound value. They do not compare
    # Gamma_ball and Gamma_std for the same release; fig09 does that explicitly.
    line_styles = {include_mode: "-", "rdp": ":"}
    include_modes = [include_mode, "rdp"]
    for model_name in PRIVATE_MODEL_ORDER:
        for mode_key in include_modes:
            sub = curves[
                (curves["model"] == model_name) & (curves["mode"] == mode_key)
            ].sort_values("kappa")
            if sub.empty:
                continue
            ax.plot(
                sub["kappa"],
                sub["gamma_ball"],
                color=MODEL_COLORS[model_name],
                linestyle=line_styles[mode_key],
                linewidth=2.0 if mode_key == include_mode else 1.8,
                label=rf"{model_name}, {MODE_LABELS[mode_key]}",
            )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"prior mass $\kappa$")
    ax.set_ylabel(r"plotted bound value $\Gamma_{\mathrm{ball}}(\kappa)$")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=1)
    fig.text(
        0.5,
        -0.02,
        r"Lower curves indicate stronger protection by the trained mechanism; direct Ball-vs-standard bound tightness is shown separately in Fig. 09.",
        ha="center",
        va="top",
        fontsize=9.0,
    )
    savefig(fig, fig_dir / fname)


def plot_attack_mode(
    fig_dir: Path,
    attack_summary: pd.DataFrame,
    finite_bound_summary: pd.DataFrame,
    mode: str,
    fname: str,
    title: str,
) -> None:
    if attack_summary.empty:
        return

    # Force all three model categories to appear. ERM/unknown is usually absent because
    # the zero-noise hidden-inclusion likelihood is singular, so we mark it N/A.
    full_models = pd.DataFrame({"model": MODEL_ORDER})
    att = attack_summary[attack_summary["mode"] == mode].copy()
    att = full_models.merge(att, on="model", how="left")
    att["mode"] = att["mode"].fillna(mode)
    att["mode_label"] = att["mode_label"].fillna(MODE_LABELS[mode])

    if not finite_bound_summary.empty:
        att = att.merge(
            finite_bound_summary[
                ["model", "mode", "mean_gamma_ball", "mean_gamma_standard"]
            ],
            on=["model", "mode"],
            how="left",
        )

    fig, ax = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
    x = np.arange(len(att))

    empirical = finite_float(att["empirical_exact_id"])
    mask_bar = np.isfinite(empirical)
    if np.any(mask_bar):
        bar_heights = empirical[mask_bar]
        err_low = bar_heights - finite_float(att.loc[mask_bar, "exact_id_ci_low"])
        err_high = finite_float(att.loc[mask_bar, "exact_id_ci_high"]) - bar_heights
        colors = [MODEL_COLORS.get(m, "tab:gray") for m in att.loc[mask_bar, "model"]]
        ax.bar(
            x[mask_bar],
            bar_heights,
            yerr=np.vstack([err_low, err_high]),
            capsize=4,
            alpha=0.88,
            color=colors,
        )

    # Mark missing rows explicitly, rather than silently dropping a model category.
    for i, row in enumerate(att.itertuples()):
        if not np.isfinite(getattr(row, "empirical_exact_id", np.nan)):
            ax.text(
                i,
                0.08,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=10,
                fontstyle="italic",
            )

    chance_vals = finite_float(att["chance_kappa"])
    chance = (
        float(np.nanmean(chance_vals)) if np.any(np.isfinite(chance_vals)) else np.nan
    )
    if np.isfinite(chance):
        ax.axhline(
            chance,
            linestyle="--",
            linewidth=1.5,
            color="black",
            label=rf"chance baseline $\kappa=1/m={chance:.3f}$",
        )

    priv = att[att["model"].isin(PRIVATE_MODEL_ORDER)].copy()
    model_to_x = {m: i for i, m in enumerate(att["model"].tolist())}
    x_priv = np.array([model_to_x[m] for m in priv["model"]])

    if "mean_gamma_ball" in priv.columns:
        vals = finite_float(priv["mean_gamma_ball"])
        mask = np.isfinite(vals)
        ax.scatter(
            x_priv[mask] - 0.08,
            vals[mask],
            s=78,
            marker="o",
            facecolors="none",
            linewidths=1.8,
            edgecolors="tab:blue",
            label=r"Ball bound $\overline{\Gamma}_{\mathrm{ball}}$",
            zorder=5,
        )
    if "mean_gamma_standard" in priv.columns:
        vals = finite_float(priv["mean_gamma_standard"])
        mask = np.isfinite(vals)
        ax.scatter(
            x_priv[mask] + 0.08,
            vals[mask],
            s=78,
            marker="s",
            facecolors="none",
            linewidths=1.8,
            edgecolors="tab:red",
            label=r"standard bound $\overline{\Gamma}_{\mathrm{std}}$",
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(att["model"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(r"empirical exact-ID success")
    ax.set_title(title)

    # Put legend above the plot so it cannot cover the ERM bar.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),  # 1.02
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    for i, row in enumerate(att.itertuples()):
        if np.isfinite(getattr(row, "empirical_exact_id", np.nan)) and np.isfinite(
            getattr(row, "n_trials", np.nan)
        ):
            ax.text(
                i,
                min(1.02, row.empirical_exact_id + 0.05),
                rf"$n={int(row.n_trials)}$",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    savefig(fig, fig_dir / fname)


def plot_noise_vs_utility(
    fig_dir: Path, utility_summary: pd.DataFrame, attack_summary: pd.DataFrame
) -> None:
    if utility_summary.empty:
        return
    protection = (
        attack_summary.groupby("model", as_index=False).agg(
            mean_exact_id=("empirical_exact_id", "mean")
        )
        if not attack_summary.empty
        else pd.DataFrame(columns=["model", "mean_exact_id"])
    )
    df = utility_summary.merge(protection, on="model", how="left")
    models = ordered_models(df["model"])
    df = df.set_index("model").loc[models].reset_index()

    fig, ax = plt.subplots(figsize=(7.3, 4.9), constrained_layout=True)
    max_noise = max(1.0, float(np.nanmax(finite_float(df["mean_noise_multiplier"]))))
    for row in df.itertuples():
        x = float(row.mean_noise_multiplier)
        y = float(row.mean_accuracy)
        c = MODEL_COLORS.get(row.model, "tab:gray")
        ax.scatter([x], [y], s=120, color=c, zorder=4)
        label = row.model
        if np.isfinite(getattr(row, "mean_exact_id", np.nan)):
            label += f"\nmean exact-ID={row.mean_exact_id:.3f}"
        ax.text(x + 0.03 * max_noise, y + 0.005, label, fontsize=9.2, va="bottom")

    ax.set_xlabel(r"mean calibrated noise multiplier $\bar{\sigma}$")
    ax.set_ylabel(r"mean public accuracy")
    ax.set_title(r"Protection at lower noise: utility versus calibrated noise")
    ax.set_xlim(left=-0.05 * max_noise)
    ax.set_ylim(0.0, 1.05)
    fig.text(
        0.5,
        -0.02,
        "Interpretation: Ball-DP and standard DP both suppress the Bayesian attack, but Ball-DP uses less noise and keeps higher utility.",
        ha="center",
        va="top",
        fontsize=9.2,
    )
    savefig(fig, fig_dir / "fig07_noise_vs_utility")


def plot_training_curves(fig_dir: Path, history: pd.DataFrame) -> None:
    if history.empty:
        return
    rows: list[dict[str, object]] = []
    for (model, step), grp in history.groupby(["model", "step"]):
        mu, lo, hi = mean_ci(grp["public_eval_accuracy"])
        rows.append(
            {"model": model, "step": int(step), "mean": mu, "ci_low": lo, "ci_high": hi}
        )
    hist_summary = pd.DataFrame(rows)
    hist_summary.to_csv(fig_dir / "history_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.4, 4.7), constrained_layout=True)
    for model in ordered_models(hist_summary["model"]):
        sub = hist_summary[hist_summary["model"] == model].sort_values("step")
        ax.plot(
            sub["step"],
            sub["mean"],
            linewidth=2.0,
            color=MODEL_COLORS.get(model, "tab:gray"),
            label=model,
        )
        ax.fill_between(
            sub["step"],
            sub["ci_low"],
            sub["ci_high"],
            alpha=0.18,
            color=MODEL_COLORS.get(model, "tab:gray"),
        )
    ax.set_xlabel(r"training step $t$")
    ax.set_ylabel(r"public accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(r"Aggregate public utility during training")
    ax.legend(loc="lower right")
    savefig(fig, fig_dir / "fig08_training_curves")


def plot_bound_tightness(fig_dir: Path, finite_bound_summary: pd.DataFrame) -> None:
    """Directly compare Gamma_ball and Gamma_std for each private release/mode.

    This is the figure to use for the claim that the Ball-side finite-prior
    certificate is tighter than the standard certificate for the same trained
    release and attack mode.
    """
    if finite_bound_summary.empty:
        return
    df = finite_bound_summary[
        finite_bound_summary["mode"].isin(["known_inclusion", "unknown_inclusion"])
        & finite_bound_summary["model"].isin(PRIVATE_MODEL_ORDER)
    ].copy()
    if df.empty:
        return

    model_rank = {m: i for i, m in enumerate(PRIVATE_MODEL_ORDER)}
    mode_rank = {"known_inclusion": 0, "unknown_inclusion": 1}
    df["_model_rank"] = df["model"].map(model_rank)
    df["_mode_rank"] = df["mode"].map(mode_rank)
    df = df.sort_values(["_model_rank", "_mode_rank"]).reset_index(drop=True)

    labels = [f"{row.model}\n{MODE_LABELS[row.mode]}" for row in df.itertuples()]
    y = np.arange(len(df))[::-1]
    gamma_ball = finite_float(df["mean_gamma_ball"])
    gamma_std = finite_float(df["mean_gamma_standard"])

    fig, ax = plt.subplots(figsize=(8.0, 4.9), constrained_layout=True)
    for i, row in enumerate(df.itertuples()):
        yi = y[i]
        gb = float(gamma_ball[i])
        gs = float(gamma_std[i])
        color = MODEL_COLORS.get(row.model, "tab:gray")
        ax.plot([gb, gs], [yi, yi], color=color, linewidth=2.0, alpha=0.75)
        ax.scatter([gb], [yi], s=85, marker="o", color="tab:blue", zorder=5)
        ax.scatter(
            [gs],
            [yi],
            s=85,
            marker="s",
            facecolors="white",
            edgecolors="tab:red",
            linewidths=1.9,
            zorder=5,
        )
        improvement = gs - gb
        ax.text(
            min(1.02, max(gb, gs) + 0.025),
            yi,
            rf"$\Delta={improvement:.3f}$",
            va="center",
            fontsize=9.0,
        )

    chance = None
    if "mean_kappa" in df.columns:
        vals = finite_float(df["mean_kappa"])
        if np.any(np.isfinite(vals)):
            chance = float(np.nanmean(vals))
    if chance is not None and np.isfinite(chance):
        ax.axvline(
            chance,
            linestyle="--",
            linewidth=1.3,
            color="black",
            label=rf"chance baseline $\kappa={chance:.3f}$",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel(r"finite-prior upper bound $\Gamma$")
    ax.set_title(
        r"Direct bound tightness: $\Gamma_{\mathrm{ball}}$ vs. $\Gamma_{\mathrm{std}}$"
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="tab:blue",
            linestyle="None",
            markersize=8,
            label=r"Ball bound $\Gamma_{\mathrm{ball}}$",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markerfacecolor="white",
            markeredgecolor="tab:red",
            color="tab:red",
            linestyle="None",
            markersize=8,
            label=r"standard bound $\Gamma_{\mathrm{std}}$",
        ),
    ]
    if chance is not None and np.isfinite(chance):
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                lw=1.3,
                label=r"chance baseline",
            )
        )
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    fig.text(
        0.5,
        -0.02,
        r"For each fixed release and attack mode, smaller $\Gamma$ means a tighter reconstruction-risk certificate.",
        ha="center",
        va="top",
        fontsize=9.2,
    )
    savefig(fig, fig_dir / "fig09_bound_tightness")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Make improved figures2 for a completed nonconvex thesis run."
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing the CSV outputs from run_nonconvex_thesis_experiment.py",
    )
    args = ap.parse_args()

    configure_matplotlib()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    fig_dir = ensure_dir(run_dir / "figures2")

    utility_summary = load_csv(run_dir, "utility_summary.csv")
    attack_summary = load_csv(run_dir, "attack_summary.csv")
    finite_bound_summary = load_csv(run_dir, "finite_bound_summary.csv")
    support_feasibility = load_csv(run_dir, "support_feasibility.csv")
    supports = load_csv(run_dir, "supports.csv")
    curves = load_csv(run_dir, "curves.csv")
    history = load_csv(run_dir, "history.csv")
    theorem_summary = load_csv(run_dir, "theorem_summary.csv")
    data_summary = load_csv(run_dir, "data_summary.csv")

    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        config = json.loads(cfg_path.read_text())
        (fig_dir / "copied_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True)
        )

    write_notes(fig_dir, data_summary, support_feasibility, theorem_summary)
    plot_feasibility(fig_dir, support_feasibility, supports)
    plot_utility(fig_dir, utility_summary)
    plot_rero_split(
        fig_dir,
        curves,
        "known_inclusion",
        "fig03_rero_known",
        r"Mechanism-level ReRo curves: known / revealed and RDP",
    )
    plot_rero_split(
        fig_dir,
        curves,
        "unknown_inclusion",
        "fig04_rero_unknown",
        r"Mechanism-level ReRo curves: unknown / hidden and RDP",
    )
    plot_attack_mode(
        fig_dir,
        attack_summary,
        finite_bound_summary,
        "known_inclusion",
        "fig05_attack_known",
        r"Known-inclusion exact-ID success vs. theorem-side bounds",
    )
    plot_attack_mode(
        fig_dir,
        attack_summary,
        finite_bound_summary,
        "unknown_inclusion",
        "fig06_attack_unknown",
        r"Unknown-inclusion exact-ID success vs. theorem-side bounds",
    )
    plot_noise_vs_utility(fig_dir, utility_summary, attack_summary)
    plot_training_curves(fig_dir, history)
    plot_bound_tightness(fig_dir, finite_bound_summary)

    print(f"Wrote improved figures to: {fig_dir}")


if __name__ == "__main__":
    main()
