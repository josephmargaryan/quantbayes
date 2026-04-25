#!/usr/bin/env python3
"""Aggregate official Paper 3 decentralized experiment outputs.

Compatible with:

    run_decentralized_observer_experiment.py
    run_decentralized_map_attack_experiment.py
    run_decentralized_prototype_utility_experiment.py

Expected layout:

    <results-root>/paper3/decentralized_observer/<dataset_tag>/
        observer_rows.csv
        observer_summary.csv
        metadata.json

    <results-root>/paper3/decentralized_map_attack/<dataset_tag>/
        map_attack_rows.csv
        map_attack_summary.csv
        metadata.json

    <results-root>/paper3/decentralized_prototype_utility/<dataset_tag>/
        prototype_utility_rows.csv
        prototype_utility_summary.csv
        metadata.json

This script writes:
    <results-root>/paper3/aggregate/
        all_<kind>_rows.csv
        all_<kind>_summary.csv
        publication tables .csv/.tex
        publication figures .pdf/.png
        aggregate_decentralized_paper3_manifest.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from quantbayes.ball_dp.serialization import save_dataframe

KIND_TO_DIR_AND_FILES = {
    "observer": ("decentralized_observer", "observer_rows.csv", "observer_summary.csv"),
    "map_attack": (
        "decentralized_map_attack",
        "map_attack_rows.csv",
        "map_attack_summary.csv",
    ),
    "utility": (
        "decentralized_prototype_utility",
        "prototype_utility_rows.csv",
        "prototype_utility_summary.csv",
    ),
}

BALL_COLOR = "#0072B2"
STANDARD_COLOR = "#D55E00"
BASELINE_COLOR = "#4D4D4D"
RDP_COLOR = "#009E73"
ALT_COLOR = "#CC79A7"

DEFAULT_REP_EPSILON = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=str, default="results_paper3")
    parser.add_argument(
        "--kind",
        choices=["observer", "map_attack", "utility", "all"],
        default="all",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset tags or display names to include.",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Optional graph to use for graph-specific publication slices.",
    )
    parser.add_argument(
        "--observer-mode",
        type=str,
        default=None,
        help="Optional observer mode for observer/map-attack publication slices.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Optional numeric radius. If omitted, inferred from available radii.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help="Optional noise std for observer/map-attack publication slices.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Optional target epsilon for utility publication slices.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.8, 4.9),
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
        }
    )


def normalize_name(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")


def wanted_dataset_filter(datasets: Iterable[str] | None) -> set[str] | None:
    if datasets is None:
        return None
    return {normalize_name(ds) for ds in datasets}


def savefig_stem(fig: plt.Figure, stem: str | Path) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def latex_escape_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("_", "\\_") for c in out.columns]
    return out


def save_table(df: pd.DataFrame, out_stem: str | Path) -> None:
    out_stem = Path(out_stem)
    save_dataframe(df, out_stem.with_suffix(".csv"), save_parquet_if_possible=False)
    tex_df = latex_escape_columns(df)
    tex = tex_df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.4f}")
    out_stem.with_suffix(".tex").write_text(tex)


def safe_read_csv(path: Path, *, strict: bool) -> pd.DataFrame | None:
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"Expected file not found: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Could not read {path}: {exc}") from exc
        print(f"Skipping unreadable {path}: {exc}", flush=True)
        return None


def read_metadata(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def dataset_matches(
    dataset_dir: Path, metadata: dict[str, Any], wanted: set[str] | None
) -> bool:
    if wanted is None:
        return True
    keys = {
        normalize_name(dataset_dir.name),
        normalize_name(str(metadata.get("dataset_tag", ""))),
        normalize_name(str(metadata.get("dataset", ""))),
    }
    return bool(keys & wanted)


def add_source_columns(
    df: pd.DataFrame,
    *,
    dataset_dir: Path,
    source_path: Path,
    metadata: dict[str, Any],
    kind: str,
) -> pd.DataFrame:
    out = df.copy()

    dataset_tag = str(metadata.get("dataset_tag", dataset_dir.name))
    dataset_name = str(metadata.get("dataset", dataset_tag))

    if "dataset_tag" not in out.columns:
        out["dataset_tag"] = dataset_tag
    if "dataset_name" not in out.columns:
        out["dataset_name"] = dataset_name

    out["source_kind"] = str(kind)
    out["source_path"] = str(source_path)
    out["source_dataset_dir"] = str(dataset_dir)
    out["metadata_dataset_tag"] = dataset_tag
    out["metadata_dataset_name"] = dataset_name

    if "feature_dim" in metadata:
        out["metadata_feature_dim"] = metadata.get("feature_dim")
    if "num_classes" in metadata:
        out["metadata_num_classes"] = metadata.get("num_classes")

    return out


def discover_dataset_dirs(search_root: Path) -> list[Path]:
    if not search_root.exists():
        return []
    return sorted(
        p
        for p in search_root.iterdir()
        if p.is_dir() and p.name not in {"aggregate", "_aggregate", "figures"}
    )


def concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def deduplicate(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    for key in keys:
        if key in df.columns:
            return df.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)
    return df.drop_duplicates().reset_index(drop=True)


def metric_col(df: pd.DataFrame, base: str) -> str | None:
    for col in (f"{base}_mean", base):
        if col in df.columns:
            return col
    return None


def metric_values(df: pd.DataFrame, base: str) -> pd.Series:
    col = metric_col(df, base)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def finite_unique(series: pd.Series) -> list[float]:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return sorted(float(v) for v in np.unique(vals))


def choose_numeric(
    series: pd.Series,
    *,
    requested: float | None,
    prefer: float | None = None,
    policy: str = "middle",
) -> float:
    vals = finite_unique(series)
    if not vals:
        raise RuntimeError(
            "No finite numeric values available for representative selection."
        )

    if requested is not None:
        return min(vals, key=lambda v: abs(v - float(requested)))

    if prefer is not None:
        return min(vals, key=lambda v: abs(v - float(prefer)))

    if policy == "first":
        return vals[0]
    if policy == "second":
        return vals[min(1, len(vals) - 1)]
    if policy == "last":
        return vals[-1]

    return vals[len(vals) // 2]


def choose_category(
    series: pd.Series,
    *,
    requested: str | None,
    prefer_order: Sequence[str],
) -> str:
    vals = [str(v) for v in sorted(series.dropna().astype(str).unique())]
    if not vals:
        raise RuntimeError("No category values available for representative selection.")

    if requested is not None:
        req = str(requested)
        if req in vals:
            return req
        normalized = {normalize_name(v): v for v in vals}
        if normalize_name(req) in normalized:
            return normalized[normalize_name(req)]
        return vals[0]

    for preferred in prefer_order:
        if preferred in vals:
            return preferred

    return vals[0]


def isclose_series(series: pd.Series, value: float) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return pd.Series(
        np.isclose(vals, float(value), atol=1e-9, rtol=1e-9), index=series.index
    )


def compact_label(row: pd.Series) -> str:
    dataset = str(row.get("dataset_name", row.get("dataset_tag", "")))
    graph = str(row.get("graph", ""))
    if dataset and graph:
        return f"{dataset}\n{graph}"
    return dataset or graph


def aggregate_kind(
    *,
    results_root: Path,
    kind: str,
    wanted: set[str] | None,
    strict: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    subdir, rows_name, summary_name = KIND_TO_DIR_AND_FILES[kind]
    search_root = results_root / "paper3" / subdir

    row_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    issues: list[dict[str, Any]] = []
    included_datasets: list[str] = []

    if not search_root.exists():
        issue = {
            "kind": kind,
            "reason": "missing search root",
            "path": str(search_root),
        }
        if strict:
            raise FileNotFoundError(f"Missing search root: {search_root}")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "status": "missing_root",
                "search_root": str(search_root),
                "issues": [issue],
                "included_datasets": [],
            },
        )

    for dataset_dir in discover_dataset_dirs(search_root):
        metadata = read_metadata(dataset_dir)
        if not dataset_matches(dataset_dir, metadata, wanted):
            continue

        included_datasets.append(dataset_dir.name)

        rows_path = dataset_dir / rows_name
        summary_path = dataset_dir / summary_name

        rows_df = safe_read_csv(rows_path, strict=strict)
        if rows_df is not None:
            row_frames.append(
                add_source_columns(
                    rows_df,
                    dataset_dir=dataset_dir,
                    source_path=rows_path,
                    metadata=metadata,
                    kind=kind,
                )
            )
        else:
            issues.append(
                {
                    "dataset_dir": dataset_dir.name,
                    "reason": f"missing {rows_name}",
                    "path": str(rows_path),
                }
            )

        summary_df = safe_read_csv(summary_path, strict=strict)
        if summary_df is not None:
            summary_frames.append(
                add_source_columns(
                    summary_df,
                    dataset_dir=dataset_dir,
                    source_path=summary_path,
                    metadata=metadata,
                    kind=kind,
                )
            )
        else:
            issues.append(
                {
                    "dataset_dir": dataset_dir.name,
                    "reason": f"missing {summary_name}",
                    "path": str(summary_path),
                }
            )

    rows = concat_or_empty(row_frames)
    summaries = concat_or_empty(summary_frames)

    rows = deduplicate(rows, keys=["row_key", "trial_key"])
    summaries = summaries.drop_duplicates().reset_index(drop=True)

    status = "ok" if not rows.empty or not summaries.empty else "empty"
    meta = {
        "status": status,
        "search_root": str(search_root),
        "included_datasets": included_datasets,
        "num_datasets": int(len(included_datasets)),
        "num_rows": int(len(rows)),
        "num_summary_rows": int(len(summaries)),
        "issues": issues,
    }
    return rows, summaries, meta


def make_bar_figure(
    df: pd.DataFrame,
    *,
    label_col: str,
    bars: list[tuple[str, str, str, str | None]],
    y_label: str,
    title: str,
    out_stem: Path,
    ylim: tuple[float, float] | None = None,
    yscale: str | None = None,
    baseline_col: str | None = None,
    baseline_label: str | None = None,
) -> None:
    if df.empty or not bars:
        return

    valid_bars = [
        (col, label, color, hatch)
        for col, label, color, hatch in bars
        if col in df.columns
    ]
    if not valid_bars:
        return

    n = len(valid_bars)
    x = np.arange(len(df), dtype=float)
    total_width = min(0.82, 0.24 * n)
    width = total_width / float(n)

    fig, ax = plt.subplots(figsize=(max(8.4, 0.9 * len(df) + 3.5), 4.9))

    for i, (col, label, color, hatch) in enumerate(valid_bars):
        offset = (i - (n - 1) / 2.0) * width
        ax.bar(
            x + offset,
            pd.to_numeric(df[col], errors="coerce"),
            width=width,
            color=color,
            edgecolor="#222222",
            linewidth=0.5,
            hatch=hatch,
            label=label,
        )

    if baseline_col is not None and baseline_col in df.columns:
        ax.plot(
            x,
            pd.to_numeric(df[baseline_col], errors="coerce"),
            color=BASELINE_COLOR,
            linestyle="--",
            marker=".",
            linewidth=1.6,
            label=baseline_label or "Baseline",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df[label_col], rotation=28, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if ylim:
        ax.set_ylim(*ylim)
    if yscale:
        ax.set_yscale(yscale)

    ax.legend(ncol=min(3, max(1, n)))
    savefig_stem(fig, out_stem)


def make_line_figure(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    group_cols: Sequence[str],
    x_label: str,
    y_label: str,
    title: str,
    out_stem: Path,
    xscale: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for key, g in df.groupby(list(group_cols), dropna=False):
        g = g.sort_values(x_col)
        x = pd.to_numeric(g[x_col], errors="coerce")
        y = pd.to_numeric(g[y_col], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            continue
        label = "/".join(map(str, key if isinstance(key, tuple) else (key,)))
        ax.plot(x[mask], y[mask], marker="o", label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if xscale:
        ax.set_xscale(xscale)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, ncol=2)
    savefig_stem(fig, out_stem)


def observer_publication_outputs(
    *,
    summary: pd.DataFrame,
    out_dir: Path,
    requested_graph: str | None,
    requested_observer_mode: str | None,
    requested_radius: float | None,
    requested_noise_std: float | None,
    no_plots: bool,
) -> dict[str, Any]:
    if summary.empty:
        return {"status": "empty"}

    required = {"graph", "mechanism", "observer_mode", "radius", "noise_std"}
    missing = sorted(required - set(summary.columns))
    if missing:
        return {
            "status": "skipped",
            "reason": "missing required columns",
            "missing": missing,
        }

    observer_mode = choose_category(
        summary["observer_mode"],
        requested=requested_observer_mode,
        prefer_order=["farthest", "all", "nearest"],
    )
    radius = choose_numeric(
        summary["radius"], requested=requested_radius, policy="second"
    )
    noise_std = choose_numeric(
        summary["noise_std"], requested=requested_noise_std, policy="second"
    )

    rep = summary[
        (summary["observer_mode"].astype(str) == observer_mode)
        & isclose_series(summary["radius"], radius)
        & isclose_series(summary["noise_std"], noise_std)
    ].copy()

    if requested_graph is not None and "graph" in rep.columns:
        rep = rep[rep["graph"].astype(str) == str(requested_graph)].copy()

    if rep.empty:
        return {
            "status": "empty_representative_slice",
            "observer_mode": observer_mode,
            "radius": radius,
            "noise_std": noise_std,
        }

    for metric in [
        "direct_gaussian_rero_bound",
        "rdp_rero_bound",
        "transferred_sensitivity",
        "gaussian_dp_mu",
        "dp_epsilon",
        "rdp_eps_alpha_16",
    ]:
        rep[metric] = metric_values(rep, metric)

    group_cols = ["dataset_tag", "dataset_name", "graph", "mechanism"]
    grouped = (
        rep.groupby(group_cols, dropna=False)
        .agg(
            direct_gaussian_rero_bound=("direct_gaussian_rero_bound", "mean"),
            rdp_rero_bound=("rdp_rero_bound", "mean"),
            transferred_sensitivity=("transferred_sensitivity", "mean"),
            gaussian_dp_mu=("gaussian_dp_mu", "mean"),
            dp_epsilon=("dp_epsilon", "mean"),
            rdp_eps_alpha_16=("rdp_eps_alpha_16", "mean"),
        )
        .reset_index()
    )

    records: list[dict[str, Any]] = []
    for key, g in grouped.groupby(
        ["dataset_tag", "dataset_name", "graph"], dropna=False
    ):
        dataset_tag, dataset_name, graph = key
        rec: dict[str, Any] = {
            "dataset_tag": dataset_tag,
            "dataset_name": dataset_name,
            "graph": graph,
        }
        for _, row in g.iterrows():
            mech = str(row["mechanism"])
            for metric in [
                "direct_gaussian_rero_bound",
                "rdp_rero_bound",
                "transferred_sensitivity",
                "gaussian_dp_mu",
                "dp_epsilon",
                "rdp_eps_alpha_16",
            ]:
                rec[f"{metric}_{mech}"] = row[metric]
        records.append(rec)

    wide = pd.DataFrame(records)
    if wide.empty:
        return {"status": "empty_wide"}

    wide["plot_label"] = wide.apply(compact_label, axis=1)
    wide = wide.sort_values(["dataset_name", "graph"]).reset_index(drop=True)

    save_dataframe(
        wide,
        out_dir / "observer_publication_representative_wide.csv",
        save_parquet_if_possible=False,
    )

    table = pd.DataFrame(
        {
            "Dataset": wide["dataset_name"],
            "Graph": wide["graph"],
            "Ball bound": wide.get("direct_gaussian_rero_bound_ball"),
            "Std bound": wide.get("direct_gaussian_rero_bound_standard"),
            "Ball sensitivity": wide.get("transferred_sensitivity_ball"),
            "Std sensitivity": wide.get("transferred_sensitivity_standard"),
            "Ball DP eps": wide.get("dp_epsilon_ball"),
            "Std DP eps": wide.get("dp_epsilon_standard"),
            "Ball RDP bound": wide.get("rdp_rero_bound_ball"),
            "Std RDP bound": wide.get("rdp_rero_bound_standard"),
        }
    )
    save_table(table, out_dir / "observer_publication_table")

    files = [
        str(out_dir / "observer_publication_representative_wide.csv"),
        str(out_dir / "observer_publication_table.csv"),
        str(out_dir / "observer_publication_table.tex"),
    ]

    if not no_plots:
        make_bar_figure(
            wide,
            label_col="plot_label",
            bars=[
                ("direct_gaussian_rero_bound_ball", "Ball", BALL_COLOR, None),
                (
                    "direct_gaussian_rero_bound_standard",
                    "Standard",
                    STANDARD_COLOR,
                    "//",
                ),
            ],
            y_label=r"Direct Gaussian ReRo bound $\gamma$",
            title=f"Observer-specific direct bound\nmode={observer_mode}, r={radius:.3g}, sigma={noise_std:g}",
            out_stem=out_dir / "fig_observer_direct_bound_ball_vs_standard",
            ylim=(0.0, 1.02),
        )
        make_bar_figure(
            wide,
            label_col="plot_label",
            bars=[
                ("transferred_sensitivity_ball", "Ball", BALL_COLOR, None),
                ("transferred_sensitivity_standard", "Standard", STANDARD_COLOR, "//"),
            ],
            y_label=r"Transferred sensitivity $\Delta_{A\leftarrow j}$",
            title=f"Observer-specific transferred sensitivity\nmode={observer_mode}, r={radius:.3g}, sigma={noise_std:g}",
            out_stem=out_dir / "fig_observer_transferred_sensitivity_ball_vs_standard",
        )
        make_bar_figure(
            wide,
            label_col="plot_label",
            bars=[
                ("dp_epsilon_ball", "Ball", BALL_COLOR, None),
                ("dp_epsilon_standard", "Standard", STANDARD_COLOR, "//"),
            ],
            y_label=r"DP accountant $\varepsilon$",
            title=f"Observer-specific DP epsilon\nmode={observer_mode}, r={radius:.3g}, sigma={noise_std:g}",
            out_stem=out_dir / "fig_observer_dp_epsilon_ball_vs_standard",
        )

        noise_df = summary[
            (summary["observer_mode"].astype(str) == observer_mode)
            & isclose_series(summary["radius"], radius)
        ].copy()
        if requested_graph is not None:
            noise_df = noise_df[noise_df["graph"].astype(str) == str(requested_graph)]
        noise_df["direct_gaussian_rero_bound"] = metric_values(
            noise_df, "direct_gaussian_rero_bound"
        )
        noise_group = (
            noise_df.groupby(["graph", "mechanism", "noise_std"], dropna=False)[
                "direct_gaussian_rero_bound"
            ]
            .mean()
            .reset_index()
        )
        make_line_figure(
            noise_group,
            x_col="noise_std",
            y_col="direct_gaussian_rero_bound",
            group_cols=["graph", "mechanism"],
            x_label="Gaussian noise std sigma",
            y_label=r"Mean direct ReRo bound $\gamma$",
            title=f"Observer noise ablation, mode={observer_mode}, r={radius:.3g}",
            out_stem=out_dir / "fig_observer_noise_ablation",
            xscale="log",
            ylim=(0.0, 1.02),
        )

        radius_df = summary[
            (summary["observer_mode"].astype(str) == observer_mode)
            & isclose_series(summary["noise_std"], noise_std)
        ].copy()
        if requested_graph is not None:
            radius_df = radius_df[
                radius_df["graph"].astype(str) == str(requested_graph)
            ]
        radius_df["direct_gaussian_rero_bound"] = metric_values(
            radius_df, "direct_gaussian_rero_bound"
        )
        radius_group = (
            radius_df.groupby(["graph", "mechanism", "radius"], dropna=False)[
                "direct_gaussian_rero_bound"
            ]
            .mean()
            .reset_index()
        )
        make_line_figure(
            radius_group,
            x_col="radius",
            y_col="direct_gaussian_rero_bound",
            group_cols=["graph", "mechanism"],
            x_label="Ball radius r",
            y_label=r"Mean direct ReRo bound $\gamma$",
            title=f"Observer radius ablation, mode={observer_mode}, sigma={noise_std:g}",
            out_stem=out_dir / "fig_observer_radius_ablation",
            ylim=(0.0, 1.02),
        )

        files.extend(
            [
                str(
                    (
                        out_dir / "fig_observer_direct_bound_ball_vs_standard"
                    ).with_suffix(".pdf")
                ),
                str(
                    (
                        out_dir
                        / "fig_observer_transferred_sensitivity_ball_vs_standard"
                    ).with_suffix(".pdf")
                ),
                str(
                    (out_dir / "fig_observer_dp_epsilon_ball_vs_standard").with_suffix(
                        ".pdf"
                    )
                ),
                str((out_dir / "fig_observer_noise_ablation").with_suffix(".pdf")),
                str((out_dir / "fig_observer_radius_ablation").with_suffix(".pdf")),
            ]
        )

    return {
        "status": "ok",
        "representative": {
            "observer_mode": observer_mode,
            "radius": float(radius),
            "noise_std": float(noise_std),
            "graph": requested_graph,
        },
        "num_rows": int(len(wide)),
        "files": files,
    }


def map_attack_publication_outputs(
    *,
    summary: pd.DataFrame,
    out_dir: Path,
    requested_graph: str | None,
    requested_observer_mode: str | None,
    requested_radius: float | None,
    requested_noise_std: float | None,
    no_plots: bool,
) -> dict[str, Any]:
    if summary.empty:
        return {"status": "empty"}

    required = {"graph", "observer_mode", "radius", "noise_std"}
    missing = sorted(required - set(summary.columns))
    if missing:
        return {
            "status": "skipped",
            "reason": "missing required columns",
            "missing": missing,
        }

    observer_mode = choose_category(
        summary["observer_mode"],
        requested=requested_observer_mode,
        prefer_order=["farthest", "all", "nearest"],
    )
    radius = choose_numeric(
        summary["radius"], requested=requested_radius, policy="first"
    )
    noise_std = choose_numeric(
        summary["noise_std"], requested=requested_noise_std, policy="middle"
    )

    rep = summary[
        (summary["observer_mode"].astype(str) == observer_mode)
        & isclose_series(summary["radius"], radius)
        & isclose_series(summary["noise_std"], noise_std)
    ].copy()

    if requested_graph is not None:
        rep = rep[rep["graph"].astype(str) == str(requested_graph)].copy()

    if rep.empty:
        return {
            "status": "empty_representative_slice",
            "observer_mode": observer_mode,
            "radius": radius,
            "noise_std": noise_std,
        }

    for metric in [
        "exact_identification_success",
        "direct_gaussian_rero_bound",
        "standard_direct_gaussian_rero_bound",
        "prior_rank",
        "prior_hit_at_5",
        "mse",
        "l2_error",
        "transferred_sensitivity",
        "standard_transferred_sensitivity",
    ]:
        rep[metric] = metric_values(rep, metric)

    group_cols = ["dataset_tag", "dataset_name", "graph", "observer_mode"]
    grouped = (
        rep.groupby(group_cols, dropna=False)
        .agg(
            exact_identification_success=("exact_identification_success", "mean"),
            direct_gaussian_rero_bound=("direct_gaussian_rero_bound", "mean"),
            standard_direct_gaussian_rero_bound=(
                "standard_direct_gaussian_rero_bound",
                "mean",
            ),
            prior_rank=("prior_rank", "mean"),
            prior_hit_at_5=("prior_hit_at_5", "mean"),
            mse=("mse", "mean"),
            l2_error=("l2_error", "mean"),
            transferred_sensitivity=("transferred_sensitivity", "mean"),
            standard_transferred_sensitivity=(
                "standard_transferred_sensitivity",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(["dataset_name", "graph"])
        .reset_index(drop=True)
    )
    grouped["plot_label"] = grouped.apply(compact_label, axis=1)

    save_dataframe(
        grouped,
        out_dir / "map_attack_publication_representative.csv",
        save_parquet_if_possible=False,
    )

    table = pd.DataFrame(
        {
            "Dataset": grouped["dataset_name"],
            "Graph": grouped["graph"],
            "Observer mode": grouped["observer_mode"],
            "Exact-ID success": grouped["exact_identification_success"],
            "Ball direct bound": grouped["direct_gaussian_rero_bound"],
            "Std same-noise bound": grouped["standard_direct_gaussian_rero_bound"],
            "Prior rank": grouped["prior_rank"],
            "Hit@5": grouped["prior_hit_at_5"],
            "MSE": grouped["mse"],
            "$\\Delta_{Ball}$": grouped["transferred_sensitivity"],
            "$\\Delta_{Std}$": grouped["standard_transferred_sensitivity"],
        }
    )
    save_table(table, out_dir / "map_attack_publication_table")

    files = [
        str(out_dir / "map_attack_publication_representative.csv"),
        str(out_dir / "map_attack_publication_table.csv"),
        str(out_dir / "map_attack_publication_table.tex"),
    ]

    if not no_plots:
        make_bar_figure(
            grouped,
            label_col="plot_label",
            bars=[
                ("exact_identification_success", "Empirical attack", BALL_COLOR, None),
                ("direct_gaussian_rero_bound", "Ball bound", RDP_COLOR, "//"),
                (
                    "standard_direct_gaussian_rero_bound",
                    "Std same-noise bound",
                    STANDARD_COLOR,
                    "..",
                ),
            ],
            y_label="Probability",
            title=f"Decentralized MAP attack vs bounds\nmode={observer_mode}, r={radius:.3g}, sigma={noise_std:g}",
            out_stem=out_dir / "fig_map_attack_success_vs_bounds",
            ylim=(0.0, 1.02),
        )
        make_bar_figure(
            grouped,
            label_col="plot_label",
            bars=[
                ("transferred_sensitivity", "Ball", BALL_COLOR, None),
                ("standard_transferred_sensitivity", "Standard", STANDARD_COLOR, "//"),
            ],
            y_label=r"Transferred sensitivity $\Delta_{A\leftarrow j}$",
            title=f"MAP attack transferred sensitivity\nmode={observer_mode}, r={radius:.3g}, sigma={noise_std:g}",
            out_stem=out_dir / "fig_map_attack_transferred_sensitivity",
        )
        make_bar_figure(
            grouped,
            label_col="plot_label",
            bars=[
                ("prior_rank", "Mean prior rank", BALL_COLOR, None),
            ],
            y_label="Mean prior rank",
            title=f"MAP attack prior rank\nmode={observer_mode}, r={radius:.3g}, sigma={noise_std:g}",
            out_stem=out_dir / "fig_map_attack_prior_rank",
        )

        noise_df = summary[
            (summary["observer_mode"].astype(str) == observer_mode)
            & isclose_series(summary["radius"], radius)
        ].copy()
        if requested_graph is not None:
            noise_df = noise_df[noise_df["graph"].astype(str) == str(requested_graph)]
        for metric in [
            "exact_identification_success",
            "direct_gaussian_rero_bound",
            "standard_direct_gaussian_rero_bound",
        ]:
            noise_df[metric] = metric_values(noise_df, metric)
        noise_group = (
            noise_df.groupby(["graph", "noise_std"], dropna=False)
            .agg(
                exact_identification_success=("exact_identification_success", "mean"),
                direct_gaussian_rero_bound=("direct_gaussian_rero_bound", "mean"),
                standard_direct_gaussian_rero_bound=(
                    "standard_direct_gaussian_rero_bound",
                    "mean",
                ),
            )
            .reset_index()
        )
        make_line_figure(
            noise_group,
            x_col="noise_std",
            y_col="exact_identification_success",
            group_cols=["graph"],
            x_label="Gaussian noise std sigma",
            y_label="Empirical exact-ID success",
            title=f"MAP attack noise ablation, mode={observer_mode}, r={radius:.3g}",
            out_stem=out_dir / "fig_map_attack_noise_ablation_success",
            xscale="log",
            ylim=(0.0, 1.02),
        )
        make_line_figure(
            noise_group,
            x_col="noise_std",
            y_col="direct_gaussian_rero_bound",
            group_cols=["graph"],
            x_label="Gaussian noise std sigma",
            y_label=r"Ball direct bound $\gamma$",
            title=f"MAP attack bound noise ablation, mode={observer_mode}, r={radius:.3g}",
            out_stem=out_dir / "fig_map_attack_noise_ablation_bound",
            xscale="log",
            ylim=(0.0, 1.02),
        )

        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        x = pd.to_numeric(grouped["direct_gaussian_rero_bound"], errors="coerce")
        y = pd.to_numeric(grouped["exact_identification_success"], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            ax.scatter(
                x[mask], y[mask], color=BALL_COLOR, marker="o", label="Ball bound"
            )
            for i in np.flatnonzero(mask.to_numpy()):
                ax.annotate(
                    str(grouped["graph"].iloc[i]),
                    (float(x.iloc[i]), float(y.iloc[i])),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
        xs = np.linspace(0, 1, 100)
        ax.plot(
            xs, xs, color=BASELINE_COLOR, linestyle="--", linewidth=1.4, label="y=x"
        )
        ax.set_xlabel(r"Ball direct bound $\gamma$")
        ax.set_ylabel("Empirical exact-ID success")
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(
            f"MAP attack success vs theorem bound\nmode={observer_mode}, r={radius:.3g}, sigma={noise_std:g}"
        )
        ax.legend()
        savefig_stem(fig, out_dir / "fig_map_attack_success_vs_bound_scatter")

        files.extend(
            [
                str((out_dir / "fig_map_attack_success_vs_bounds").with_suffix(".pdf")),
                str(
                    (out_dir / "fig_map_attack_transferred_sensitivity").with_suffix(
                        ".pdf"
                    )
                ),
                str((out_dir / "fig_map_attack_prior_rank").with_suffix(".pdf")),
                str(
                    (out_dir / "fig_map_attack_noise_ablation_success").with_suffix(
                        ".pdf"
                    )
                ),
                str(
                    (out_dir / "fig_map_attack_noise_ablation_bound").with_suffix(
                        ".pdf"
                    )
                ),
                str(
                    (out_dir / "fig_map_attack_success_vs_bound_scatter").with_suffix(
                        ".pdf"
                    )
                ),
            ]
        )

    return {
        "status": "ok",
        "representative": {
            "observer_mode": observer_mode,
            "radius": float(radius),
            "noise_std": float(noise_std),
            "graph": requested_graph,
        },
        "num_rows": int(len(grouped)),
        "files": files,
    }


def utility_publication_outputs(
    *,
    summary: pd.DataFrame,
    out_dir: Path,
    requested_graph: str | None,
    requested_radius: float | None,
    requested_epsilon: float | None,
    no_plots: bool,
) -> dict[str, Any]:
    if summary.empty:
        return {"status": "empty"}

    required = {"graph", "mechanism", "radius", "target_epsilon"}
    missing = sorted(required - set(summary.columns))
    if missing:
        return {
            "status": "skipped",
            "reason": "missing required columns",
            "missing": missing,
        }

    nonbaseline = summary[summary["mechanism"].astype(str) != "none"].copy()
    if nonbaseline.empty:
        return {"status": "empty_nonbaseline"}

    radius = choose_numeric(
        nonbaseline["radius"], requested=requested_radius, policy="second"
    )
    epsilon = choose_numeric(
        nonbaseline["target_epsilon"],
        requested=requested_epsilon,
        prefer=DEFAULT_REP_EPSILON,
        policy="middle",
    )

    rep = nonbaseline[
        isclose_series(nonbaseline["radius"], radius)
        & isclose_series(nonbaseline["target_epsilon"], epsilon)
    ].copy()

    if requested_graph is not None:
        rep = rep[rep["graph"].astype(str) == str(requested_graph)].copy()

    if rep.empty:
        return {
            "status": "empty_representative_slice",
            "radius": radius,
            "epsilon": epsilon,
        }

    for metric in [
        "accuracy_mean",
        "accuracy_min_node",
        "accuracy_std_node",
        "consensus_state_disagreement",
        "noise_std",
        "sensitivity",
    ]:
        rep[metric] = metric_values(rep, metric)

    group_cols = ["dataset_tag", "dataset_name", "graph", "mechanism"]
    grouped = (
        rep.groupby(group_cols, dropna=False)
        .agg(
            accuracy_mean=("accuracy_mean", "mean"),
            accuracy_min_node=("accuracy_min_node", "mean"),
            accuracy_std_node=("accuracy_std_node", "mean"),
            consensus_state_disagreement=("consensus_state_disagreement", "mean"),
            noise_std=("noise_std", "mean"),
            sensitivity=("sensitivity", "mean"),
        )
        .reset_index()
    )

    records: list[dict[str, Any]] = []
    for key, g in grouped.groupby(
        ["dataset_tag", "dataset_name", "graph"], dropna=False
    ):
        dataset_tag, dataset_name, graph = key
        rec: dict[str, Any] = {
            "dataset_tag": dataset_tag,
            "dataset_name": dataset_name,
            "graph": graph,
        }
        for _, row in g.iterrows():
            mech = str(row["mechanism"])
            for metric in [
                "accuracy_mean",
                "accuracy_min_node",
                "accuracy_std_node",
                "consensus_state_disagreement",
                "noise_std",
                "sensitivity",
            ]:
                rec[f"{metric}_{mech}"] = row[metric]
        records.append(rec)

    wide = pd.DataFrame(records)
    if wide.empty:
        return {"status": "empty_wide"}

    baseline = summary[summary["mechanism"].astype(str) == "none"].copy()
    if not baseline.empty:
        baseline["accuracy_mean"] = metric_values(baseline, "accuracy_mean")
        baseline_group = (
            baseline.groupby(["dataset_tag", "dataset_name", "graph"], dropna=False)
            .agg(baseline_accuracy=("accuracy_mean", "mean"))
            .reset_index()
        )
        wide = wide.merge(
            baseline_group,
            on=["dataset_tag", "dataset_name", "graph"],
            how="left",
        )

    wide["plot_label"] = wide.apply(compact_label, axis=1)
    wide = wide.sort_values(["dataset_name", "graph"]).reset_index(drop=True)

    save_dataframe(
        wide,
        out_dir / "utility_publication_representative_wide.csv",
        save_parquet_if_possible=False,
    )

    table = pd.DataFrame(
        {
            "Dataset": wide["dataset_name"],
            "Graph": wide["graph"],
            "Baseline acc": wide.get("baseline_accuracy"),
            "Ball acc": wide.get("accuracy_mean_ball"),
            "Std acc": wide.get("accuracy_mean_standard"),
            "Ball min-node acc": wide.get("accuracy_min_node_ball"),
            "Std min-node acc": wide.get("accuracy_min_node_standard"),
            "Ball sigma": wide.get("noise_std_ball"),
            "Std sigma": wide.get("noise_std_standard"),
            "Ball sensitivity": wide.get("sensitivity_ball"),
            "Std sensitivity": wide.get("sensitivity_standard"),
            "Ball disagreement": wide.get("consensus_state_disagreement_ball"),
            "Std disagreement": wide.get("consensus_state_disagreement_standard"),
        }
    )
    save_table(table, out_dir / "utility_publication_table")

    files = [
        str(out_dir / "utility_publication_representative_wide.csv"),
        str(out_dir / "utility_publication_table.csv"),
        str(out_dir / "utility_publication_table.tex"),
    ]

    if not no_plots:
        make_bar_figure(
            wide,
            label_col="plot_label",
            bars=[
                ("accuracy_mean_ball", "Ball", BALL_COLOR, None),
                ("accuracy_mean_standard", "Standard", STANDARD_COLOR, "//"),
            ],
            y_label="Mean node accuracy",
            title=f"Decentralized prototype utility\nr={radius:.3g}, target epsilon={epsilon:g}",
            out_stem=out_dir / "fig_utility_accuracy_ball_vs_standard",
            ylim=(0.0, 1.02),
            baseline_col="baseline_accuracy",
            baseline_label="No-noise baseline",
        )
        make_bar_figure(
            wide,
            label_col="plot_label",
            bars=[
                ("noise_std_ball", "Ball", BALL_COLOR, None),
                ("noise_std_standard", "Standard", STANDARD_COLOR, "//"),
            ],
            y_label="Calibrated Gaussian noise std",
            title=f"Noise calibration\nr={radius:.3g}, target epsilon={epsilon:g}",
            out_stem=out_dir / "fig_utility_noise_ball_vs_standard",
        )
        make_bar_figure(
            wide,
            label_col="plot_label",
            bars=[
                ("sensitivity_ball", "Ball", BALL_COLOR, None),
                ("sensitivity_standard", "Standard", STANDARD_COLOR, "//"),
            ],
            y_label="Step sensitivity",
            title=f"Prototype sensitivity\nr={radius:.3g}, target epsilon={epsilon:g}",
            out_stem=out_dir / "fig_utility_sensitivity_ball_vs_standard",
        )
        make_bar_figure(
            wide,
            label_col="plot_label",
            bars=[
                ("consensus_state_disagreement_ball", "Ball", BALL_COLOR, None),
                (
                    "consensus_state_disagreement_standard",
                    "Standard",
                    STANDARD_COLOR,
                    "//",
                ),
            ],
            y_label="Consensus state disagreement",
            title=f"Consensus disagreement\nr={radius:.3g}, target epsilon={epsilon:g}",
            out_stem=out_dir / "fig_utility_consensus_disagreement",
        )

        eps_df = nonbaseline[isclose_series(nonbaseline["radius"], radius)].copy()
        if requested_graph is not None:
            eps_df = eps_df[eps_df["graph"].astype(str) == str(requested_graph)]
        eps_df["accuracy_mean"] = metric_values(eps_df, "accuracy_mean")
        eps_group = (
            eps_df.groupby(["graph", "mechanism", "target_epsilon"], dropna=False)[
                "accuracy_mean"
            ]
            .mean()
            .reset_index()
        )
        make_line_figure(
            eps_group,
            x_col="target_epsilon",
            y_col="accuracy_mean",
            group_cols=["graph", "mechanism"],
            x_label="Target epsilon",
            y_label="Mean node accuracy",
            title=f"Utility/privacy ablation, r={radius:.3g}",
            out_stem=out_dir / "fig_utility_accuracy_vs_epsilon",
            xscale="log",
            ylim=(0.0, 1.02),
        )

        radius_df = nonbaseline[
            isclose_series(nonbaseline["target_epsilon"], epsilon)
        ].copy()
        if requested_graph is not None:
            radius_df = radius_df[
                radius_df["graph"].astype(str) == str(requested_graph)
            ]
        radius_df["accuracy_mean"] = metric_values(radius_df, "accuracy_mean")
        radius_group = (
            radius_df.groupby(["graph", "mechanism", "radius"], dropna=False)[
                "accuracy_mean"
            ]
            .mean()
            .reset_index()
        )
        make_line_figure(
            radius_group,
            x_col="radius",
            y_col="accuracy_mean",
            group_cols=["graph", "mechanism"],
            x_label="Ball radius r",
            y_label="Mean node accuracy",
            title=f"Radius/utility ablation, target epsilon={epsilon:g}",
            out_stem=out_dir / "fig_utility_accuracy_vs_radius",
            ylim=(0.0, 1.02),
        )

        files.extend(
            [
                str(
                    (out_dir / "fig_utility_accuracy_ball_vs_standard").with_suffix(
                        ".pdf"
                    )
                ),
                str(
                    (out_dir / "fig_utility_noise_ball_vs_standard").with_suffix(".pdf")
                ),
                str(
                    (out_dir / "fig_utility_sensitivity_ball_vs_standard").with_suffix(
                        ".pdf"
                    )
                ),
                str(
                    (out_dir / "fig_utility_consensus_disagreement").with_suffix(".pdf")
                ),
                str((out_dir / "fig_utility_accuracy_vs_epsilon").with_suffix(".pdf")),
                str((out_dir / "fig_utility_accuracy_vs_radius").with_suffix(".pdf")),
            ]
        )

    return {
        "status": "ok",
        "representative": {
            "radius": float(radius),
            "target_epsilon": float(epsilon),
            "graph": requested_graph,
        },
        "num_rows": int(len(wide)),
        "files": files,
    }


def main() -> None:
    configure_matplotlib()
    args = parse_args()

    results_root = Path(args.results_root)
    aggregate_dir = results_root / "paper3" / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    wanted = wanted_dataset_filter(args.datasets)
    kinds = list(KIND_TO_DIR_AND_FILES) if args.kind == "all" else [args.kind]

    manifest: dict[str, Any] = {
        "results_root": str(results_root),
        "aggregate_dir": str(aggregate_dir),
        "requested_kind": args.kind,
        "requested_datasets": (
            list(args.datasets) if args.datasets is not None else None
        ),
        "requested_graph": args.graph,
        "requested_observer_mode": args.observer_mode,
        "requested_radius": args.radius,
        "requested_noise_std": args.noise_std,
        "requested_epsilon": args.epsilon,
        "strict": bool(args.strict),
        "no_plots": bool(args.no_plots),
        "kinds": {},
    }

    any_success = False

    for kind in kinds:
        rows, summary, kind_meta = aggregate_kind(
            results_root=results_root,
            kind=kind,
            wanted=wanted,
            strict=bool(args.strict),
        )

        subdir, rows_name, summary_name = KIND_TO_DIR_AND_FILES[kind]
        rows_out = aggregate_dir / f"all_{kind}_rows.csv"
        summary_out = aggregate_dir / f"all_{kind}_summary.csv"

        if not rows.empty:
            save_dataframe(rows, rows_out, save_parquet_if_possible=False)
            kind_meta["rows_output"] = str(rows_out)

        if not summary.empty:
            save_dataframe(summary, summary_out, save_parquet_if_possible=False)
            kind_meta["summary_output"] = str(summary_out)

        publication_meta: dict[str, Any]
        if kind == "observer":
            publication_meta = observer_publication_outputs(
                summary=summary,
                out_dir=aggregate_dir,
                requested_graph=args.graph,
                requested_observer_mode=args.observer_mode,
                requested_radius=args.radius,
                requested_noise_std=args.noise_std,
                no_plots=bool(args.no_plots),
            )
        elif kind == "map_attack":
            publication_meta = map_attack_publication_outputs(
                summary=summary,
                out_dir=aggregate_dir,
                requested_graph=args.graph,
                requested_observer_mode=args.observer_mode,
                requested_radius=args.radius,
                requested_noise_std=args.noise_std,
                no_plots=bool(args.no_plots),
            )
        elif kind == "utility":
            publication_meta = utility_publication_outputs(
                summary=summary,
                out_dir=aggregate_dir,
                requested_graph=args.graph,
                requested_radius=args.radius,
                requested_epsilon=args.epsilon,
                no_plots=bool(args.no_plots),
            )
        else:
            publication_meta = {"status": "unknown_kind"}

        kind_meta["publication_outputs"] = publication_meta

        if kind_meta.get("status") == "ok":
            any_success = True

        manifest["kinds"][kind] = kind_meta

    manifest_path = aggregate_dir / "aggregate_decentralized_paper3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    if not any_success:
        raise RuntimeError(
            "No decentralized Paper 3 aggregate outputs were produced. "
            f"See manifest: {manifest_path}"
        )

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
