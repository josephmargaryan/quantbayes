#!/usr/bin/env python3
"""Run the missing ERM epsilon sweeps for additional local radii.

This is a small launcher around your existing ERM experiment driver. It reads
existing per-dataset summaries under

    <results-root>/erm/<model>/<dataset>/erm_summary.csv

infers the epsilon grid from a reference radius, and launches only the
(dataset, radius, epsilon) configurations that are absent unless --force is set.

Example from the repository root:

    python quantbayes/ball_dp/experiments/run_missing_radius_epsilon_sweeps.py \
        --results-root quantbayes/results \
        --model ridge_prototype \
        --m 10 \
        --radii q50 q95 \
        --jobs 1 \
        -- --data-root ./data

If auto-detection cannot find your ERM driver, pass it explicitly:

    --erm-script quantbayes/ball_dp/experiments/<YOUR_ERM_SCRIPT>.py
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

DEFAULT_RADII: tuple[str, ...] = ("q50", "q95")
DEFAULT_REFERENCE_RADIUS = "q80"
DEFAULT_RESULTS_ROOT = "quantbayes/results"


@dataclass(frozen=True)
class DatasetInfo:
    dataset_dir: Path
    summary_path: Path
    dataset_tag: str
    dataset_name: str
    summary: pd.DataFrame


def normalize_key(value: str) -> str:
    return (
        str(value).strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    )


def first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def format_float(value: float) -> str:
    return f"{float(value):.12g}"


def sorted_unique_floats(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    for value in sorted(float(v) for v in values):
        if not any(np.isclose(value, prev, rtol=1e-10, atol=1e-12) for prev in out):
            out.append(value)
    return out


def contains_close(values: Sequence[float], target: float) -> bool:
    return any(
        np.isclose(float(v), float(target), rtol=1e-10, atol=1e-12) for v in values
    )


def resolve_results_root(path_arg: str) -> Path:
    root = Path(path_arg)
    if root.exists():
        return root
    # Convenience fallback if the script is run from quantbayes/ instead of repo root.
    if path_arg == DEFAULT_RESULTS_ROOT and Path("results").exists():
        return Path("results")
    return root


def auto_find_erm_script(explicit: str | None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Explicit --erm-script does not exist: {p}")
        return p

    candidates = [
        Path("quantbayes/ball_dp/experiments/run_erm_experiment.py"),
        Path("quantbayes/ball_dp/experiments/run_erm_experiments.py"),
        Path("quantbayes/ball_dp/experiments/run_erm.py"),
        Path("quantbayes/ball_dp/experiments/run_attack_erm_experiment.py"),
    ]
    for p in candidates:
        if p.exists():
            return p

    experiments_dir = Path("quantbayes/ball_dp/experiments")
    if experiments_dir.exists():
        for p in sorted(experiments_dir.glob("*.py")):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            # Robust signature of the ERM driver pasted in the prompt.
            if (
                "rebuild_erm_dataset_outputs" in text
                and "fit_release" in text
                and "--sweep" in text
                and "erm_seed_rows.csv" in text
            ):
                return p

    raise FileNotFoundError(
        "Could not auto-detect the ERM driver. Pass it explicitly with, e.g., "
        "--erm-script quantbayes/ball_dp/experiments/<YOUR_ERM_SCRIPT>.py"
    )


def discover_datasets(
    model_dir: Path, dataset_queries: Sequence[str] | None
) -> list[DatasetInfo]:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model results directory not found: {model_dir}")

    infos: list[DatasetInfo] = []
    for dataset_dir in sorted(
        p for p in model_dir.iterdir() if p.is_dir() and p.name != "_aggregate"
    ):
        summary_path = dataset_dir / "erm_summary.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        if summary.empty:
            continue
        required = {"dataset_tag", "dataset_name", "radius_tag", "epsilon", "m"}
        missing = required.difference(summary.columns)
        if missing:
            raise ValueError(
                f"{summary_path} is missing required columns: {sorted(missing)}"
            )
        infos.append(
            DatasetInfo(
                dataset_dir=dataset_dir,
                summary_path=summary_path,
                dataset_tag=str(first_non_null(summary["dataset_tag"])),
                dataset_name=str(first_non_null(summary["dataset_name"])),
                summary=summary,
            )
        )

    if not infos:
        raise FileNotFoundError(
            f"No per-dataset erm_summary.csv files found under {model_dir}"
        )

    if dataset_queries:
        wanted = {normalize_key(q) for q in dataset_queries}
        selected: list[DatasetInfo] = []
        for info in infos:
            keys = {
                normalize_key(info.dataset_dir.name),
                normalize_key(info.dataset_tag),
                normalize_key(info.dataset_name),
            }
            if keys & wanted:
                selected.append(info)
        matched = {
            key
            for info in selected
            for key in (
                normalize_key(info.dataset_dir.name),
                normalize_key(info.dataset_tag),
                normalize_key(info.dataset_name),
            )
        }
        missing_queries = sorted(q for q in wanted if q not in matched)
        if missing_queries:
            available = ", ".join(sorted(info.dataset_dir.name for info in infos))
            raise FileNotFoundError(
                f"Could not match datasets {missing_queries}. Available: {available}"
            )
        infos = selected

    return infos


def eps_grid_for_radius(summary: pd.DataFrame, *, radius: str, m: int) -> list[float]:
    mask = (summary["radius_tag"].astype(str) == str(radius)) & np.isclose(
        summary["m"].astype(float), float(m)
    )
    return sorted_unique_floats(summary.loc[mask, "epsilon"].astype(float).tolist())


def infer_epsilon_grid(
    summary: pd.DataFrame,
    *,
    m: int,
    reference_radius: str,
    explicit_grid: Sequence[float] | None,
) -> list[float]:
    if explicit_grid:
        return sorted_unique_floats([float(v) for v in explicit_grid])

    ref = eps_grid_for_radius(summary, radius=reference_radius, m=m)
    if ref:
        return ref

    # Fall back to all epsilons available at this m.
    mask = np.isclose(summary["m"].astype(float), float(m))
    inferred = sorted_unique_floats(summary.loc[mask, "epsilon"].astype(float).tolist())
    if inferred:
        return inferred

    raise RuntimeError(
        f"Could not infer epsilon grid at m={m}. Pass --epsilon-grid explicitly."
    )


def missing_epsilons(
    summary: pd.DataFrame,
    *,
    radius: str,
    m: int,
    target_eps: Sequence[float],
    force: bool,
) -> list[float]:
    if force:
        return list(target_eps)
    present = eps_grid_for_radius(summary, radius=radius, m=m)
    return [float(eps) for eps in target_eps if not contains_close(present, float(eps))]


def build_commands(
    args: argparse.Namespace, passthrough: list[str]
) -> tuple[Path, list[dict[str, Any]]]:
    results_root = resolve_results_root(args.results_root)
    model_dir = results_root / "erm" / normalize_key(args.model)
    erm_script = auto_find_erm_script(args.erm_script)
    datasets = discover_datasets(model_dir, args.datasets)

    plan: list[dict[str, Any]] = []
    for info in datasets:
        target_eps = infer_epsilon_grid(
            info.summary,
            m=int(args.m),
            reference_radius=str(args.reference_radius),
            explicit_grid=args.epsilon_grid,
        )
        for radius in args.radii:
            missing = missing_epsilons(
                info.summary,
                radius=str(radius),
                m=int(args.m),
                target_eps=target_eps,
                force=bool(args.force),
            )
            if not missing:
                continue
            cmd = [
                str(args.python),
                str(erm_script),
                "--dataset",
                info.dataset_tag,
                "--model",
                str(args.model),
                "--results-root",
                str(results_root),
                "--sweep",
                "epsilon",
                "--fixed-radius",
                str(radius),
                "--fixed-m",
                str(int(args.m)),
                "--epsilon-grid",
                *[format_float(eps) for eps in missing],
            ]
            if args.fixed_epsilon is not None:
                cmd.extend(["--fixed-epsilon", format_float(float(args.fixed_epsilon))])
            cmd.extend(passthrough)
            plan.append(
                {
                    "dataset_tag": info.dataset_tag,
                    "dataset_name": info.dataset_name,
                    "radius": str(radius),
                    "m": int(args.m),
                    "missing_epsilons": missing,
                    "command": cmd,
                }
            )
    return erm_script, plan


def run_one(entry: dict[str, Any]) -> tuple[dict[str, Any], int]:
    cmd = entry["command"]
    print("\n$ " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    completed = subprocess.run(cmd, check=False)
    return entry, int(completed.returncode)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run missing ERM epsilon sweeps for q50/q95 or other local radii."
    )
    parser.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--model", type=str, default="ridge_prototype")
    parser.add_argument("--erm-script", type=str, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--radii", nargs="+", type=str, default=list(DEFAULT_RADII))
    parser.add_argument(
        "--reference-radius", type=str, default=DEFAULT_REFERENCE_RADIUS
    )
    parser.add_argument(
        "--epsilon-grid",
        nargs="+",
        type=float,
        default=None,
        help="Explicit epsilon grid. Default: infer from --reference-radius at the chosen m.",
    )
    parser.add_argument("--fixed-epsilon", type=float, default=None)
    parser.add_argument("--datasets", nargs="*", type=str, default=None)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of subprocesses to run concurrently.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run all target epsilons even if they already appear present.",
    )
    parser.add_argument(
        "--plan-json",
        type=str,
        default=None,
        help="Optional path where the launch plan should be saved as JSON.",
    )
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def main() -> None:
    args, passthrough = parse_args()
    erm_script, plan = build_commands(args, passthrough)

    compact_plan = [
        {
            "dataset_tag": item["dataset_tag"],
            "radius": item["radius"],
            "m": item["m"],
            "missing_epsilons": item["missing_epsilons"],
        }
        for item in plan
    ]
    payload = {
        "erm_script": str(erm_script),
        "num_commands": len(plan),
        "jobs": int(args.jobs),
        "dry_run": bool(args.dry_run),
        "plan": compact_plan,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.plan_json is not None:
        path = Path(args.plan_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if not plan:
        print("No missing sweeps found. Nothing to run.")
        return

    if args.dry_run:
        print("\nDry run commands:")
        for item in plan:
            print(" ".join(shlex.quote(str(x)) for x in item["command"]))
        return

    jobs = max(1, int(args.jobs))
    failures: list[tuple[dict[str, Any], int]] = []
    if jobs == 1:
        for item in plan:
            entry, code = run_one(item)
            if code != 0:
                failures.append((entry, code))
                break
    else:
        with cf.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(run_one, item) for item in plan]
            for fut in cf.as_completed(futures):
                entry, code = fut.result()
                if code != 0:
                    failures.append((entry, code))

    if failures:
        details = "; ".join(
            f"{entry['dataset_tag']}:{entry['radius']} exited {code}"
            for entry, code in failures
        )
        raise SystemExit(f"Some sweeps failed: {details}")

    print("All requested missing sweeps completed successfully.")


if __name__ == "__main__":
    main()
