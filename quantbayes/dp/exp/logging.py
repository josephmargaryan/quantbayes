from __future__ import annotations
import csv, os
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional


@dataclass
class MetricRecord:
    tag: str  # e.g., "output", "objective", "dpgd_zcdp"
    epsilon: float
    delta: float
    lam: float
    n: int
    d: int
    seed: int
    value: float  # e.g., train log-loss or excess risk
    extra: Optional[Dict[str, Any]] = None


def write_csv(path: str, rows: List[MetricRecord]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # flatten 'extra'
    base = asdict(rows[0])
    base.pop("extra", None)
    base_fields = list(base.keys())
    extra_keys = sorted({k for r in rows for k in (r.extra or {}).keys()})
    fieldnames = base_fields + extra_keys
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = asdict(r)
            extra = row.pop("extra", {}) or {}
            row.update({k: extra.get(k) for k in extra_keys})
            w.writerow(row)
