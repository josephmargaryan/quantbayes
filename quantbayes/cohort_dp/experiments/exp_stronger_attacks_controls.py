from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import os
import numpy as np

from quantbayes.cohort_dp.metrics import L2Metric
from quantbayes.cohort_dp.synthetic import make_synthetic_patients
from quantbayes.cohort_dp.analysis import estimate_r_global, attacker_exact_within_r
from quantbayes.cohort_dp.registry import MechanismSpec, build_api, seed_from_spec
from quantbayes.cohort_dp.candidates import AllCandidates
from quantbayes.cohort_dp.io import write_csv
from quantbayes.cohort_dp.eval import FrequencyAttacker
from quantbayes.cohort_dp.attacks_stronger import CoverageIntersectionAttacker
from quantbayes.cohort_dp.policies import StickyOutputPolicy


@dataclass
class Config:
    seed: int = 0
    out_dir: str = "results_stronger_attacks_controls"

    n: int = 2000
    d: int = 25
    n_clusters: int = 10

    k_service: int = 10
    k0_for_r: int = 20

    n_attack_targets: int = 80
    Q_freq: int = 50

    # Intersection attacker
    n_query_points: int = 4
    enum_rounds_per_point: int = 12

    # Sticky policy
    sticky_decimals: int = 2


def run(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    metric = L2Metric()

    X = make_synthetic_patients(cfg.n, cfg.d, cfg.n_clusters, rng)
    r_global = estimate_r_global(
        X, metric, k0=cfg.k0_for_r, rng=np.random.default_rng(cfg.seed + 1), m=200
    )
    print(f"r_global ≈ {r_global:.4f}")

    candgen = AllCandidates(n=X.shape[0])
    targets = rng.choice(cfg.n, size=min(cfg.n_attack_targets, cfg.n), replace=False)

    specs = [
        MechanismSpec("NONPRIVATE", cfg.k_service, 0.0, name="NonPrivate kNN"),
        MechanismSpec("NOISY_TOPK", cfg.k_service, 2.0, name="NoisyTopK eps=2"),
        MechanismSpec("ABU", cfg.k_service, 0.0, k0=20, name="ABU k0=20"),
        MechanismSpec(
            "ABM", cfg.k_service, 2.0, k0=20, mix_uniform=0.7, name="ABM k0=20 mix=0.7"
        ),
        MechanismSpec(
            "ABO", cfg.k_service, 0.0, k0=20, eps_ball=2.0, name="ABO k0=20 eps_ball=2"
        ),
    ]

    rows: List[Dict[str, Any]] = []

    for s in specs:
        seed_s = seed_from_spec(cfg.seed, s, namespace="strong_attacks")

        # Plain API
        api_plain = build_api(
            s,
            X_db=X,
            metric=metric,
            r=r_global,
            seed=seed_s,
            candidate_generator=candgen,
            no_repeat=False,
            sticky_policy=None,
        )

        # no-repeat API
        api_norep = build_api(
            s,
            X_db=X,
            metric=metric,
            r=r_global,
            seed=seed_s + 1,
            candidate_generator=candgen,
            no_repeat=True,
            sticky_policy=None,
        )

        # sticky API (no-repeat OFF; sticky works through caching)
        sticky = StickyOutputPolicy(
            rng=np.random.default_rng(seed_s + 999),
            decimals=int(cfg.sticky_decimals),
            max_cache=50_000,
        )
        api_sticky = build_api(
            s,
            X_db=X,
            metric=metric,
            r=r_global,
            seed=seed_s + 2,
            candidate_generator=candgen,
            no_repeat=False,
            sticky_policy=sticky,
        )

        # Frequency attacker (classic)
        freq = FrequencyAttacker(
            query_noise_std=0.05,
            Q=int(cfg.Q_freq),
            k_attack=int(cfg.k_service),
            rng=np.random.default_rng(seed_s + 123),
            count_all_returned=True,
            session_id="attacker",
            new_session_per_query=False,
        )

        # Intersection attacker (designed to exploit no-repeat)
        inter = CoverageIntersectionAttacker(
            query_noise_std=0.03,
            n_query_points=int(cfg.n_query_points),
            enum_rounds_per_point=int(cfg.enum_rounds_per_point),
            k_attack=int(cfg.k_service),
            rng=np.random.default_rng(seed_s + 456),
            session_id="attacker",
            new_session_per_point=False,
        )

        freq_plain, _ = attacker_exact_within_r(
            api_plain, X, metric, r_global, freq, targets
        )
        freq_norep, _ = attacker_exact_within_r(
            api_norep, X, metric, r_global, freq, targets
        )
        freq_sticky, _ = attacker_exact_within_r(
            api_sticky, X, metric, r_global, freq, targets
        )

        inter_plain, _ = attacker_exact_within_r(
            api_plain, X, metric, r_global, inter, targets
        )
        inter_norep, _ = attacker_exact_within_r(
            api_norep, X, metric, r_global, inter, targets
        )
        inter_sticky, _ = attacker_exact_within_r(
            api_sticky, X, metric, r_global, inter, targets
        )

        rows.append(
            {
                "name": s.label(),
                "kind": s.kind,
                "k_service": int(cfg.k_service),
                "r_global": float(r_global),
                "freq_exact_plain": float(freq_plain),
                "freq_exact_no_repeat": float(freq_norep),
                "freq_exact_sticky": float(freq_sticky),
                "inter_exact_plain": float(inter_plain),
                "inter_exact_no_repeat": float(inter_norep),
                "inter_exact_sticky": float(inter_sticky),
                "sticky_decimals": int(cfg.sticky_decimals),
                "Q_freq": int(cfg.Q_freq),
                "n_query_points": int(cfg.n_query_points),
                "enum_rounds_per_point": int(cfg.enum_rounds_per_point),
            }
        )

        print(
            f"[attacks] {s.label():>20} | freq plain={freq_plain:.3f} norep={freq_norep:.3f} sticky={freq_sticky:.3f} "
            f"| inter plain={inter_plain:.3f} norep={inter_norep:.3f} sticky={inter_sticky:.3f}"
        )

    out_csv = os.path.join(cfg.out_dir, "strong_attacks_controls.csv")
    write_csv(out_csv, rows)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    run(Config())
