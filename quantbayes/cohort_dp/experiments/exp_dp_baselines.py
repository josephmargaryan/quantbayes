# quantbayes/cohort_dp/experiments/exp_ab_benchmark.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import os
import numpy as np

from quantbayes.cohort_dp.metrics import L2Metric
from quantbayes.cohort_dp.eval import true_knn, precision_at_k, FrequencyAttacker
from quantbayes.cohort_dp.analysis import (
    mean_distance,
    estimate_r_from_local_density,
    compute_ball_density_stats,
    attacker_exact_within_r,
)
from quantbayes.cohort_dp.synthetic import make_synthetic_patients
from quantbayes.cohort_dp.registry import (
    MechanismSpec,
    build_api,
    build_candidate_generator,
    seed_from_spec,
)
from quantbayes.cohort_dp.io import write_csv
from quantbayes.cohort_dp.plotting import plot_eps_sweep, plot_q_sweep


@dataclass
class Config:
    seed: int = 0
    out_root: str = "results_dp_baselines_sweep"

    candidate_mode: str = "all"  # all | lsh | proto

    n: int = 2000
    d: int = 25
    n_clusters: int = 10

    n_utility_queries: int = 200
    utility_query_noise_std: float = 0.20

    n_attack_targets: int = 50
    Q_fixed_for_eps_sweep: int = 50
    Q_list: List[int] = None

    eps_list: List[float] = None
    eps_for_Q_sweep: List[float] = None

    k0_for_r: int = 20

    def __post_init__(self):
        if self.Q_list is None:
            self.Q_list = [1, 5, 10, 25, 50, 100]
        if self.eps_list is None:
            self.eps_list = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
        if self.eps_for_Q_sweep is None:
            self.eps_for_Q_sweep = [0.5, 1.0, 2.0]


def run(cfg: Config) -> None:
    os.makedirs(cfg.out_root, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    metric = L2Metric()

    X = make_synthetic_patients(cfg.n, cfg.d, cfg.n_clusters, rng)

    r = estimate_r_from_local_density(
        X, metric, k0=cfg.k0_for_r, rng=np.random.default_rng(cfg.seed + 1)
    )
    print(f"Estimated radius r ≈ {r:.4f} | candidate_mode={cfg.candidate_mode}")

    dens = compute_ball_density_stats(
        X, metric, r, rng=np.random.default_rng(cfg.seed + 3), m=200
    )
    write_csv(os.path.join(cfg.out_root, "r_ball_density.csv"), [dens])

    # build candidate generator
    candgen = build_candidate_generator(
        cfg.candidate_mode, X, metric, seed=cfg.seed + 2
    )

    # utility queries
    base_idxs = rng.choice(cfg.n, size=min(cfg.n_utility_queries, cfg.n), replace=False)
    base_points = X[base_idxs]
    utility_queries = base_points + rng.normal(
        scale=cfg.utility_query_noise_std, size=base_points.shape
    )

    # attack targets
    targets = rng.choice(cfg.n, size=min(cfg.n_attack_targets, cfg.n), replace=False)

    # scenarios
    scenarios: List[MechanismSpec] = [
        MechanismSpec("NONPRIVATE", k_service=10, eps_total=0.0)
    ]
    for eps in cfg.eps_list:
        scenarios.append(MechanismSpec("EM", k_service=1, eps_total=eps))
        scenarios.append(MechanismSpec("EM", k_service=10, eps_total=eps))
        scenarios.append(MechanismSpec("NOISY_TOPK", k_service=10, eps_total=eps))
        scenarios.append(MechanismSpec("LAPLACE_TOPK", k_service=10, eps_total=eps))

    # EPS sweep
    eps_rows: List[Dict[str, Any]] = []
    for s in scenarios:
        seed_s = seed_from_spec(cfg.seed, s, namespace="eps_sweep")

        api = build_api(
            s,
            X_db=X,
            metric=metric,
            r=r,
            seed=seed_s,
            candidate_generator=candgen,
            no_repeat=False,
            sticky_policy=None,
        )

        precisions, dist_ratios = [], []
        for z in utility_queries:
            got = api.query(z=z, k=s.k_service)
            truth = true_knn(metric, X, z, k=s.k_service)
            precisions.append(precision_at_k(got, truth))

            d_got = mean_distance(metric, X, z, got)
            d_true = mean_distance(metric, X, z, truth)
            dist_ratios.append(d_got / max(d_true, 1e-3))

        attacker = FrequencyAttacker(
            query_noise_std=0.05,
            Q=cfg.Q_fixed_for_eps_sweep,
            k_attack=s.k_service,
            rng=np.random.default_rng(seed_s + 123),
            count_all_returned=True,
            session_id="attacker",
            new_session_per_query=False,
        )
        attack_exact, attack_within = attacker_exact_within_r(
            api, X, metric, r, attacker, targets
        )

        eps_rows.append(
            {
                "candidate_mode": cfg.candidate_mode,
                "n": cfg.n,
                "d": cfg.d,
                "r": r,
                "mechanism": s.kind,
                "k_service": s.k_service,
                "eps_total": float(s.eps_total),
                "Q_fixed": cfg.Q_fixed_for_eps_sweep,
                "precision": float(np.mean(precisions)),
                "dist_ratio": float(np.mean(dist_ratios)),
                "attacker_exact": float(attack_exact),
                "attacker_within_r": float(attack_within),
            }
        )

        print(
            f"{s.label():>30} | dist_ratio={np.mean(dist_ratios):.3f} | exact={attack_exact:.3f} within={attack_within:.3f}"
        )

    write_csv(os.path.join(cfg.out_root, "eps_sweep.csv"), eps_rows)
    plot_eps_sweep(
        eps_rows,
        out_dir=os.path.join(cfg.out_root, "figures"),
        title_suffix=f" ({cfg.candidate_mode})",
    )

    # Q sweep (selected eps)
    q_rows: List[Dict[str, Any]] = []
    q_scenarios = [MechanismSpec("NONPRIVATE", 10, 0.0)]
    for eps in cfg.eps_for_Q_sweep:
        q_scenarios.append(MechanismSpec("EM", 10, eps))
        q_scenarios.append(MechanismSpec("NOISY_TOPK", 10, eps))
        q_scenarios.append(MechanismSpec("LAPLACE_TOPK", 10, eps))

    for s in q_scenarios:
        for Q in cfg.Q_list:
            seed_s = seed_from_spec(cfg.seed, s, namespace=f"q_sweep_Q={Q}")
            api = build_api(
                s, X_db=X, metric=metric, r=r, seed=seed_s, candidate_generator=candgen
            )

            attacker = FrequencyAttacker(
                query_noise_std=0.05,
                Q=Q,
                k_attack=s.k_service,
                rng=np.random.default_rng(seed_s + 999),
                count_all_returned=True,
                session_id="attacker",
                new_session_per_query=False,
            )
            attack_exact, attack_within = attacker_exact_within_r(
                api, X, metric, r, attacker, targets
            )

            q_rows.append(
                {
                    "candidate_mode": cfg.candidate_mode,
                    "n": cfg.n,
                    "d": cfg.d,
                    "r": r,
                    "mechanism": s.kind,
                    "k_service": s.k_service,
                    "eps_total": float(s.eps_total),
                    "Q": int(Q),
                    "attacker_exact": float(attack_exact),
                    "attacker_within_r": float(attack_within),
                }
            )

    write_csv(os.path.join(cfg.out_root, "q_sweep.csv"), q_rows)
    plot_q_sweep(
        q_rows,
        out_dir=os.path.join(cfg.out_root, "figures"),
        title_suffix=f" ({cfg.candidate_mode})",
    )

    print(f"\nSaved under: {cfg.out_root}/")


if __name__ == "__main__":
    run(Config())
