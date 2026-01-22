# quantbayes/cohort_dp/experiments/exp_retrieval_augmented.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import os
import numpy as np
import matplotlib.pyplot as plt

from quantbayes.cohort_dp.metrics import L2Metric
from quantbayes.cohort_dp.synthetic import make_synthetic_patients
from quantbayes.cohort_dp.analysis import (
    estimate_r_global,
    mean_distance,
    attacker_exact_within_r,
)
from quantbayes.cohort_dp.eval import true_knn, precision_at_k, FrequencyAttacker
from quantbayes.cohort_dp.registry import MechanismSpec, build_api, seed_from_spec
from quantbayes.cohort_dp.candidates import AllCandidates
from quantbayes.cohort_dp.io import write_csv


@dataclass
class Config:
    seed: int = 0
    out_dir: str = "results_ab_uniform_k0_q_sweep"

    n: int = 2000
    d: int = 25
    n_clusters: int = 10

    k_service: int = 10
    k0_list: List[int] = None
    Q_list: List[int] = None

    n_utility: int = 300
    utility_noise_std: float = 0.20

    n_attack_targets: int = 80

    def __post_init__(self):
        if self.k0_list is None:
            self.k0_list = [5, 10, 20, 40, 80]
        if self.Q_list is None:
            self.Q_list = [1, 5, 10, 25, 50, 100]


def run(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    metric = L2Metric()

    X = make_synthetic_patients(cfg.n, cfg.d, cfg.n_clusters, rng)
    r_global = estimate_r_global(
        X, metric, k0=20, rng=np.random.default_rng(cfg.seed + 1), m=200
    )
    print(f"r_global ≈ {r_global:.4f}")

    candgen = AllCandidates(n=cfg.n)

    base = X[rng.choice(cfg.n, size=min(cfg.n_utility, cfg.n), replace=False)]
    utility_queries = base + rng.normal(scale=cfg.utility_noise_std, size=base.shape)

    targets = rng.choice(cfg.n, size=min(cfg.n_attack_targets, cfg.n), replace=False)

    rows: List[Dict[str, Any]] = []

    for k0 in cfg.k0_list:
        spec = MechanismSpec(
            kind="ABU",
            k_service=cfg.k_service,
            eps_total=0.0,
            k0=int(k0),
            name=f"AB-Uniform k0={k0}",
        )
        seed_s = seed_from_spec(cfg.seed, spec, namespace="abu_sweep")

        # utility API (no-repeat irrelevant for utility here)
        api_util = build_api(
            spec,
            X_db=X,
            metric=metric,
            r=r_global,
            seed=seed_s,
            candidate_generator=candgen,
            no_repeat=False,
        )

        precisions, dist_ratios = [], []
        for z in utility_queries:
            got = api_util.query(z=z, k=cfg.k_service)
            truth = true_knn(metric, X, z, k=cfg.k_service)
            precisions.append(precision_at_k(got, truth))

            d_g = mean_distance(metric, X, z, got)
            d_t = mean_distance(metric, X, z, truth)
            dist_ratios.append(d_g / max(d_t, 1e-3))

        util_prec = float(np.mean(precisions))
        util_ratio = float(np.mean(dist_ratios))

        for Q in cfg.Q_list:
            # plain
            api_plain = build_api(
                spec,
                X_db=X,
                metric=metric,
                r=r_global,
                seed=seed_s + 10,
                candidate_generator=candgen,
                no_repeat=False,
            )
            attacker_plain = FrequencyAttacker(
                query_noise_std=0.05,
                Q=int(Q),
                k_attack=cfg.k_service,
                rng=np.random.default_rng(seed_s + 200 + int(Q)),
                count_all_returned=True,
                session_id="attacker",
                new_session_per_query=False,
            )
            exact_plain, within_plain = attacker_exact_within_r(
                api_plain, X, metric, r_global, attacker_plain, targets
            )

            # no-repeat same session
            api_norep = build_api(
                spec,
                X_db=X,
                metric=metric,
                r=r_global,
                seed=seed_s + 20,
                candidate_generator=candgen,
                no_repeat=True,
            )
            attacker_same = FrequencyAttacker(
                query_noise_std=0.05,
                Q=int(Q),
                k_attack=cfg.k_service,
                rng=np.random.default_rng(seed_s + 300 + int(Q)),
                count_all_returned=True,
                session_id="attacker",
                new_session_per_query=False,
            )
            exact_same, within_same = attacker_exact_within_r(
                api_norep, X, metric, r_global, attacker_same, targets
            )

            # no-repeat rotating sessions
            attacker_rot = FrequencyAttacker(
                query_noise_std=0.05,
                Q=int(Q),
                k_attack=cfg.k_service,
                rng=np.random.default_rng(seed_s + 400 + int(Q)),
                count_all_returned=True,
                session_id="attacker",
                new_session_per_query=True,
            )
            exact_rot, within_rot = attacker_exact_within_r(
                api_norep, X, metric, r_global, attacker_rot, targets
            )

            rows.append(
                {
                    "k0": int(k0),
                    "Q": int(Q),
                    "k_service": int(cfg.k_service),
                    "r_global": float(r_global),
                    "utility_precision": float(util_prec),
                    "utility_dist_ratio": float(util_ratio),
                    "exact_plain": float(exact_plain),
                    "within_plain": float(within_plain),
                    "exact_no_repeat_same_session": float(exact_same),
                    "within_no_repeat_same_session": float(within_same),
                    "exact_no_repeat_rotating_sessions": float(exact_rot),
                    "within_no_repeat_rotating_sessions": float(within_rot),
                }
            )

            print(
                f"k0={k0:<3} Q={Q:<3} | util dist_ratio={util_ratio:.3f} "
                f"| plain exact={exact_plain:.3f} | norep exact={exact_same:.3f} | rot exact={exact_rot:.3f}"
            )

    csv_path = os.path.join(cfg.out_dir, "ab_uniform_sweep.csv")
    write_csv(csv_path, rows)
    print(f"\nSaved: {csv_path}")

    # plots (exact vs Q)
    for key, title, col in [
        (
            "exact_plain",
            "AB-Uniform: attacker exact vs Q (plain)",
            "exact_plain_vs_Q.png",
        ),
        (
            "exact_no_repeat_same_session",
            "AB-Uniform: attacker exact vs Q (no-repeat, same session)",
            "exact_norepeat_same_vs_Q.png",
        ),
        (
            "exact_no_repeat_rotating_sessions",
            "AB-Uniform: attacker exact vs Q (no-repeat, rotating sessions)",
            "exact_norepeat_rotate_vs_Q.png",
        ),
    ]:
        plt.figure()
        for k0 in cfg.k0_list:
            xs = [r["Q"] for r in rows if r["k0"] == int(k0)]
            ys = [r[key] for r in rows if r["k0"] == int(k0)]
            plt.plot(xs, ys, marker="o", label=f"k0={k0}")
        plt.xlabel("Q")
        plt.ylabel(key)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, col), dpi=160)
        plt.close()

    print(f"Saved plots under: {cfg.out_dir}/")


if __name__ == "__main__":
    run(Config())
