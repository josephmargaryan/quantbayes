# quantbayes/cohort_dp/experiments/exp_ab_uniform_sensitivity.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt

from quantbayes.cohort_dp.metrics import L2Metric
from quantbayes.cohort_dp.eval import true_knn, precision_at_k, FrequencyAttacker
from quantbayes.cohort_dp.synthetic import make_synthetic_patients
from quantbayes.cohort_dp.analysis import (
    mean_distance,
    estimate_r_global,
    attacker_exact_within_r,
)
from quantbayes.cohort_dp.registry import MechanismSpec, build_api, seed_from_spec
from quantbayes.cohort_dp.candidates import AllCandidates
from quantbayes.cohort_dp.io import write_csv


@dataclass
class Config:
    seed: int = 0
    out_dir: str = "results_ab_mechanisms_benchmark"
    n: int = 2000
    d: int = 25
    n_clusters: int = 10
    k_service: int = 10
    k0_for_r: int = 20
    n_utility: int = 300
    utility_noise_std: float = 0.20
    n_attack_targets: int = 80
    Q: int = 50


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

    base = X[rng.choice(cfg.n, size=min(cfg.n_utility, cfg.n), replace=False)]
    utility_queries = base + rng.normal(scale=cfg.utility_noise_std, size=base.shape)

    targets = rng.choice(cfg.n, size=min(cfg.n_attack_targets, cfg.n), replace=False)

    scenarios = [
        MechanismSpec("NONPRIVATE", cfg.k_service, 0.0, name="NonPrivate kNN"),
        MechanismSpec("NOISY_TOPK", cfg.k_service, 1.0, name="NoisyTopK eps=1"),
        MechanismSpec("NOISY_TOPK", cfg.k_service, 2.0, name="NoisyTopK eps=2"),
        MechanismSpec("EM", cfg.k_service, 2.0, name="EM(k=10) eps=2"),
        MechanismSpec("ABU", cfg.k_service, 0.0, k0=20, name="AB-Uniform k0=20"),
        MechanismSpec("ABU", cfg.k_service, 0.0, k0=40, name="AB-Uniform k0=40"),
        MechanismSpec("ABE", cfg.k_service, 2.0, k0=20, name="AB-Exp k0=20 eps=2"),
        MechanismSpec(
            "ABM",
            cfg.k_service,
            2.0,
            k0=20,
            mix_uniform=0.7,
            name="AB-Mix k0=20 eps=2 mix=0.7",
        ),
        MechanismSpec(
            "ABM",
            cfg.k_service,
            2.0,
            k0=40,
            mix_uniform=0.7,
            name="AB-Mix k0=40 eps=2 mix=0.7",
        ),
    ]

    attacker = FrequencyAttacker(
        query_noise_std=0.05,
        Q=cfg.Q,
        k_attack=cfg.k_service,
        rng=np.random.default_rng(cfg.seed + 123),
        count_all_returned=True,
        session_id="attacker",
        new_session_per_query=False,
    )

    rows: List[Dict[str, Any]] = []

    for s in scenarios:
        seed_s = seed_from_spec(cfg.seed, s, namespace="demo_novel")
        api_plain = build_api(
            s,
            X_db=X,
            metric=metric,
            r=r_global,
            seed=seed_s,
            candidate_generator=candgen,
            no_repeat=False,
        )

        precisions, dist_ratios = [], []
        for z in utility_queries:
            got = api_plain.query(z=z, k=s.k_service)
            truth = true_knn(metric, X, z, k=s.k_service)
            precisions.append(precision_at_k(got, truth))

            d_g = mean_distance(metric, X, z, got)
            d_t = mean_distance(metric, X, z, truth)
            dist_ratios.append(d_g / max(d_t, 1e-3))

        exact_plain, within_plain = attacker_exact_within_r(
            api_plain, X, metric, r_global, attacker, targets
        )

        api_norep = build_api(
            s,
            X_db=X,
            metric=metric,
            r=r_global,
            seed=seed_s + 1,
            candidate_generator=candgen,
            no_repeat=True,
        )
        exact_norep, within_norep = attacker_exact_within_r(
            api_norep, X, metric, r_global, attacker, targets
        )

        rows.append(
            {
                "name": s.label(),
                "kind": s.kind,
                "k_service": s.k_service,
                "eps_total": float(s.eps_total),
                "k0": int(s.k0),
                "mix_uniform": float(s.mix_uniform),
                "r_global": float(r_global),
                "utility_precision": float(np.mean(precisions)),
                "utility_dist_ratio": float(np.mean(dist_ratios)),
                "attack_exact_plain": float(exact_plain),
                "attack_within_plain": float(within_plain),
                "attack_exact_no_repeat": float(exact_norep),
                "attack_within_no_repeat": float(within_norep),
            }
        )

        print(
            f"{s.label():>28} | dist_ratio={np.mean(dist_ratios):.3f} | exact={exact_plain:.3f} | norep={exact_norep:.3f}"
        )

    write_csv(os.path.join(cfg.out_dir, "novel_summary.csv"), rows)

    # a couple of quick plots
    labels = [r["name"] for r in rows]
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 4))
    plt.bar(x, [r["utility_dist_ratio"] for r in rows])
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("dist_ratio (lower is better)")
    plt.title("Utility (dist_ratio) across scenarios")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "utility_dist_ratio_bar.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    w = 0.4
    plt.bar(x - w / 2, [r["attack_exact_plain"] for r in rows], width=w, label="plain")
    plt.bar(
        x + w / 2,
        [r["attack_exact_no_repeat"] for r in rows],
        width=w,
        label="no-repeat",
    )
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("attacker exact success")
    plt.title("Attack exact success: plain vs no-repeat")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(cfg.out_dir, "attack_exact_plain_vs_norepeat.png"), dpi=160
    )
    plt.close()

    print(f"\nSaved under: {cfg.out_dir}/")


if __name__ == "__main__":
    run(Config())
