from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import os
import numpy as np

from quantbayes.cohort_dp.metrics import L2Metric
from quantbayes.cohort_dp.synthetic import make_synthetic_patients
from quantbayes.cohort_dp.analysis import estimate_r_global
from quantbayes.cohort_dp.registry import MechanismSpec, build_api, seed_from_spec
from quantbayes.cohort_dp.candidates import AllCandidates
from quantbayes.cohort_dp.io import write_csv
from quantbayes.cohort_dp.abi import compute_abi_metrics


@dataclass
class Config:
    seed: int = 0
    out_dir: str = "results_abi_report"

    n: int = 2000
    d: int = 25
    n_clusters: int = 10

    k_service: int = 10
    k0_eval: int = 20  # ABI ball size definition for evaluation
    n_abi_queries: int = 60
    abi_query_noise_std: float = 0.20

    mc_samples: int = 800  # used only when retriever has no exact pmf


def run(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    metric = L2Metric()

    X = make_synthetic_patients(cfg.n, cfg.d, cfg.n_clusters, rng)

    r_global = estimate_r_global(
        X, metric, k0=cfg.k0_eval, rng=np.random.default_rng(cfg.seed + 1), m=200
    )
    print(f"r_global ≈ {r_global:.4f}")

    candgen = AllCandidates(n=X.shape[0])

    base = X[rng.choice(cfg.n, size=min(cfg.n_abi_queries, cfg.n), replace=False)]
    abi_queries = base + rng.normal(scale=cfg.abi_query_noise_std, size=base.shape)

    # Include at least one of each family + AB-Optimal (ABO)
    specs = [
        MechanismSpec(
            "NONPRIVATE", k_service=cfg.k_service, eps_total=0.0, name="NonPrivate kNN"
        ),
        MechanismSpec(
            "NOISY_TOPK", k_service=cfg.k_service, eps_total=2.0, name="NoisyTopK eps=2"
        ),
        MechanismSpec("EM", k_service=cfg.k_service, eps_total=2.0, name="EM eps=2"),
        MechanismSpec(
            "ABU", k_service=cfg.k_service, eps_total=0.0, k0=20, name="ABU k0=20"
        ),
        MechanismSpec(
            "ABM",
            k_service=cfg.k_service,
            eps_total=2.0,
            k0=20,
            mix_uniform=0.7,
            name="ABM k0=20 mix=0.7 (eps->gamma)",
        ),
        MechanismSpec(
            "ABO",
            k_service=cfg.k_service,
            eps_total=0.0,
            k0=20,
            eps_ball=1.0,
            name="ABO k0=20 eps_ball=1",
        ),
        MechanismSpec(
            "ABO",
            k_service=cfg.k_service,
            eps_total=0.0,
            k0=20,
            eps_ball=2.0,
            name="ABO k0=20 eps_ball=2",
        ),
    ]

    rows: List[Dict[str, Any]] = []

    for s in specs:
        seed_s = seed_from_spec(cfg.seed, s, namespace="abi_report")
        api = build_api(
            s,
            X_db=X,
            metric=metric,
            r=r_global,
            seed=seed_s,
            candidate_generator=candgen,
            no_repeat=False,
            sticky_policy=None,
        )

        ms = []
        for z in abi_queries:
            abi = compute_abi_metrics(
                X=X,
                metric=metric,
                retriever=api.retriever,  # mechanism-level ABI
                z=z,
                k0=int(cfg.k0_eval),
                candidates=None,
                mc_samples=int(cfg.mc_samples),
                rng=np.random.default_rng(seed_s + 999),
            )
            ms.append(abi)

        m_arr = np.array([a.m for a in ms], dtype=float)
        delta_arr = np.array([a.delta for a in ms], dtype=float)
        eps_arr = np.array([a.eps_ball_cond for a in ms], dtype=float)
        pmax_arr = np.array([a.pmax_in_ball_uncond for a in ms], dtype=float)

        rows.append(
            {
                "name": s.label(),
                "kind": s.kind,
                "k_service": int(cfg.k_service),
                "k0_eval": int(cfg.k0_eval),
                "eps_total": float(s.eps_total),
                "eps_ball": float(s.eps_ball) if s.eps_ball is not None else None,
                "mix_uniform": float(s.mix_uniform),
                "gamma": float(s.gamma) if s.gamma is not None else None,
                "m_mean": float(np.mean(m_arr)),
                "m_p10": float(np.percentile(m_arr, 10)),
                "m_p50": float(np.percentile(m_arr, 50)),
                "m_p90": float(np.percentile(m_arr, 90)),
                "delta_mean": float(np.mean(delta_arr)),
                "eps_ball_cond_mean": (
                    float(np.mean(eps_arr[np.isfinite(eps_arr)]))
                    if np.any(np.isfinite(eps_arr))
                    else float("inf")
                ),
                "pmax_in_ball_uncond_mean": float(np.mean(pmax_arr)),
            }
        )

        print(
            f"[ABI] {s.label():>28} | m~{np.mean(m_arr):.1f} | delta~{np.mean(delta_arr):.3f} | eps_ball~{np.mean(eps_arr[np.isfinite(eps_arr)]):.3f}"
        )

    out_csv = os.path.join(cfg.out_dir, "abi_report.csv")
    write_csv(out_csv, rows)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    run(Config())
