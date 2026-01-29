# cohort_dp/experiments/decentralized_sweep.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import os
import numpy as np
import matplotlib.pyplot as plt

from quantbayes.cohort_dp.metrics import L2Metric
from quantbayes.cohort_dp.synthetic import make_synthetic_multihospital
from quantbayes.cohort_dp.analysis import estimate_r_global
from quantbayes.cohort_dp.clustering import kmeans_centers
from quantbayes.cohort_dp.decentralized import (
    HospitalNode,
    PrototypeFirstRouter,
    majority_vote,
)
from quantbayes.cohort_dp.registry import MechanismSpec, build_api, seed_from_spec
from quantbayes.cohort_dp.baselines import NonPrivateKNNRetriever
from quantbayes.cohort_dp.novel_mechanisms import AdaptiveBallUniformRetriever
from quantbayes.cohort_dp.io import write_csv


@dataclass
class Config:
    seed: int = 0
    out_dir: str = "results_decentralized_prototype_router_sweep"

    # dataset
    n_total: int = 8000
    d: int = 25
    n_classes: int = 10
    n_hospitals: int = 5
    class_sep: float = 1.9
    noise_std: float = 1.7
    hospital_shift_std: float = 1.0

    # retrieval
    k_service: int = 10
    k0_local: int = 20
    n_prototypes_per_h: int = 40
    proto_kmeans_iters: int = 25

    # sweeps
    n_probe_list: List[int] = None
    proto_noise_list: List[float] = None
    local_no_repeat_list: List[bool] = None
    output_no_repeat_list: List[bool] = None

    # attack
    Q: int = 50
    n_attack_targets: int = 80
    attacker_modes: List[str] = None
    session_pool_size: int = 10  # for pool_sessions

    def __post_init__(self):
        if self.n_probe_list is None:
            self.n_probe_list = [1, 2, 3]
        if self.proto_noise_list is None:
            self.proto_noise_list = [0.0, 0.05, 0.10, 0.20]
        if self.local_no_repeat_list is None:
            self.local_no_repeat_list = [False, True]
        if self.output_no_repeat_list is None:
            self.output_no_repeat_list = [False, True]
        if self.attacker_modes is None:
            self.attacker_modes = ["same_session", "pool_sessions", "rotating_sessions"]


def attacker_metrics_router(
    router: PrototypeFirstRouter,
    X_train: np.ndarray,
    h_train: np.ndarray,
    metric: L2Metric,
    r_global: float,
    targets: np.ndarray,
    *,
    Q: int,
    k_attack: int,
    attacker_mode: str,
    session_pool_size: int,
    seed: int,
) -> Dict[str, float]:
    """
    Router attacker metrics with per-target session isolation.
    Without this, router/output no-repeat state leaks across targets and biases results.
    """
    exact = 0
    within_r = 0
    hospital_hit = 0

    for t in targets:
        t = int(t)
        x_t = X_train[t]
        true_h = int(h_train[t])

        # Per-target RNG for reproducibility + comparability
        rng_t = np.random.default_rng(int(seed) + 10_000_000 + t)

        counts: Dict[int, int] = {}
        for q in range(int(Q)):
            z = x_t + rng_t.normal(loc=0.0, scale=0.05, size=x_t.shape[0])

            # IMPORTANT: make session IDs unique per target
            if attacker_mode == "same_session":
                sid = f"attacker::t{t}"
            elif attacker_mode == "rotating_sessions":
                sid = f"attacker::t{t}::q{q}"
            elif attacker_mode == "pool_sessions":
                sid = f"attacker::t{t}::p{q % max(1, session_pool_size)}"
            else:
                raise ValueError("unknown attacker_mode")

            idx = router.query(z=z, k_total=k_attack, session_id=sid)
            for i in np.asarray(idx, dtype=int).tolist():
                counts[i] = counts.get(i, 0) + 1

        pred = max(counts.items(), key=lambda kv: kv[1])[0] if counts else 0
        exact += int(pred == t)

        pred_h = int(h_train[pred])
        hospital_hit += int(pred_h == true_h)

        dist = metric.pairwise(x_t.reshape(1, -1), X_train[pred].reshape(1, -1))[0, 0]
        within_r += int(dist <= r_global)

    n = float(len(targets))
    return {
        "attack_exact": exact / n,
        "attack_within_r": within_r / n,
        "attack_hospital": hospital_hit / n,
    }


def run(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    metric = L2Metric()

    X, y, h = make_synthetic_multihospital(
        n_total=cfg.n_total,
        d=cfg.d,
        n_classes=cfg.n_classes,
        n_hospitals=cfg.n_hospitals,
        rng=rng,
        class_sep=cfg.class_sep,
        noise_std=cfg.noise_std,
        hospital_shift_std=cfg.hospital_shift_std,
    )

    idx = rng.permutation(cfg.n_total)
    n_train = int(0.70 * cfg.n_total)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X_train, y_train, h_train = X[train_idx], y[train_idx], h[train_idx]
    X_test, y_test, h_test = X[test_idx], y[test_idx], h[test_idx]

    r_global = estimate_r_global(
        X_train, metric, k0=20, rng=np.random.default_rng(cfg.seed + 1), m=200
    )
    print(f"r_global ≈ {r_global:.4f}")

    # centralized utility refs
    central_np = NonPrivateKNNRetriever(X=X_train, metric=metric)
    central_abu = AdaptiveBallUniformRetriever(
        X=X_train,
        metric=metric,
        k0=cfg.k0_local,
        rng=np.random.default_rng(cfg.seed + 9999),
        eps_total=0.0,
    )

    targets = np.random.default_rng(cfg.seed + 777).choice(
        X_train.shape[0],
        size=min(cfg.n_attack_targets, X_train.shape[0]),
        replace=False,
    )

    rows: List[Dict[str, Any]] = []

    # build base hospital partitions
    base_nodes: List[Dict[str, Any]] = []
    for hid in range(cfg.n_hospitals):
        local_idx = np.where(h_train == hid)[0]
        X_local = X_train[local_idx]
        y_local = y_train[local_idx]
        if X_local.shape[0] < 10:
            raise RuntimeError(f"Hospital {hid} too small: {X_local.shape[0]}")

        # prototypes
        centers = kmeans_centers(
            X_local,
            n_clusters=min(cfg.n_prototypes_per_h, X_local.shape[0]),
            n_iters=cfg.proto_kmeans_iters,
            rng=np.random.default_rng(cfg.seed + 2000 + hid),
        )

        base_nodes.append(
            {
                "hid": hid,
                "global_indices": local_idx.astype(int),
                "X_local": X_local,
                "y_local": y_local,
                "prototypes": centers,
            }
        )

    # local mechanism (fixed as ABU)
    local_spec = MechanismSpec(
        kind="ABU", k_service=cfg.k_service, eps_total=0.0, k0=cfg.k0_local
    )

    for local_no_repeat in cfg.local_no_repeat_list:
        for output_no_repeat in cfg.output_no_repeat_list:
            for proto_noise in cfg.proto_noise_list:
                # instantiate hospital nodes for this sweep setting
                hospitals: List[HospitalNode] = []
                for node in base_nodes:
                    hid = int(node["hid"])
                    prot = np.array(node["prototypes"], copy=True)
                    if proto_noise > 0:
                        prot = prot + np.random.default_rng(
                            cfg.seed + 3000 + hid
                        ).normal(scale=proto_noise, size=prot.shape)

                    seed_local = seed_from_spec(
                        cfg.seed,
                        local_spec,
                        namespace=f"local_h{hid}_nr={local_no_repeat}",
                    )
                    api_local = build_api(
                        local_spec,
                        X_db=node["X_local"],
                        metric=metric,
                        r=r_global,  # unused by ABU but fine
                        seed=seed_local,
                        candidate_generator=None,  # local uses full pool by default
                        candidate_mode=None,
                        no_repeat=bool(local_no_repeat),
                    )

                    hospitals.append(
                        HospitalNode(
                            hid=hid,
                            global_indices=node["global_indices"],
                            X_local=node["X_local"],
                            y_local=node["y_local"],
                            api=api_local,
                            prototypes=prot,
                        )
                    )

                for n_probe in cfg.n_probe_list:
                    router = PrototypeFirstRouter(
                        hospitals=hospitals,
                        X_train_global=X_train,
                        y_train_global=y_train,
                        metric=metric,
                        n_probe_hospitals=int(n_probe),
                        output_no_repeat=bool(output_no_repeat),
                        overfetch_factor=3,
                    )

                    # utility: router retrieval-only classifier (majority vote)
                    y_pred = np.zeros_like(y_test)
                    for i in range(X_test.shape[0]):
                        y_pred[i] = router.predict_label(
                            z=X_test[i], k_total=cfg.k_service, session_id="clinician"
                        )
                    acc_router = float(np.mean(y_pred == y_test))

                    # centralized refs
                    y_pred_np = []
                    y_pred_abu = []
                    for i in range(X_test.shape[0]):
                        idx_knn = central_np.query(X_test[i], k=cfg.k_service)
                        y_pred_np.append(majority_vote(y_train[idx_knn]))
                        idx_abu = central_abu.query(X_test[i], k=cfg.k_service)
                        y_pred_abu.append(majority_vote(y_train[idx_abu]))
                    acc_np = float(np.mean(np.array(y_pred_np, dtype=int) == y_test))
                    acc_abu = float(np.mean(np.array(y_pred_abu, dtype=int) == y_test))

                    # attacks
                    for amode in cfg.attacker_modes:
                        seed_attack = (
                            cfg.seed
                            + 999
                            + int(proto_noise * 1000)
                            + int(n_probe) * 17
                            + (1 if local_no_repeat else 0) * 100
                            + (1 if output_no_repeat else 0) * 1000
                            + (
                                0
                                if amode == "same_session"
                                else 10 if amode == "pool_sessions" else 20
                            )
                        )
                        attack = attacker_metrics_router(
                            router=router,
                            X_train=X_train,
                            h_train=h_train,
                            metric=metric,
                            r_global=r_global,
                            targets=targets,
                            Q=cfg.Q,
                            k_attack=cfg.k_service,
                            attacker_mode=amode,
                            session_pool_size=cfg.session_pool_size,
                            seed=seed_attack,
                        )

                        rows.append(
                            {
                                "local_no_repeat": bool(local_no_repeat),
                                "output_no_repeat": bool(output_no_repeat),
                                "proto_noise": float(proto_noise),
                                "n_probe_hospitals": int(n_probe),
                                "k0_local": int(cfg.k0_local),
                                "k_service": int(cfg.k_service),
                                "n_hospitals": int(cfg.n_hospitals),
                                "r_global": float(r_global),
                                "acc_router": float(acc_router),
                                "acc_central_nonprivate": float(acc_np),
                                "acc_central_ABU": float(acc_abu),
                                "attacker_mode": str(amode),
                                "attack_exact": float(attack["attack_exact"]),
                                "attack_hospital": float(attack["attack_hospital"]),
                                "attack_within_r": float(attack["attack_within_r"]),
                                "session_pool_size": int(cfg.session_pool_size),
                            }
                        )

                        print(
                            f"[decent] localNR={local_no_repeat} outNR={output_no_repeat} proto={proto_noise:.2f} probe={n_probe} "
                            f"| acc={acc_router:.3f} | mode={amode} exact={attack['attack_exact']:.3f} hosp={attack['attack_hospital']:.3f}"
                        )

    csv_path = os.path.join(cfg.out_dir, "decentralized_sweep.csv")
    write_csv(csv_path, rows)
    print(f"\nSaved: {csv_path}")

    # Basic plot: attack exact vs prototype noise for each mode/policy (optional quick view)
    for amode in cfg.attacker_modes:
        for local_no_repeat in cfg.local_no_repeat_list:
            for output_no_repeat in cfg.output_no_repeat_list:
                plt.figure()
                for n_probe in cfg.n_probe_list:
                    xs, ys = [], []
                    for r in rows:
                        if r["attacker_mode"] != amode:
                            continue
                        if r["local_no_repeat"] != bool(local_no_repeat):
                            continue
                        if r["output_no_repeat"] != bool(output_no_repeat):
                            continue
                        if r["n_probe_hospitals"] != int(n_probe):
                            continue
                        xs.append(r["proto_noise"])
                        ys.append(r["attack_exact"])
                    if len(xs) == 0:
                        continue
                    order = np.argsort(np.array(xs))
                    xs = np.array(xs)[order]
                    ys = np.array(ys)[order]
                    plt.plot(xs, ys, marker="o", label=f"n_probe={n_probe}")

                plt.xlabel("Prototype noise std")
                plt.ylabel("Attacker exact success")
                plt.title(
                    f"Attack exact vs proto noise | mode={amode} | localNR={local_no_repeat} outNR={output_no_repeat}"
                )
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                fname = f"attack_exact__{amode}__localNR_{local_no_repeat}__outNR_{output_no_repeat}.png"
                plt.savefig(os.path.join(cfg.out_dir, fname), dpi=160)
                plt.close()

    print(f"Saved plots under: {cfg.out_dir}/")


if __name__ == "__main__":
    run(Config())
