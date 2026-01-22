# quantbayes/cohort_dp/experiments/exp_decentralized_routing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import os
import numpy as np
import matplotlib.pyplot as plt

from quantbayes.cohort_dp.metrics import L2Metric
from quantbayes.cohort_dp.synthetic import make_synthetic_embeddings_with_labels
from quantbayes.cohort_dp.analysis import estimate_r_global, attacker_exact_within_r
from quantbayes.cohort_dp.registry import MechanismSpec, build_api, seed_from_spec
from quantbayes.cohort_dp.candidates import AllCandidates
from quantbayes.cohort_dp.eval import FrequencyAttacker
from quantbayes.cohort_dp.simple_models import SoftmaxRegression, accuracy, blend_probs
from quantbayes.cohort_dp.retrieval_models import (
    retrieval_predict,
    retrieval_predict_proba,
    tune_alpha,
)
from quantbayes.cohort_dp.io import write_csv


@dataclass
class Config:
    seed: int = 0
    out_dir: str = "results_retrieval_augmented_classifier"

    # synthetic dataset
    n_total: int = 6000
    d: int = 25
    n_classes: int = 10
    class_sep: float = 3.0
    noise_std: float = 1.0

    # splits
    n_train: int = 4000
    n_val: int = 1000

    # retrieval + attack
    k_service: int = 10
    Q: int = 50
    n_attack_targets: int = 120


def run(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    metric = L2Metric()

    X, y = make_synthetic_embeddings_with_labels(
        n=cfg.n_total,
        d=cfg.d,
        n_classes=cfg.n_classes,
        rng=rng,
        class_sep=cfg.class_sep,
        noise_std=cfg.noise_std,
    )

    idx = rng.permutation(cfg.n_total)
    train_idx = idx[: cfg.n_train]
    val_idx = idx[cfg.n_train : cfg.n_train + cfg.n_val]
    test_idx = idx[cfg.n_train + cfg.n_val :]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    r_global = estimate_r_global(
        X_train, metric, k0=20, rng=np.random.default_rng(cfg.seed + 1), m=200
    )
    print(f"r_global ≈ {r_global:.4f}")

    # baseline softmax on embeddings
    clf = SoftmaxRegression(
        n_classes=cfg.n_classes, lr=0.2, l2=1e-3, epochs=500, seed=cfg.seed
    )
    clf.fit(X_train, y_train)
    p_val_base = clf.predict_proba(X_val)
    p_test_base = clf.predict_proba(X_test)
    acc_base = accuracy(y_test, np.argmax(p_test_base, axis=1))
    print(f"[Baseline Softmax] test acc = {acc_base:.3f}")

    candgen = AllCandidates(n=X_train.shape[0])

    specs = [
        MechanismSpec(
            kind="NONPRIVATE",
            k_service=cfg.k_service,
            eps_total=0.0,
            name="NonPrivate kNN",
        ),
        MechanismSpec(
            kind="NOISY_TOPK",
            k_service=cfg.k_service,
            eps_total=1.0,
            name="NoisyTopK eps=1",
        ),
        MechanismSpec(
            kind="NOISY_TOPK",
            k_service=cfg.k_service,
            eps_total=2.0,
            name="NoisyTopK eps=2",
        ),
        MechanismSpec(
            kind="LAPLACE_TOPK",
            k_service=cfg.k_service,
            eps_total=2.0,
            name="LaplaceTopK eps=2",
        ),
        MechanismSpec(
            kind="ABU",
            k_service=cfg.k_service,
            eps_total=0.0,
            k0=20,
            name="AB-Uniform k0=20",
        ),
        MechanismSpec(
            kind="ABU",
            k_service=cfg.k_service,
            eps_total=0.0,
            k0=40,
            name="AB-Uniform k0=40",
        ),
        MechanismSpec(
            kind="ABM",
            k_service=cfg.k_service,
            eps_total=2.0,
            k0=20,
            mix_uniform=0.7,
            name="AB-Mix k0=20 eps=2 mix=0.7",
        ),
    ]

    targets = rng.choice(
        cfg.n_train, size=min(cfg.n_attack_targets, cfg.n_train), replace=False
    )

    rows: List[Dict[str, Any]] = []

    for s in specs:
        seed_s = seed_from_spec(cfg.seed, s, namespace="month3")

        # plain API
        api_plain = build_api(
            s,
            X_db=X_train,
            metric=metric,
            r=r_global,
            seed=seed_s,
            candidate_generator=candgen,
            no_repeat=False,
        )

        # retrieval-only
        y_pred_retr = retrieval_predict(
            api_plain, X_test, y_train, k=cfg.k_service, session_id=None
        )
        acc_retr = accuracy(y_test, y_pred_retr)

        # retrieval probs for blending
        p_val_retr = retrieval_predict_proba(
            api_plain, X_val, y_train, cfg.n_classes, k=cfg.k_service
        )
        p_test_retr = retrieval_predict_proba(
            api_plain, X_test, y_train, cfg.n_classes, k=cfg.k_service
        )

        alpha = tune_alpha(p_val_base, p_val_retr, y_val)
        p_test_mix = blend_probs(p_test_base, p_test_retr, alpha=alpha)
        acc_mix = accuracy(y_test, np.argmax(p_test_mix, axis=1))

        # attacks: frequency
        attacker_plain = FrequencyAttacker(
            query_noise_std=0.05,
            Q=cfg.Q,
            k_attack=cfg.k_service,
            rng=np.random.default_rng(seed_s + 200),
            count_all_returned=True,
            session_id="attacker",
            new_session_per_query=False,
        )
        exact_plain, within_plain = attacker_exact_within_r(
            api_plain, X_train, metric, r_global, attacker_plain, targets
        )

        # no-repeat same session
        api_norep = build_api(
            s,
            X_db=X_train,
            metric=metric,
            r=r_global,
            seed=seed_s + 1,
            candidate_generator=candgen,
            no_repeat=True,
        )
        attacker_same = FrequencyAttacker(
            query_noise_std=0.05,
            Q=cfg.Q,
            k_attack=cfg.k_service,
            rng=np.random.default_rng(seed_s + 300),
            count_all_returned=True,
            session_id="attacker",
            new_session_per_query=False,
        )
        exact_norep, within_norep = attacker_exact_within_r(
            api_norep, X_train, metric, r_global, attacker_same, targets
        )

        # no-repeat but rotating sessions
        attacker_rot = FrequencyAttacker(
            query_noise_std=0.05,
            Q=cfg.Q,
            k_attack=cfg.k_service,
            rng=np.random.default_rng(seed_s + 400),
            count_all_returned=True,
            session_id="attacker",
            new_session_per_query=True,
        )
        exact_rot, within_rot = attacker_exact_within_r(
            api_norep, X_train, metric, r_global, attacker_rot, targets
        )

        rows.append(
            {
                "name": s.label(),
                "kind": s.kind,
                "k_service": cfg.k_service,
                "r_global": float(r_global),
                "baseline_softmax_test_acc": float(acc_base),
                "retrieval_only_test_acc": float(acc_retr),
                "blend_alpha": float(alpha),
                "retrieval_augmented_test_acc": float(acc_mix),
                "attack_exact_plain": float(exact_plain),
                "attack_within_plain": float(within_plain),
                "attack_exact_no_repeat_same_session": float(exact_norep),
                "attack_within_no_repeat_same_session": float(within_norep),
                "attack_exact_no_repeat_rotating_sessions": float(exact_rot),
                "attack_within_no_repeat_rotating_sessions": float(within_rot),
            }
        )

        print(
            f"{s.label():>26} | acc_mix={acc_mix:.3f} alpha={alpha:.2f} "
            f"| plain exact={exact_plain:.3f} | norep exact={exact_norep:.3f} | rot exact={exact_rot:.3f}"
        )

    csv_path = os.path.join(cfg.out_dir, "usecase_summary.csv")
    write_csv(csv_path, rows)
    print(f"\nSaved: {csv_path}")

    # quick plot: utility vs attack
    labels = [r["name"] for r in rows]
    x = np.arange(len(labels))

    plt.figure(figsize=(11, 4))
    plt.bar(
        x - 0.25,
        [r["retrieval_only_test_acc"] for r in rows],
        width=0.25,
        label="Retrieval-only",
    )
    plt.bar(
        x,
        [r["retrieval_augmented_test_acc"] for r in rows],
        width=0.25,
        label="Retrieval-augmented",
    )
    plt.bar(
        x + 0.25,
        [r["baseline_softmax_test_acc"] for r in rows],
        width=0.25,
        label="Baseline softmax",
    )
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Test accuracy")
    plt.title("Utility with retrieval layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "accuracy_bars.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    for r in rows:
        plt.scatter([r["attack_exact_plain"]], [r["retrieval_augmented_test_acc"]])
        plt.text(
            r["attack_exact_plain"] + 0.01,
            r["retrieval_augmented_test_acc"],
            r["kind"],
            fontsize=9,
        )
    plt.xlabel("Attacker exact success (plain)")
    plt.ylabel("Retrieval-augmented test accuracy")
    plt.title("Utility vs reconstruction risk (Month 3)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "utility_vs_attack_scatter.png"), dpi=160)
    plt.close()

    print(f"Saved plots under: {cfg.out_dir}/")


if __name__ == "__main__":
    run(Config())
