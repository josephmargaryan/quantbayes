from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from quantbayes.ball_dp.experiments.cifar10_embed_cache import (
    CIFAR10EmbedConfig,
    get_or_compute_cifar10_embeddings,
)
from quantbayes.ball_dp.utils.io import ensure_dir, write_json, write_csv_rows
from quantbayes.ball_dp.utils.plotting import save_errorbar_plot
from quantbayes.ball_dp.utils.seeding import set_global_seed

from quantbayes.retrieval_dp.radius import (
    within_class_nn_distances,
    radii_from_percentiles,
)
from quantbayes.retrieval_dp.metrics import l2_norm_rows
from quantbayes.retrieval_dp.sensitivity import bounded_replacement_radius
from quantbayes.retrieval_dp.mechanisms import (
    NonPrivateTopKRetriever,
    NoisyScoresTopKLaplaceRetriever,
    NoisyScoresTopKGaussianRetriever,
)
from quantbayes.retrieval_dp.eval import (
    eval_accuracy_trials,
    eval_retrieval_classifier_accuracy,
)


def subsample_per_class(
    Z: np.ndarray, y: np.ndarray, n_per_class: int, *, num_classes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_all = []
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        if idx.size < n_per_class:
            raise RuntimeError(
                f"class {c} has only {idx.size} samples; need {n_per_class}"
            )
        sub = rng.choice(idx, size=n_per_class, replace=False)
        idx_all.append(sub)
    idx_all = np.concatenate(idx_all, axis=0)
    rng.shuffle(idx_all)
    return Z[idx_all], y[idx_all]


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", type=str, default="./runs/cifar10_retrieval_ball_dp")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument(
        "--cache_npz", type=str, default="./cache/cifar10_resnet18_embeds.npz"
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--weights", type=str, default="DEFAULT")
    ap.add_argument("--l2_normalize", action="store_true")

    ap.add_argument("--score", type=str, default="neg_l2", choices=["dot", "neg_l2"])
    ap.add_argument(
        "--mechanism", type=str, default="gaussian", choices=["gaussian", "laplace"]
    )

    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--delta", type=float, default=1e-5)  # only for gaussian
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--nn_sample_per_class", type=int, default=400)
    ap.add_argument("--r_percentiles", type=str, default="10,25,50,75,90")
    ap.add_argument("--B_quantile", type=float, default=0.999)

    ap.add_argument("--eps_list", type=str, default="0.05,0.1,0.2,0.5,1,2,5,10")
    ap.add_argument("--n_per_class_list", type=str, default="100,2000,5000")

    ap.add_argument("--n_test_eval", type=int, default=2000)
    ap.add_argument("--eval_batch", type=int, default=128)

    args = ap.parse_args()
    set_global_seed(args.seed)

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    emb_cfg = CIFAR10EmbedConfig(
        data_dir=args.data_dir,
        cache_npz=args.cache_npz,
        batch_size=args.batch_size,
        device=args.device,
        weights=args.weights,
        l2_normalize=bool(args.l2_normalize),
        seed=args.seed,
    )
    Ztr, ytr, Zte, yte = get_or_compute_cifar10_embeddings(emb_cfg)

    # optionally subsample test for speed
    if int(args.n_test_eval) > 0 and int(args.n_test_eval) < int(Zte.shape[0]):
        rng = np.random.default_rng(args.seed + 999)
        idx = rng.choice(Zte.shape[0], size=int(args.n_test_eval), replace=False)
        Zte_eval = Zte[idx]
        yte_eval = yte[idx]
    else:
        Zte_eval = Zte
        yte_eval = yte

    # radii (computed on full training set for consistency)
    nn_dists = within_class_nn_distances(
        Ztr,
        ytr,
        num_classes=10,
        per_class=int(args.nn_sample_per_class),
        seed=int(args.seed),
    )
    r_percentiles = [
        float(s.strip()) for s in args.r_percentiles.split(",") if s.strip()
    ]
    r_vals = radii_from_percentiles(nn_dists, r_percentiles)

    # bounded-replacement baseline radius r_std = 2B
    norms = l2_norm_rows(Ztr)
    B = float(np.quantile(norms, float(args.B_quantile)))
    r_std = bounded_replacement_radius(B)

    # query norm bound (needed for dot scoring). If embeddings are L2-normalized, this is ~1.
    # If not normalized, we clip queries to a conservative bound (still clean DP after clipping).
    if bool(args.l2_normalize):
        q_norm_bound = 1.0
    else:
        q_norm_bound = float(
            np.quantile(l2_norm_rows(Zte_eval), float(args.B_quantile))
        )

    write_json(
        out_dir / "retrieval_setup.json",
        {
            "score": str(args.score),
            "mechanism": str(args.mechanism),
            "k": int(args.k),
            "delta": float(args.delta),
            "trials": int(args.trials),
            "l2_normalize": int(bool(args.l2_normalize)),
            "B_quantile": float(args.B_quantile),
            "B": float(B),
            "r_std": float(r_std),
            "r_percentiles": r_percentiles,
            "r_values": r_vals,
            "q_norm_bound": float(q_norm_bound),
            "n_test_eval": int(Zte_eval.shape[0]),
        },
    )

    eps_list = [float(s.strip()) for s in args.eps_list.split(",") if s.strip()]
    n_list = [int(s.strip()) for s in args.n_per_class_list.split(",") if s.strip()]

    results: List[Dict[str, object]] = []

    for n_per_class in n_list:
        # corpus = subsampled training embeddings
        Zdb, ydb = subsample_per_class(
            Ztr, ytr, n_per_class, num_classes=10, seed=args.seed
        )

        # non-private baseline accuracy
        base_retr = NonPrivateTopKRetriever(
            V=Zdb, score=str(args.score), q_norm_bound=q_norm_bound
        )
        acc_np = eval_retrieval_classifier_accuracy(
            base_retr,
            Zte_eval,
            yte_eval,
            ydb,
            k=int(args.k),
            batch_size=int(args.eval_batch),
        )

        curves: Dict[str, Tuple[List[float], List[float]]] = {}

        def make_factory(r_val: float, eps: float):
            def _factory(seed_local: int):
                rng = np.random.default_rng(seed_local)
                if str(args.mechanism) == "laplace":
                    return NoisyScoresTopKLaplaceRetriever(
                        V=Zdb,
                        score=str(args.score),
                        r=float(r_val),
                        eps=float(eps),
                        rng=rng,
                        q_norm_bound=float(q_norm_bound),
                    )
                else:
                    return NoisyScoresTopKGaussianRetriever(
                        V=Zdb,
                        score=str(args.score),
                        r=float(r_val),
                        eps=float(eps),
                        delta=float(args.delta),
                        rng=rng,
                        q_norm_bound=float(q_norm_bound),
                    )

            return _factory

        def eval_curve(
            label: str, r_kind: str, r_val: float
        ) -> Tuple[List[float], List[float]]:
            means, stds = [], []
            for eps in eps_list:
                m, s = eval_accuracy_trials(
                    make_factory(r_val=float(r_val), eps=float(eps)),
                    Zte_eval,
                    yte_eval,
                    ydb,
                    k=int(args.k),
                    trials=int(args.trials),
                    seed=int(args.seed) + 123,
                    batch_size=int(args.eval_batch),
                    n_classes=10,
                )
                means.append(float(m))
                stds.append(float(s))

                results.append(
                    {
                        "head": "retrieval",
                        "score": str(args.score),
                        "mechanism": str(args.mechanism),
                        "k": int(args.k),
                        "n_per_class": int(n_per_class),
                        "m_db": int(Zdb.shape[0]),
                        "r_kind": str(r_kind),
                        "r_value": float(r_val),
                        "eps": float(eps),
                        "delta": (
                            float(args.delta)
                            if str(args.mechanism) == "gaussian"
                            else None
                        ),
                        "q_norm_bound": float(q_norm_bound),
                        "B": float(B),
                        "r_std": float(r_std),
                        "acc_mean": float(m),
                        "acc_std": float(s),
                        "acc_non_private": float(acc_np),
                        "l2_normalize": int(bool(args.l2_normalize)),
                    }
                )
            return means, stds

        # Ball radii
        for p in r_percentiles:
            rv = float(r_vals[float(p)])
            label = f"Ball r=p{int(p)} ({rv:.3g})"
            curves[label] = eval_curve(label, r_kind=f"ball_p{int(p)}", r_val=rv)

        # Baseline radius
        curves[f"Bounded r_std=2B ({r_std:.3g})"] = eval_curve(
            f"Bounded r_std=2B ({r_std:.3g})",
            r_kind="std_2B",
            r_val=float(r_std),
        )

        save_errorbar_plot(
            eps_list,
            curves,
            title=f"CIFAR-10 | Retrieval ({args.score},{args.mechanism}) | n/class={n_per_class} | base={acc_np:.3f}",
            xlabel="epsilon (log scale)",
            ylabel="test accuracy",
            out_path=fig_dir / f"acc_vs_eps_retrieval_n{n_per_class}.png",
            xscale_log=True,
        )

    # write CSV
    cols = [
        "head",
        "score",
        "mechanism",
        "k",
        "n_per_class",
        "m_db",
        "r_kind",
        "r_value",
        "eps",
        "delta",
        "q_norm_bound",
        "B",
        "r_std",
        "acc_mean",
        "acc_std",
        "acc_non_private",
        "l2_normalize",
    ]
    write_csv_rows(out_dir / "results.csv", results, fieldnames=cols)


if __name__ == "__main__":
    main()
