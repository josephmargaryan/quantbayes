# quantbayes/ball_dp/reconstruction/experiments/mnist_convex_reconstruction.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

from quantbayes.ball_dp.api import (
    compute_radius_policy,
    dp_release_eqx_model_gaussian,
    dp_release_ridge_prototypes_gaussian,
)
from quantbayes.ball_dp.lz import lz_softmax_linear_bound
from quantbayes.ball_dp.reconstruction.types import Candidate
from quantbayes.ball_dp.reconstruction.informed import InformedDataset
from quantbayes.ball_dp.reconstruction.priors import PoolBallPrior
from quantbayes.ball_dp.reconstruction.reporting import (
    ensure_dir,
    save_csv,
    plot_image_grid,
)
from quantbayes.ball_dp.reconstruction.vectorizers import (
    vectorize_prototypes,
    vectorize_eqx_model,
)
from quantbayes.ball_dp.reconstruction.convex.gaussian_identifier import (
    GaussianOutputIdentifier,
)
from quantbayes.ball_dp.reconstruction.convex.equation_solvers import (
    RidgePrototypesEquationSolver,
    SoftmaxEquationSolver,
)
from quantbayes.ball_dp.reconstruction.convex.eqx_trainers import (
    EqxFullBatchLBFGSTrainer,
    FullBatchLBFGSConfig,
    softmax_params_numpy,
)

from quantbayes.ball_dp.heads.prototypes import fit_ridge_prototypes
from quantbayes.ball_dp.heads.softmax_eqx import SoftmaxLinearEqx, softmax_objective


def load_mnist_numpy(
    root: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch  # noqa: F401
    from torchvision import datasets, transforms

    tfm = transforms.ToTensor()
    tr = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    te = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    def _to_numpy(ds):
        X, y = [], []
        for x, yy in ds:
            X.append(x.numpy())
            y.append(int(yy))
        return np.stack(X, 0).astype(np.float32), np.asarray(y, dtype=np.int64)

    Xtr, ytr = _to_numpy(tr)
    Xte, yte = _to_numpy(te)
    return Xtr, ytr, Xte, yte


def flatten_nchw(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./_out_convex")
    ap.add_argument("--fixed_size", type=int, default=2000)
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--candidate_m", type=int, default=30)

    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--eps", type=float, default=2.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument(
        "--sigma_method", type=str, choices=["classic", "analytic"], default="analytic"
    )

    ap.add_argument("--radius_percentile", type=float, default=50.0)

    # LBFGS for softmax ERM
    ap.add_argument("--softmax_lbfgs_epochs", type=int, default=80)
    ap.add_argument("--softmax_lbfgs_patience", type=int, default=20)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    rng = np.random.default_rng(int(args.seed))

    Xtr, ytr, Xte, yte = load_mnist_numpy(args.data_root)
    Ztr = flatten_nchw(Xtr).astype(np.float32)
    Zte = flatten_nchw(Xte).astype(np.float32)

    # Public bound for pixels in [0,1]: ||x||_2 <= sqrt(784)=28
    B_public = float(np.sqrt(Ztr.shape[1]))
    r_std = 2.0 * B_public

    # compute Ball radius from within-class NN distances (train set)
    policy = compute_radius_policy(
        Ztr,
        ytr,
        percentiles=(10, 25, 50, 75, 90),
        nn_sample_per_class=400,
        seed=int(args.seed),
        B_mode="public",
        B_public=float(B_public),
        check_public_bound=True,
    )
    r_ball = float(policy.r_values[float(args.radius_percentile)])

    print(f"[radius] B_public={policy.B:.4f} -> r_std=2B={float(policy.r_std):.4f}")
    print(f"[radius] r_ball(p={args.radius_percentile})={r_ball:.4f}")

    prior = PoolBallPrior(pool_X=Ztr, pool_y=ytr, radius=r_ball, label_fixed=None)

    def decode_to_img(Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float32)
        return Z.reshape(Z.shape[0], 1, 28, 28)

    rows: List[Dict[str, Any]] = []

    for trial in range(int(args.n_trials)):
        # pick a target from test such that at least one train point lies in the ball
        max_attempts = 200
        z_target = None
        y_target = None
        for _ in range(max_attempts):
            t = int(rng.integers(0, Zte.shape[0]))
            y_try = int(yte[t])
            z_try = Zte[t].astype(np.float64)
            prior.label_fixed = y_try
            if prior.candidate_indices(center=z_try).size > 0:
                z_target = z_try
                y_target = y_try
                break
        if z_target is None:
            raise RuntimeError(
                "Could not find a target with any candidates inside the ball; increase radius_percentile."
            )

        # D^- from train
        fixed_idx = rng.choice(
            np.arange(Ztr.shape[0]), size=int(args.fixed_size), replace=False
        )
        X_minus = Ztr[fixed_idx].astype(np.float64)
        y_minus = ytr[fixed_idx].astype(np.int64)

        dset = InformedDataset(
            X_minus=X_minus,
            y_minus=y_minus,
            x_target=z_target.reshape(-1),
            y_target=int(y_target),
        )
        X_full, y_full = dset.full_dataset()
        n_total = int(X_full.shape[0])

        # candidate set from ball + include true target
        prior.label_fixed = int(y_target)
        candidates = list(
            prior.sample(center=z_target, m=max(1, int(args.candidate_m) - 1), rng=rng)
        )
        candidates.append(
            Candidate(
                record=z_target.copy(),
                label=int(y_target),
                meta={"is_target": True, "id": "TARGET"},
            )
        )
        rng.shuffle(candidates)

        # nearest neighbor baseline (train, same label)
        train_same = np.where(ytr == int(y_target))[0]
        P = Ztr[train_same].astype(np.float64)
        dists = np.linalg.norm(P - z_target.reshape(1, -1), axis=1)
        nn_idx = train_same[int(np.argmin(dists))]
        nn_rec = Ztr[nn_idx].astype(np.float64)

        # ----------------------------
        # Ridge prototypes (closed-form)
        # ----------------------------
        mus, counts = fit_ridge_prototypes(
            Z=np.asarray(X_full, dtype=np.float32),
            y=np.asarray(y_full, dtype=np.int64),
            num_classes=10,
            lam=float(args.lam),
        )

        proto_solver = RidgePrototypesEquationSolver(
            lam=float(args.lam), n_total=n_total
        )
        proto_rec = proto_solver.reconstruct(
            release_mus=mus, d_minus=dset.d_minus, label_known=int(y_target)
        )
        proto_eq_mse = (
            float(np.mean((proto_rec.record_hat - z_target) ** 2))
            if proto_rec.record_hat is not None
            else float("nan")
        )

        dp_proto_std = dp_release_ridge_prototypes_gaussian(
            mus,
            counts,
            r=float(policy.r_std),
            lam=float(args.lam),
            eps=float(args.eps),
            delta=float(args.delta),
            sigma_method=str(args.sigma_method),
            rng=rng,
        )
        dp_proto_ball = dp_release_ridge_prototypes_gaussian(
            mus,
            counts,
            r=float(r_ball),
            lam=float(args.lam),
            eps=float(args.eps),
            delta=float(args.delta),
            sigma_method=str(args.sigma_method),
            rng=rng,
        )

        class ProtoTrainer:
            def fit(self, X, y):
                mu2, _ = fit_ridge_prototypes(
                    Z=np.asarray(X, dtype=np.float32),
                    y=np.asarray(y, dtype=np.int64),
                    num_classes=10,
                    lam=float(args.lam),
                )
                return mu2

        ident_proto_std = GaussianOutputIdentifier(
            trainer=ProtoTrainer(),
            vectorizer=vectorize_prototypes,
            sigma=float(dp_proto_std["sigma"]),
            cache=True,
        )
        out_p_std = ident_proto_std.identify(
            release_params=dp_proto_std["mus_noisy"],
            X_minus=dset.X_minus,
            y_minus=dset.y_minus,
            candidates=candidates,
        )
        cand_p_std = candidates[int(out_p_std.best_idx)]
        p_std_hit = bool(cand_p_std.meta.get("is_target", False))
        p_std_mse = float(np.mean((cand_p_std.record - z_target) ** 2))

        ident_proto_ball = GaussianOutputIdentifier(
            trainer=ProtoTrainer(),
            vectorizer=vectorize_prototypes,
            sigma=float(dp_proto_ball["sigma"]),
            cache=True,
        )
        out_p_ball = ident_proto_ball.identify(
            release_params=dp_proto_ball["mus_noisy"],
            X_minus=dset.X_minus,
            y_minus=dset.y_minus,
            candidates=candidates,
        )
        cand_p_ball = candidates[int(out_p_ball.best_idx)]
        p_ball_hit = bool(cand_p_ball.meta.get("is_target", False))
        p_ball_mse = float(np.mean((cand_p_ball.record - z_target) ** 2))

        rows.append(
            dict(
                trial=trial,
                head="ridge_prototypes",
                y_target=int(y_target),
                n_total=n_total,
                r_ball=float(r_ball),
                r_std=float(policy.r_std),
                mse_eqsolve=float(proto_eq_mse),
                mse_dp_std=float(p_std_mse),
                mse_dp_ball=float(p_ball_mse),
                hit_dp_std=int(p_std_hit),
                hit_dp_ball=int(p_ball_hit),
                sigma_std=float(dp_proto_std["sigma"]),
                sigma_ball=float(dp_proto_ball["sigma"]),
            )
        )

        # ----------------------------
        # Softmax linear (convex ERM) — train with LBFGS
        # ----------------------------
        d_in = int(X_full.shape[1])
        lam = float(args.lam)

        def make_softmax(key):
            return SoftmaxLinearEqx(d_in=d_in, n_classes=10, key=key)

        def loss_fn(m, state, xb, yb, key):
            return softmax_objective(m, state, xb, yb, key, lam=lam)

        trainer_sm = EqxFullBatchLBFGSTrainer(
            make_model=make_softmax,
            loss_fn=loss_fn,
            cfg=FullBatchLBFGSConfig(
                num_epochs=int(args.softmax_lbfgs_epochs),
                patience=int(args.softmax_lbfgs_patience),
                batch_size_full=True,
                seed=int(args.seed),
            ),
        )

        sm_model = trainer_sm.fit(
            np.asarray(X_full, dtype=np.float32), np.asarray(y_full, dtype=np.int64)
        )

        # equation solve
        W, b = softmax_params_numpy(sm_model)
        sm_solver = SoftmaxEquationSolver(
            lam=lam, n_total=n_total, include_bias=True, batch_size=8192
        )
        sm_rec = sm_solver.reconstruct(W=W, b=b, d_minus=dset.d_minus)

        sm_status = str(sm_rec.status)
        sm_rank1_resid = (
            float(sm_rec.details.get("rank1_resid", np.nan))
            if sm_rec.details
            else float("nan")
        )
        sm_eq_mse = (
            float(np.mean((sm_rec.record_hat - z_target) ** 2))
            if sm_rec.record_hat is not None
            else float("nan")
        )

        # DP releases (std vs ball)
        Lz = float(
            lz_softmax_linear_bound(
                B=float(B_public), lam=float(lam), include_bias=True
            )
        )

        sm_dp_std = dp_release_eqx_model_gaussian(
            sm_model,
            lz=Lz,
            r=float(policy.r_std),
            lam=lam,
            n=n_total,
            eps=float(args.eps),
            delta=float(args.delta),
            sigma_method=str(args.sigma_method),
            rng=rng,
        )
        sm_dp_ball = dp_release_eqx_model_gaussian(
            sm_model,
            lz=Lz,
            r=float(r_ball),
            lam=lam,
            n=n_total,
            eps=float(args.eps),
            delta=float(args.delta),
            sigma_method=str(args.sigma_method),
            rng=rng,
        )

        ident_sm_std = GaussianOutputIdentifier(
            trainer=trainer_sm,
            vectorizer=lambda mdl: vectorize_eqx_model(mdl, include_state=False),
            sigma=float(sm_dp_std["dp_report"]["sigma"]),
            cache=True,
        )
        out_s_std = ident_sm_std.identify(
            release_params=sm_dp_std["model_private"],
            X_minus=dset.X_minus,
            y_minus=dset.y_minus,
            candidates=candidates,
        )
        cand_s_std = candidates[int(out_s_std.best_idx)]
        s_std_hit = bool(cand_s_std.meta.get("is_target", False))
        s_std_mse = float(np.mean((cand_s_std.record - z_target) ** 2))

        ident_sm_ball = GaussianOutputIdentifier(
            trainer=trainer_sm,
            vectorizer=lambda mdl: vectorize_eqx_model(mdl, include_state=False),
            sigma=float(sm_dp_ball["dp_report"]["sigma"]),
            cache=True,
        )
        out_s_ball = ident_sm_ball.identify(
            release_params=sm_dp_ball["model_private"],
            X_minus=dset.X_minus,
            y_minus=dset.y_minus,
            candidates=candidates,
        )
        cand_s_ball = candidates[int(out_s_ball.best_idx)]
        s_ball_hit = bool(cand_s_ball.meta.get("is_target", False))
        s_ball_mse = float(np.mean((cand_s_ball.record - z_target) ** 2))

        rows.append(
            dict(
                trial=trial,
                head="softmax_linear",
                y_target=int(y_target),
                n_total=n_total,
                r_ball=float(r_ball),
                r_std=float(policy.r_std),
                Lz=float(Lz),
                softmax_eq_status=sm_status,
                softmax_rank1_resid=float(sm_rank1_resid),
                mse_eqsolve=float(sm_eq_mse),
                mse_dp_std=float(s_std_mse),
                mse_dp_ball=float(s_ball_mse),
                hit_dp_std=int(s_std_hit),
                hit_dp_ball=int(s_ball_hit),
                sigma_std=float(sm_dp_std["dp_report"]["sigma"]),
                sigma_ball=float(sm_dp_ball["dp_report"]["sigma"]),
            )
        )

        # ----------------------------
        # Visualization
        # ----------------------------
        tgt_img = decode_to_img(z_target.reshape(1, -1))[0]
        nn_img = decode_to_img(nn_rec.reshape(1, -1))[0]

        proto_eq_img = (
            decode_to_img(proto_rec.record_hat.reshape(1, -1))[0]
            if proto_rec.record_hat is not None
            else nn_img
        )
        proto_std_img = decode_to_img(np.asarray(cand_p_std.record).reshape(1, -1))[0]
        proto_ball_img = decode_to_img(np.asarray(cand_p_ball.record).reshape(1, -1))[0]

        sm_eq_img = (
            decode_to_img(sm_rec.record_hat.reshape(1, -1))[0]
            if sm_rec.record_hat is not None
            else nn_img
        )
        sm_std_img = decode_to_img(np.asarray(cand_s_std.record).reshape(1, -1))[0]
        sm_ball_img = decode_to_img(np.asarray(cand_s_ball.record).reshape(1, -1))[0]

        def _hitmark(x: bool) -> str:
            return "✓" if x else "✗"

        titles = [
            "target",
            "NN (train,same y)",
            "proto eq-solve",
            f"proto DP std {_hitmark(p_std_hit)}",
            f"proto Ball-DP {_hitmark(p_ball_hit)}",
            f"softmax eq-solve ({sm_status})",
            f"softmax DP std {_hitmark(s_std_hit)}",
            f"softmax Ball-DP {_hitmark(s_ball_hit)}",
        ]
        imgs = [
            tgt_img,
            nn_img,
            proto_eq_img,
            proto_std_img,
            proto_ball_img,
            sm_eq_img,
            sm_std_img,
            sm_ball_img,
        ]

        plot_image_grid(
            images=imgs,
            titles=titles,
            nrows=2,
            ncols=4,
            save_path=Path(out_dir) / f"trial_{trial:03d}_grid.png",
            show=False,
        )

        print(f"[trial {trial}] done (softmax rank1_resid={sm_rank1_resid:.4g})")

    save_csv(Path(out_dir) / "results.csv", rows)
    print(f"[done] wrote {Path(out_dir)/'results.csv'}")


if __name__ == "__main__":
    main()
