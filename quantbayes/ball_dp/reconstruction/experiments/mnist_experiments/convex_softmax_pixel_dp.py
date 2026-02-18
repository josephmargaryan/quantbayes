# quantbayes/ball_dp/reconstruction/experiments/mnist/convex_softmax_pixel_dp.py
from __future__ import annotations

import argparse
import math
import numpy as np

from quantbayes.ball_dp.reconstruction.convex.equation_solvers import (
    SoftmaxEquationSolver,
)
from quantbayes.ball_dp.reconstruction.reporting import (
    ensure_dir,
    save_csv,
    save_json,
    plot_hist,
    plot_image_grid,
)
from quantbayes.ball_dp.reconstruction.experiments.mnist_experiments.common import (
    set_global_seed,
    load_mnist_numpy,
    flatten_pixels,
    train_softmax_lbfgs_torch,
    softmax_precompute,
    softmax_pseudo_missing_gradient,
    softmax_stationarity_norm,
    softmax_accuracy,
    mnist_img_mse_psnr,
    calibrate_softmax_output_perturbation,
    add_noise_softmax,
)

if __package__ is None or __package__ == "":
    import sys, pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[5]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--lbfgs-max-iter", type=int, default=800)
    p.add_argument("--lbfgs-tol-grad", type=float, default=1e-11)

    p.add_argument("--epsilon", type=float, default=3.0)
    p.add_argument("--delta", type=float, default=1e-5)

    p.add_argument("--n-targets", type=int, default=32)
    p.add_argument("--n-show", type=int, default=12)

    p.add_argument(
        "--save-dir", type=str, default="./artifacts/mnist/convex_softmax_pixel_dp"
    )
    args = p.parse_args()

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    outdir = ensure_dir(args.save_dir)

    Xtr, ytr, Xte, yte = load_mnist_numpy(
        n_train=args.n_train, n_test=args.n_test, seed=args.seed
    )
    Xpix = flatten_pixels(Xtr)

    # public pixel bound for MNIST in [0,1]^784: ||x||2 <= sqrt(784)=28
    B = 28.0
    r_std = 2.0 * B  # bounded replacement special case

    # Train noiseless ERM
    W, b, _ = train_softmax_lbfgs_torch(
        X=Xpix,
        y=ytr,
        lam=args.lam,
        max_iter=args.lbfgs_max_iter,
        tol_grad=args.lbfgs_tol_grad,
    )

    # DP calibration + noise
    cal = calibrate_softmax_output_perturbation(
        epsilon=args.epsilon,
        delta=args.delta,
        lam=args.lam,
        n=int(Xpix.shape[0]),
        B=B,
        r=r_std,
        include_bias=True,
    )
    Wn, bn = add_noise_softmax(W, b, sigma=cal.sigma, rng=rng)

    pre = softmax_precompute(Wn, bn, Xpix, ytr)
    stat = softmax_stationarity_norm(pre, args.lam)

    acc_tr = softmax_accuracy(Wn, bn, Xpix, ytr)
    acc_te = softmax_accuracy(Wn, bn, flatten_pixels(Xte), yte)

    print(
        f"[DP pixel] eps={args.epsilon} delta={args.delta}  sigma={cal.sigma:.4e}  Δ={cal.delta2:.4e}  Lz={cal.Lz:.4e}"
    )
    print(f"[DP pixel] stationarity_norm(release)={stat:.4e}")
    print(f"[DP pixel] acc_train={acc_tr:.4f}  acc_test={acc_te:.4f}")

    solver = SoftmaxEquationSolver(
        lam=float(args.lam), n_total=int(Xpix.shape[0]), include_bias=True
    )

    idx = rng.choice(
        np.arange(Xpix.shape[0]), size=min(args.n_targets, Xpix.shape[0]), replace=False
    )

    rows = []
    l2errs = []
    mses = []
    psnrs = []
    resid = []

    viz_imgs = []
    viz_titles = []

    for t, j in enumerate(idx):
        j = int(j)
        Gm = softmax_pseudo_missing_gradient(pre, target_idx=j, lam=args.lam)
        res = solver.factorize_missing_gradient(Gm)

        x_hat = res.record_hat.reshape(28, 28)
        x_true = Xtr[j, 0]

        l2 = float(np.linalg.norm(x_hat.reshape(-1) - x_true.reshape(-1)))
        m, p = mnist_img_mse_psnr(x_hat, x_true)

        l2errs.append(l2)
        mses.append(m)
        psnrs.append(p)
        resid.append(float(res.details.get("rank1_resid", np.nan)))

        rows.append(
            {
                "target_idx": j,
                "y_true": int(ytr[j]),
                "y_hat": int(res.label_hat),
                "l2_err": l2,
                "mse": m,
                "psnr": p,
                "rank1_resid": float(res.details.get("rank1_resid", np.nan)),
            }
        )

        if t < args.n_show:
            viz_imgs.extend([x_true, x_hat])
            viz_titles.extend([f"orig y={int(ytr[j])}", f"DP recon mse={m:.2e}"])

    save_csv(outdir / "metrics.csv", rows)
    summary = {
        "setting": "convex_softmax_pixel_dp",
        "n_train": int(Xpix.shape[0]),
        "lam": float(args.lam),
        "epsilon": float(args.epsilon),
        "delta": float(args.delta),
        "B": float(B),
        "r_std": float(r_std),
        "Lz": float(cal.Lz),
        "Delta2": float(cal.delta2),
        "sigma": float(cal.sigma),
        "stationarity_norm_release": float(stat),
        "acc_train": float(acc_tr),
        "acc_test": float(acc_te),
        "mean_l2": float(np.mean(l2errs)),
        "median_l2": float(np.median(l2errs)),
        "mean_mse": float(np.mean(mses)),
        "mean_psnr": float(np.mean(psnrs)),
        "mean_rank1_resid": float(np.nanmean(resid)),
    }
    save_json(outdir / "summary.json", summary)

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    plot_hist(
        l2errs,
        title="Pixel L2 recon error (DP)",
        xlabel="||x_hat-x||2",
        save_path=outdir / "hist_l2.png",
    )
    plot_hist(
        mses, title="Pixel MSE (DP)", xlabel="MSE", save_path=outdir / "hist_mse.png"
    )
    plot_hist(
        resid,
        title="Rank-1 residual (DP)",
        xlabel="||G-ae^T||/||G||",
        save_path=outdir / "hist_rank1.png",
    )

    if viz_imgs:
        ncols = 2
        nrows = int(math.ceil(len(viz_imgs) / ncols))
        plot_image_grid(
            viz_imgs,
            viz_titles,
            nrows=nrows,
            ncols=ncols,
            save_path=outdir / "recon_grid.png",
        )

    print(f"\n[Saved] {outdir}/summary.json, metrics.csv, recon_grid.png, hist_*.png")


if __name__ == "__main__":
    main()
