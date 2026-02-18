# quantbayes/ball_dp/reconstruction/experiments/mnist/convex_softmax_embed_ball_dp.py
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
    plot_curve,
)
from quantbayes.ball_dp.reconstruction.experiments.mnist_experiments.common import (
    set_global_seed,
    load_mnist_numpy,
    get_or_train_ae,
    encode_numpy,
    decode_numpy,
    get_device,
    train_softmax_lbfgs_torch,
    softmax_precompute,
    softmax_pseudo_missing_gradient,
    softmax_stationarity_norm,
    softmax_accuracy,
    calibrate_softmax_output_perturbation,
    add_noise_softmax,
    l2_clip_rows,
    mnist_img_mse_psnr,
)

if __package__ is None or __package__ == "":
    import sys, pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[5]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)

    # AE
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--ae-epochs", type=int, default=10)
    p.add_argument("--ae-batch-size", type=int, default=256)
    p.add_argument("--ae-lr", type=float, default=1e-3)
    p.add_argument(
        "--B-embed",
        type=float,
        default=0.0,
        help="If >0, clip embeddings to this public bound; else use max-norm (MNIST public).",
    )

    # Convex softmax
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--lbfgs-max-iter", type=int, default=800)
    p.add_argument("--lbfgs-tol-grad", type=float, default=1e-11)

    # Ball-DP policy
    p.add_argument("--epsilon", type=float, default=3.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Ball-DP policy radius r in embedding L2",
    )

    p.add_argument("--n-targets", type=int, default=32)
    p.add_argument("--n-show", type=int, default=8)

    p.add_argument(
        "--save-dir", type=str, default="./artifacts/mnist/convex_softmax_embed_ball_dp"
    )
    args = p.parse_args()

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    outdir = ensure_dir(args.save_dir)

    Xtr, ytr, Xte, yte = load_mnist_numpy(
        n_train=args.n_train, n_test=args.n_test, seed=args.seed
    )

    # AE: train/load + compute embeddings
    ae_art = get_or_train_ae(
        save_dir=outdir,
        Xtr=Xtr,
        Xte=Xte,
        embed_dim=args.embed_dim,
        ae_epochs=args.ae_epochs,
        ae_batch_size=args.ae_batch_size,
        ae_lr=args.ae_lr,
        seed=args.seed,
        B_embed=None if args.B_embed <= 0 else float(args.B_embed),
    )
    ae = ae_art.ae
    device = get_device()

    Etr = encode_numpy(ae, Xtr, device=device, batch_size=512)
    Ete = encode_numpy(ae, Xte, device=device, batch_size=512)

    B_embed = float(ae_art.B_embed)
    Etr = l2_clip_rows(Etr, B_embed)
    Ete = l2_clip_rows(Ete, B_embed)

    # sanity: AE recon quality on a sample
    X_ae = decode_numpy(ae, Etr[:128], device=device, batch_size=128)
    ae_mse = float(np.mean((X_ae - Xtr[:128]) ** 2))
    print(
        f"[AE] embed_dim={args.embed_dim}  B_embed={B_embed:.4f}  recon_mse(sample)={ae_mse:.6f}"
    )

    # Train convex softmax on embeddings
    W, b, _ = train_softmax_lbfgs_torch(
        X=Etr,
        y=ytr,
        lam=args.lam,
        max_iter=args.lbfgs_max_iter,
        tol_grad=args.lbfgs_tg if hasattr(args, "lbfgs_tg") else args.lbfgs_tol_grad,
    )

    # Ball-DP output perturbation calibration at radius r
    cal = calibrate_softmax_output_perturbation(
        epsilon=args.epsilon,
        delta=args.delta,
        lam=args.lam,
        n=int(Etr.shape[0]),
        B=B_embed,
        r=float(args.radius),
        include_bias=True,
    )
    Wn, bn = add_noise_softmax(W, b, sigma=cal.sigma, rng=rng)

    pre = softmax_precompute(Wn, bn, Etr, ytr)
    stat = softmax_stationarity_norm(pre, args.lam)

    acc_tr = softmax_accuracy(Wn, bn, Etr, ytr)
    acc_te = softmax_accuracy(Wn, bn, Ete, yte)

    print(
        f"[Ball-DP embed] eps={args.epsilon} delta={args.delta}  r={args.radius}  sigma={cal.sigma:.4e}  Δ={cal.delta2:.4e}  Lz={cal.Lz:.4e}"
    )
    print(f"[Ball-DP embed] stationarity_norm(release)={stat:.4e}")
    print(f"[Ball-DP embed] acc_train={acc_tr:.4f}  acc_test={acc_te:.4f}")

    solver = SoftmaxEquationSolver(
        lam=float(args.lam), n_total=int(Etr.shape[0]), include_bias=True
    )

    idx = rng.choice(
        np.arange(Etr.shape[0]), size=min(args.n_targets, Etr.shape[0]), replace=False
    )

    rows = []
    e_l2 = []
    img_mses = []
    img_psnrs = []
    resid = []

    # 4-column viz: orig | AE(orig) | decoded(recon) | NN (D^-)
    viz_imgs = []
    viz_titles = []

    for t, j in enumerate(idx):
        j = int(j)
        Gm = softmax_pseudo_missing_gradient(pre, target_idx=j, lam=args.lam)
        res = solver.factorize_missing_gradient(Gm)

        e_hat = np.asarray(res.record_hat, dtype=np.float64).reshape(-1)
        e_true = Etr[j].reshape(-1)
        eerr = float(np.linalg.norm(e_hat - e_true))
        e_l2.append(eerr)

        # decode recon embedding
        X_dec = decode_numpy(ae, e_hat[None, :], device=device, batch_size=1)[
            0, 0
        ]  # (28,28)
        X_true = Xtr[j, 0]
        m, p = mnist_img_mse_psnr(X_dec, X_true)
        img_mses.append(m)
        img_psnrs.append(p)

        resid.append(float(res.details.get("rank1_resid", np.nan)))

        # NN baseline in D^- (exclude target itself)
        mask = np.ones((Etr.shape[0],), dtype=bool)
        mask[j] = False
        pool = Etr[mask]
        idx_map = np.where(mask)[0]
        nn_pool = int(np.argmin(np.linalg.norm(pool - e_hat[None, :], axis=1)))
        nn_idx = int(idx_map[nn_pool])
        X_nn = Xtr[nn_idx, 0]

        # AE(orig) baseline
        X_ae1 = decode_numpy(ae, e_true[None, :], device=device, batch_size=1)[0, 0]

        rows.append(
            {
                "target_idx": j,
                "y_true": int(ytr[j]),
                "y_hat": int(res.label_hat),
                "embed_l2_err": eerr,
                "img_mse": m,
                "img_psnr": p,
                "rank1_resid": float(res.details.get("rank1_resid", np.nan)),
                "nn_idx": nn_idx,
                "nn_y": int(ytr[nn_idx]),
            }
        )

        if t < args.n_show:
            viz_imgs.extend([X_true, X_ae1, X_dec, X_nn])
            viz_titles.extend(
                [
                    f"orig y={int(ytr[j])}",
                    "AE(orig)",
                    f"dec(recon) mse={m:.2e}",
                    f"NN y={int(ytr[nn_idx])}",
                ]
            )

    save_csv(outdir / "metrics.csv", rows)
    summary = {
        "setting": "convex_softmax_embed_ball_dp",
        "n_train": int(Etr.shape[0]),
        "embed_dim": int(args.embed_dim),
        "B_embed": float(B_embed),
        "lam": float(args.lam),
        "epsilon": float(args.epsilon),
        "delta": float(args.delta),
        "radius": float(args.radius),
        "Lz": float(cal.Lz),
        "Delta2": float(cal.delta2),
        "sigma": float(cal.sigma),
        "stationarity_norm_release": float(stat),
        "acc_train": float(acc_tr),
        "acc_test": float(acc_te),
        "mean_embed_l2": float(np.mean(e_l2)),
        "mean_img_mse": float(np.mean(img_mses)),
        "mean_img_psnr": float(np.mean(img_psnrs)),
        "mean_rank1_resid": float(np.nanmean(resid)),
        "ae_recon_mse_sample": float(ae_mse),
    }
    save_json(outdir / "summary.json", summary)

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    plot_hist(
        e_l2,
        title="Embedding L2 recon error (Ball-DP)",
        xlabel="||e_hat-e||2",
        save_path=outdir / "hist_embed_l2.png",
    )
    plot_hist(
        img_mses,
        title="Image MSE after decoding (Ball-DP)",
        xlabel="MSE",
        save_path=outdir / "hist_img_mse.png",
    )
    plot_hist(
        resid,
        title="Rank-1 residual (Ball-DP)",
        xlabel="||G-ae^T||/||G||",
        save_path=outdir / "hist_rank1.png",
    )

    # AE curves if trained now
    if ae_art.ae_hist.get("train"):
        plot_curve(
            ae_art.ae_hist["train"],
            title="AE train MSE",
            xlabel="epoch",
            ylabel="MSE",
            save_path=outdir / "ae_train_curve.png",
        )
    if ae_art.ae_hist.get("val"):
        plot_curve(
            ae_art.ae_hist["val"],
            title="AE val MSE",
            xlabel="epoch",
            ylabel="MSE",
            save_path=outdir / "ae_val_curve.png",
        )

    if viz_imgs:
        ncols = 4
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
