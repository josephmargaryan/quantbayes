# quantbayes/ball_dp/reconstruction/scripts/run_cifar10_full.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import jax.random as jr

from quantbayes.ball_dp.api import compute_radius_policy, eqx_flatten_adapter
from quantbayes.ball_dp.lz import lz_softmax_linear_bound
from quantbayes.ball_dp.sensitivity import erm_sensitivity_l2
from quantbayes.ball_dp.mechanisms import gaussian_sigma, add_gaussian_noise
from quantbayes.ball_dp.utils.io import ensure_dir, write_json
from quantbayes.ball_dp.utils.plotting import save_errorbar_plot

from quantbayes.ball_dp.reconstruction.splits import make_informed_split
from quantbayes.ball_dp.reconstruction.convex_softmax import (
    fit_softmax_erm_precise_eqx,
    reconstruct_missing_softmax_from_release,
)

from quantbayes.ball_dp.reconstruction.vision.datasets import load_cifar10_numpy
from quantbayes.ball_dp.reconstruction.vision.preprocess import (
    flatten_chw,
    unflatten_cifar10,
    pixel_l2_bound_unit_box,
    clip01,
)
from quantbayes.ball_dp.reconstruction.vision.plotting import save_recon_grid


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean((a - b) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="cifar10_recon_out")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--sigma_method", type=str, default="analytic", choices=["classic", "analytic"]
    )
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--lam", type=float, default=0.2)

    ap.add_argument("--eps_list", type=str, default="0.2,0.5,1,2,5")
    ap.add_argument("--ball_percentiles", type=str, default="10,50,90")

    # smaller defaults than MNIST due to compute
    ap.add_argument("--n_fixed_per_class", type=int, default=200)
    ap.add_argument("--n_shadow_per_class", type=int, default=200)
    ap.add_argument("--n_target_per_class", type=int, default=20)

    ap.add_argument("--max_targets", type=int, default=200)
    ap.add_argument("--save_examples", action="store_true")
    ap.add_argument("--examples_eps", type=float, default=1.0)
    ap.add_argument("--examples_percentile", type=float, default=50.0)
    ap.add_argument("--examples_n", type=int, default=12)

    # ERM solve knobs
    ap.add_argument("--max_iters", type=int, default=600)
    ap.add_argument("--grad_tol", type=float, default=1e-7)
    ap.add_argument("--stall_patience", type=int, default=50)
    ap.add_argument("--stall_grad_tol", type=float, default=1e-6)

    args = ap.parse_args()
    out_dir = Path(ensure_dir(args.out_dir))

    eps_list = [float(x.strip()) for x in args.eps_list.split(",") if x.strip()]
    ball_ps = [float(x.strip()) for x in args.ball_percentiles.split(",") if x.strip()]

    # Load CIFAR10 train split
    Xchw, y = load_cifar10_numpy(
        root=args.data_root, train=True, download=True, flatten=False
    )
    X = flatten_chw(Xchw)  # (N,3072)
    K = int(np.max(y) + 1)

    # Public bound for pixels in [0,1]^3072
    B_public = pixel_l2_bound_unit_box(3072, 1.0)

    split = make_informed_split(
        y,
        n_fixed_per_class=int(args.n_fixed_per_class),
        n_shadow_per_class=int(args.n_shadow_per_class),
        n_target_per_class=int(args.n_target_per_class),
        num_classes=K,
        seed=int(args.seed),
    )

    X_fix, y_fix = X[split.fixed_idx], y[split.fixed_idx]
    X_sh, y_sh = X[split.shadow_idx], y[split.shadow_idx]
    X_tg, y_tg = X[split.target_idx], y[split.target_idx]

    if X_tg.shape[0] > int(args.max_targets):
        X_tg = X_tg[: int(args.max_targets)]
        y_tg = y_tg[: int(args.max_targets)]

    rp = compute_radius_policy(
        X_fix,
        y_fix,
        percentiles=tuple(ball_ps),
        nn_sample_per_class=min(400, int(args.n_fixed_per_class)),
        seed=int(args.seed),
        B_mode="public",
        B_public=float(B_public),
    )
    r_std = float(rp.r_std)
    r_ball = {p: float(rp.r_values[p]) for p in rp.percentiles}

    Lz = float(
        lz_softmax_linear_bound(
            B=float(B_public), lam=float(args.lam), include_bias=True
        )
    )
    n_full = int(X_fix.shape[0] + 1)

    print(f"[CIFAR10] Fitting θ* for {X_tg.shape[0]} targets...")
    target_cache: List[Tuple[np.ndarray, callable]] = []
    for i in range(X_tg.shape[0]):
        x = X_tg[i]
        yc = int(y_tg[i])
        X_full = np.concatenate([X_fix, x[None, :]], axis=0)
        y_full = np.concatenate([y_fix, np.array([yc], dtype=np.int64)], axis=0)

        mdl, info = fit_softmax_erm_precise_eqx(
            X=X_full,
            y=y_full,
            num_classes=K,
            lam=float(args.lam),
            key=jr.PRNGKey(0),
            max_iters=int(args.max_iters),
            grad_tol=float(args.grad_tol),
            stall_patience=int(args.stall_patience),
            stall_grad_tol=float(args.stall_grad_tol),
            verbose=False,
            warmup_compile=(i == 0),
        )
        theta_hat, unflatten = eqx_flatten_adapter(mdl)
        target_cache.append((theta_hat.astype(np.float32, copy=False), unflatten))
        if (i + 1) % 10 == 0 or (i + 1) == X_tg.shape[0]:
            print(f"  fitted {i+1}/{X_tg.shape[0]}")

    curves_mse: Dict[str, Tuple[List[float], List[float]]] = {}
    curves_l2: Dict[str, Tuple[List[float], List[float]]] = {}

    radius_items = [("bounded_rstd", r_std)] + [
        (f"ball_p{int(p)}", r_ball[p]) for p in ball_ps
    ]

    def eval_one_setting(r: float, eps: float) -> Tuple[np.ndarray, np.ndarray]:
        errs_l2, errs_mse = [], []
        rng = np.random.default_rng(int(args.seed) + 1234)
        for i in range(X_tg.shape[0]):
            theta_hat, unflatten = target_cache[i]
            Delta = float(
                erm_sensitivity_l2(
                    lz=Lz, r=float(r), lam=float(args.lam), n=n_full
                ).delta_l2
            )
            sigma = float(
                gaussian_sigma(
                    Delta, float(eps), float(args.delta), method=str(args.sigma_method)
                )
            )
            theta_noisy = add_gaussian_noise(theta_hat, sigma, rng=rng).astype(
                np.float32
            )

            mdl_priv = unflatten(theta_noisy)
            rec = reconstruct_missing_softmax_from_release(
                W_release=np.asarray(mdl_priv.W),
                b_release=np.asarray(mdl_priv.b),
                X_minus=X_fix,
                y_minus=y_fix,
                lam=float(args.lam),
                n_total_full=n_full,
            )
            x_hat = rec["x_hat"]
            x_true = X_tg[i]
            errs_l2.append(float(np.linalg.norm(x_hat - x_true)))
            errs_mse.append(_mse(x_hat, x_true))
        return np.asarray(errs_l2, dtype=np.float32), np.asarray(
            errs_mse, dtype=np.float32
        )

    for name, r in radius_items:
        mses_mean, mses_std = [], []
        l2_mean, l2_std = [], []
        for eps in eps_list:
            l2s, mses = eval_one_setting(r, eps)
            mses_mean.append(float(mses.mean()))
            mses_std.append(float(mses.std() + 1e-12))
            l2_mean.append(float(l2s.mean()))
            l2_std.append(float(l2s.std() + 1e-12))
            print(
                f"[{name}] eps={eps:g} | MSE={mses_mean[-1]:.6f} | L2={l2_mean[-1]:.4f}"
            )
        curves_mse[name] = (mses_mean, mses_std)
        curves_l2[name] = (l2_mean, l2_std)

    save_errorbar_plot(
        x=eps_list,
        curves=curves_mse,
        title="CIFAR10 pixel-space convex softmax recon: MSE vs epsilon",
        xlabel="epsilon",
        ylabel="MSE(x_hat, x)",
        out_path=out_dir / "cifar10_convex_softmax_mse_vs_eps.png",
        xscale_log=True,
    )
    save_errorbar_plot(
        x=eps_list,
        curves=curves_l2,
        title="CIFAR10 pixel-space convex softmax recon: L2 vs epsilon",
        xlabel="epsilon",
        ylabel="||x_hat - x||_2",
        out_path=out_dir / "cifar10_convex_softmax_l2_vs_eps.png",
        xscale_log=True,
    )

    report = {
        "dataset": "CIFAR10",
        "head": "softmax_convex",
        "lam": float(args.lam),
        "delta": float(args.delta),
        "sigma_method": str(args.sigma_method),
        "B_public": float(B_public),
        "Lz_bound": float(Lz),
        "n_fixed": int(X_fix.shape[0]),
        "n_shadow": int(X_sh.shape[0]),
        "n_targets": int(X_tg.shape[0]),
        "r_std": float(r_std),
        "r_ball": r_ball,
        "eps_list": eps_list,
        "curves_mse": {k: {"mean": v[0], "std": v[1]} for k, v in curves_mse.items()},
        "curves_l2": {k: {"mean": v[0], "std": v[1]} for k, v in curves_l2.items()},
    }
    write_json(out_dir / "cifar10_convex_softmax_report.json", report)

    if args.save_examples:
        p = float(args.examples_percentile)
        r_ex = r_ball[p]
        eps_ex = float(args.examples_eps)
        print(f"[CIFAR10] Saving example grid at p={p} r={r_ex:.4f} eps={eps_ex:g}")

        Nshow = min(int(args.examples_n), X_tg.shape[0])
        orig_imgs = unflatten_cifar10(X_tg[:Nshow])
        recon_flat = []

        rng2 = np.random.default_rng(999)
        for i in range(Nshow):
            theta_hat, unflatten = target_cache[i]
            Delta = float(
                erm_sensitivity_l2(
                    lz=Lz, r=float(r_ex), lam=float(args.lam), n=n_full
                ).delta_l2
            )
            sigma = float(
                gaussian_sigma(
                    Delta, eps_ex, float(args.delta), method=str(args.sigma_method)
                )
            )
            theta_noisy = add_gaussian_noise(theta_hat, sigma, rng=rng2).astype(
                np.float32
            )
            mdl_priv = unflatten(theta_noisy)
            rec = reconstruct_missing_softmax_from_release(
                W_release=np.asarray(mdl_priv.W),
                b_release=np.asarray(mdl_priv.b),
                X_minus=X_fix,
                y_minus=y_fix,
                lam=float(args.lam),
                n_total_full=n_full,
            )
            recon_flat.append(rec["x_hat"])

        recon_flat = np.stack(recon_flat, axis=0).astype(np.float32)
        recon_imgs = clip01(unflatten_cifar10(recon_flat))
        save_recon_grid(
            orig_imgs,
            recon_imgs,
            out_dir / f"cifar10_examples_softmax_eps{eps_ex:g}_p{int(p)}.png",
            n=Nshow,
            title=f"CIFAR10 recon (softmax) eps={eps_ex:g} p{int(p)}",
        )

    print(f"[OK] CIFAR10 outputs in: {out_dir}")


if __name__ == "__main__":
    main()
