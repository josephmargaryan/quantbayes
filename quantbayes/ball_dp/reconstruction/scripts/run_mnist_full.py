# quantbayes/ball_dp/reconstruction/scripts/run_mnist_full.py
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import jax
import jax.random as jr

from quantbayes.ball_dp.api import (
    compute_radius_policy,
    eqx_flatten_adapter,
    dp_release_ridge_prototypes_gaussian,
)
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
from quantbayes.ball_dp.reconstruction.convex_prototypes import (
    class_sums_counts_np,
    reconstruct_missing_from_prototypes_given_label,
)

from quantbayes.ball_dp.reconstruction.vision.datasets import (
    load_mnist_numpy,
    make_mnist_paired_loader_for_resnet,
    extract_embeddings_and_targets,
)
from quantbayes.ball_dp.reconstruction.vision.preprocess import (
    flatten_chw,
    unflatten_mnist,
    pixel_l2_bound_unit_box,
    clip01,
)
from quantbayes.ball_dp.reconstruction.vision.plotting import save_recon_grid
from quantbayes.ball_dp.reconstruction.vision.encoders_torch import (
    build_resnet18_embedder,
)
from quantbayes.ball_dp.reconstruction.vision.train_decoder_eqx import (
    train_decoder_mlp,
    decode_images_from_embeddings,
)


def _print(s: str):
    print(s, flush=True)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean((a - b) ** 2))


def _mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(x.mean()), float(x.std() + 1e-12)


def _grid4_save(
    *,
    orig: np.ndarray,
    noiseless: np.ndarray,
    dp_ball: np.ndarray,
    dp_bounded: np.ndarray,
    out_path: Path,
    n: int,
    title: str,
):
    """
    Save 4-column grid using save_recon_grid twice (2 cols + 2 cols) and stitch vertically.
    Keeps dependencies minimal.
    """
    import tempfile
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a.png"
        p2 = Path(td) / "b.png"
        save_recon_grid(
            orig, noiseless, p1, n=n, title=title + " | target vs noiseless"
        )
        save_recon_grid(
            dp_ball, dp_bounded, p2, n=n, title=title + " | DP ball vs DP bounded"
        )

        im1 = Image.open(p1)
        im2 = Image.open(p2)

        w = max(im1.size[0], im2.size[0])
        h = im1.size[1] + im2.size[1]
        canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
        canvas.paste(im1.convert("RGB"), (0, 0))
        canvas.paste(im2.convert("RGB"), (0, im1.size[1]))
        canvas.save(out_path)


def _prepare_pixel(args):
    Xchw, y = load_mnist_numpy(
        root=args.data_root, train=True, download=True, flatten=False
    )
    X = flatten_chw(Xchw)  # (N,784)
    K = int(np.max(y) + 1)
    B_public = pixel_l2_bound_unit_box(784, 1.0)  # 28
    return dict(
        mode="pixel",
        X=X,
        X_img=Xchw,
        y=y,
        K=K,
        B_public=B_public,
        img_shape=(1, 28, 28),
    )


def _prepare_embed(args):
    _print("[MNIST/embed] Building ResNet18 encoder...")
    enc = build_resnet18_embedder(device=args.torch_device)

    _print("[MNIST/embed] Building paired loader...")
    loader = make_mnist_paired_loader_for_resnet(
        root=args.data_root,
        train=True,
        batch_size=args.embed_batch_size,
        num_workers=args.num_workers,
        download=True,
        shuffle=False,
    )

    _print("[MNIST/embed] Extracting embeddings (can take time on full MNIST)...")
    E, X_img, y = extract_embeddings_and_targets(
        loader=loader,
        encoder=enc,
        device=args.torch_device,
        l2_normalize_embeddings=bool(args.embed_l2_normalize),
        max_batches=args.embed_max_batches if args.embed_max_batches > 0 else None,
    )
    K = int(np.max(y) + 1)

    if not args.embed_l2_normalize:
        _print(
            "[WARN] embed_l2_normalize is OFF. Baseline r_std=2B is not policy-clean unless you enforce a public bound."
        )
        B_public = float(np.max(np.linalg.norm(E, axis=1)))
    else:
        B_public = 1.0

    _print(f"[MNIST/embed] E={E.shape}, B_public={B_public:.3f}")
    return dict(
        mode="embed",
        X=E,
        X_img=X_img,
        y=y,
        K=K,
        B_public=B_public,
        img_shape=(1, 28, 28),
    )


def _train_decoder_if_needed(
    args,
    E: np.ndarray,
    X_img: np.ndarray,
    out_dir: Path,
    img_shape: Tuple[int, int, int],
):
    if not args.train_decoder:
        return None
    C, H, W = img_shape
    X_flat = X_img.reshape(X_img.shape[0], -1).astype(np.float32)

    if args.decoder_train_max > 0 and X_flat.shape[0] > args.decoder_train_max:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(
            X_flat.shape[0], size=int(args.decoder_train_max), replace=False
        )
        Etr, Xtr = E[idx], X_flat[idx]
    else:
        Etr, Xtr = E, X_flat

    _print(f"[decoder] Training decoder on {Etr.shape[0]} pairs...")
    out = train_decoder_mlp(
        E=Etr,
        X_target_flat=Xtr,
        hidden=tuple(int(x) for x in args.decoder_hidden.split(",") if x.strip()),
        act=args.decoder_act,
        out_act="sigmoid",
        lr=args.decoder_lr,
        epochs=args.decoder_epochs,
        batch_size=args.decoder_batch_size,
        seed=args.seed,
        val_frac=args.decoder_val_frac,
    )
    write_json(
        out_dir / "decoder_train_report.json",
        {
            "decoder_hidden": args.decoder_hidden,
            "decoder_epochs": args.decoder_epochs,
            "decoder_batch_size": args.decoder_batch_size,
            "decoder_lr": args.decoder_lr,
            "train_last": float(out["train_hist"][-1]) if out["train_hist"] else None,
            "val_last": float(out["val_hist"][-1]) if out["val_hist"] else None,
        },
    )
    _print("[decoder] done.")
    return out["decoder"]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="mnist_recon_out")

    ap.add_argument("--mode", type=str, default="pixel", choices=["pixel", "embed"])
    ap.add_argument(
        "--head", type=str, default="prototypes", choices=["prototypes", "softmax"]
    )

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--sigma_method", type=str, default="analytic", choices=["classic", "analytic"]
    )
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--lam", type=float, default=0.2)

    ap.add_argument("--eps_list", type=str, default="0.1,0.2,0.5,1,2,5")
    ap.add_argument("--ball_percentiles", type=str, default="10,50,90")

    ap.add_argument("--n_fixed_per_class", type=int, default=200)
    ap.add_argument("--n_shadow_per_class", type=int, default=200)
    ap.add_argument("--n_target_per_class", type=int, default=30)
    ap.add_argument("--max_targets", type=int, default=200)

    # solver knobs (softmax)
    ap.add_argument("--max_iters", type=int, default=600)
    ap.add_argument("--grad_tol", type=float, default=1e-7)
    ap.add_argument("--stall_patience", type=int, default=50)
    ap.add_argument("--stall_grad_tol", type=float, default=1e-6)
    ap.add_argument("--memory_size", type=int, default=20)

    # progress
    ap.add_argument("--progress_every", type=int, default=5)
    ap.add_argument("--fit_verbose_first", action="store_true")
    ap.add_argument("--fit_print_every", type=int, default=25)

    # metrics
    ap.add_argument(
        "--metric_clip",
        action="store_true",
        help="Compute metrics after clipping recon to [0,1] in pixel mode.",
    )
    ap.set_defaults(metric_clip=True)

    # examples
    ap.add_argument("--save_examples", action="store_true")
    ap.add_argument("--examples_eps", type=float, default=1.0)
    ap.add_argument("--examples_percentile", type=float, default=50.0)
    ap.add_argument("--examples_n", type=int, default=16)

    # embed options
    ap.add_argument("--torch_device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--embed_batch_size", type=int, default=256)
    ap.add_argument("--embed_max_batches", type=int, default=0)
    ap.add_argument("--embed_l2_normalize", action="store_true")

    # decoder
    ap.add_argument("--train_decoder", action="store_true")
    ap.add_argument("--decoder_hidden", type=str, default="1024,1024")
    ap.add_argument("--decoder_act", type=str, default="relu")
    ap.add_argument("--decoder_epochs", type=int, default=20)
    ap.add_argument("--decoder_batch_size", type=int, default=256)
    ap.add_argument("--decoder_lr", type=float, default=1e-3)
    ap.add_argument("--decoder_val_frac", type=float, default=0.1)
    ap.add_argument("--decoder_train_max", type=int, default=20000)

    args = ap.parse_args()
    out_dir = Path(ensure_dir(args.out_dir))

    eps_list = [float(x.strip()) for x in args.eps_list.split(",") if x.strip()]
    ball_ps = [float(x.strip()) for x in args.ball_percentiles.split(",") if x.strip()]

    np.random.seed(args.seed)

    data = _prepare_pixel(args) if args.mode == "pixel" else _prepare_embed(args)
    X_all = data["X"]
    X_img_all = data["X_img"]
    y_all = data["y"]
    K = data["K"]
    B_public = float(data["B_public"])
    img_shape = data["img_shape"]

    split = make_informed_split(
        y_all,
        n_fixed_per_class=int(args.n_fixed_per_class),
        n_shadow_per_class=int(args.n_shadow_per_class),
        n_target_per_class=int(args.n_target_per_class),
        num_classes=K,
        seed=int(args.seed),
    )

    X_fix, y_fix = X_all[split.fixed_idx], y_all[split.fixed_idx]
    X_sh, y_sh = X_all[split.shadow_idx], y_all[split.shadow_idx]
    X_tg, y_tg = X_all[split.target_idx], y_all[split.target_idx]
    X_img_tg = X_img_all[split.target_idx]

    if X_tg.shape[0] > int(args.max_targets):
        X_tg = X_tg[: int(args.max_targets)]
        y_tg = y_tg[: int(args.max_targets)]
        X_img_tg = X_img_tg[: int(args.max_targets)]

    _print(
        f"[MNIST/{args.mode}/{args.head}] fixed={X_fix.shape[0]} shadow={X_sh.shape[0]} targets={X_tg.shape[0]}"
    )

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
    radius_items = [("bounded_rstd", r_std)] + [
        (f"ball_p{int(p)}", r_ball[p]) for p in ball_ps
    ]

    decoder = None
    if args.mode == "embed" and args.train_decoder:
        decoder = _train_decoder_if_needed(args, X_all, X_img_all, out_dir, img_shape)

    # ---------- caches ----------
    n_full = int(X_fix.shape[0] + 1)

    theta_cache: List[Tuple[np.ndarray, callable]] = []
    proto_cache: List[Tuple[np.ndarray, np.ndarray]] = []

    if args.head == "softmax":
        Lz = float(
            lz_softmax_linear_bound(
                B=float(B_public), lam=float(args.lam), include_bias=True
            )
        )
        _print(f"[softmax] Lz={Lz:.4f} B={B_public:.4f} n_full={n_full}")

        _print(f"[MNIST] Fitting θ* for {X_tg.shape[0]} targets...")
        for i in range(X_tg.shape[0]):
            t0 = time.time()
            x = X_tg[i]
            yc = int(y_tg[i])
            X_full = np.concatenate([X_fix, x[None, :]], axis=0)
            y_full = np.concatenate([y_fix, np.array([yc], dtype=np.int64)], axis=0)

            verbose = bool(args.fit_verbose_first and i == 0)
            warmup = bool(i == 0)

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
                memory_size=int(args.memory_size),
                verbose=verbose,
                print_every=int(args.fit_print_every),
                warmup_compile=warmup,
            )
            _ = jax.block_until_ready(mdl.W)
            theta_hat, unflatten = eqx_flatten_adapter(mdl)
            theta_cache.append((theta_hat.astype(np.float32, copy=False), unflatten))

            dt = time.time() - t0
            if (i + 1) % int(args.progress_every) == 0 or (i + 1) == X_tg.shape[0]:
                _print(
                    f"  fitted {i+1}/{X_tg.shape[0]} y={yc} stop={info.get('stop_reason','?')} iters={info.get('n_iters','?')} dt={dt:.2f}s"
                )

    else:
        sums_fix, counts_fix = class_sums_counts_np(X_fix, y_fix, num_classes=K)
        _print(f"[prototypes] caching mus for {X_tg.shape[0]} targets...")
        for i in range(X_tg.shape[0]):
            x = X_tg[i].astype(np.float64)
            yc = int(y_tg[i])

            sums = sums_fix.astype(np.float64).copy()
            counts = counts_fix.astype(np.int64).copy()
            sums[yc] += x
            counts[yc] += 1

            mus = np.zeros((K, X_fix.shape[1]), dtype=np.float64)
            for c in range(K):
                den = 2.0 * float(counts[c]) + float(args.lam) * float(n_full)
                if den > 0:
                    mus[c] = (2.0 * sums[c]) / den

            proto_cache.append((mus.astype(np.float32), counts.astype(np.int64)))

            if (i + 1) % int(args.progress_every) == 0 or (i + 1) == X_tg.shape[0]:
                _print(f"  cached {i+1}/{X_tg.shape[0]}")

    # ---------- evaluation ----------
    curves_l2: Dict[str, Tuple[List[float], List[float]]] = {}
    curves_mse: Dict[str, Tuple[List[float], List[float]]] = {}
    curves_img_mse: Dict[str, Tuple[List[float], List[float]]] = {}

    def eval_softmax(r: float, eps: float):
        rng = np.random.default_rng(int(args.seed) + 999)
        errs_l2, errs_mse = [], []
        errs_img = [] if decoder is not None else None

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

        for i in range(X_tg.shape[0]):
            theta_hat, unflatten = theta_cache[i]
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

            if args.mode == "pixel" and args.metric_clip:
                x_hat_eval = np.clip(x_hat, 0.0, 1.0)
            else:
                x_hat_eval = x_hat

            errs_l2.append(float(np.linalg.norm(x_hat_eval - x_true)))
            errs_mse.append(_mse(x_hat_eval, x_true))

            if errs_img is not None:
                img_hat = decode_images_from_embeddings(
                    decoder=decoder, E=x_hat[None, :], out_shape=img_shape
                )[0]
                img_true = X_img_tg[i]
                errs_img.append(_mse(img_hat, img_true))

        return (
            np.asarray(errs_l2, np.float32),
            np.asarray(errs_mse, np.float32),
            (np.asarray(errs_img, np.float32) if errs_img is not None else None),
        )

    def eval_prototypes(r: float, eps: float):
        errs_l2, errs_mse = [], []
        errs_img = [] if decoder is not None else None

        sums_minus, counts_minus = class_sums_counts_np(X_fix, y_fix, num_classes=K)

        for i in range(X_tg.shape[0]):
            mus, counts_full = proto_cache[i]
            dp = dp_release_ridge_prototypes_gaussian(
                mus=mus,
                counts=counts_full,
                r=float(r),
                lam=float(args.lam),
                eps=float(eps),
                delta=float(args.delta),
                sigma_method=str(args.sigma_method),
                rng=np.random.default_rng(int(args.seed) + 1234 + i),
            )
            mus_noisy = dp["mus_noisy"]
            yc = int(y_tg[i])

            x_hat = reconstruct_missing_from_prototypes_given_label(
                mu_y_release=mus_noisy[yc],
                sum_y_minus=sums_minus[yc],
                n_y_minus=int(counts_minus[yc]),
                n_total_full=n_full,
                lam=float(args.lam),
            )
            x_true = X_tg[i]

            if args.mode == "pixel" and args.metric_clip:
                x_hat_eval = np.clip(x_hat, 0.0, 1.0)
            else:
                x_hat_eval = x_hat

            errs_l2.append(float(np.linalg.norm(x_hat_eval - x_true)))
            errs_mse.append(_mse(x_hat_eval, x_true))

            if errs_img is not None:
                img_hat = decode_images_from_embeddings(
                    decoder=decoder, E=x_hat[None, :], out_shape=img_shape
                )[0]
                img_true = X_img_tg[i]
                errs_img.append(_mse(img_hat, img_true))

        return (
            np.asarray(errs_l2, np.float32),
            np.asarray(errs_mse, np.float32),
            (np.asarray(errs_img, np.float32) if errs_img is not None else None),
        )

    eval_fn = eval_softmax if args.head == "softmax" else eval_prototypes

    _print("[MNIST] Evaluating curves...")
    for name, r in radius_items:
        m_l2, s_l2 = [], []
        m_mse, s_mse = [], []
        m_img, s_img = [], []

        for eps in eps_list:
            t0 = time.time()
            l2s, mses, imgmses = eval_fn(r, eps)
            dt = time.time() - t0

            mu_l2, sd_l2 = _mean_std(l2s)
            mu_mse, sd_mse = _mean_std(mses)
            m_l2.append(mu_l2)
            s_l2.append(sd_l2)
            m_mse.append(mu_mse)
            s_mse.append(sd_mse)

            if imgmses is not None:
                mu_img, sd_img = _mean_std(imgmses)
                m_img.append(mu_img)
                s_img.append(sd_img)

            _print(
                f"  [{name}] eps={eps:g}  L2={mu_l2:.4f}  MSE={mu_mse:.6f}  dt={dt:.2f}s"
            )

        curves_l2[name] = (m_l2, s_l2)
        curves_mse[name] = (m_mse, s_mse)
        if decoder is not None:
            curves_img_mse[name] = (m_img, s_img)

    save_errorbar_plot(
        x=eps_list,
        curves=curves_l2,
        title=f"MNIST {args.mode}/{args.head}: L2(recon,target) vs ε",
        xlabel="epsilon",
        ylabel="L2 error",
        out_path=out_dir / f"mnist_{args.mode}_{args.head}_l2_vs_eps.png",
        xscale_log=True,
    )
    save_errorbar_plot(
        x=eps_list,
        curves=curves_mse,
        title=f"MNIST {args.mode}/{args.head}: MSE(recon,target) vs ε",
        xlabel="epsilon",
        ylabel="MSE",
        out_path=out_dir / f"mnist_{args.mode}_{args.head}_mse_vs_eps.png",
        xscale_log=True,
    )
    if decoder is not None:
        save_errorbar_plot(
            x=eps_list,
            curves=curves_img_mse,
            title=f"MNIST embed+decoder/{args.head}: image MSE(decoded(recon),target) vs ε",
            xlabel="epsilon",
            ylabel="decoded image MSE",
            out_path=out_dir / f"mnist_embed_{args.head}_decoded_mse_vs_eps.png",
            xscale_log=True,
        )

    report = {
        "dataset": "MNIST",
        "mode": args.mode,
        "head": args.head,
        "lam": float(args.lam),
        "delta": float(args.delta),
        "sigma_method": str(args.sigma_method),
        "metric_clip": bool(args.metric_clip),
        "B_public": float(B_public),
        "n_fixed": int(X_fix.shape[0]),
        "n_shadow": int(X_sh.shape[0]),
        "n_targets": int(X_tg.shape[0]),
        "r_std": float(r_std),
        "r_ball": r_ball,
        "eps_list": eps_list,
        "curves_l2": {k: {"mean": v[0], "std": v[1]} for k, v in curves_l2.items()},
        "curves_mse": {k: {"mean": v[0], "std": v[1]} for k, v in curves_mse.items()},
        "decoder_used": decoder is not None,
    }
    if args.head == "softmax":
        report["Lz_bound"] = float(Lz)
    if decoder is not None:
        report["curves_img_mse"] = {
            k: {"mean": v[0], "std": v[1]} for k, v in curves_img_mse.items()
        }

    write_json(out_dir / f"mnist_{args.mode}_{args.head}_report.json", report)

    # ---------- examples: 4-column compare ----------
    if args.save_examples:
        p = float(args.examples_percentile)
        eps_ex = float(args.examples_eps)
        r_ex = r_ball[p]
        Nshow = min(int(args.examples_n), X_tg.shape[0])

        _print(
            f"[MNIST] Saving example comparisons: p{int(p)} eps={eps_ex:g} N={Nshow}"
        )

        if args.mode == "embed" and decoder is None:
            _print("[MNIST/embed] Skipping examples: no decoder. Use --train_decoder.")
        else:
            # build four sets: target, noiseless, dp_ball, dp_bounded
            orig_imgs = X_img_tg[:Nshow]

            noiseless_imgs = []
            dp_ball_imgs = []
            dp_bounded_imgs = []

            if args.head == "softmax":
                # noiseless: use theta_hat directly (sigma=0)
                for i in range(Nshow):
                    theta_hat, unflatten = theta_cache[i]
                    mdl0 = unflatten(theta_hat)
                    rec0 = reconstruct_missing_softmax_from_release(
                        W_release=np.asarray(mdl0.W),
                        b_release=np.asarray(mdl0.b),
                        X_minus=X_fix,
                        y_minus=y_fix,
                        lam=float(args.lam),
                        n_total_full=n_full,
                    )
                    x0 = rec0["x_hat"]

                    if args.mode == "pixel":
                        x0 = np.clip(x0, 0.0, 1.0)
                        noiseless_imgs.append(unflatten_mnist(x0[None, :])[0])
                    else:
                        img0 = decode_images_from_embeddings(
                            decoder=decoder, E=x0[None, :], out_shape=img_shape
                        )[0]
                        noiseless_imgs.append(img0)

                # DP ball and DP bounded
                def _dp_img_for_r(r_use: float) -> List[np.ndarray]:
                    rng = np.random.default_rng(2026)
                    out = []
                    Delta = float(
                        erm_sensitivity_l2(
                            lz=Lz, r=float(r_use), lam=float(args.lam), n=n_full
                        ).delta_l2
                    )
                    sigma = float(
                        gaussian_sigma(
                            Delta,
                            eps_ex,
                            float(args.delta),
                            method=str(args.sigma_method),
                        )
                    )
                    for i in range(Nshow):
                        theta_hat, unflatten = theta_cache[i]
                        theta_noisy = add_gaussian_noise(
                            theta_hat, sigma, rng=rng
                        ).astype(np.float32)
                        mdl = unflatten(theta_noisy)
                        rec = reconstruct_missing_softmax_from_release(
                            W_release=np.asarray(mdl.W),
                            b_release=np.asarray(mdl.b),
                            X_minus=X_fix,
                            y_minus=y_fix,
                            lam=float(args.lam),
                            n_total_full=n_full,
                        )
                        xh = rec["x_hat"]
                        if args.mode == "pixel":
                            xh = np.clip(xh, 0.0, 1.0)
                            out.append(unflatten_mnist(xh[None, :])[0])
                        else:
                            out.append(
                                decode_images_from_embeddings(
                                    decoder=decoder, E=xh[None, :], out_shape=img_shape
                                )[0]
                            )
                    return out

                dp_ball_imgs = _dp_img_for_r(r_ex)
                dp_bounded_imgs = _dp_img_for_r(r_std)

            else:
                # prototypes noiseless / dp
                sums_minus, counts_minus = class_sums_counts_np(
                    X_fix, y_fix, num_classes=K
                )

                # noiseless recon
                for i in range(Nshow):
                    mus, _ = proto_cache[i]
                    yc = int(y_tg[i])
                    x0 = reconstruct_missing_from_prototypes_given_label(
                        mu_y_release=mus[yc],
                        sum_y_minus=sums_minus[yc],
                        n_y_minus=int(counts_minus[yc]),
                        n_total_full=n_full,
                        lam=float(args.lam),
                    )
                    if args.mode == "pixel":
                        x0 = np.clip(x0, 0.0, 1.0)
                        noiseless_imgs.append(unflatten_mnist(x0[None, :])[0])
                    else:
                        noiseless_imgs.append(
                            decode_images_from_embeddings(
                                decoder=decoder, E=x0[None, :], out_shape=img_shape
                            )[0]
                        )

                def _dp_img_for_r(r_use: float) -> List[np.ndarray]:
                    out = []
                    for i in range(Nshow):
                        mus, counts_full = proto_cache[i]
                        dp = dp_release_ridge_prototypes_gaussian(
                            mus=mus,
                            counts=counts_full,
                            r=float(r_use),
                            lam=float(args.lam),
                            eps=float(eps_ex),
                            delta=float(args.delta),
                            sigma_method=str(args.sigma_method),
                            rng=np.random.default_rng(3000 + i),
                        )
                        mus_noisy = dp["mus_noisy"]
                        yc = int(y_tg[i])
                        xh = reconstruct_missing_from_prototypes_given_label(
                            mu_y_release=mus_noisy[yc],
                            sum_y_minus=sums_minus[yc],
                            n_y_minus=int(counts_minus[yc]),
                            n_total_full=n_full,
                            lam=float(args.lam),
                        )
                        if args.mode == "pixel":
                            xh = np.clip(xh, 0.0, 1.0)
                            out.append(unflatten_mnist(xh[None, :])[0])
                        else:
                            out.append(
                                decode_images_from_embeddings(
                                    decoder=decoder, E=xh[None, :], out_shape=img_shape
                                )[0]
                            )
                    return out

                dp_ball_imgs = _dp_img_for_r(r_ex)
                dp_bounded_imgs = _dp_img_for_r(r_std)

            noiseless_imgs = clip01(np.stack(noiseless_imgs, axis=0).astype(np.float32))
            dp_ball_imgs = clip01(np.stack(dp_ball_imgs, axis=0).astype(np.float32))
            dp_bounded_imgs = clip01(
                np.stack(dp_bounded_imgs, axis=0).astype(np.float32)
            )

            _grid4_save(
                orig=clip01(orig_imgs[:Nshow]),
                noiseless=noiseless_imgs,
                dp_ball=dp_ball_imgs,
                dp_bounded=dp_bounded_imgs,
                out_path=out_dir
                / f"mnist_{args.mode}_{args.head}_examples_eps{eps_ex:g}_p{int(p)}.png",
                n=Nshow,
                title=f"MNIST {args.mode}/{args.head} eps={eps_ex:g} p{int(p)}",
            )

    _print(f"[OK] Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
