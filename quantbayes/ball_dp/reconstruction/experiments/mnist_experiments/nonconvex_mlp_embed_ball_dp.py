# quantbayes/ball_dp/reconstruction/experiments/mnist/nonconvex_mlp_embed_ball_dp.py
from __future__ import annotations

import argparse
import math
import numpy as np

import jax.random as jr
import optax
import equinox as eqx

from quantbayes.ball_dp.reconstruction.reporting import (
    ensure_dir,
    save_json,
    plot_image_grid,
)
from quantbayes.ball_dp.reconstruction.vectorizers import vectorize_eqx_model
from quantbayes.ball_dp.reconstruction.nonconvex.shadow_identifier import (
    ShadowModelIdentifier,
)
from quantbayes.ball_dp.reconstruction.nonconvex.trainers_eqx import (
    EqxTrainerConfig,
    EqxBallDPSGDTrainer,
)
from quantbayes.ball_dp.reconstruction.models import MLPClassifierEqx, multiclass_loss
from quantbayes.ball_dp.reconstruction.experiments.mnist_experiments.common import (
    set_global_seed,
    load_mnist_numpy,
    get_or_train_ae,
    get_device,
    encode_numpy,
    decode_numpy,
    l2_clip_rows,
    make_candidates_including_target,
    mnist_img_mse_psnr,
)

from quantbayes.ball_dp.privacy.rdp_wor_gaussian import calibrate_noise_multiplier
from quantbayes.ball_dp.privacy.ball_dpsgd import BallDPSGDConfig

if __package__ is None or __package__ == "":
    import sys, pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[5]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)

    # AE
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--ae-epochs", type=int, default=10)
    p.add_argument("--ae-batch-size", type=int, default=256)
    p.add_argument("--ae-lr", type=float, default=1e-3)
    p.add_argument("--B-embed", type=float, default=0.0)

    # target + candidates in embedding space
    p.add_argument("--target-idx", type=int, default=0)
    p.add_argument("--radius", type=float, default=2.0)
    p.add_argument("--m-candidates", type=int, default=12)
    p.add_argument("--shadows-per-candidate", type=int, default=2)

    # MLP
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)

    # Ball-DP-SGD
    p.add_argument("--epsilon", type=float, default=3.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument(
        "--lz",
        type=float,
        default=1.0,
        help="Assumed Lz for Ball-DP-SGD (we’ll certify later)",
    )
    p.add_argument(
        "--clip",
        type=float,
        default=0.0,
        help="Optional clipping for optimization stability only (0 disables)",
    )

    p.add_argument(
        "--save-dir", type=str, default="./artifacts/mnist/nonconvex_mlp_embed_ball_dp"
    )
    args = p.parse_args()

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    outdir = ensure_dir(args.save_dir)

    Xtr, ytr, Xte, yte = load_mnist_numpy(
        n_train=args.n_train, n_test=2000, seed=args.seed
    )

    # AE (public)
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
    B_embed = float(ae_art.B_embed)
    Etr = l2_clip_rows(Etr, B_embed)

    j = int(args.target_idx) % Etr.shape[0]
    yj = int(ytr[j])

    mask = np.ones((Etr.shape[0],), dtype=bool)
    mask[j] = False
    X_minus = Etr[mask]
    y_minus = ytr[mask]

    cands = make_candidates_including_target(
        pool_X=Etr,
        pool_y=ytr,
        target_idx=j,
        radius=float(args.radius),
        m=int(args.m_candidates),
        rng=rng,
        label_preserving=True,
    )
    print(
        f"[Nonconvex embed Ball-DP] target_idx={j} y={yj} |C|={len(cands)} shadows/cand={args.shadows_per_candidate}"
    )

    # MLP trainer
    def make_model(seed: int):
        key = jr.PRNGKey(int(seed))
        model = MLPClassifierEqx(
            in_dim=Etr.shape[1],
            n_classes=10,
            width=int(args.width),
            depth=int(args.depth),
            key=key,
            activation="gelu",
        )
        return model, None

    def make_optimizer(model):
        opt = optax.adamw(learning_rate=float(args.lr), weight_decay=0.0)
        params = eqx.filter(model, eqx.is_inexact_array)
        opt_state = opt.init(params)
        return opt, opt_state

    def loss_fn(model, state, xb, yb, key):
        return multiclass_loss(
            model, state, xb, yb, key, weight_decay=float(args.weight_decay)
        )

    # Ball-DP-SGD: calibrate noise_multiplier nm to hit target epsilon for WOR accountant.
    N = int(Etr.shape[0])
    q = float(args.batch_size) / max(1, N)
    steps_per_epoch = int(math.ceil(N / max(1, args.batch_size)))
    total_steps = int(args.epochs) * steps_per_epoch

    nm = calibrate_noise_multiplier(
        target_epsilon=float(args.epsilon),
        delta=float(args.delta),
        q=float(q),
        steps=int(total_steps),
    )

    ball_cfg = BallDPSGDConfig(
        radius=float(args.radius),
        lz=float(args.lz),  # assumed for now
        noise_multiplier=float(nm),
        delta=float(args.delta),
        clipping_norm=(float(args.clip) if float(args.clip) > 0 else None),
    )

    trainer = EqxBallDPSGDTrainer(
        make_model=make_model,
        make_optimizer=make_optimizer,
        loss_fn=loss_fn,
        ball_dp_config=ball_cfg,
        cfg=EqxTrainerConfig(
            batch_size=int(args.batch_size),
            num_epochs=int(args.epochs),
            patience=int(args.patience),
            shuffle=True,
        ),
    )

    # released model on full D
    release_model, release_state = trainer.fit(Etr, ytr, seed=int(args.seed) + 123)

    ident = ShadowModelIdentifier(
        trainer=trainer,
        vectorizer=lambda params, state=None: vectorize_eqx_model(
            params, state=state, include_state=False
        ),
        shadows_per_candidate=int(args.shadows_per_candidate),
        score_mode="nearest_mean",
        cache_dir=str(outdir / "shadow_cache"),
    )

    res = ident.identify(
        release_params=release_model,
        release_state=release_state,
        X_minus=X_minus,
        y_minus=y_minus,
        candidates=cands,
        rng=rng,
    )

    true_rank = int(np.where(res.ranked_idx == 0)[0][0])
    success = res.best_idx == 0
    best = cands[res.best_idx]
    best_pool = int(best.meta.get("pool_index", -1))

    # decode original + decoded(best embedding) for visualization
    X_true = Xtr[j, 0]
    e_best = np.asarray(best.record, dtype=np.float64).reshape(1, -1)
    X_best_dec = decode_numpy(ae, e_best, device=device, batch_size=1)[0, 0]
    m, p = mnist_img_mse_psnr(X_best_dec, X_true)

    summary = {
        "setting": "nonconvex_mlp_embed_ball_dp",
        "n_train": int(Etr.shape[0]),
        "embed_dim": int(args.embed_dim),
        "B_embed": float(B_embed),
        "target_idx": int(j),
        "target_label": int(yj),
        "radius": float(args.radius),
        "m_candidates": int(len(cands)),
        "shadows_per_candidate": int(args.shadows_per_candidate),
        "epsilon_target": float(args.epsilon),
        "delta": float(args.delta),
        "sampling_rate_q": float(q),
        "steps_total": int(total_steps),
        "noise_multiplier_nm": float(nm),
        "lz_assumed": float(args.lz),
        "best_candidate_idx": int(res.best_idx),
        "best_pool_index": int(best_pool),
        "true_candidate_rank": int(true_rank),
        "success_top1": bool(success),
        "decoded_best_mse_vs_true": float(m),
        "decoded_best_psnr_vs_true": float(p),
    }
    save_json(outdir / "summary.json", summary)

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Viz: orig | decoded(best) | best-candidate true image (from pool)
    X_best_img = Xtr[best_pool, 0] if best_pool >= 0 else np.zeros_like(X_true)
    plot_image_grid(
        [X_true, X_best_dec, X_best_img],
        [
            f"orig y={yj}",
            f"decoded(best) mse={m:.2e}",
            f"best cand img y={int(ytr[best_pool])}",
        ],
        nrows=1,
        ncols=3,
        save_path=outdir / "viz.png",
    )

    print(f"\n[Saved] {outdir}/summary.json, viz.png, shadow_cache/")


if __name__ == "__main__":
    main()
