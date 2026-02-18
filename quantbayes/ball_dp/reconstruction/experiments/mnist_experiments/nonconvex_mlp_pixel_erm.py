# quantbayes/ball_dp/reconstruction/experiments/mnist/nonconvex_mlp_pixel_erm.py
from __future__ import annotations

import argparse
import math
import numpy as np

import jax.random as jr
import optax
import equinox as eqx
import jax.numpy as jnp

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
    EqxNonPrivateTrainer,
)
from quantbayes.ball_dp.reconstruction.models import MLPClassifierEqx, multiclass_loss
from quantbayes.ball_dp.reconstruction.experiments.mnist_experiments.common import (
    set_global_seed,
    load_mnist_numpy,
    flatten_pixels,
    make_candidates_including_target,
)

if __package__ is None or __package__ == "":
    import sys, pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[5]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--target-idx", type=int, default=0)
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--m-candidates", type=int, default=12)
    p.add_argument("--shadows-per-candidate", type=int, default=3)

    # MLP
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)

    p.add_argument(
        "--save-dir", type=str, default="./artifacts/mnist/nonconvex_mlp_pixel_erm"
    )
    args = p.parse_args()

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    outdir = ensure_dir(args.save_dir)

    Xtr, ytr, _, _ = load_mnist_numpy(n_train=args.n_train, n_test=1, seed=args.seed)
    Xpix = flatten_pixels(Xtr)

    j = int(args.target_idx) % Xpix.shape[0]
    yj = int(ytr[j])

    # D^- for informed adversary
    mask = np.ones((Xpix.shape[0],), dtype=bool)
    mask[j] = False
    X_minus = Xpix[mask]
    y_minus = ytr[mask]

    # Candidate set (includes true target at index 0)
    cands = make_candidates_including_target(
        pool_X=Xpix,
        pool_y=ytr,
        target_idx=j,
        radius=float(args.radius),
        m=int(args.m_candidates),
        rng=rng,
        label_preserving=True,
    )
    print(
        f"[Nonconvex pixel ERM] target_idx={j} y={yj} |C|={len(cands)} shadows/cand={args.shadows_per_candidate}"
    )

    # ---- trainer wrapper using your existing stochax trainer ----
    def make_model(seed: int):
        key = jr.PRNGKey(int(seed))
        model = MLPClassifierEqx(
            in_dim=Xpix.shape[1],
            n_classes=10,
            width=int(args.width),
            depth=int(args.depth),
            key=key,
            activation="gelu",
        )
        state = None
        return model, state

    def make_optimizer(model):
        opt = optax.adamw(learning_rate=float(args.lr), weight_decay=0.0)
        params = eqx.filter(model, eqx.is_inexact_array)
        opt_state = opt.init(params)
        return opt, opt_state

    def loss_fn(model, state, xb, yb, key):
        return multiclass_loss(
            model, state, xb, yb, key, weight_decay=float(args.weight_decay)
        )

    trainer = EqxNonPrivateTrainer(
        make_model=make_model,
        make_optimizer=make_optimizer,
        loss_fn=loss_fn,
        cfg=EqxTrainerConfig(
            batch_size=int(args.batch_size),
            num_epochs=int(args.epochs),
            patience=int(args.patience),
            shuffle=True,
        ),
    )

    # Train the released model on full D (same dataset used in candidates)
    release_model, release_state = trainer.fit(Xpix, ytr, seed=int(args.seed) + 123)

    # Shadow identifier (cache shadows to disk)
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

    best = cands[res.best_idx]
    best_pool = int(best.meta.get("pool_index", -1))
    true_rank = int(
        np.where(res.ranked_idx == 0)[0][0]
    )  # candidate 0 is true by construction
    success = res.best_idx == 0

    summary = {
        "setting": "nonconvex_mlp_pixel_erm",
        "n_train": int(Xpix.shape[0]),
        "target_idx": int(j),
        "target_label": int(yj),
        "radius": float(args.radius),
        "m_candidates": int(len(cands)),
        "shadows_per_candidate": int(args.shadows_per_candidate),
        "best_candidate_idx": int(res.best_idx),
        "best_pool_index": int(best_pool),
        "true_candidate_rank": int(true_rank),
        "success_top1": bool(success),
    }
    save_json(outdir / "summary.json", summary)

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Viz: orig | best-candidate image | (optional) NN style baseline not needed
    X_true = Xtr[j, 0]
    X_best = Xtr[best_pool, 0] if best_pool >= 0 else np.zeros_like(X_true)

    imgs = [X_true, X_best]
    titles = [f"orig y={yj}", f"best cand y={int(ytr[best_pool])} (idx={best_pool})"]
    plot_image_grid(imgs, titles, nrows=1, ncols=2, save_path=outdir / "viz.png")

    print(f"\n[Saved] {outdir}/summary.json, viz.png, shadow_cache/")


if __name__ == "__main__":
    main()
