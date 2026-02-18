# quantbayes/ball_dp/reconstruction/experiments/mnist/nonconvex_mlp_pixel_dp.py
from __future__ import annotations

import argparse
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
    EqxDPSGDTrainer,
)
from quantbayes.ball_dp.reconstruction.models import MLPClassifierEqx, multiclass_loss
from quantbayes.ball_dp.reconstruction.experiments.mnist_experiments.common import (
    set_global_seed,
    load_mnist_numpy,
    flatten_pixels,
    make_candidates_including_target,
)

from quantbayes.stochax.privacy.dp import DPSGDConfig, rdp_epsilon_for_sgm

if __package__ is None or __package__ == "":
    import sys, pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[5]))


def calibrate_dp_noise_multiplier(
    *,
    target_epsilon: float,
    delta: float,
    q: float,
    steps: int,
    orders=tuple(list(range(2, 65)) + [80, 96, 128, 256]),
    lo: float = 0.2,
    hi: float = 50.0,
    iters: int = 50,
) -> float:
    target_epsilon = float(target_epsilon)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        eps = rdp_epsilon_for_sgm(
            q=float(q),
            sigma=float(mid),
            steps=int(steps),
            delta=float(delta),
            orders=orders,
        )
        if eps <= target_epsilon:
            hi = mid
        else:
            lo = mid
    return float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--target-idx", type=int, default=0)
    p.add_argument("--radius", type=float, default=8.0)
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

    # DP
    p.add_argument("--epsilon", type=float, default=3.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument(
        "--poisson",
        action="store_true",
        help="Use Poisson sampling in DP-SGD accounting/training",
    )
    p.add_argument(
        "--steps-per-epoch",
        type=int,
        default=0,
        help="If >0, used for Poisson sampling steps/epoch",
    )

    p.add_argument(
        "--save-dir", type=str, default="./artifacts/mnist/nonconvex_mlp_pixel_dp"
    )
    args = p.parse_args()

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    outdir = ensure_dir(args.save_dir)

    Xtr, ytr, _, _ = load_mnist_numpy(n_train=args.n_train, n_test=1, seed=args.seed)
    Xpix = flatten_pixels(Xtr)

    j = int(args.target_idx) % Xpix.shape[0]
    yj = int(ytr[j])

    mask = np.ones((Xpix.shape[0],), dtype=bool)
    mask[j] = False
    X_minus = Xpix[mask]
    y_minus = ytr[mask]

    cands = make_candidates_including_target(
        pool_X=Xpix,
        pool_y=ytr,
        target_idx=j,
        radius=float(args.radius),
        m=int(args.m_candidates),
        rng=rng,
        label_preserving=True,
    )

    # trainer defs
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

    # DP calibration (target epsilon)
    N = int(Xpix.shape[0])
    q = float(args.batch_size) / max(1, N)

    steps_per_epoch = (
        int(args.steps_per_epoch)
        if args.steps_per_epoch > 0
        else int(math.ceil(N / max(1, args.batch_size)))
    )
    total_steps = int(args.epochs) * steps_per_epoch

    noise_mult = calibrate_dp_noise_multiplier(
        target_epsilon=float(args.epsilon),
        delta=float(args.delta),
        q=q,
        steps=total_steps,
    )

    dp_cfg = DPSGDConfig(
        clipping_norm=float(args.clip),
        noise_multiplier=float(noise_mult),
        delta=float(args.delta),
        poisson_sampling=bool(args.poisson),
        sampling_rate=q,
        microbatch_size=None,
    )

    trainer = EqxDPSGDTrainer(
        make_model=make_model,
        make_optimizer=make_optimizer,
        loss_fn=loss_fn,
        dp_config=dp_cfg,
        cfg=EqxTrainerConfig(
            batch_size=int(args.batch_size),
            num_epochs=int(args.epochs),
            patience=int(args.patience),
            shuffle=True,
        ),
    )

    release_model, release_state = trainer.fit(Xpix, ytr, seed=int(args.seed) + 123)

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

    summary = {
        "setting": "nonconvex_mlp_pixel_dp",
        "n_train": int(Xpix.shape[0]),
        "target_idx": int(j),
        "target_label": int(yj),
        "radius": float(args.radius),
        "m_candidates": int(len(cands)),
        "shadows_per_candidate": int(args.shadows_per_candidate),
        "epsilon_target": float(args.epsilon),
        "delta": float(args.delta),
        "clip": float(args.clip),
        "sampling_rate_q": float(q),
        "steps_total": int(total_steps),
        "noise_multiplier": float(noise_mult),
        "best_candidate_idx": int(res.best_idx),
        "best_pool_index": int(best_pool),
        "true_candidate_rank": int(true_rank),
        "success_top1": bool(success),
    }
    save_json(outdir / "summary.json", summary)

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    X_true = Xtr[j, 0]
    X_best = Xtr[best_pool, 0] if best_pool >= 0 else np.zeros_like(X_true)

    plot_image_grid(
        [X_true, X_best],
        [f"orig y={yj}", f"best cand y={int(ytr[best_pool])} (idx={best_pool})"],
        nrows=1,
        ncols=2,
        save_path=outdir / "viz.png",
    )

    print(f"\n[Saved] {outdir}/summary.json, viz.png, shadow_cache/")


if __name__ == "__main__":
    import math

    main()
