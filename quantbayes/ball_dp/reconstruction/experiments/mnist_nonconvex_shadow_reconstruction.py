# quantbayes/ball_dp/reconstruction/experiments/mnist_nonconvex_shadow_reconstruction.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

from quantbayes.ball_dp.api import compute_radius_policy
from quantbayes.ball_dp.privacy.rdp_wor_gaussian import calibrate_noise_multiplier
from quantbayes.ball_dp.privacy.dpsgd import DPSGDConfig
from quantbayes.ball_dp.privacy.ball_dpsgd import BallDPSGDConfig

from quantbayes.ball_dp.reconstruction.types import Candidate
from quantbayes.ball_dp.reconstruction.priors import PoolBallPrior
from quantbayes.ball_dp.reconstruction.reporting import (
    ensure_dir,
    save_csv,
    plot_image_grid,
)
from quantbayes.ball_dp.reconstruction.nonconvex.shadow_identifier import (
    ShadowModelIdentifier,
)
from quantbayes.ball_dp.reconstruction.nonconvex.trainers_eqx import (
    EqxTrainerConfig,
    EqxNonPrivateTrainer,
    EqxStandardDPSGDTrainer,
    EqxBallDPSGDTrainer,
)
from quantbayes.ball_dp.reconstruction.vectorizers import vectorize_eqx_model

from quantbayes.ball_dp.reconstruction.models.mlp_eqx import (
    MLPClassifierEqx,
    multiclass_loss,
)


# -------------------------
# MNIST + utils
# -------------------------
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


def clip_l2_rows(Z: np.ndarray, B: float) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float32)
    norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    scale = np.minimum(1.0, float(B) / norms)
    return Z * scale


# -------------------------
# Optional VAE for embedding mode
# -------------------------
def train_or_load_vae_mnist(
    *,
    ckpt_path: Path,
    X_train_nchw: np.ndarray,
    latent_dim: int,
    epochs: int,
    seed: int,
):
    import jax.random as jr
    import equinox as eqx
    import jax.numpy as jnp

    from quantbayes.stochax.vae.components import ConvVAE
    from quantbayes.stochax.vae.train_vae import train_vae, TrainConfig

    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    key = jr.PRNGKey(int(seed))
    model_template = ConvVAE(
        image_size=28,
        channels=1,
        hidden_channels=32,
        latent_dim=int(latent_dim),
        key=key,
    )

    if ckpt_path.exists():
        return eqx.tree_deserialise_leaves(str(ckpt_path), model_template)

    data = jnp.asarray(X_train_nchw.astype(np.float32))
    cfg = TrainConfig(
        epochs=int(epochs),
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=int(seed),
        verbose=True,
        likelihood="bernoulli",
        drop_last=True,
    )
    model = train_vae(model_template, data, cfg)
    eqx.tree_serialise_leaves(str(ckpt_path), model)
    return model


def vae_encode_mu(model, X_nchw: np.ndarray) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    X = jnp.asarray(X_nchw.astype(np.float32))

    @eqx.filter_jit
    def _enc(xb):
        mu, _ = model.encoder(xb, train=False)
        return mu

    mu = _enc(X)
    return np.asarray(jax.device_get(mu), dtype=np.float32)


def vae_decode(model, Z: np.ndarray) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    Zj = jnp.asarray(Z.astype(np.float32))

    @eqx.filter_jit
    def _dec(zb):
        logits = model.decoder(zb, train=False)
        return jax.nn.sigmoid(logits)

    out = _dec(Zj)
    return np.asarray(jax.device_get(out), dtype=np.float32)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./_out_nonconvex")
    ap.add_argument(
        "--mode", type=str, choices=["pixel", "embedding"], default="embedding"
    )

    ap.add_argument("--fixed_size", type=int, default=2000)
    ap.add_argument("--candidate_m", type=int, default=40)
    ap.add_argument("--n_trials", type=int, default=5)

    # training
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)

    # privacy targets
    ap.add_argument("--target_eps", type=float, default=8.0)
    ap.add_argument("--delta", type=float, default=1e-5)

    # DP-SGD baseline
    ap.add_argument("--clip_C", type=float, default=1.0)

    # Ball-DP-SGD
    ap.add_argument(
        "--ball_lz", type=float, default=5.0
    )  # assumed/certified Lz for nonconvex
    ap.add_argument("--radius_percentile", type=float, default=50.0)

    # VAE (embedding mode)
    ap.add_argument("--vae_latent_dim", type=int, default=16)
    ap.add_argument("--vae_epochs", type=int, default=10)
    ap.add_argument("--vae_seed", type=int, default=0)
    ap.add_argument("--embed_B_public", type=float, default=5.0)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    rng = np.random.default_rng(int(args.seed))

    Xtr, ytr, Xte, yte = load_mnist_numpy(args.data_root)

    if args.mode == "pixel":
        Ztr = flatten_nchw(Xtr).astype(np.float32)
        Zte = flatten_nchw(Xte).astype(np.float32)
        B_public = float(np.sqrt(Ztr.shape[1]))

        def decode_to_img(Z: np.ndarray) -> np.ndarray:
            Z = np.asarray(Z, dtype=np.float32)
            return Z.reshape(Z.shape[0], 1, 28, 28)

    else:
        ckpt = Path(out_dir) / f"mnist_convvae_lat{args.vae_latent_dim}.eqx"
        vae = train_or_load_vae_mnist(
            ckpt_path=ckpt,
            X_train_nchw=Xtr,
            latent_dim=int(args.vae_latent_dim),
            epochs=int(args.vae_epochs),
            seed=int(args.vae_seed),
        )
        Ztr = vae_encode_mu(vae, Xtr)
        Zte = vae_encode_mu(vae, Xte)

        B_public = float(args.embed_B_public)
        Ztr = clip_l2_rows(Ztr, B_public)
        Zte = clip_l2_rows(Zte, B_public)

        def decode_to_img(Z: np.ndarray) -> np.ndarray:
            return vae_decode(vae, np.asarray(Z, dtype=np.float32))

    # radius policy on TRAIN
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
    print(
        f"[radius] r_ball(p={args.radius_percentile})={r_ball:.4f}, r_std={policy.r_std:.4f}"
    )

    prior = PoolBallPrior(
        pool_X=Ztr, pool_y=ytr, radius=float(r_ball), label_fixed=None
    )

    # training config (IMPORTANT: trainers_eqx.py trains on ALL points)
    train_cfg = EqxTrainerConfig(
        batch_size=int(args.batch_size),
        num_epochs=int(args.epochs),
        patience=int(args.epochs) + 1,
        shuffle=True,
        val_fraction=0.0,
    )

    # model/optimizer factories
    def make_model(seed: int):
        import jax.random as jr

        return (
            MLPClassifierEqx(
                in_dim=int(Ztr.shape[1]),
                n_classes=10,
                width=256,
                depth=2,
                key=jr.PRNGKey(seed),
            ),
            None,
        )

    def make_optimizer(model):
        import optax
        import equinox as eqx

        tx = optax.adamw(1e-3, weight_decay=1e-4)
        opt_state = tx.init(eqx.filter(model, eqx.is_inexact_array))
        return tx, opt_state

    def loss_fn(model, state, xb, yb, key):
        return multiclass_loss(model, state, xb, yb, key, weight_decay=0.0)

    rows: List[Dict[str, Any]] = []

    for trial in range(int(args.n_trials)):
        # choose target from TEST with at least one candidate in the ball
        max_attempts = 200
        chosen = False
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
                chosen = True
                break
        if not chosen:
            raise RuntimeError(
                f"Failed to find a test target with any training candidates inside r_ball={r_ball:.4f} "
                f"after {max_attempts} attempts. Increase --radius_percentile or use pixel mode."
            )

        assert z_target is not None and y_target is not None

        # D^- from TRAIN, target from TEST
        fixed_idx = rng.choice(
            np.arange(Ztr.shape[0]), size=int(args.fixed_size), replace=False
        )
        X_minus = Ztr[fixed_idx].astype(np.float64)
        y_minus = ytr[fixed_idx].astype(np.int64)

        X_full = np.concatenate([X_minus, z_target.reshape(1, -1)], axis=0)
        y_full = np.concatenate(
            [y_minus, np.asarray([y_target], dtype=np.int64)], axis=0
        )

        # candidate set from ball (label-preserving)
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

        # NN oracle for viz
        train_same = np.where(ytr == int(y_target))[0]
        P = Ztr[train_same].astype(np.float64)
        nn_dists = np.linalg.norm(P - z_target.reshape(1, -1), axis=1)
        nn_idx = train_same[int(np.argmin(nn_dists))]
        nn_rec = Ztr[nn_idx].astype(np.float64)

        # calibrate nm to hit target_eps (same nm used for DP-SGD and Ball-DP-SGD)
        N = int(X_full.shape[0])
        q = min(1.0, float(args.batch_size) / max(1, N))
        steps = int(args.epochs) * int(np.ceil(N / int(args.batch_size)))
        nm = calibrate_noise_multiplier(
            target_epsilon=float(args.target_eps),
            delta=float(args.delta),
            q=q,
            steps=steps,
        )
        print(f"[trial {trial}] nm≈{nm:.3f} for eps≈{args.target_eps} (δ={args.delta})")

        # (1) ERM (non-private)
        trainer_np = EqxNonPrivateTrainer(
            make_model=make_model,
            make_optimizer=make_optimizer,
            loss_fn=loss_fn,
            cfg=train_cfg,
        )
        mdl_np, st_np = trainer_np.fit(X_full, y_full, seed=int(args.seed))

        # (2) Standard DP-SGD
        dp_cfg = DPSGDConfig(
            clipping_norm=float(args.clip_C),
            noise_multiplier=float(nm),
            delta=float(args.delta),
        )
        trainer_dp = EqxStandardDPSGDTrainer(
            make_model=make_model,
            make_optimizer=make_optimizer,
            loss_fn=loss_fn,
            dp_config=dp_cfg,
            cfg=train_cfg,
        )
        mdl_dp, st_dp = trainer_dp.fit(X_full, y_full, seed=int(args.seed))

        # (3) Ball-DP-SGD
        ball_cfg = BallDPSGDConfig(
            radius=float(r_ball),
            lz=float(args.ball_lz),
            noise_multiplier=float(nm),
            delta=float(args.delta),
        )
        trainer_ball = EqxBallDPSGDTrainer(
            make_model=make_model,
            make_optimizer=make_optimizer,
            loss_fn=loss_fn,
            ball_dp_config=ball_cfg,
            cfg=train_cfg,
        )
        mdl_ball, st_ball = trainer_ball.fit(X_full, y_full, seed=int(args.seed))

        # Shadow identification helper
        def run_shadow_attack(release_model, release_state, trainer, tag: str):
            ident = ShadowModelIdentifier(
                trainer=trainer,
                vectorizer=lambda m, state=None: vectorize_eqx_model(
                    m, state=state, include_state=False
                ),
                shadows_per_candidate=4,
                score_mode="nearest_mean",
                include_state_in_vector=False,
                cache_dir=str(Path(out_dir) / "shadow_cache" / tag),
            )
            out = ident.identify(
                release_params=release_model,
                release_state=release_state,
                X_minus=X_minus,
                y_minus=y_minus,
                candidates=candidates,
                rng=rng,
            )
            cand = candidates[int(out.best_idx)]
            mse_feat = float(np.mean((cand.record - z_target) ** 2))
            return cand, mse_feat

        cand_np, mse_np = run_shadow_attack(mdl_np, st_np, trainer_np, "ERM")
        cand_dp, mse_dp = run_shadow_attack(mdl_dp, st_dp, trainer_dp, "DP")
        cand_ball, mse_ball = run_shadow_attack(
            mdl_ball, st_ball, trainer_ball, "BallDP"
        )

        # record results
        rows.append(
            dict(
                trial=trial,
                mode=args.mode,
                y_target=int(y_target),
                r_ball=r_ball,
                target_eps=float(args.target_eps),
                delta=float(args.delta),
                nm=float(nm),
                clip_C=float(args.clip_C),
                ball_lz=float(args.ball_lz),
                sigma_sum_dp=float(nm) * float(args.clip_C),
                sigma_sum_ball=float(nm) * float(args.ball_lz) * float(r_ball),
                mse_erm=mse_np,
                mse_dp=mse_dp,
                mse_ball=mse_ball,
            )
        )

        # visualize
        tgt_img = decode_to_img(z_target.reshape(1, -1))[0]
        nn_img = decode_to_img(nn_rec.reshape(1, -1))[0]
        np_img = decode_to_img(
            np.asarray(cand_np.record, dtype=np.float64).reshape(1, -1)
        )[0]
        dp_img = decode_to_img(
            np.asarray(cand_dp.record, dtype=np.float64).reshape(1, -1)
        )[0]
        ball_img = decode_to_img(
            np.asarray(cand_ball.record, dtype=np.float64).reshape(1, -1)
        )[0]

        plot_image_grid(
            images=[tgt_img, nn_img, np_img, dp_img, ball_img],
            titles=[
                "target",
                "NN (train, same y)",
                "ERM (shadow-id)",
                "DP-SGD (shadow-id)",
                "Ball-DP-SGD (shadow-id)",
            ],
            nrows=1,
            ncols=5,
            save_path=Path(out_dir) / f"trial_{trial:03d}_grid.png",
            show=False,
        )

        print(
            f"[trial {trial}] mse(ERM)={mse_np:.6f} mse(DP)={mse_dp:.6f} mse(BallDP)={mse_ball:.6f}"
        )

    save_csv(Path(out_dir) / "results.csv", rows)
    print(f"[done] wrote {Path(out_dir)/'results.csv'}")


if __name__ == "__main__":
    main()
