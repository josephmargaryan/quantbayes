# quantbayes/ball_dp/reconstruction/experiments/mnist_convex_reconstruction.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

# ---- quantbayes.ball_dp ----
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
    EqxFullBatchGDTrainer,
    FullBatchGDConfig,
    softmax_params_numpy,
)

# ---- heads ----
from quantbayes.ball_dp.heads.prototypes import fit_ridge_prototypes
from quantbayes.ball_dp.heads.softmax_eqx import SoftmaxLinearEqx, softmax_objective


# -------------------------
# Utilities
# -------------------------
def load_mnist_numpy(
    root: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Xtr: (N,1,28,28) float32 in [0,1]
      ytr: (N,) int64
      Xte: (M,1,28,28) float32
      yte: (M,) int64
    """
    import torch  # noqa: F401
    from torchvision import datasets, transforms

    tfm = transforms.ToTensor()
    tr = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    te = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    def _to_numpy(ds):
        X, y = [], []
        for x, yy in ds:
            X.append(x.numpy())  # (1,28,28)
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
    """
    Trains a public ConvVAE (Bernoulli likelihood) and caches it with eqx serialization.
    """
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
    """
    Encode images -> latent mean mu (as embeddings).
    """
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    X = jnp.asarray(X_nchw.astype(np.float32))

    @eqx.filter_jit
    def _enc(xb):
        mu, _logvar = model.encoder(xb, train=False)
        return mu

    mu = _enc(X)
    return np.asarray(jax.device_get(mu), dtype=np.float32)


def vae_decode(model, Z: np.ndarray) -> np.ndarray:
    """
    Decode latents -> images (N,1,28,28) in [0,1] via sigmoid(Bernoulli logits).
    """
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
# Main experiment
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./_out_convex")
    ap.add_argument("--mode", type=str, choices=["pixel", "embedding"], default="pixel")

    ap.add_argument("--fixed_size", type=int, default=2000)
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--candidate_m", type=int, default=30)

    # ERM + DP
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--eps", type=float, default=2.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument(
        "--sigma_method", type=str, choices=["classic", "analytic"], default="analytic"
    )

    # Ball radius selection
    ap.add_argument("--radius_percentile", type=float, default=50.0)

    # VAE settings (embedding mode)
    ap.add_argument("--vae_latent_dim", type=int, default=16)
    ap.add_argument("--vae_epochs", type=int, default=10)
    ap.add_argument("--vae_seed", type=int, default=0)

    # Public embedding bound (embedding mode) enforced by clipping
    ap.add_argument("--embed_B_public", type=float, default=5.0)

    # softmax ERM trainer
    ap.add_argument("--softmax_steps", type=int, default=1200)
    ap.add_argument("--softmax_lr", type=float, default=0.15)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    rng = np.random.default_rng(int(args.seed))

    Xtr, ytr, Xte, yte = load_mnist_numpy(args.data_root)

    # Build feature space Z
    if args.mode == "pixel":
        Ztr = flatten_nchw(Xtr).astype(np.float32)
        Zte = flatten_nchw(Xte).astype(np.float32)

        # Public bound for pixels in [0,1]: ||x||_2 <= sqrt(d)=28
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

        # Enforce public bound by clipping to embed_B_public (keeps decoding meaningful)
        B_public = float(args.embed_B_public)
        Ztr = clip_l2_rows(Ztr, B_public)
        Zte = clip_l2_rows(Zte, B_public)

        def decode_to_img(Z: np.ndarray) -> np.ndarray:
            return vae_decode(vae, np.asarray(Z, dtype=np.float32))

    # Compute radius policy on train embeddings
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
    r_std = float(policy.r_std)

    print(f"[radius] B_public={policy.B:.4f} -> r_std=2B={r_std:.4f}")
    print(f"[radius] r_ball(p={args.radius_percentile})={r_ball:.4f}")

    # Prior pool over TRAIN points; we will set label_fixed per target
    prior = PoolBallPrior(pool_X=Ztr, pool_y=ytr, radius=r_ball, label_fixed=None)

    rows: List[Dict[str, Any]] = []

    # Share caches across std vs ball identification to avoid recomputing means twice
    proto_ident_cache: Dict[str, np.ndarray] = {}
    softmax_ident_cache: Dict[str, np.ndarray] = {}

    # Helper: stable candidate meta id for caching
    def _ensure_candidate_ids(cands: List[Candidate]) -> None:
        for i, c in enumerate(cands):
            if "id" not in c.meta:
                # prefer pool_index if present; else fallback to loop id
                if "pool_index" in c.meta:
                    c.meta["id"] = f"pool_{int(c.meta['pool_index'])}"
                elif c.meta.get("is_target", False):
                    c.meta["id"] = "TARGET"
                else:
                    c.meta["id"] = f"cand_{i}"

    for trial in range(int(args.n_trials)):
        # ---- choose a target from TEST with at least one candidate in the ball ----
        max_attempts = 200
        chosen = False
        z_target = None
        y_target = None
        for _ in range(max_attempts):
            t = int(rng.integers(0, Zte.shape[0]))
            y_try = int(yte[t])
            z_try = Zte[t].astype(np.float64)

            prior.label_fixed = y_try
            idx = prior.candidate_indices(center=z_try)

            if idx.size > 0:
                z_target = z_try
                y_target = y_try
                chosen = True
                break

        if not chosen:
            raise RuntimeError(
                f"Failed to find a test target with any training candidates inside radius r_ball={r_ball:.4f} "
                f"after {max_attempts} attempts. Increase --radius_percentile or use embedding mode."
            )

        assert z_target is not None and y_target is not None

        # ---- D^- from TRAIN (known to adversary), target is from TEST (missing record) ----
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

        # ---- candidates sampled from the ball around the true target (label-preserving) ----
        prior.label_fixed = int(y_target)
        candidates = list(
            prior.sample(center=z_target, m=max(1, int(args.candidate_m) - 1), rng=rng)
        )
        # include the true target as one candidate
        candidates.append(
            Candidate(
                record=z_target.copy(),
                label=int(y_target),
                meta={"is_target": True, "id": "TARGET"},
            )
        )
        rng.shuffle(candidates)
        _ensure_candidate_ids(candidates)

        # ---- Simple NN oracle baseline (not DP-related; for visualization only) ----
        # We search within TRAIN, same label
        train_same = np.where(ytr == int(y_target))[0]
        P = Ztr[train_same].astype(np.float64)
        dists = np.linalg.norm(P - z_target.reshape(1, -1), axis=1)
        nn_idx = train_same[int(np.argmin(dists))]
        nn_rec = Ztr[nn_idx].astype(np.float64)

        # =========================================================
        # (A) Ridge prototypes (closed-form convex ERM)
        # =========================================================
        mus, counts = fit_ridge_prototypes(
            Z=np.asarray(X_full, dtype=np.float32),
            y=np.asarray(y_full, dtype=np.int64),
            num_classes=10,
            lam=float(args.lam),
        )

        # ERM exact equation solve
        proto_solver = RidgePrototypesEquationSolver(
            lam=float(args.lam), n_total=n_total
        )
        proto_rec = proto_solver.reconstruct(
            release_mus=mus, d_minus=dset.d_minus, label_known=None
        )
        proto_erm_mse = (
            float(np.mean((proto_rec.record_hat - z_target) ** 2))
            if proto_rec.record_hat is not None
            else float("nan")
        )

        # DP releases (std baseline r_std and ball r_ball)
        dp_proto_std = dp_release_ridge_prototypes_gaussian(
            mus,
            counts,
            r=r_std,
            lam=float(args.lam),
            eps=float(args.eps),
            delta=float(args.delta),
            sigma_method=str(args.sigma_method),
            rng=rng,
        )
        dp_proto_ball = dp_release_ridge_prototypes_gaussian(
            mus,
            counts,
            r=r_ball,
            lam=float(args.lam),
            eps=float(args.eps),
            delta=float(args.delta),
            sigma_method=str(args.sigma_method),
            rng=rng,
        )

        # Gaussian identification (reuse cache across std/ball by reusing one identifier object)
        class ProtoTrainer:
            def fit(self, X, y):
                mus2, _ = fit_ridge_prototypes(
                    Z=np.asarray(X, dtype=np.float32),
                    y=np.asarray(y, dtype=np.int64),
                    num_classes=10,
                    lam=float(args.lam),
                )
                return mus2

        proto_ident = GaussianOutputIdentifier(
            trainer=ProtoTrainer(),
            vectorizer=vectorize_prototypes,
            sigma=float(dp_proto_std["sigma"]),
            cache=True,
        )
        # manually inject a shared cache
        proto_ident._cache = proto_ident_cache  # type: ignore[attr-defined]

        out_std = proto_ident.identify(
            release_params=dp_proto_std["mus_noisy"],
            X_minus=dset.X_minus,
            y_minus=dset.y_minus,
            candidates=candidates,
        )
        cand_std = candidates[int(out_std.best_idx)]
        proto_std_mse = float(np.mean((cand_std.record - z_target) ** 2))

        proto_ident.sigma = float(dp_proto_ball["sigma"])
        out_ball = proto_ident.identify(
            release_params=dp_proto_ball["mus_noisy"],
            X_minus=dset.X_minus,
            y_minus=dset.y_minus,
            candidates=candidates,
        )
        cand_ball = candidates[int(out_ball.best_idx)]
        proto_ball_mse = float(np.mean((cand_ball.record - z_target) ** 2))

        rows.append(
            dict(
                trial=trial,
                mode=args.mode,
                head="ridge_prototypes",
                y_target=int(y_target),
                n_total=n_total,
                r_ball=r_ball,
                r_std=r_std,
                mse_erm=proto_erm_mse,
                mse_dp_std=proto_std_mse,
                mse_dp_ball=proto_ball_mse,
                sigma_std=float(dp_proto_std["sigma"]),
                sigma_ball=float(dp_proto_ball["sigma"]),
                candidate_count=len(candidates),
            )
        )

        # =========================================================
        # (B) Softmax linear (multiclass convex ERM)
        # =========================================================
        d_in = int(X_full.shape[1])
        lam = float(args.lam)

        def make_softmax(key):
            return SoftmaxLinearEqx(d_in=d_in, n_classes=10, key=key)

        def loss_fn(m, state, xb, yb, key):
            return softmax_objective(m, state, xb, yb, key, lam=lam)

        trainer = EqxFullBatchGDTrainer(
            make_model=make_softmax,
            loss_fn=loss_fn,
            cfg=FullBatchGDConfig(
                steps=int(args.softmax_steps),
                learning_rate=float(args.softmax_lr),
                grad_clip_norm=None,
                seed=int(args.seed),
                jit=True,
            ),
        )

        softmax_model = trainer.fit(
            np.asarray(X_full, dtype=np.float32),
            np.asarray(y_full, dtype=np.int64),
        )

        # ERM equation solve (exact in theory; approximate if optimizer not fully converged)
        W, b = softmax_params_numpy(softmax_model)
        sm_solver = SoftmaxEquationSolver(
            lam=lam,
            n_total=n_total,
            include_bias=True,
            batch_size=8192,
        )
        sm_rec = sm_solver.reconstruct(W=W, b=b, d_minus=dset.d_minus)
        sm_erm_mse = (
            float(np.mean((sm_rec.record_hat - z_target) ** 2))
            if sm_rec.record_hat is not None
            else float("nan")
        )

        # Lz bound for softmax head (uses B_public)
        Lz = float(
            lz_softmax_linear_bound(B=float(B_public), lam=lam, include_bias=True)
        )

        sm_dp_std = dp_release_eqx_model_gaussian(
            softmax_model,
            lz=Lz,
            r=r_std,
            lam=lam,
            n=n_total,
            eps=float(args.eps),
            delta=float(args.delta),
            sigma_method=str(args.sigma_method),
            rng=rng,
        )
        sm_dp_ball = dp_release_eqx_model_gaussian(
            softmax_model,
            lz=Lz,
            r=r_ball,
            lam=lam,
            n=n_total,
            eps=float(args.eps),
            delta=float(args.delta),
            sigma_method=str(args.sigma_method),
            rng=rng,
        )

        # Gaussian identification on model objects; reuse cache across std/ball
        sm_ident = GaussianOutputIdentifier(
            trainer=trainer,
            vectorizer=lambda mdl: vectorize_eqx_model(mdl, include_state=False),
            sigma=float(sm_dp_std["dp_report"]["sigma"]),
            cache=True,
        )
        sm_ident._cache = softmax_ident_cache  # type: ignore[attr-defined]

        out_sm_std = sm_ident.identify(
            release_params=sm_dp_std["model_private"],
            X_minus=dset.X_minus,
            y_minus=dset.y_minus,
            candidates=candidates,
        )
        cand_sm_std = candidates[int(out_sm_std.best_idx)]
        sm_std_mse = float(np.mean((cand_sm_std.record - z_target) ** 2))

        sm_ident.sigma = float(sm_dp_ball["dp_report"]["sigma"])
        out_sm_ball = sm_ident.identify(
            release_params=sm_dp_ball["model_private"],
            X_minus=dset.X_minus,
            y_minus=dset.y_minus,
            candidates=candidates,
        )
        cand_sm_ball = candidates[int(out_sm_ball.best_idx)]
        sm_ball_mse = float(np.mean((cand_sm_ball.record - z_target) ** 2))

        rows.append(
            dict(
                trial=trial,
                mode=args.mode,
                head="softmax_linear",
                y_target=int(y_target),
                n_total=n_total,
                r_ball=r_ball,
                r_std=r_std,
                Lz=Lz,
                mse_erm=sm_erm_mse,
                mse_dp_std=sm_std_mse,
                mse_dp_ball=sm_ball_mse,
                sigma_std=float(sm_dp_std["dp_report"]["sigma"]),
                sigma_ball=float(sm_dp_ball["dp_report"]["sigma"]),
                candidate_count=len(candidates),
            )
        )

        # =========================================================
        # Visualization (one grid per trial)
        # =========================================================
        tgt_img = decode_to_img(z_target.reshape(1, -1))[0]
        nn_img = decode_to_img(nn_rec.reshape(1, -1))[0]

        # prototypes
        proto_erm_img = (
            decode_to_img(proto_rec.record_hat.reshape(1, -1))[0]
            if proto_rec.record_hat is not None
            else nn_img
        )
        proto_std_img = decode_to_img(
            np.asarray(cand_std.record, dtype=np.float64).reshape(1, -1)
        )[0]
        proto_ball_img = decode_to_img(
            np.asarray(cand_ball.record, dtype=np.float64).reshape(1, -1)
        )[0]

        # softmax
        sm_erm_img = (
            decode_to_img(sm_rec.record_hat.reshape(1, -1))[0]
            if sm_rec.record_hat is not None
            else nn_img
        )
        sm_std_img = decode_to_img(
            np.asarray(cand_sm_std.record, dtype=np.float64).reshape(1, -1)
        )[0]
        sm_ball_img = decode_to_img(
            np.asarray(cand_sm_ball.record, dtype=np.float64).reshape(1, -1)
        )[0]

        grid_imgs = [
            tgt_img,
            nn_img,
            proto_erm_img,
            proto_std_img,
            proto_ball_img,
            sm_erm_img,
            sm_std_img,
            sm_ball_img,
        ]
        titles = [
            "target",
            "NN (train, same y)",
            "proto ERM (eq-solve)",
            "proto DP std (identify)",
            "proto Ball-DP (identify)",
            "softmax ERM (eq-solve)",
            "softmax DP std (identify)",
            "softmax Ball-DP (identify)",
        ]

        plot_image_grid(
            images=grid_imgs,
            titles=titles,
            nrows=2,
            ncols=4,
            save_path=Path(out_dir) / f"trial_{trial:03d}_grid.png",
            show=False,
        )

        print(f"[trial {trial}] done. saved grid.")

    save_csv(Path(out_dir) / "results.csv", rows)
    print(f"[done] wrote {Path(out_dir)/'results.csv'}")


if __name__ == "__main__":
    main()
