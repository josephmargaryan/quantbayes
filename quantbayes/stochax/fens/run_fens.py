# quantbayes/stochax/fens/run_fens.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.fens.trainer import (
    FENSAggregatorFLTrainerEqx,
    collect_outputs_dataset,
)
from quantbayes.stochax.fens.aggregators import make_fens_aggregator
from quantbayes.stochax.robust_inference.data import (
    load_dataset,
    dirichlet_label_split,
)
from quantbayes.stochax.robust_inference.clients import train_clients
from quantbayes.stochax.robust_inference.eval import aggregator_clean_acc
from quantbayes.stochax.trainer.train import BoundLogger

# Diagnostics
from quantbayes.stochax.utils.research_diagnostics import (
    compute_and_save_diagnostics,
    pretty_print_diagnostics,
    plot_margin_distribution,
)
from quantbayes.stochax.utils.lip_upper import make_lipschitz_upper_fn
from quantbayes.stochax.privacy.dp import DPSGDConfig


# ============================ TOGGLES ============================ #
USE_LMT: bool = True  # LMT at aggregator level
USE_SPECTRAL_REG: bool = True  # enable Σσ regularization for aggregator locals
USE_DP: bool = False  # DP on aggregator locals

# Optional logging (turn both off to skip all bound plots)
BOUND_LOG_EVERY: Optional[int] = 1  # aggregator locals & global; None disables
LOG_CLIENT_BOUNDS_EVERY: Optional[int] = 1  # client per-epoch L; None disables

OUT_DIR = "runs/fens_end_to_end"
os.makedirs(OUT_DIR, exist_ok=True)

# LMT config (aggregator-level)
LMT_KW = dict(
    eps=0.5,
    alpha=1.0,
    conv_mode="tn",
    conv_tn_iters=8,
    conv_gram_iters=5,
    conv_fft_shape=None,
    conv_input_shape=None,
    stop_grad_L=True,
)

# Spectral regularization config (aggregator-level)
SPEC_REG: Dict = dict(
    lambda_spec=0.0,
    lambda_frob=0.0,
    lambda_specnorm=1e-4,  # Σ per-layer σ
    lambda_sob_jac=0.0,
    lambda_sob_kernel=0.0,
    lambda_liplog=0.0,  # τ·log(L) (gentle)
    specnorm_conv_mode="tn",
    specnorm_conv_tn_iters=8,
    specnorm_conv_gram_iters=5,
    specnorm_conv_fft_shape=None,
    specnorm_conv_input_shape=None,
    lip_conv_mode="tn",
    lip_conv_tn_iters=8,
    lip_conv_gram_iters=5,
    lip_conv_fft_shape=None,
    lip_conv_input_shape=None,
)


# ============================ CONFIG ============================ #
@dataclass
class Config:
    seed: int = 0
    n_clients: int = 6
    dirichlet_alpha: float = 0.8
    client_width: int = 256
    client_epochs: int = 10
    client_batch: int = 256
    dataset: str = "synthetic"  # "synthetic" | "mnist" | "cifar-10" | "cifar-100"

    # aggregator
    agg_name: str = (
        "mlp"  # "mlp" | "per_client_class" | "linear" | "deepset" | "mlp_sn"
    )
    # "mlp_sn"
    target: float = 1.0
    agg_hidden: int = 128

    # FENS phase-2 (FL)
    rounds: int = 40
    local_epochs: int = 1
    local_batch: int = 256
    local_lr: float = 1e-3
    local_wd: float = 0.0
    patience: int = 2

    # DP (optional)
    dp_clip: float = 1.0
    dp_noise: float = 1.0
    dp_delta: float = 1e-5


# ============================ PLOTTING HELPERS ============================ #
def _apply_pub_style():
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "figure.dpi": 120,
            "legend.frameon": False,
        }
    )


def _plot_series(xs, ys, title, ylabel, path):
    _apply_pub_style()
    plt.figure(figsize=(6.3, 3.4))
    plt.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.8)
    plt.title(title)
    plt.xlabel("round")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_band(xs, mean, lo, hi, title, ylabel, path):
    _apply_pub_style()
    plt.figure(figsize=(6.3, 3.4))
    plt.plot(xs, mean, linewidth=2.0)
    plt.fill_between(xs, lo, hi, alpha=0.18)
    plt.title(title)
    plt.xlabel("round")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_clients_L(recs: List[Dict], n_clients: int, out_path: str):
    """One line per client: L vs epoch (different lengths are fine)."""
    _apply_pub_style()
    plt.figure(figsize=(7.6, 4.1))
    colors = plt.cm.tab10.colors  # nice palette for up to 10 clients

    drew = False
    for c in range(n_clients):
        pts = [
            (r["epoch"], r["L_raw"])
            for r in recs
            if r.get("client") == c and "L_raw" in r
        ]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        plt.plot(
            xs,
            ys,
            color=colors[c % len(colors)],
            linewidth=2.0,
            alpha=0.95,
            marker="o",
            markersize=3.8,
            label=f"client {c}",
        )
        drew = True

    if not drew:
        plt.close()
        return
    plt.xlabel("epoch")
    plt.ylabel("client Lipschitz L")
    plt.title("Clients: Lipschitz per epoch")
    if n_clients <= 10:
        plt.legend(ncol=min(3, n_clients))
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_clients_vs_agg_overlay(
    client_recs: List[Dict],
    agg_recs: List[Dict],
    n_clients: int,
    n_rounds: int,
    out_path: str,
):
    """
    Overlay: clients (epoch-scale) vs aggregator global L (rounds mapped to epoch-scale).
    Different client lengths are fine; we don't pad.
    """
    # collect client traces
    per_c = {}
    emax = 0
    for c in range(n_clients):
        pts = [
            (r["epoch"], r["L_raw"])
            for r in client_recs
            if r.get("client") == c and "L_raw" in r
        ]
        if not pts:
            continue
        pts = sorted(pts)
        emax = max(emax, pts[-1][0] if pts else 0)
        per_c[c] = pts

    # aggregator global L per round
    gL = sorted(
        [
            (r["round"], r["L_raw"])
            for r in agg_recs
            if r.get("client") == -1 and "L_raw" in r
        ]
    )

    if not per_c and not gL:
        return

    _apply_pub_style()
    plt.figure(figsize=(7.6, 4.1))
    colors = plt.cm.tab10.colors

    # plot clients
    for c, pts in per_c.items():
        xs, ys = zip(*pts)
        plt.plot(
            xs,
            ys,
            color=colors[c % len(colors)],
            linewidth=2.0,
            alpha=0.95,
            marker="o",
            markersize=3.5,
            label=f"client {c}",
        )

    # overlay aggregator (map rounds to epoch-scale using max client epoch)
    if gL and emax > 0:
        xs_r, ys_r = zip(*gL)
        xs_aggr = [
            1 + (xr - 1) * (max(1, emax - 1) / max(1, n_rounds - 1)) for xr in xs_r
        ]
        plt.plot(
            xs_aggr,
            ys_r,
            color="#7B61FF",
            linewidth=2.8,
            alpha=0.95,
            linestyle="--",
            marker=None,
            label="aggregator (global)",
        )

    plt.xlabel("training progress (epoch-scale)")
    plt.ylabel("Lipschitz L")
    plt.yscale("log")
    plt.title("Clients vs Aggregator: Lipschitz progression")
    if n_clients <= 10:
        plt.legend(ncol=min(3, n_clients + 1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# ============================ MAIN ============================ #
def main(cfg: Config):
    print(f"[config] {cfg}")

    # 1) dataset
    Xtr, ytr, Xte, yte, K = load_dataset(cfg.dataset)

    # 2) client partitions
    parts = dirichlet_label_split(
        Xtr,
        ytr,
        n_clients=cfg.n_clients,
        n_classes=K,
        alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
    )

    # 3) train clients (one-shot) — optional per-epoch L logging
    client_logger = BoundLogger() if LOG_CLIENT_BOUNDS_EVERY else None

    models, states = train_clients(
        parts,
        d_in=Xtr.shape[1],
        k=K,
        width=cfg.client_width,
        epochs=cfg.client_epochs,
        batch=cfg.client_batch,
        lr=1e-3,
        wd=1e-4,
        seed=cfg.seed,
        X_val=Xte[:1024],
        y_val=yte[:1024],
        log_client_bounds_every=LOG_CLIENT_BOUNDS_EVERY,
        client_bound_recorder=client_logger if LOG_CLIENT_BOUNDS_EVERY else None,
    )

    # 4) collect logits meta-datasets for the aggregator
    Ps_parts: List[jnp.ndarray] = []
    y_parts: List[jnp.ndarray] = []
    for i, (Xi, yi) in enumerate(parts):
        P_i = collect_outputs_dataset(
            models, states, Xi, batch_size=512, key=jr.PRNGKey(cfg.seed + 100 + i)
        )
        Ps_parts.append(P_i)
        y_parts.append(yi.astype(jnp.int32))

    Ps_test = collect_outputs_dataset(
        models, states, Xte, batch_size=512, key=jr.PRNGKey(cfg.seed + 999)
    )
    y_test = yte.astype(jnp.int32)

    # 5) baseline mean(probs)
    Ps_test_probs = jax.nn.softmax(Ps_test, axis=-1)
    from quantbayes.stochax.robust_inference.aggregators import MeanAgg

    mean_acc = aggregator_clean_acc(MeanAgg(K=K), Ps_test_probs, y_test)

    # 6) aggregator factory
    def agg_init(k):
        return make_fens_aggregator(
            cfg.agg_name,
            n_clients=cfg.n_clients,
            n_classes=K,
            key=k,
            hidden=cfg.agg_hidden,
            target=cfg.target,
        )

    dp_cfg = (
        DPSGDConfig(
            clipping_norm=cfg.dp_clip, noise_multiplier=cfg.dp_noise, delta=cfg.dp_delta
        )
        if USE_DP
        else None
    )

    # 7) trainer (aggregator) — optional bound logging for locals/global
    trainer = FENSAggregatorFLTrainerEqx(
        agg_init,
        n_clients=cfg.n_clients,
        outer_rounds=cfg.rounds,
        inner_epochs=cfg.local_epochs,
        batch_size=cfg.local_batch,
        local_lr=cfg.local_lr,
        local_weight_decay=cfg.local_wd,
        patience=cfg.patience,
        server_opt={
            "name": "adam",
            "lr": 1e-2,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
        dp_config=dp_cfg,
        key=jr.PRNGKey(cfg.seed + 123),
        agg_loss=("lmt" if USE_LMT else "ce"),
        lmt_kwargs=(LMT_KW if USE_LMT else None),
        spec_reg=(SPEC_REG if USE_SPECTRAL_REG else None),
        bound_log_every=BOUND_LOG_EVERY,
        bound_conv_mode="tn",
        bound_tn_iters=8,
        bound_gram_iters=5,
        bound_fft_shape=None,
        bound_input_shape=None,
        apply_spec_in_dp=False,
    )

    agg_logger = BoundLogger() if BOUND_LOG_EVERY else None
    if agg_logger is not None:
        trainer.bound_recorder = agg_logger

    # 8) train agg (FENS rounds)
    agg_model, hist = trainer.train(Ps_parts, y_parts, Ps_test, y_test)
    final_acc = aggregator_clean_acc(agg_model, Ps_test, y_test)

    print(f"\n[FENS] Mean(probs) baseline acc = {mean_acc*100:.2f}%")
    print(f"[FENS] FENS({cfg.agg_name}) final acc = {final_acc*100:.2f}%")

    # curves
    rounds = np.arange(1, len(hist["ce"]) + 1)
    _plot_series(
        rounds,
        hist["ce"],
        "FENS Test CE vs Rounds",
        "cross-entropy",
        os.path.join(OUT_DIR, "fens_ce.png"),
    )
    _plot_series(
        rounds,
        np.array(hist["acc"]) * 100.0,
        "FENS Test Accuracy vs Rounds",
        "accuracy (%)",
        os.path.join(OUT_DIR, "fens_acc.png"),
    )

    # 9) OPTIONAL: bounds & Σσ plots at the end (aggregator + clients)
    if agg_logger is not None and agg_logger.data:
        recs = agg_logger.data

        # global L
        gL = sorted(
            [
                (r["round"], r["L_raw"])
                for r in recs
                if r.get("client") == -1 and ("L_raw" in r)
            ]
        )
        if gL:
            xs, ys = zip(*gL)
            _plot_series(
                xs,
                ys,
                "Global aggregator L (post-aggregation)",
                "L (certified UB)",
                os.path.join(OUT_DIR, "agg_global_L.png"),
            )

        # local L band
        by_round = {}
        for r in recs:
            if ("L_raw" in r) and (r.get("client", 0) >= 0):
                by_round.setdefault(r["round"], []).append(r["L_raw"])
        if by_round:
            xs = sorted(by_round.keys())
            means = [float(np.mean(by_round[x])) for x in xs]
            lo = [float(np.min(by_round[x])) for x in xs]
            hi = [float(np.max(by_round[x])) for x in xs]
            _plot_band(
                xs,
                means,
                lo,
                hi,
                "Local aggregator L per round",
                "L (certified UB)",
                os.path.join(OUT_DIR, "agg_local_L_band.png"),
            )

        # global Σσ
        gS = sorted(
            [
                (r["round"], r["sigma_sum"])
                for r in recs
                if r.get("client") == -1 and ("sigma_sum" in r)
            ]
        )
        if gS:
            xs, ys = zip(*gS)
            _plot_series(
                xs,
                ys,
                "Global aggregator Σσ (post-aggregation)",
                "Σσ",
                os.path.join(OUT_DIR, "agg_global_sigma_sum.png"),
            )

        # local Σσ band
        bS = {}
        for r in recs:
            if ("sigma_sum" in r) and (r.get("client", 0) >= 0):
                bS.setdefault(r["round"], []).append(r["sigma_sum"])
        if bS:
            xs = sorted(bS.keys())
            means = [float(np.mean(bS[x])) for x in xs]
            lo = [float(np.min(bS[x])) for x in xs]
            hi = [float(np.max(bS[x])) for x in xs]
            _plot_band(
                xs,
                means,
                lo,
                hi,
                "Local aggregator Σσ per round",
                "Σσ",
                os.path.join(OUT_DIR, "agg_local_sigma_sum_band.png"),
            )

    # per-client L per epoch & overlay (clients vs aggregator)
    if client_logger is not None and client_logger.data:
        _plot_clients_L(
            client_logger.data,
            cfg.n_clients,
            os.path.join(OUT_DIR, "clients_L_per_epoch.png"),
        )
        if agg_logger is not None and agg_logger.data:
            _plot_clients_vs_agg_overlay(
                client_logger.data,
                agg_logger.data,
                cfg.n_clients,
                len(rounds),
                os.path.join(OUT_DIR, "clients_vs_agg_L_overlay.png"),
            )

    # 10) research diagnostics (aggregator-only)
    def agg_predict(m, s, X, key=None):
        return jax.vmap(lambda Pi: m(Pi, None, None)[0])(X)

    L_fn = make_lipschitz_upper_fn(conv_mode="tn")
    d = compute_and_save_diagnostics(
        agg_model,
        None,
        margin_subset=(Ps_test[:4000], y_test[:4000]),
        predict_fn=agg_predict,  # wrapper avoids PRNG key issues
        lipschitz_upper_bound_fn=L_fn,
        include_coverage=True,
        include_sigma_lists=True,
        save_path=os.path.join(OUT_DIR, "agg_diag.npz"),
        save_meta={
            "dataset": cfg.dataset,
            "split": "test",
            "loss": "lmt" if USE_LMT else "ce",
            "spec_reg": USE_SPECTRAL_REG,
        },
    )
    print("\n=== Aggregator diagnostics ===")
    print(pretty_print_diagnostics(d))
    plot_margin_distribution(d, which=("raw", "normalized", "radius"), show=False)
    plt.savefig(
        os.path.join(OUT_DIR, "fens_margin_dist.png"), dpi=220, bbox_inches="tight"
    )
    plt.close()

    print(f"[FENS] artifacts saved under: {OUT_DIR}")


if __name__ == "__main__":
    main(Config())
