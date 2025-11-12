# quantbayes/stochax/robust_inference/run_robust_inference.py
from __future__ import annotations
import os, glob
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from quantbayes.stochax.robust_inference.data import (
    load_dataset,
    dirichlet_label_split,
    collect_probits_dataset,
)
from quantbayes.stochax.robust_inference.clients import train_clients
from quantbayes.stochax.robust_inference.aggregators import make_aggregator, DeepSetTM
from quantbayes.stochax.robust_inference.agg_trainer import (
    train_aggregator_erm,
    train_aggregator_robust,
)
from quantbayes.stochax.robust_inference.eval import (
    aggregator_clean_acc,
    quick_attack_bench,
    pgd_cw_vs_f,
)
from quantbayes.stochax.trainer.train import BoundLogger
from quantbayes.stochax.utils.research_diagnostics import (
    compute_and_save_diagnostics,
    pretty_print_diagnostics,
    plot_margin_distribution,
)
from quantbayes.stochax.utils.lip_upper import make_lipschitz_upper_fn
from quantbayes.stochax.utils.regularizers import global_spectral_norm_penalty


# ---------------------------- Toggles ---------------------------- #
TRAIN_MODE = "robust"  # "erm" | "robust"
USE_LMT = True
USE_SPECTRAL_REG = True  # applies in both modes

# Optional bound logging (clients + aggregator)
LOG_CLIENT_BOUNDS_EVERY: Optional[int] = 1  # None disables client per-epoch L logging
BOUND_LOG_EVERY: Optional[int] = 1  # None disables aggregator per-epoch L logging

# Optional checkpointing to compute accuracy/Σσ/L per epoch
SAVE_CLIENT_CKPTS: bool = True
SAVE_AGG_CKPTS: bool = True

# Attack switches for quick bench
ATTACKS: Dict[str, bool] = {
    "pgd_cw": True,  # main robust metric
    "runnerup": False,
    "cwtm_aware": False,
    "sia_bb": False,
    "sia_wb": False,
    "lma": False,
}

OUT_DIR = "runs/robust_inference_end_to_end"
os.makedirs(OUT_DIR, exist_ok=True)
CLIENT_CKPT_DIR = os.path.join(OUT_DIR, "clients_ckpts") if SAVE_CLIENT_CKPTS else None
AGG_CKPT_DIR = os.path.join(OUT_DIR, "agg_ckpts") if SAVE_AGG_CKPTS else None

# LMT config (aggregator-level; ERM mode)
LMT_KW = dict(
    eps=1.0,
    alpha=1.0,
    conv_mode="tn",
    conv_tn_iters=8,
    conv_gram_iters=5,
    conv_fft_shape=None,
    conv_input_shape=None,
    stop_grad_L=True,
)

# Spectral regularization config (aggregator-level)
SPEC_REG = dict(
    lambda_spec=0.0,
    lambda_frob=0.0,
    lambda_specnorm=1e-4,
    lambda_sob_jac=0.0,
    lambda_sob_kernel=0.0,
    lambda_liplog=0.0,
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


# ---------------------------- Config ---------------------------- #
@dataclass
class Config:
    dataset: str = "synthetic"  # "synthetic" | "mnist" | "cifar-10" | "cifar-100"
    n_clients: int = 6
    f_adv: Optional[int] = None
    dirichlet_alpha: float = 0.5
    seed: int = 0

    # Client training
    client_width: int = 256
    client_epochs: int = 8
    client_batch: int = 256
    client_lr: float = 1e-3
    client_wd: float = 1e-4

    # Aggregator
    aggregator: str = (
        "deepset_tm"  # "mean"|"cwtm"|"cwmed"|"linear"|"deepset"|"deepset_tm"
    )
    deepset_hidden: int = 128

    # Aggregator training budget
    agg_epochs: int = 8
    agg_batch: int = 128
    agg_lr: float = 5e-5
    agg_patience: int = 3

    # Adversarial inner (Alg.1)
    pgd_steps: int = 20
    pgd_step_size: float = 5e-2
    tries_per_batch: int = 1  # <— keep small (1–3) for dev; **must be an int**

    # Eval
    eval_subset: Optional[int] = None  # e.g., 2000 for quick CIFAR-10
    eval_pgd_tries: int = 1
    eval_batch_size: int = 256
    eval_seed: int = 1

    # Viz
    viz_sample_for_cert: int = 4000


# ---------------------------- Plot helpers ---------------------------- #
def _style():
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "legend.frameon": False,
            "figure.dpi": 120,
        }
    )


def _plot_series(xs, ys, title, ylabel, path, xlabel="x"):
    _style()
    plt.figure(figsize=(6.2, 3.4))
    plt.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_multi_lines(
    curves: Dict[str, Tuple[List[float], List[float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    path: str,
):
    _style()
    plt.figure(figsize=(7.6, 4.1))
    colors = plt.cm.tab10.colors
    for idx, (lab, (xs, ys)) in enumerate(curves.items()):
        plt.plot(
            xs,
            ys,
            color=colors[idx % len(colors)],
            linewidth=2.0,
            marker="o",
            markersize=3.6,
            label=lab,
        )
    if curves:
        plt.legend(ncol=min(3, len(curves)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _agg_logits(agg, Ps: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(lambda P: agg(P, None, None)[0])(Ps)


def _acc_from_logits(logits: jnp.ndarray, y: jnp.ndarray) -> float:
    return float(jnp.mean((jnp.argmax(logits, axis=-1) == y).astype(jnp.float32)))


def _overlay_clients_vs_agg(
    client_logger: Optional[BoundLogger], agg_L_ep: Optional[List[float]], out_path: str
):
    """Overlay client L (per epoch) and aggregator L (per epoch)."""
    if client_logger is None or not client_logger.data or not agg_L_ep:
        return
    # collect clients
    per_c: Dict[int, Tuple[List[int], List[float]]] = {}
    for rec in client_logger.data:
        if "L_raw" in rec and "client" in rec and "epoch" in rec:
            c = int(rec["client"])
            per_c.setdefault(c, ([], []))
            per_c[c][0].append(int(rec["epoch"]))
            per_c[c][1].append(float(rec["L_raw"]))
    if not per_c:
        return
    _style()
    plt.figure(figsize=(7.6, 4.1))
    colors = plt.cm.tab10.colors
    for idx, (c, (xs, ys)) in enumerate(sorted(per_c.items())):
        xs, ys = zip(*sorted(zip(xs, ys)))
        plt.plot(
            xs,
            ys,
            color=colors[idx % len(colors)],
            linewidth=2.0,
            marker="o",
            markersize=3.6,
            label=f"client {c}",
        )
    # aggregator
    xs_ag = list(range(1, len(agg_L_ep) + 1))
    plt.plot(
        xs_ag,
        agg_L_ep,
        color="#7B61FF",
        linewidth=2.8,
        linestyle="--",
        label="aggregator (global)",
    )
    plt.xlabel("epoch")
    plt.ylabel("Lipschitz L")
    plt.yscale("log")
    plt.title("Clients vs Aggregator: Lipschitz progression")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# ---------------------------- MAIN ---------------------------- #
def main(cfg: Config):
    print(f"[config] {cfg}")

    # 1) Load data
    Xtr, ytr, Xte, yte, K = load_dataset(cfg.dataset)

    # 2) Server split
    N = Xtr.shape[0]
    N_server = max(1, int(0.10 * N))
    X_server, y_server = Xtr[:N_server], ytr[:N_server]
    X_tr_clients, y_tr_clients = Xtr[N_server:], ytr[N_server:]

    # 3) Dirichlet partition across clients
    parts = dirichlet_label_split(
        X_tr_clients,
        y_tr_clients,
        n_clients=cfg.n_clients,
        n_classes=K,
        alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
        equalize_sizes=False,
        min_per_client=0,
    )

    # 4) Train client models (optional per-epoch L, optional checkpoints for per-epoch acc)
    client_logger = BoundLogger() if LOG_CLIENT_BOUNDS_EVERY else None
    models, states, client_hist = train_clients(
        parts,
        d_in=int(Xtr.shape[1]),
        k=K,
        width=cfg.client_width,
        epochs=cfg.client_epochs,
        batch=cfg.client_batch,
        lr=cfg.client_lr,
        wd=cfg.client_wd,
        seed=cfg.seed,
        # optional Lipschitz logging
        log_client_bounds_every=LOG_CLIENT_BOUNDS_EVERY,
        client_bound_recorder=client_logger if LOG_CLIENT_BOUNDS_EVERY else None,
        # optional per-epoch loss/acc
        return_history=True,
        save_checkpoints_dir=CLIENT_CKPT_DIR,
        X_val=X_server[: max(1, N_server // 5)],
        y_val=y_server[: max(1, N_server // 5)],
    )

    # 5) Collect probits (for aggregator training/eval)
    n_val = max(1, X_server.shape[0] // 5)
    X_srv_tr, y_srv_tr = X_server[:-n_val], y_server[:-n_val]
    X_srv_val, y_srv_val = X_server[-n_val:], y_server[-n_val:]

    Ps_tr = collect_probits_dataset(
        models, states, X_srv_tr, batch_size=256, key=jr.PRNGKey(0)
    )
    Ps_val = collect_probits_dataset(
        models, states, X_srv_val, batch_size=256, key=jr.PRNGKey(1)
    )
    Ps_te = collect_probits_dataset(
        models, states, Xte, batch_size=256, key=jr.PRNGKey(2)
    )

    n_clients = Ps_tr.shape[1]
    print(
        f"[data] server-train {Ps_tr.shape}, server-val {Ps_val.shape}, test {Ps_te.shape}"
    )

    # 6) Effective f
    f_eff = cfg.f_adv if cfg.f_adv is not None else (n_clients - 1) // 2
    f_eff = min(f_eff, (n_clients - 1) // 2)  # ensure 2f < n
    if cfg.f_adv is None:
        print(f"[note] default f_adv set to {f_eff} (≈ n/2, satisfies 2f < n).")
    elif f_eff != cfg.f_adv:
        print(f"[note] clamping f_adv from {cfg.f_adv} to {f_eff} (2f < n).")

    # 7) Build aggregator
    k_model = jr.PRNGKey(42)
    if cfg.aggregator == "deepset_tm":
        base = make_aggregator(
            "deepset", n_clients=n_clients, K=K, key=k_model, hidden=cfg.deepset_hidden
        )
        agg_for_training = base
        wrap_cwtm = True
    else:
        agg_for_training = make_aggregator(
            cfg.aggregator,
            n_clients=n_clients,
            K=K,
            key=k_model,
            hidden=cfg.deepset_hidden,
            f=f_eff,
        )
        wrap_cwtm = False

    # 8) Train aggregator (ERM or robust), with optional bound logging & checkpoints
    agg_logger = BoundLogger() if BOUND_LOG_EVERY else None
    if TRAIN_MODE.lower() == "robust" and cfg.aggregator in {
        "deepset",
        "deepset_tm",
        "linear",
    }:
        print("[train] robust (adversarial) aggregator training (RERM)")
        best_agg, hist = train_aggregator_robust(
            Ps_tr,
            y_srv_tr,
            Ps_val,
            y_srv_val,
            agg_for_training,
            f=f_eff,
            epochs=cfg.agg_epochs,
            batch_size=cfg.agg_batch,
            lr=cfg.agg_lr,
            patience=cfg.agg_patience,
            pgd_steps=cfg.pgd_steps,
            pgd_step_size=cfg.pgd_step_size,
            tries_per_batch=cfg.tries_per_batch,
            project="softmax",
            key=jr.PRNGKey(cfg.seed + 123),
            use_lmt_logits=USE_LMT,
            lmt_eps=LMT_KW["eps"],
            lmt_alpha=LMT_KW["alpha"],
            lmt_stop_grad_L=LMT_KW["stop_grad_L"],
            **SPEC_REG,
            log_global_bound_every=BOUND_LOG_EVERY,
            bound_conv_mode="tn",
            bound_tn_iters=8,
            bound_gram_iters=5,
            bound_fft_shape=None,
            bound_input_shape=None,
            bound_recorder=agg_logger if BOUND_LOG_EVERY else None,
            ckpt_dir=AGG_CKPT_DIR,
        )
    else:
        print("[train] ERM aggregator training")
        best_agg, hist = train_aggregator_erm(
            Ps_tr,
            y_srv_tr,
            Ps_val,
            y_srv_val,
            agg_for_training,
            epochs=cfg.agg_epochs,
            batch_size=cfg.agg_batch,
            lr=cfg.agg_lr,
            patience=cfg.agg_patience,
            key=jr.PRNGKey(cfg.seed + 123),
            loss_kind=("lmt" if USE_LMT else "ce"),
            lmt_kwargs=(LMT_KW if USE_LMT else None),
            **SPEC_REG,
            log_global_bound_every=BOUND_LOG_EVERY,
            bound_conv_mode="tn",
            bound_tn_iters=8,
            bound_gram_iters=5,
            bound_fft_shape=None,
            bound_input_shape=None,
            bound_recorder=agg_logger if BOUND_LOG_EVERY else None,
            ckpt_dir=AGG_CKPT_DIR,
        )

    agg_eval = DeepSetTM(best_agg, f=f_eff) if wrap_cwtm else best_agg

    # 9) Eval (subset option)
    if cfg.eval_subset is not None:
        Ps_eval = Ps_te[: cfg.eval_subset]
        y_eval = yte[: cfg.eval_subset]
        print(f"[eval] using subset {Ps_eval.shape[0]} / {Ps_te.shape[0]}")
    else:
        Ps_eval, y_eval = Ps_te, yte

    clean = aggregator_clean_acc(agg_eval, Ps_eval, y_eval)
    print(f"[eval] clean acc: {clean:.4f}")

    # Configurable attack bench
    bench = quick_attack_bench(
        agg_eval,
        Ps_eval,
        y_eval,
        f=f_eff,
        seed=cfg.eval_seed,
        pgd_steps=cfg.pgd_steps,
        pgd_step_size=cfg.pgd_step_size,
        pgd_tries=cfg.eval_pgd_tries,
        pgd_batch_size=cfg.eval_batch_size,
        attacks=ATTACKS,
    )
    print(
        "[eval] attacks:",
        {k: f"{v:.4f}" for k, v in bench.items() if ATTACKS.get(k, False)},
    )

    # PGD-cw vs f
    f_max = min(f_eff, (n_clients - 1) // 2)
    if f_max >= 1:
        xs, ys = pgd_cw_vs_f(
            agg_eval,
            Ps_eval,
            y_eval,
            list(range(1, f_max + 1)),
            steps=cfg.pgd_steps,
            step_size=cfg.pgd_step_size,
            tries=cfg.eval_pgd_tries,
            batch_size=cfg.eval_batch_size,
            seed=cfg.eval_seed,
        )
        _plot_series(
            xs,
            ys,
            f"PGD-cw robustness vs f | {cfg.dataset}, n={cfg.n_clients}",
            "PGD-cw accuracy",
            os.path.join(OUT_DIR, "pgd_vs_f.png"),
            xlabel="adversarial clients (f)",
        )

    # 10) Clients: loss/acc per epoch (pretty multi-line plots)
    if client_hist is not None:
        # Losses
        curves_tr, curves_va = {}, {}
        for i, h in enumerate(client_hist):
            xs_tr = list(range(1, len(h["train_loss"]) + 1))
            xs_va = list(range(1, len(h["val_loss"]) + 1))
            curves_tr[f"client {i}"] = (xs_tr, h["train_loss"])
            curves_va[f"client {i}"] = (xs_va, h["val_loss"])
        if curves_tr:
            _plot_multi_lines(
                curves_tr,
                "Clients: train loss per epoch",
                "epoch",
                "loss",
                os.path.join(OUT_DIR, "clients_loss_train.png"),
            )
        if curves_va:
            _plot_multi_lines(
                curves_va,
                "Clients: val loss per epoch",
                "epoch",
                "loss",
                os.path.join(OUT_DIR, "clients_loss_val.png"),
            )
        # Accuracies (if checkpoints were enabled)
        curves_atr, curves_ava = {}, {}
        for i, h in enumerate(client_hist):
            if h["train_acc"]:
                xs_atr = list(range(1, len(h["train_acc"]) + 1))
                curves_atr[f"client {i}"] = (xs_atr, h["train_acc"])
            if h["val_acc"]:
                xs_ava = list(range(1, len(h["val_acc"]) + 1))
                curves_ava[f"client {i}"] = (xs_ava, h["val_acc"])
        if curves_atr:
            _plot_multi_lines(
                curves_atr,
                "Clients: train accuracy per epoch",
                "epoch",
                "accuracy",
                os.path.join(OUT_DIR, "clients_acc_train.png"),
            )
        if curves_ava:
            _plot_multi_lines(
                curves_ava,
                "Clients: val accuracy per epoch",
                "epoch",
                "accuracy",
                os.path.join(OUT_DIR, "clients_acc_val.png"),
            )

    # 11) Aggregator: losses/acc per epoch + Σσ/L from checkpoints
    xs = list(range(1, len(hist["train_loss"]) + 1))
    title_mode = "robust" if TRAIN_MODE.lower() == "robust" else "ERM"
    _plot_series(
        xs,
        hist["train_loss"],
        f"Aggregator {title_mode} train loss per epoch",
        "loss",
        os.path.join(OUT_DIR, "agg_loss_train.png"),
        xlabel="epoch",
    )
    _plot_series(
        xs,
        hist["val_loss"],
        f"Aggregator {title_mode} val loss per epoch",
        "loss",
        os.path.join(OUT_DIR, "agg_loss_val.png"),
        xlabel="epoch",
    )

    agg_L_from_ckpt: List[float] = []

    if hist.get("ckpt_dir"):
        ckpt_files = sorted(
            glob.glob(os.path.join(hist["ckpt_dir"], "agg_epoch=*.eqx")),
            key=lambda p: int(p.split("agg_epoch=")[1].split(".eqx")[0]),
        )
        acc_tr, acc_va, sigmas, Ls = [], [], [], []
        template = {"model": agg_for_training, "state": None}
        for e, path in enumerate(ckpt_files, start=1):
            bundle = eqx.tree_deserialise_leaves(path, template)
            mt = bundle["model"]
            # acc
            logits_tr = _agg_logits(mt, Ps_tr)
            acc_tr.append(_acc_from_logits(logits_tr, y_srv_tr))
            logits_va = _agg_logits(mt, Ps_val)
            acc_va.append(_acc_from_logits(logits_va, y_srv_val))
            # Σσ & L
            sigmas.append(float(global_spectral_norm_penalty(mt, conv_mode="tn")))
            from quantbayes.stochax.utils.lip_upper import network_lipschitz_upper

            Ls.append(float(network_lipschitz_upper(mt, state=None, conv_mode="tn")))
        xe = list(range(1, len(acc_tr) + 1))
        if acc_tr:
            _plot_series(
                xe,
                acc_tr,
                "Aggregator train accuracy per epoch",
                "accuracy",
                os.path.join(OUT_DIR, "agg_acc_train.png"),
                xlabel="epoch",
            )
        if acc_va:
            _plot_series(
                xe,
                acc_va,
                "Aggregator val accuracy per epoch",
                "accuracy",
                os.path.join(OUT_DIR, "agg_acc_val.png"),
                xlabel="epoch",
            )
        if sigmas:
            _plot_series(
                xe,
                sigmas,
                "Aggregator Σσ per epoch",
                "Σσ",
                os.path.join(OUT_DIR, "agg_sigma_sum_per_epoch.png"),
                xlabel="epoch",
            )
        if Ls:
            _plot_series(
                xe,
                Ls,
                "Aggregator Lipschitz L per epoch",
                "L",
                os.path.join(OUT_DIR, "agg_L_per_epoch.png"),
                xlabel="epoch",
            )
            agg_L_from_ckpt = Ls

    # 12) Overlay: clients vs aggregator Lipschitz (epoch scale)
    _overlay_clients_vs_agg(
        client_logger,
        agg_L_from_ckpt,
        os.path.join(OUT_DIR, "clients_vs_agg_L_overlay.png"),
    )

    # NEW: clients-only Lipschitz per epoch (parity with FENS)
    if client_logger is not None and client_logger.data:
        per_c: Dict[int, Tuple[List[int], List[float]]] = {}
        for rec in client_logger.data:
            if "L_raw" in rec and "client" in rec and "epoch" in rec:
                c = int(rec["client"])
                per_c.setdefault(c, ([], []))
                per_c[c][0].append(int(rec["epoch"]))
                per_c[c][1].append(float(rec["L_raw"]))
        if per_c:
            curves = {}
            for c, (xs, ys) in sorted(per_c.items()):
                x_sorted, y_sorted = zip(*sorted(zip(xs, ys)))
                curves[f"client {c}"] = (list(x_sorted), list(y_sorted))
            _plot_multi_lines(
                curves,
                "Clients: Lipschitz per epoch",
                "epoch",
                "Lipschitz L",
                os.path.join(OUT_DIR, "clients_L_per_epoch.png"),
            )

    # 13) Research diagnostics (aggregator-only)
    def agg_predict(m, s, X, key=None):  # keyless wrapper
        return jax.vmap(lambda Pi: m(Pi, None, None)[0])(X)

    L_fn = make_lipschitz_upper_fn(
        conv_mode="tn", conv_tn_iters=8, conv_input_shape=None
    )
    d = compute_and_save_diagnostics(
        agg_eval,
        None,
        margin_subset=(Ps_eval[:4000], y_eval[:4000]),
        predict_fn=agg_predict,
        lipschitz_upper_bound_fn=L_fn,
        include_coverage=True,
        include_sigma_lists=True,
        save_path=os.path.join(OUT_DIR, "agg_diag.npz"),
        save_meta={
            "dataset": cfg.dataset,
            "split": "test",
            "mode": TRAIN_MODE,
            "loss": "lmt" if (TRAIN_MODE == "erm" and USE_LMT) else "ce",
            "spec_reg": USE_SPECTRAL_REG,
        },
    )
    print("\n=== Aggregator diagnostics ===")
    print(pretty_print_diagnostics(d))
    plot_margin_distribution(
        d, which=("raw", "normalized", "radius"), show=False
    )  # save only
    plt.savefig(os.path.join(OUT_DIR, "margin_dist.png"), dpi=220, bbox_inches="tight")
    plt.close()

    print(f"[RobustInference] artifacts saved under: {OUT_DIR}")


if __name__ == "__main__":
    main(Config())
