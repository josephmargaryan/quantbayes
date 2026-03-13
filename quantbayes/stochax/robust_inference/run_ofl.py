# quantbayes/stochax/robust_inference/run_ofl_demo.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.robust_inference.data import (
    load_mnist,
    load_cifar10,
    load_cifar100,
    load_synthetic,
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
    aggregator_pgd_cw_acc,
    pgd_cw_vs_f,
)
from quantbayes.stochax.robust_inference.probits import sigma_x, margin
from quantbayes.stochax.robust_inference.certificate import kappa_const


# ---------------------------- Config ---------------------------- #


@dataclass
class Config:
    dataset: str = "mnist"  # "synthetic" | "mnist" | "cifar-10" | "cifar-100"
    n_clients: int = 17
    f_adv: int = 4
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
    robust_train: bool = True
    agg_epochs: int = 10
    agg_batch: int = 128
    agg_lr: float = 5e-5
    agg_patience: int = 3

    # Adversarial inner (Alg.1)
    pgd_steps: int = 50
    pgd_step_size: float = 5e-2
    tries_per_batch: Optional[int] = None  # None => inferred from n

    # Eval
    eval_subset: Optional[int] = None  # e.g., 2000 for quick CIFAR-10
    eval_pgd_tries: int = 1
    eval_batch_size: int = 256
    eval_seed: int = 1

    # Viz
    viz_sample_for_cert: int = (
        4000  # number of points used in cert scatter (cap for speed)
    )


# ---------------------------- Data loaders ---------------------------- #


def _load_dataset(
    name: str,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    name = name.lower()
    if name == "mnist":
        Xtr, ytr, Xte, yte = load_mnist(seed=0)
        K = 10
    elif name == "cifar-10":
        Xtr, ytr, Xte, yte = load_cifar10(seed=0)
        K = 10
    elif name == "cifar-100":
        Xtr, ytr, Xte, yte = load_cifar100(seed=0)
        K = 100
    elif name == "synthetic":
        Xtr, ytr, Xte, yte = load_synthetic(k=6, seed=0)
        K = int(jnp.max(ytr)) + 1
    else:
        raise ValueError(f"Unknown dataset {name}")
    return Xtr, ytr, Xte, yte, K


# ---------------------------- Viz helpers ---------------------------- #


def _plot_attack_bars(clean_acc: float, attack_dict: dict, title: str):
    names = ["clean"] + list(attack_dict.keys())
    vals = [clean_acc] + [attack_dict[k] for k in attack_dict]
    plt.figure(figsize=(6.8, 3.6))
    plt.bar(np.arange(len(names)), vals)
    plt.ylim(0, 1.0)
    plt.xticks(np.arange(len(names)), names, rotation=20)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def _plot_pgd_vs_f(f_list: List[int], accs: List[float], title: str):
    plt.figure(figsize=(5.2, 3.6))
    plt.plot(f_list, accs, marker="o")
    plt.xlabel("Number of adversarial clients (f)")
    plt.ylabel("PGD-cw accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_theorem1_scatter(Ps: jnp.ndarray, f: int, title: str, cap: int = 4000):
    """
    Scatter of (sigma_x, margin(mean probits)) with certification boundary.
    Uses at most `cap` points for speed.
    """
    N = Ps.shape[0]
    use = min(N, int(cap))
    Ps_use = Ps[:use]
    n = Ps_use.shape[1]

    Pbar = Ps_use.mean(axis=1)  # (use, K)
    sig = jax.vmap(sigma_x)(Ps_use)  # (use,)
    mar = jax.vmap(margin)(Pbar)  # (use,)

    kap = kappa_const(int(n), int(f))
    C = 2.0 * (np.sqrt(kap * n / (n - f)) + np.sqrt(f / (n - f)))
    certified = mar > C * sig

    xs = np.asarray(sig)
    ys = np.asarray(mar)
    plt.figure(figsize=(5.6, 4.2))
    plt.scatter(xs[~certified], ys[~certified], s=8, alpha=0.35, label="not cert.")
    plt.scatter(xs[certified], ys[certified], s=8, alpha=0.6, label="certified")
    xx = np.linspace(max(1e-8, xs.min()), xs.max() if xs.max() > 0 else 1.0, 200)
    plt.plot(xx, C * xx, "k--", lw=1.25, label="margin = C·σ")
    plt.xlabel(r"$\sigma_x$")
    plt.ylabel(r"margin$(\bar h)$")
    plt.title(title + f"\nCertified rate = {float(certified.mean()):.4f}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------- Main ---------------------------- #


def main(cfg: Config):
    print(f"[config] {cfg}")

    # 1) Load data
    Xtr, ytr, Xte, yte, K = _load_dataset(cfg.dataset)

    # Server slice (10%)
    N = Xtr.shape[0]
    N_server = max(1, int(0.10 * N))
    X_server = Xtr[:N_server]
    y_server = ytr[:N_server]
    X_tr_clients = Xtr[N_server:]
    y_tr_clients = ytr[N_server:]

    # 2) Dirichlet partition
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

    # 3) Train client models (MLP baseline; easy to swap to ResNet/ViT later)
    models, states = train_clients(
        parts,
        d_in=int(Xtr.shape[1]),
        k=K,
        width=cfg.client_width,
        epochs=cfg.client_epochs,
        batch=cfg.client_batch,
        lr=cfg.client_lr,
        wd=cfg.client_wd,
        seed=cfg.seed,
    )

    # 4) Collect probits
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

    # 5) Build & train aggregator
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
            f=cfg.f_adv,
        )
        wrap_cwtm = False

    if cfg.robust_train and cfg.aggregator in {"deepset", "deepset_tm", "linear"}:
        print("[train] robust (adversarial) aggregator training (RERM)")
        best_agg, _ = train_aggregator_robust(
            Ps_tr,
            y_srv_tr,
            Ps_val,
            y_srv_val,
            agg_for_training,
            f=cfg.f_adv,
            epochs=cfg.agg_epochs,
            batch_size=cfg.agg_batch,
            lr=cfg.agg_lr,
            patience=cfg.agg_patience,
            pgd_steps=cfg.pgd_steps,
            pgd_step_size=cfg.pgd_step_size,
            tries_per_batch=cfg.tries_per_batch,
            mask_mode="paper",
            project="softmax",
            key=jr.PRNGKey(cfg.seed + 123),
        )
    else:
        print("[train] ERM aggregator training")
        best_agg, _ = train_aggregator_erm(
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
        )

    agg_eval = DeepSetTM(best_agg, f=cfg.f_adv) if wrap_cwtm else best_agg

    # 6) Eval (subset option)
    if cfg.eval_subset is not None:
        Ps_eval = Ps_te[: cfg.eval_subset]
        y_eval = yte[: cfg.eval_subset]
        print(f"[eval] using subset {Ps_eval.shape[0]} / {Ps_te.shape[0]}")
    else:
        Ps_eval = Ps_te
        y_eval = yte

    clean = aggregator_clean_acc(agg_eval, Ps_eval, y_eval)
    print(f"[eval] clean acc: {clean:.4f}")

    bench = quick_attack_bench(
        agg_eval,
        Ps_eval,
        y_eval,
        f=cfg.f_adv,
        seed=cfg.eval_seed,
        pgd_steps=cfg.pgd_steps,
        pgd_step_size=cfg.pgd_step_size,
        pgd_tries=cfg.eval_pgd_tries,
        pgd_batch_size=cfg.eval_batch_size,
        attacks={
            "pgd_cw": True,  # keep this on by default
            "runnerup": False,
            "cwtm_aware": False,
            "sia_bb": False,
            "sia_wb": False,
            "lma": False,
        },
    )
    print("[eval] attacks:", {k: f"{v:.4f}" for k, v in bench.items()})

    # 7) Visualizations (publication-ready)
    ds_title = f"{cfg.dataset} | n={cfg.n_clients}, f={cfg.f_adv}, α={cfg.dirichlet_alpha} | agg={cfg.aggregator}"
    _plot_attack_bars(clean, bench, title=f"Clean vs Attacks\n{ds_title}")

    # Worst-case PGD-cw vs f
    f_max = min(cfg.f_adv, (cfg.n_clients - 1) // 2)
    f_list = list(range(1, max(1, f_max) + 1))
    if len(f_list) > 0:
        xs, ys = pgd_cw_vs_f(
            agg_eval,
            Ps_eval,
            y_eval,
            f_list,
            steps=cfg.pgd_steps,
            step_size=cfg.pgd_step_size,
            tries=cfg.eval_pgd_tries,
            batch_size=cfg.eval_batch_size,
            seed=cfg.eval_seed,
        )
        _plot_pgd_vs_f(xs, ys, title=f"PGD-cw robustness vs f\n{ds_title}")

    # Theorem-1 certification scatter on probits (no attack)
    if cfg.aggregator.lower() == "cwtm":
        _plot_theorem1_scatter(
            Ps_eval,
            cfg.f_adv,
            title=f"Certification via Theorem 1\n{ds_title}",
            cap=cfg.viz_sample_for_cert,
        )


if __name__ == "__main__":
    # Default: MNIST on Colab (or flip to "synthetic" for quick smoke test)
    cfg = Config(
        dataset="synthetic",
        n_clients=17,
        f_adv=4,
        dirichlet_alpha=0.5,
        client_epochs=8,
        aggregator="deepset_tm",
        robust_train=True,
        agg_epochs=10,
        eval_subset=None,  # set e.g. 2000 for quick CIFAR dev
        eval_pgd_tries=1,
        eval_batch_size=256,
        viz_sample_for_cert=4000,
    )
    main(cfg)
