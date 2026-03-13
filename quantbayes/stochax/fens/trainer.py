# quantbayes/stochax/fens/trainer.py
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib.pyplot as plt

from quantbayes.stochax.trainer.train import (
    train as eqx_train,
    multiclass_loss,
    make_lmt_multiclass_loss,  # allow LMT at aggregator level
    predict as eqx_predict,  # model-level predict (used in __main__ demo only)
)
from quantbayes.stochax.distributed_training.fedavg import FedOptServer
from quantbayes.stochax.robust_inference.data import (
    dirichlet_label_split,
    load_dataset,  # <-- use generic dataset switch
)
from quantbayes.stochax.robust_inference.clients import train_clients
from quantbayes.stochax.robust_inference.eval import aggregator_clean_acc
from quantbayes.stochax.robust_inference.aggregators import MeanAgg  # baseline on probs
from quantbayes.stochax.privacy.dp import DPSGDConfig, rdp_epsilon_for_sgm
from quantbayes.stochax.privacy.dp_train import dp_eqx_train

from quantbayes.stochax.fens.aggregators import make_fens_aggregator
from quantbayes.stochax.utils.lip_upper import network_lipschitz_upper
from quantbayes.stochax.utils.regularizers import (
    global_spectral_norm_penalty,
)  # Σ per-layer σ

Array = jnp.ndarray
PRNG = jax.Array


# ------------------------ helpers ------------------------ #


def _clone_module(m: eqx.Module) -> eqx.Module:
    params, static = eqx.partition(m, eqx.is_inexact_array)
    params_copy = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), params)
    return eqx.combine(params_copy, static)


def _weighted_avg(models: List[eqx.Module], weights: List[float]) -> eqx.Module:
    ps = [eqx.filter(m, eqx.is_inexact_array) for m in models]
    w = jnp.asarray(weights, dtype=jnp.float32)

    def wsum(*leaves):
        return sum(wi * leaf for wi, leaf in zip(w, leaves))

    avg = jax.tree_util.tree_map(wsum, *ps)
    _, static = eqx.partition(models[0], eqx.is_inexact_array)
    return eqx.combine(avg, static)


def collect_outputs_dataset(
    models: List[eqx.Module],
    states: List[Any],
    X: Array,
    *,
    batch_size: int = 512,
    key: Optional[PRNG] = None,
) -> Array:
    """
    Returns (N, n_clients, K) of client *logits*.
    """
    n = len(models)
    N = int(X.shape[0])
    chunks: List[Array] = []
    for start in range(0, N, batch_size):
        xb = X[start : start + batch_size]  # (B, d)
        per_client: List[Array] = []
        for i, m in enumerate(models):
            k_i = jr.fold_in(
                jr.PRNGKey(0) if key is None else key, i * 1_000_003 + start
            )
            logits = eqx_predict(m, states[i], xb, k_i)  # (B, K), pre-softmax
            per_client.append(logits)
        chunks.append(jnp.stack(per_client, axis=1))  # (B, n_clients, K)
    return jnp.concatenate(chunks, axis=0)  # (N, n_clients, K)


# ------------------------ FENS aggregator FL ------------------------ #


class FENSAggregatorFLTrainerEqx:
    """
    FENS Phase 2: FL training of a light aggregator f_λ on *frozen* client models.
    Only λ is communicated. Server optionally uses FedOpt on deltas.

    Knobs:
      • agg_loss: "ce" or "lmt" to swap in LMT multiclass loss.
      • lmt_kwargs: forwarded to make_lmt_multiclass_loss(eps=..., alpha=..., conv_*...).
      • spec_reg: spectral regularization knobs applied on aggregator locals (non-DP).
          {
            "lambda_spec": 0.0,
            "lambda_frob": 0.0,
            "lambda_specnorm": 0.0,
            "lambda_sob_jac": 0.0,
            "lambda_sob_kernel": 0.0,
            "lambda_liplog": 0.0,
            "specnorm_conv_mode": "tn",
            "specnorm_conv_tn_iters": 8,
            "specnorm_conv_gram_iters": 5,
            "specnorm_conv_fft_shape": None,
            "specnorm_conv_input_shape": None,
            "lip_conv_mode": "tn",
            "lip_conv_tn_iters": 8,
            "lip_conv_gram_iters": 5,
            "lip_conv_fft_shape": None,
            "lip_conv_input_shape": None,
          }
      • bound logging (optional) per-epoch on locals and global-after-server:
          bound_log_every, bound_conv_mode, bound_tn_iters, bound_gram_iters, bound_fft_shape, bound_input_shape

      If you attach `self.bound_recorder = BoundLogger()`, the trainer will emit:
        - local logs via `eqx_train(..., bound_recorder=_rec)` with {"epoch","L_raw","L_eval","mode","round","client"}
        - global log (post aggregation) with {"round","client":-1,"epoch":0,"L_raw","L_eval","mode","sigma_sum"}
        - local Σσ (after each local update) with {"round","client", "epoch":E, "sigma_sum", "type":"local"}
    """

    def __init__(
        self,
        aggregator_init_fn: Callable[[PRNG], eqx.Module],
        n_clients: int,
        *,
        outer_rounds: int = 150,
        inner_epochs: int = 1,
        batch_size: int = 256,
        local_lr: float = 1e-3,
        local_weight_decay: float = 0.0,
        patience: int = 3,
        server_opt: Optional[Dict[str, float | str]] = None,  # FedOpt on deltas
        dp_config: Optional[DPSGDConfig] = None,  # Optional DP on local steps
        key: Optional[PRNG] = None,
        agg_loss: str = "ce",  # "ce" | "lmt"
        lmt_kwargs: Optional[Dict[str, Any]] = None,
        spec_reg: Optional[Dict[str, Any]] = None,
        # bound logging during local/global (optional)
        bound_log_every: Optional[int] = None,
        bound_conv_mode: str = "tn",
        bound_tn_iters: int = 8,
        bound_gram_iters: int = 5,
        bound_fft_shape: Optional[Tuple[int, int]] = None,
        bound_input_shape: Optional[Tuple[int, int]] = None,
        apply_spec_in_dp: bool = False,
    ):
        self.make_agg = aggregator_init_fn
        self.M = int(n_clients)
        self.R = int(outer_rounds)
        self.E = int(inner_epochs)
        self.B = int(batch_size)
        self.lr = float(local_lr)
        self.wd = float(local_weight_decay)
        self.patience = int(patience)
        self.key = jr.PRNGKey(0) if key is None else key

        so = server_opt or {
            "name": "adam",
            "lr": 1e-2,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        }
        self.server = (
            FedOptServer(**so)
            if so.get("name", "none").lower() in {"adam", "yogi", "adagrad", "momentum"}
            else None
        )

        self.local_opt = optax.adamw(learning_rate=self.lr, weight_decay=self.wd)
        self.dp_config = dp_config

        # global aggregator init
        self.key, sub = jr.split(self.key)
        self.global_agg = self.make_agg(sub)

        # loss selection & knobs
        self.agg_loss = str(agg_loss).lower().strip()
        self.lmt_kwargs = {} if lmt_kwargs is None else dict(lmt_kwargs)

        # spectral regularization knobs
        _sr = {} if spec_reg is None else dict(spec_reg)
        self.spec_reg = {
            "lambda_spec": float(_sr.get("lambda_spec", 0.0)),
            "lambda_frob": float(_sr.get("lambda_frob", 0.0)),
            "lambda_specnorm": float(_sr.get("lambda_specnorm", 0.0)),
            "lambda_sob_jac": float(_sr.get("lambda_sob_jac", 0.0)),
            "lambda_sob_kernel": float(_sr.get("lambda_sob_kernel", 0.0)),
            "lambda_liplog": float(_sr.get("lambda_liplog", 0.0)),
            "specnorm_conv_mode": _sr.get("specnorm_conv_mode", "tn"),
            "specnorm_conv_tn_iters": int(_sr.get("specnorm_conv_tn_iters", 8)),
            "specnorm_conv_gram_iters": int(_sr.get("specnorm_conv_gram_iters", 5)),
            "specnorm_conv_fft_shape": _sr.get("specnorm_conv_fft_shape", None),
            "specnorm_conv_input_shape": _sr.get("specnorm_conv_input_shape", None),
            "lip_conv_mode": _sr.get("lip_conv_mode", "tn"),
            "lip_conv_tn_iters": int(_sr.get("lip_conv_tn_iters", 8)),
            "lip_conv_gram_iters": int(_sr.get("lip_conv_gram_iters", 5)),
            "lip_conv_fft_shape": _sr.get("lip_conv_fft_shape", None),
            "lip_conv_input_shape": _sr.get("lip_conv_input_shape", None),
        }
        self.apply_spec_in_dp = bool(apply_spec_in_dp)

        # bound logging knobs
        self.bound_log_every = bound_log_every
        self.bound_conv_mode = bound_conv_mode
        self.bound_tn_iters = int(bound_tn_iters)
        self.bound_gram_iters = int(bound_gram_iters)
        self.bound_fft_shape = bound_fft_shape
        self.bound_input_shape = bound_input_shape

        # external sink: set trainer.bound_recorder = BoundLogger() in your script if you want logs
        self.bound_recorder: Optional[Callable[[Dict[str, Any]], None]] = None

    # ------------------------ evaluation helpers ------------------------ #

    def _make_local_loss_fn(self) -> Callable:
        if self.agg_loss == "lmt":
            kwargs = {
                "eps": 1.0,
                "alpha": 1.0,
                "conv_mode": "tn",
                "conv_tn_iters": 8,
                "conv_gram_iters": 5,
                "conv_fft_shape": None,
                "conv_input_shape": None,
                "stop_grad_L": True,
            }
            kwargs.update(self.lmt_kwargs)
            return make_lmt_multiclass_loss(**kwargs)
        return multiclass_loss

    def _eval_ce(self, model: eqx.Module, X: Array, y: Array) -> float:
        def one(P):
            logits, _ = model(P, None, None)
            return logits

        logits = jax.vmap(one)(X)  # (N,K)
        ce = optax.softmax_cross_entropy_with_integer_labels(
            logits, y.astype(jnp.int32)
        )
        return float(jnp.mean(ce))

    def _eval_acc(self, model: eqx.Module, X: Array, y: Array) -> float:
        return aggregator_clean_acc(model, X, y)

    # ------------------------ main train loop ------------------------ #
    def train(
        self,
        Ps_parts: List[Array],
        y_parts: List[Array],
        Ps_test: Array,
        y_test: Array,
    ) -> Tuple[eqx.Module, Dict[str, List[float]]]:
        """
        Federated training of aggregator parameters on *per-client* meta-datasets.
        Ps_parts[i]: (N_i, M, K), y_parts[i]: (N_i,)
        Ps_test: (N_test, M, K), y_test: (N_test,)
        """
        assert len(Ps_parts) == self.M and len(y_parts) == self.M

        # client sizes and FedAvg weights
        sizes = [int(Pi.shape[0]) for Pi in Ps_parts]
        total = float(sum(sizes))
        weights = [s / (total + 1e-12) for s in sizes]

        # DP trackers
        steps_cum = [0 for _ in range(self.M)]
        eps_cum: List[float] = [0.0 for _ in range(self.M)]
        dp_eps_max_hist: List[float] = []
        dp_eps_mean_hist: List[float] = []

        test_ce: List[float] = []
        test_acc: List[float] = []

        def _make_loss():
            return self._make_local_loss_fn()

        for rnd in range(1, self.R + 1):
            local_models: List[eqx.Module] = []

            # Local FL step on each client's meta-data (Ps_i, y_i)
            for i in range(self.M):
                lm = _clone_module(self.global_agg)
                opt_state = self.local_opt.init(eqx.filter(lm, eqx.is_inexact_array))
                loss_fn = _make_loss()

                if self.dp_config is None:
                    # recorder wrapper (add round & client)
                    rec_cb = self.bound_recorder
                    if (
                        (self.bound_log_every is not None)
                        and (self.bound_log_every > 0)
                        and (rec_cb is not None)
                    ):

                        def _rec(rec, r=rnd, c=i, cb=rec_cb):
                            rec = dict(rec)
                            rec["round"] = int(r)
                            rec["client"] = int(c)
                            cb(rec)

                    else:
                        _rec = None

                    lm_trained, _, *_ = eqx_train(
                        model=lm,
                        state=None,
                        opt_state=opt_state,
                        optimizer=self.local_opt,
                        loss_fn=loss_fn,
                        X_train=Ps_parts[i],
                        y_train=y_parts[i].astype(jnp.int32),
                        X_val=Ps_parts[i],
                        y_val=y_parts[i].astype(jnp.int32),
                        batch_size=min(self.B, max(1, Ps_parts[i].shape[0])),
                        num_epochs=self.E,
                        patience=self.patience,
                        key=jr.PRNGKey(10_000 + rnd * 97 + i),
                        # penalties
                        lambda_spec=self.spec_reg["lambda_spec"],
                        lambda_frob=self.spec_reg["lambda_frob"],
                        lambda_specnorm=self.spec_reg["lambda_specnorm"],
                        lambda_sob_jac=self.spec_reg["lambda_sob_jac"],
                        lambda_sob_kernel=self.spec_reg["lambda_sob_kernel"],
                        lambda_liplog=self.spec_reg["lambda_liplog"],
                        # Σσ penalty config
                        specnorm_conv_mode=self.spec_reg["specnorm_conv_mode"],
                        specnorm_conv_tn_iters=self.spec_reg["specnorm_conv_tn_iters"],
                        specnorm_conv_gram_iters=self.spec_reg[
                            "specnorm_conv_gram_iters"
                        ],
                        specnorm_conv_fft_shape=self.spec_reg[
                            "specnorm_conv_fft_shape"
                        ],
                        specnorm_conv_input_shape=self.spec_reg[
                            "specnorm_conv_input_shape"
                        ],
                        # Lip product penalty config
                        lip_conv_mode=self.spec_reg["lip_conv_mode"],
                        lip_conv_tn_iters=self.spec_reg["lip_conv_tn_iters"],
                        lip_conv_gram_iters=self.spec_reg["lip_conv_gram_iters"],
                        lip_conv_fft_shape=self.spec_reg["lip_conv_fft_shape"],
                        lip_conv_input_shape=self.spec_reg["lip_conv_input_shape"],
                        # Optional: per-epoch certified global bound on the *local* aggregator
                        log_global_bound_every=self.bound_log_every,
                        bound_conv_mode=self.bound_conv_mode,
                        bound_tn_iters=self.bound_tn_iters,
                        bound_gram_iters=self.bound_gram_iters,
                        bound_fft_shape=self.bound_fft_shape,
                        bound_input_shape=self.bound_input_shape,
                        bound_recorder=_rec,
                    )

                    # --- NEW: record Σσ for the local aggregator after its epoch ---
                    if (
                        (self.bound_log_every is not None)
                        and (self.bound_log_every > 0)
                        and (self.bound_recorder is not None)
                    ):
                        sigma_sum_local = float(
                            global_spectral_norm_penalty(lm_trained, conv_mode="tn")
                        )
                        self.bound_recorder(
                            {
                                "round": int(rnd),
                                "client": int(i),
                                "epoch": int(self.E),
                                "sigma_sum": sigma_sum_local,
                                "type": "local",
                            }
                        )

                else:
                    # DP path (minimal)
                    dp_kwargs_base = dict(
                        model=lm,
                        state=None,
                        opt_state=opt_state,
                        optimizer=self.local_opt,
                        loss_fn=loss_fn,
                        X_train=Ps_parts[i],
                        y_train=y_parts[i].astype(jnp.int32),
                        X_val=Ps_parts[i],
                        y_val=y_parts[i].astype(jnp.int32),
                        dp_config=self.dp_config,
                        batch_size=min(self.B, max(1, Ps_parts[i].shape[0])),
                        num_epochs=self.E,
                        patience=self.patience,
                        key=jr.PRNGKey(20_000 + rnd * 101 + i),
                    )
                    lm_trained, _, *_ = dp_eqx_train(**dp_kwargs_base)

                    # DP accounting
                    n_i = max(1, int(Ps_parts[i].shape[0]))
                    steps_this = max(1, int(jnp.ceil(n_i / max(1, self.B)))) * self.E
                    steps_cum[i] += steps_this
                    if (self.dp_config.sampling_rate is not None) and (
                        self.dp_config.sampling_rate > 0
                    ):
                        q_i = float(min(1.0, self.dp_config.sampling_rate))
                    else:
                        q_i = float(min(1.0, self.B / float(n_i)))
                    eps_cum[i] = rdp_epsilon_for_sgm(
                        q=q_i,
                        sigma=float(self.dp_config.noise_multiplier),
                        steps=int(steps_cum[i]),
                        delta=float(self.dp_config.delta),
                    )

                local_models.append(lm_trained)

            # --- Server aggregation over λ ---
            if self.server is not None:
                self.global_agg = self.server.apply(
                    self.global_agg, local_models, weights
                )
            else:
                self.global_agg = _weighted_avg(local_models, weights)

            # --- NEW: global bound + Σσ after aggregation, once per round ---
            if (
                (self.bound_log_every is not None)
                and (self.bound_log_every > 0)
                and (self.bound_recorder is not None)
            ):
                Lg = float(
                    network_lipschitz_upper(
                        self.global_agg,
                        state=None,
                        conv_mode=self.bound_conv_mode,
                        conv_tn_iters=self.bound_tn_iters,
                        conv_gram_iters=self.bound_gram_iters,
                        conv_fft_shape=self.bound_fft_shape,
                        conv_input_shape=self.bound_input_shape,
                    )
                )
                sigma_sum_global = float(
                    global_spectral_norm_penalty(self.global_agg, conv_mode="tn")
                )
                self.bound_recorder(
                    {
                        "round": int(rnd),
                        "client": -1,
                        "epoch": 0,
                        "L_raw": Lg,
                        "L_eval": Lg,
                        "mode": self.bound_conv_mode,
                        "sigma_sum": sigma_sum_global,
                        "type": "global",
                    }
                )

            # Evaluate
            ce = self._eval_ce(self.global_agg, Ps_test, y_test)
            acc = self._eval_acc(self.global_agg, Ps_test, y_test)
            test_ce.append(ce)
            test_acc.append(acc)

            # Optional DP summary
            if self.dp_config is not None:
                eps_max = float(max(eps_cum)) if eps_cum else 0.0
                eps_mean = float(jnp.mean(jnp.asarray(eps_cum))) if eps_cum else 0.0
                dp_eps_max_hist.append(eps_max)
                dp_eps_mean_hist.append(eps_mean)
                print(
                    f"[FENS|DP] round {rnd:03d}/{self.R} | ε_max={eps_max:.3f}  ε_mean={eps_mean:.3f}  (δ={self.dp_config.delta:g})"
                )

            if (rnd % max(1, self.R // 10) == 0) or rnd == self.R:
                print(
                    f"[FENS] round {rnd}/{self.R} | test CE={ce:.4f} | acc={acc*100:.2f}%"
                )

        out = {"ce": test_ce, "acc": test_acc}
        if self.dp_config is not None:
            out.update({"dp_eps_max": dp_eps_max_hist, "dp_eps_mean": dp_eps_mean_hist})
        return self.global_agg, out


# ---------------------- Demo (modern knobs) ---------------------- #
if __name__ == "__main__":
    """
    Minimal modern demo:
      - dataset switch via load_dataset(...)
      - LMT toggled on
      - spectral regularization toggled on
      - optional per-round global bound logging (L)
    """
    OUT_DIR = "runs/fens_demo"
    import os

    os.makedirs(OUT_DIR, exist_ok=True)

    SEED = 0
    DATASET = "synthetic"  # "synthetic" | "mnist" | "cifar-10" | "cifar-100"
    N_CLIENTS = 6
    ALPHA = 0.8
    CLIENT_WIDTH = 256
    CLIENT_EPOCHS = 6
    CLIENT_BATCH = 256

    AGG_NAME = "mlp"
    AGG_HIDDEN = 128

    FENS_ROUNDS = 40
    FENS_LOCAL_EPOCHS = 1
    FENS_BATCH = 256

    # toggles
    USE_LMT = True
    USE_SPEC_REG = True
    BOUND_LOG_EVERY = 1  # set None to disable bound logging

    # data
    Xtr, ytr, Xte, yte, N_CLASSES = load_dataset(DATASET)

    parts = dirichlet_label_split(Xtr, ytr, N_CLIENTS, N_CLASSES, ALPHA, seed=SEED)
    models, states = train_clients(
        parts,
        d_in=Xtr.shape[1],
        k=N_CLASSES,
        width=CLIENT_WIDTH,
        epochs=CLIENT_EPOCHS,
        batch=CLIENT_BATCH,
        lr=1e-3,
        wd=1e-4,
        seed=SEED,
        X_val=Xte[:1024],
        y_val=yte[:1024],
    )

    # meta-sets
    Ps_parts: List[Array] = []
    y_parts: List[Array] = []
    for i, (Xi, yi) in enumerate(parts):
        P_i = collect_outputs_dataset(
            models, states, Xi, batch_size=512, key=jr.PRNGKey(SEED + 100 + i)
        )
        Ps_parts.append(P_i)
        y_parts.append(yi.astype(jnp.int32))
    Ps_test = collect_outputs_dataset(
        models, states, Xte, batch_size=512, key=jr.PRNGKey(SEED + 999)
    )
    y_test = yte.astype(jnp.int32)

    # baseline
    Ps_test_probs = jax.nn.softmax(Ps_test, axis=-1)
    mean_agg = MeanAgg(K=N_CLASSES)
    mean_acc = aggregator_clean_acc(mean_agg, Ps_test_probs, y_test)

    # agg init
    def agg_init(k):
        return make_fens_aggregator(
            AGG_NAME, n_clients=N_CLIENTS, n_classes=N_CLASSES, key=k, hidden=AGG_HIDDEN
        )

    # spec-reg
    SPEC_REG = dict(
        lambda_spec=0.0,
        lambda_frob=0.0,
        lambda_specnorm=1e-4 if USE_SPEC_REG else 0.0,
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

    # trainer
    trainer = FENSAggregatorFLTrainerEqx(
        agg_init,
        n_clients=N_CLIENTS,
        outer_rounds=FENS_ROUNDS,
        inner_epochs=FENS_LOCAL_EPOCHS,
        batch_size=FENS_BATCH,
        local_lr=1e-3,
        local_weight_decay=0.0,
        patience=2,
        server_opt={
            "name": "adam",
            "lr": 1e-2,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
        dp_config=None,
        key=jr.PRNGKey(SEED + 123),
        agg_loss=("lmt" if USE_LMT else "ce"),
        lmt_kwargs=dict(
            eps=0.5,
            alpha=1.0,
            conv_mode="tn",
            conv_tn_iters=8,
            conv_gram_iters=5,
            stop_grad_L=True,
        ),
        spec_reg=SPEC_REG,
        bound_log_every=BOUND_LOG_EVERY,
        bound_conv_mode="tn",
        bound_tn_iters=8,
        bound_gram_iters=5,
        bound_fft_shape=None,
        bound_input_shape=None,
        apply_spec_in_dp=False,
    )

    # simple logger (reuse the one in your generic trainer if you like)
    class _Logger:
        def __init__(self):
            self.data = []

        def __call__(self, rec):
            out = {}
            for k, v in rec.items():
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = v
            self.data.append(out)

    logger = _Logger() if BOUND_LOG_EVERY else None
    if logger is not None:
        trainer.bound_recorder = logger

    agg_model, hist = trainer.train(Ps_parts, y_parts, Ps_test, y_test)
    final_acc = aggregator_clean_acc(agg_model, Ps_test, y_test)

    print(f"\n[FENS] Mean(probs) baseline acc = {mean_acc*100:.2f}%")
    print(f"[FENS] FENS({AGG_NAME}) final acc = {final_acc*100:.2f}%")

    # curves
    rounds = jnp.arange(1, len(hist["ce"]) + 1)
    fig = plt.figure(figsize=(8.8, 3.8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(rounds, hist["ce"])
    ax1.set_title("FENS Test CE vs Rounds")
    ax1.set_xlabel("round")
    ax1.set_ylabel("cross-entropy")
    ax1.axhline(jnp.log(N_CLASSES), ls="--", lw=1.0, label="uniform log(K)")
    ax1.legend()
    ax2.plot(rounds, jnp.asarray(hist["acc"]) * 100.0, label=f"FENS ({AGG_NAME})")
    ax2.axhline(mean_acc * 100.0, ls="--", label="Mean(probs) baseline")
    ax2.set_title("FENS Test Accuracy vs Rounds")
    ax2.set_xlabel("round")
    ax2.set_ylabel("accuracy (%)")
    ax2.legend()
    fig.tight_layout()
    figpath = os.path.join(OUT_DIR, "fens_curves.png")
    fig.savefig(figpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[FENS] saved plots → {figpath}")

    # optional bound plots
    if logger is not None and logger.data:
        recs = logger.data
        # global L
        gL = sorted(
            [
                (r["round"], r["L_raw"])
                for r in recs
                if r.get("client") == -1 and "L_raw" in r
            ]
        )
        if gL:
            xs, ys = zip(*gL)
            plt.figure(figsize=(6, 3))
            plt.plot(xs, ys, marker="o")
            plt.xlabel("round")
            plt.ylabel("global L")
            plt.title("Global L per round")
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUT_DIR, "fens_global_L.png"), dpi=220, bbox_inches="tight"
            )
            plt.close()
        # local Σσ band
        bS = {}
        for r in recs:
            if (r.get("client", 0) >= 0) and ("sigma_sum" in r):
                bS.setdefault(r["round"], []).append(r["sigma_sum"])
        if bS:
            xs = sorted(bS.keys())
            means = [float(jnp.mean(jnp.asarray(bS[x]))) for x in xs]
            lo = [float(jnp.min(jnp.asarray(bS[x]))) for x in xs]
            hi = [float(jnp.max(jnp.asarray(bS[x]))) for x in xs]
            plt.figure(figsize=(6, 3))
            plt.plot(xs, means)
            plt.fill_between(xs, lo, hi, alpha=0.2)
            plt.xlabel("round")
            plt.ylabel("Σσ (local)")
            plt.title("Local Σσ per round")
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUT_DIR, "fens_local_sigma_sum.png"),
                dpi=220,
                bbox_inches="tight",
            )
            plt.close()
