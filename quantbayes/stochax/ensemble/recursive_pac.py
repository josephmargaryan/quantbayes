"""
recursive_pac_bayes_equinox.py
------------------------------
Recursive PAC-Bayes (Wu et al., NeurIPS 2024) for JAX / Equinox.

Single-stage bounds (t ≥ 2) you may choose:
    "splitkl"         – paper-exact
    "splitbernstein"  – variance-adaptive extension

Stage 1 always uses plain KL.

Author : Joseph Margaryan — updated 2025-06-29
"""

from __future__ import annotations

import math
import copy
from typing import Callable, List, Sequence, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from scipy.optimize import minimize
from sklearn.utils import check_random_state, check_X_y

# ---------------------------------------------------------------------
# helper imports from quantbayes
# ---------------------------------------------------------------------
from quantbayes.stochax.trainer.train import (  # noqa: F401
    train as _train,
    predict as _single_predict,
    binary_loss,
    multiclass_loss,
    regression_loss,
)
from quantbayes.pacbayes.recursive import (
    _kl_inv_plus as _kl_inv_plus,
    SplitKLCriterion,
    SplitBernsteinCriterion,
)

_EPS = 1e-12


# ---------------------------------------------------------------------
# single-stage criteria
# ---------------------------------------------------------------------
class PlainKLCriterion:
    uses_lambda = False

    def compute(self, losses, rho, kl, n, delta, *_):
        p_hat = float(rho @ losses)

        # build the full “kl_term” then pass n explicitly
        kl_term = kl + math.log(2.0 * math.sqrt(n) / delta)
        b = _kl_inv_plus(p_hat, kl_term, n)
        return p_hat, b


_CRIT_MAP = {
    "splitkl": SplitKLCriterion,
    "splitbernstein": SplitBernsteinCriterion,
}


# ---------------------------------------------------------------------
#  main ensemble
# ---------------------------------------------------------------------
class RecursivePACBayesEnsemble:
    """
    Recursive PAC-Bayes for Equinox models.

    Parameters
    ----------
    model_constructors : list[Callable[key] → eqx.Module]
        Each callable returns a *fresh* model when given an rng key.
        The model must implement `__call__(x, key, state) → (logits, new_state)`.
    loss_fn : ignored — kept for backward compatibility (will be chosen automatically)
    optimizer : optax optimizer.
    task : "binary" | "multiclass" | "regression"
    bound_type : "splitkl" | "splitbernstein"
    """

    def __init__(
        self,
        model_constructors: List[Callable[[jr.PRNGKey], eqx.Module]],
        optimizer: optax.GradientTransformation,
        *,
        task: str = "binary",
        bound_type: str = "splitkl",
        delta: float = 0.05,
        T: int = 2,
        gamma_grid: Optional[Sequence[float]] = None,
        seed: int = 0,
        batch_size: int = 128,
        num_epochs: int = 200,
        patience: int = 20,
        verbose: bool = False,
    ):
        if task not in ("binary", "multiclass", "regression"):
            raise ValueError("task must be binary|multiclass|regression")
        if bound_type not in _CRIT_MAP:
            raise ValueError("bound_type must be 'splitkl' or 'splitbernstein'")

        self.ctors = model_constructors
        self.K = len(model_constructors)
        self.opt = optimizer
        self.task = task
        self.bound_type = bound_type
        self.delta = float(delta)
        self.T = int(T)
        self.gamma_grid = (
            np.asarray(gamma_grid, float)
            if gamma_grid is not None
            else np.linspace(0.1, 0.9, 9)
        )
        self.seed = int(seed)
        self.bs, self.epochs, self.patience = batch_size, num_epochs, patience
        self.verbose = verbose
        self._fitted = False

    # ------------------ private helpers ---------------------------
    def _log(self, *msg):
        if self.verbose:
            print(*msg)

    def _split_chunks(self, n: int, rng: np.random.RandomState) -> List[np.ndarray]:
        """Simple nearly-equal split into T chunks."""
        sz = np.full(self.T, n // self.T, int)
        sz[: n % self.T] += 1
        idx = rng.permutation(n)
        return list(np.split(idx, np.cumsum(sz))[:-1])

    # ------------------ fit ---------------------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng_np = check_random_state(self.seed)
        chunks = self._split_chunks(len(y), rng_np)
        S1 = chunks[0]

        # choose loss fn
        loss_fn = {
            "binary": binary_loss,
            "multiclass": multiclass_loss,
            "regression": regression_loss,
        }[self.task]

        # -------- build & train each hypothesis on S1 -------------
        self.models_: List[eqx.Module] = []
        self.states_: List[dict | None] = []
        key = jr.PRNGKey(self.seed)

        for ctor in self.ctors:
            key, sub = jr.split(key)
            model = ctor(sub)
            # initialise state as empty dict so _train never gets None
            init_state = {}

            opt_state = self.opt.init(eqx.filter(model, eqx.is_inexact_array))
            model_tr, state_tr, *_ = _train(
                copy.deepcopy(model),
                init_state,
                opt_state,
                self.opt,
                loss_fn,
                X[S1],
                y[S1],
                X[S1],
                y[S1],
                self.bs,
                self.epochs,
                self.patience,
                sub,
            )
            self.models_.append(model_tr)
            self.states_.append(state_tr)

        # -------- loss matrix  L[h,i] ------------------------------
        self._log("computing loss matrix …")
        L = np.empty((self.K, len(y)))
        key, sub = jr.split(key)
        keys = jr.split(sub, self.K)

        for k, (m, st, kk) in enumerate(zip(self.models_, self.states_, keys)):
            logits = np.array(_single_predict(m, st, X, kk))
            if self.task == "binary":
                preds = (jax.nn.sigmoid(logits).ravel() >= 0.5).astype(int)
                L[k] = (preds != y).astype(float)
            elif self.task == "multiclass":
                preds = jax.nn.softmax(logits, axis=-1).argmax(-1)
                L[k] = (np.array(preds) != y).astype(float)
            else:  # regression
                # scale MSE into [0,1] by dividing by empirical max
                mse = (np.array(logits).ravel() - y) ** 2
                L[k] = mse / max(1e-8, mse.max())

        # prior is uniform
        prior = np.full(self.K, 1.0 / self.K)

        # containers
        self.pi_, self.B_, self.gamma_ = [], [], []

        δ_step = self.delta / self.T
        δ_grid = δ_step / max(1, len(self.gamma_grid))

        # ---------- stage 1 (plain-KL) -----------------------------
        crit1 = PlainKLCriterion()
        ℓ1 = L[:, S1].mean(1)

        def obj1(v):
            w = np.exp(v - v.max())
            w /= w.sum()
            kl = np.sum(w * np.log((w + _EPS) / prior))
            _, b = crit1.compute(ℓ1, w, kl, len(S1), δ_grid)
            return b

        res = minimize(obj1, np.zeros(self.K), method="Nelder-Mead")
        w_prev = np.exp(res.x - res.x.max())
        w_prev /= w_prev.sum()
        kl1 = np.sum(w_prev * np.log((w_prev + _EPS) / prior))
        _, B_prev = crit1.compute(ℓ1, w_prev, kl1, len(S1), δ_step)

        self.pi_.append(w_prev)
        self.B_.append(float(B_prev))
        self.gamma_.append(0.0)
        self._log(f"Stage 1  B₁ = {B_prev:.5f}")

        # ---------- stages t ≥ 2 ----------------------------------
        SplitCrit = _CRIT_MAP[self.bound_type]

        for t in range(2, self.T + 1):
            St, Uval = chunks[t - 1], np.concatenate(chunks[t - 1 :])
            best = (np.inf, None, None)

            for γ in self.gamma_grid:
                crit = SplitCrit(γ)
                b, α = crit.b, crit.alpha

                def make_fhat(idx):
                    loss_prev = w_prev @ L[:, idx]
                    return np.stack(
                        [(L[:, idx] - γ * loss_prev >= thr).mean(1) for thr in b[1:]]
                    )

                F_tr, F_val = make_fhat(St), make_fhat(Uval)

                def obj(v):
                    w = np.exp(v - v.max())
                    w /= w.sum()
                    kl = np.sum(w * np.log((w + _EPS) / (w_prev + _EPS)))
                    _, eps = crit.compute(F_tr, w, kl, len(St), δ_grid)
                    return eps + γ * B_prev

                res = minimize(obj, np.log(w_prev + _EPS), method="Nelder-Mead")
                w_t = np.exp(res.x - res.x.max())
                w_t /= w_t.sum()

                kl_t = np.sum(w_t * np.log((w_t + _EPS) / (w_prev + _EPS)))
                _, eps_val = crit.compute(F_val, w_t, kl_t, len(Uval), δ_step)
                B_t = float(eps_val + γ * B_prev)
                if B_t < best[0]:
                    best = (B_t, γ, w_t)

            B_prev, γ_star, w_prev = best
            self.B_.append(B_prev)
            self.gamma_.append(γ_star)
            self.pi_.append(w_prev)
            self._log(f"Stage {t}  γ={γ_star:.3f}  B_{t}={B_prev:.5f}")

        self._fitted = True
        return self

    # ------------------ inference helpers ------------------------
    def _aggregate_logits(self, X, *, key: jr.PRNGKey):
        """Return stacked logits from each member (numpy array)."""
        keys = jr.split(key, self.K)
        logits_list = [
            np.array(_single_predict(m, st, X, kk))
            for (m, st, kk) in zip(self.models_, self.states_, keys)
        ]
        return np.stack(logits_list)  # shape K × n × ...

    # ------------------ public API -------------------------------
    def predict_proba(self, X, *, seed: int | None = None):
        if not self._fitted:
            raise RuntimeError("call fit() first")
        X = np.asarray(X)
        key = jr.PRNGKey(self.seed if seed is None else seed)
        w = self.pi_[-1]

        logits = self._aggregate_logits(X, key=key)  # K×n×…
        if self.task == "binary":
            p = jax.nn.sigmoid(logits[..., 0])  # K×n
            probs = (w[:, None] * p).sum(0)
            probs = np.clip(probs, 1e-12, 1 - 1e-12)
            return np.stack([1 - probs, probs], axis=-1)

        if self.task == "multiclass":
            p = jax.nn.softmax(logits, axis=-1)  # K×n×C
            probs = (w[:, None, None] * p).sum(0)
            probs = np.clip(probs, 1e-12, 1.0)
            probs /= probs.sum(-1, keepdims=True)
            return probs

        raise AttributeError("predict_proba undefined for regression")

    def predict(self, X, *, seed: int | None = None):
        if self.task == "regression":
            raise NotImplementedError("regression prediction not implemented")
        probs = self.predict_proba(X, seed=seed)
        if self.task == "binary":
            return (probs[:, 1] >= 0.5).astype(int)
        return probs.argmax(-1)

    # ------------------ read-only attrs --------------------------
    @property
    def posterior_weights_(self):
        return self.pi_[-1].copy()

    @property
    def risk_bounds_(self):
        return self.B_.copy()

    @property
    def gammas_(self):
        return self.gamma_.copy()


# ---------------------------------------------------------------------
# smoke-test (binary)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import zero_one_loss

    class Tiny(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, k, d):
            self.lin = eqx.nn.Linear(d, 1, key=k)

        def __call__(self, x, key, state):
            return self.lin(x), state  # state unused

    X, y = make_classification(600, 10, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)

    ctors = [lambda k, i=i: Tiny(jr.fold_in(k, i), Xtr.shape[1]) for i in range(3)]

    ens = RecursivePACBayesEnsemble(
        model_constructors=ctors,
        optimizer=optax.adam(1e-3),
        task="binary",
        bound_type="splitkl",  # try "splitkl" or "splitbernstein"
        delta=0.05,
        T=3,
        seed=0,
        verbose=True,
    ).fit(Xtr, ytr)

    err = zero_one_loss(yte, ens.predict(Xte))
    print("\n=== smoke-test (Equinox) ===")
    print("π_T :", np.round(ens.posterior_weights_, 4))
    print("B_t :", [round(b, 4) for b in ens.risk_bounds_])
    print("γ_t :", ens.gammas_)
    print("0-1 error :", err)
