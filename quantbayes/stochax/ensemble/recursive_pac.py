# recursive_pac_bayes_equinox.py

import math
import copy
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from typing import List, Callable, Tuple

from scipy.optimize import minimize
from sklearn.utils import check_random_state

# Import the same PAC‐Bayes bound functions as in recursive_pac_bayes.py
from quantbayes.pacbayes.recursive import (
    compute_plain_kl_bound,
    compute_split_kl_bound,
    compute_empirical_bernstein_bound,
    compute_unexpected_bernstein_bound,
)

from quantbayes.stochax.trainer.train import (
    train,
    binary_loss,
    multiclass_loss,
    regression_loss,
)
from quantbayes.stochax.trainer.train import predict as _single_predict

CRITERIA = {
    "plain-kl",
    "split-kl",
    "emp-bernstein",
    "unexp-bernstein",
}


class RecursivePACBayesEnsemble:
    """
    Recursive PAC‐Bayes ensemble wrapper for Equinox‐based modules,
    reusing the existing `train` and `_single_predict` functions.

    Each constructor must produce an Equinox model. We train each model
    once on the entire (X_train, y_train) using `train`. After that, we
    freeze all models and adjust only the discrete mixture weights via
    the recursive PAC-Bayes bound procedure.

    Parameters
    ----------
    model_constructors : List[Callable[[jax.random.PRNGKey], eqx.Module]]
        Each callable accepts a JAX PRNGKey and returns an Equinox Module.
    task : {"binary", "multiclass", "regression"}
    loss_fn : Callable
        One of {binary_loss, multiclass_loss, regression_loss}.
    optimizer : optax.GradientTransformation
        The optimizer used during `train()`.
    bound_type : {"plain-kl","split-kl","emp-bernstein","unexp-bernstein"}
    delta : float ∈ (0,1)
        Overall failure probability for the union‐bound over T stages.
    T : int ≥ 1
        Number of recursive stages.
    gamma_grid : 1D numpy array of candidates in (0,1)
        If None: defaults to np.linspace(0.1, 0.9, 9).
    lambda_grid : 1D numpy array of candidates in (0,1) for unexpected‐Bernstein
        If None: defaults to np.linspace(1e-4, 0.9999, 200) but will be clipped per b=1+γ.
    seed : int
        Global random seed for splitting and for model RNG.
    L_max : float
        In regression mode, maximum possible loss (for clipping).
    batch_size, num_epochs, patience : training settings for each weak learner.
    """

    def __init__(
        self,
        model_constructors: List[Callable[[jax.random.PRNGKey], eqx.Module]],
        task: str,
        loss_fn: Callable,
        optimizer: optax.GradientTransformation,
        bound_type: str = "split-kl",
        delta: float = 0.05,
        T: int = 2,
        gamma_grid: np.ndarray = None,
        lambda_grid: np.ndarray = None,
        seed: int = 0,
        L_max: float = 1.0,
        batch_size: int = 64,
        num_epochs: int = 200,
        patience: int = 20,
    ):
        if bound_type not in CRITERIA:
            raise ValueError(f"Unknown bound_type '{bound_type}'.")
        if task not in ("binary", "multiclass", "regression"):
            raise ValueError("`task` must be 'binary', 'multiclass', or 'regression'.")
        if task == "regression" and bound_type == "split-kl":
            raise ValueError(
                "`split-kl` is only valid for finite‐valued losses; use a Bernstein or plain‐kl bound for regression."
            )

        self.model_constructors = model_constructors
        self.K = len(model_constructors)
        self.task = task
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.bound_type = bound_type
        self.delta = delta
        self.T = T
        self.gamma_grid = (
            gamma_grid if gamma_grid is not None else np.linspace(0.1, 0.9, 9)
        )
        self.lambda_grid = (
            lambda_grid if lambda_grid is not None else np.linspace(1e-4, 0.9999, 200)
        )
        self.seed = seed
        self.L_max = L_max

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience

        # Will be populated in .fit()
        self.trained_models_states_: List[Tuple[eqx.Module, dict]] = []
        self.pi_list_: List[np.ndarray] = []
        self.bounds_: List[float] = []
        self.chunk_indices_: List[np.ndarray] = []
        self.is_fitted = False

    def _geometric_split_indices(self, n: int) -> List[np.ndarray]:
        """
        Split indices {0,...,n-1} into T geometric chunks S₁,...,S_T with sizes ~1,2,4,...,n/2.
        """
        sizes = []
        rem = n
        for t in range(1, self.T + 1):
            if t == self.T:
                sizes.append(rem)
            else:
                size_t = min(rem - (self.T - t), 2 ** (t - 1))
                sizes.append(size_t)
                rem -= size_t
        total = sum(sizes)
        if total != n:
            sizes[-1] += n - total

        idx = np.arange(n)
        rs = check_random_state(self.seed)
        rs.shuffle(idx)

        chunks = []
        start = 0
        for s in sizes:
            chunks.append(idx[start : start + s])
            start += s
        return chunks

    def _compute_loss_matrix(
        self,
        models_states: List[Tuple[eqx.Module, dict]],
        X: np.ndarray,
        y: np.ndarray,
        rng: jax.random.PRNGKey,
    ) -> np.ndarray:
        """
        Given a list of K trained (model, state) tuples, compute a (K, n) matrix of 0-1 or clipped
        regression losses for (X,y) on each model, using `_single_predict`.
        """
        n = X.shape[0]
        K = len(models_states)
        loss_matrix = np.zeros((K, n), dtype=float)
        rngs = jr.split(rng, num=K + 1)[1:]  # one key per model

        for i, (model, state) in enumerate(models_states):
            preds = _single_predict(model, state, X, rngs[i])
            if self.task == "binary":
                # preds are logits; apply sigmoid, threshold at 0.5
                probs = jax.nn.sigmoid(preds).ravel()
                pred_labels = (np.array(probs) >= 0.5).astype(int)
                loss_matrix[i, :] = (pred_labels != y).astype(float)

            elif self.task == "multiclass":
                probs = jax.nn.softmax(preds, axis=-1)
                pred_labels = np.array(jnp.argmax(probs, axis=-1))
                loss_matrix[i, :] = (pred_labels != y).astype(float)

            else:  # regression
                preds_np = np.array(preds).ravel()
                mse = (preds_np - y) ** 2
                clipped = np.minimum(mse, self.L_max) / self.L_max
                loss_matrix[i, :] = clipped

        return loss_matrix

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "RecursivePACBayesEnsemble":
        """
        Fit the Recursive PAC‐Bayes ensemble:
          1) Geometric split of data into T chunks.
          2) Train each Equinox model once on the entire (X,y), via `train()`.
          3) Build the full (K, n) loss matrix.
          4) Stage 1: plain PAC‐Bayes‐KL on S₁.
          5) Stages 2..T: chosen bound on Uₜ = Sₜ ∪ ... ∪ S_T.
        """
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y).astype(
            int if self.task in ("binary", "multiclass") else float
        )
        n = X_np.shape[0]

        # 1) Geometric split
        chunks = self._geometric_split_indices(n)
        self.chunk_indices_ = chunks

        # 2) Train each Equinox model on full data using `train()`
        self.trained_models_states_ = []
        rng = jr.PRNGKey(self.seed)
        for ctor in self.model_constructors:
            rng, subkey = jr.split(rng)
            # Initialize model and state
            model_init_fn = lambda key: ctor(key)
            model, state = eqx.nn.make_with_state(model_init_fn)(subkey)
            # Prepare optimizer state
            opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))
            # Train on (X_np, y_np) with itself as hold‐out (no separate hold‐out split)
            best_model, best_state, _, _ = train(
                model,
                state,
                opt_state,
                self.optimizer,
                self.loss_fn,
                X_np,
                y_np,
                X_np,
                y_np,
                self.batch_size,
                self.num_epochs,
                self.patience,
                subkey,
            )
            self.trained_models_states_.append((best_model, best_state))

        # 3) Build full loss_matrix_all of shape (K, n)
        rng, loss_key = jr.split(rng)
        loss_matrix_all = self._compute_loss_matrix(
            self.trained_models_states_, X_np, y_np, loss_key
        )

        # ===== Stage 1 (plain PAC‐Bayes‐KL) =====
        S1_idx = chunks[0]
        loss_S1 = loss_matrix_all[:, S1_idx].mean(axis=1)  # (K,)
        n1 = len(S1_idx)
        delta_t = self.delta / self.T

        def objective_stage1(v: np.ndarray) -> float:
            exp_v = np.exp(v - np.max(v))
            w = exp_v / exp_v.sum()
            p_mean = float(np.dot(w, loss_S1))
            KL_w = float(np.sum(w * np.log(w * self.K + 1e-12)))
            return compute_plain_kl_bound(p_mean, KL_w, delta_t, n1)

        v0 = np.zeros(self.K)
        res1 = minimize(
            objective_stage1,
            v0,
            method="Nelder-Mead",
            options={"maxiter": 500, "disp": False},
        )
        v_opt1 = res1.x
        exp_v1 = np.exp(v_opt1 - np.max(v_opt1))
        pi1 = exp_v1 / exp_v1.sum()

        # Recompute B1
        p_mean_final = float(np.dot(pi1, loss_S1))
        KL_final1 = float(np.sum(pi1 * np.log(pi1 * self.K + 1e-12)))
        B1 = compute_plain_kl_bound(p_mean_final, KL_final1, delta_t, n1)

        self.pi_list_ = [pi1]
        self.bounds_ = [B1]

        # ===== Stages 2..T (chosen bound) =====
        B_prev = B1
        pi_prev = pi1.copy()

        # Precompute Uₜ indices = union of chunks[t-1 : T]
        future_indices = []
        for t in range(self.T):
            all_future = np.concatenate(chunks[t:], axis=0)
            future_indices.append(all_future)

        for t in range(2, self.T + 1):
            S_t_idx = chunks[t - 1]
            U_t_idx = future_indices[t - 1]
            U_t = loss_matrix_all[:, U_t_idx]  # shape (K, |U_t|)
            m_t = U_t.shape[1]
            delta_t = self.delta / self.T

            # Precompute weighted loss of π_{t-1} on U_t
            weighted_prev_U = pi_prev @ U_t  # shape (|U_t|,)

            def evaluate_for_gamma(gamma: float) -> Tuple[np.ndarray, float]:
                # Compute f_gamma_U = loss_U - γ * weighted_prev_U
                f_gamma_U = U_t - gamma * weighted_prev_U  # shape (K, m_t)

                if self.bound_type == "split-kl":
                    b0 = -gamma
                    b_vals = np.array([b0, 0.0, 1.0 - gamma, 1.0])
                    alpha_vals = np.diff(b_vals)  # [γ, 1-γ, γ]
                    indicator = np.zeros((3, self.K, m_t), dtype=float)
                    for j in range(1, 4):
                        indicator[j - 1] = (f_gamma_U >= b_vals[j]).astype(float)
                    Fhat = indicator.mean(axis=2)  # shape (3, K)

                elif self.bound_type == "plain-kl":
                    f_gamma_means = f_gamma_U.mean(axis=1)  # (K,)

                elif self.bound_type == "emp-bernstein":
                    f_gamma_means = f_gamma_U.mean(axis=1)  # (K,)
                    var_i = f_gamma_U.var(axis=1, ddof=1)  # sample var
                    mu_i = f_gamma_means.copy()
                    v_i = var_i.copy()

                elif self.bound_type == "unexp-bernstein":
                    f_gamma_means = f_gamma_U.mean(axis=1)  # (K,)
                    s_i = (f_gamma_U * f_gamma_U).mean(axis=1)  # (K,)
                    mu_i = f_gamma_means.copy()

                else:
                    raise ValueError(f"Unknown bound_type='{self.bound_type}'")

                def objective_t(v: np.ndarray) -> float:
                    exp_v = np.exp(v - np.max(v))
                    w = exp_v / exp_v.sum()
                    KL_div = float(np.sum(w * np.log((w + 1e-12) / (pi_prev + 1e-12))))

                    if self.bound_type == "split-kl":
                        eps = compute_split_kl_bound(
                            w=w,
                            Fhat=Fhat,
                            alpha_vals=alpha_vals,
                            b0=b0,
                            KL_div=KL_div,
                            delta=self.delta,
                            T=self.T,
                            n=m_t,
                        )
                    elif self.bound_type == "plain-kl":
                        p_mean = float(np.dot(w, f_gamma_means))
                        eps = compute_plain_kl_bound(p_mean, KL_div, delta_t, m_t)
                    elif self.bound_type == "emp-bernstein":
                        eps = compute_empirical_bernstein_bound(
                            w=w,
                            mu_i=mu_i,
                            v_i=v_i,
                            KL_div=KL_div,
                            delta_t=delta_t,
                            m=m_t,
                            b=1.0 + gamma,
                        )
                    elif self.bound_type == "unexp-bernstein":
                        # Build a λ‐grid in (0, 1/(1+γ))
                        b_val = 1.0 + gamma
                        eps_small = 1e-6
                        max_lam = (1.0 / b_val) - eps_small
                        if max_lam <= 0:
                            lambda_grid_t = np.array([eps_small])
                        else:
                            lambda_grid_t = np.linspace(
                                eps_small, max_lam, len(self.lambda_grid)
                            )
                        eps = compute_unexpected_bernstein_bound(
                            w=w,
                            mu_i=mu_i,
                            s_i=s_i,
                            KL_div=KL_div,
                            delta_t=delta_t,
                            m=m_t,
                            b=b_val,
                            lambda_grid=lambda_grid_t,
                        )
                    else:
                        raise ValueError(f"Unknown bound_type='{self.bound_type}'")

                    return eps + gamma * B_prev

                v0_t = np.log(pi_prev + 1e-12)
                res_t = minimize(
                    objective_t,
                    v0_t,
                    method="Nelder-Mead",
                    options={"maxiter": 500, "disp": False},
                )
                v_opt_t = res_t.x
                exp_vt = np.exp(v_opt_t - np.max(v_opt_t))
                w_opt_t = exp_vt / exp_vt.sum()
                B_t_candidate = objective_t(v_opt_t)
                return w_opt_t, B_t_candidate

            # 3) Search γ on descending grid
            best_gamma = None
            best_pi_t = None
            best_Bt = None
            for gamma_candidate in sorted(self.gamma_grid, reverse=True):
                w_cand, B_cand = evaluate_for_gamma(float(gamma_candidate))
                if B_cand < B_prev:
                    best_gamma = float(gamma_candidate)
                    best_pi_t = w_cand
                    best_Bt = B_cand
                    break

            # 4) Fallback to median γ if needed
            if best_gamma is None:
                mid_idx = len(self.gamma_grid) // 2
                gamma_fallback = float(self.gamma_grid[mid_idx])
                w_fb, B_fb = evaluate_for_gamma(gamma_fallback)
                best_gamma = gamma_fallback
                best_pi_t = w_fb
                best_Bt = B_fb

            self.pi_list_.append(best_pi_t)
            self.bounds_.append(best_Bt)

            pi_prev = best_pi_t.copy()
            B_prev = best_Bt

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, seed: int = None) -> np.ndarray:
        """
        Predict labels (or regression values) on X by weighted‐vote (final π_T).
        For classification:
          - binary: returns array in {0,1}.
          - multiclass: returns array of predicted class indices.
        For regression: returns weighted average.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit(...) before predict(...)")

        X_np = np.asarray(X, dtype=np.float32)
        n = X_np.shape[0]
        π_T = self.pi_list_[-1]
        K = self.K
        if seed is None:
            seed = self.seed
        rng = jr.PRNGKey(seed)
        rngs = jr.split(rng, num=K + 1)[1:]  # one key per model

        if self.task == "binary":
            vote_matrix = np.zeros((K, n))
            for i, (model, state) in enumerate(self.trained_models_states_):
                preds = _single_predict(model, state, X_np, rngs[i])
                probs = jax.nn.sigmoid(preds).ravel()
                vote_matrix[i, :] = np.where(np.array(probs) >= 0.5, 1, 0)
            weighted_vote = (π_T[:, None] * vote_matrix).sum(axis=0)
            return np.array(weighted_vote >= 0.5, dtype=int)

        elif self.task == "multiclass":
            sample_logits = _single_predict(
                self.trained_models_states_[0][0],
                self.trained_models_states_[0][1],
                X_np,
                rngs[0],
            )
            C = sample_logits.shape[-1]
            agg_probs = np.zeros((n, C))
            for i, (model, state) in enumerate(self.trained_models_states_):
                preds = _single_predict(model, state, X_np, rngs[i])
                probs = jax.nn.softmax(preds, axis=-1)
                agg_probs += π_T[i] * np.array(probs)
            return np.array(np.argmax(agg_probs, axis=1), dtype=int)

        else:  # regression
            agg_preds = np.zeros(n)
            for i, (model, state) in enumerate(self.trained_models_states_):
                preds = _single_predict(model, state, X_np, rngs[i])
                preds_np = np.array(preds).ravel()
                agg_preds += π_T[i] * preds_np
            return agg_preds

    @property
    def get_posteriors(self) -> List[np.ndarray]:
        return self.pi_list_

    @property
    def get_bounds(self) -> List[float]:
        return self.bounds_


if __name__ == "__main__":
    # =============================
    # Minimal sanity check for RecursivePACBayesEnsemble (Equinox version)
    # =============================

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import zero_one_loss

    # 1) Create a tiny synthetic binary classification dataset
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=0,
    )
    # Split into train/hold/test (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    X_train, X_hold, y_train, y_hold = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=2, stratify=y_temp
    )  # 0.25×0.8 = 0.2

    # 2) Define a trivial Equinox‐style module that always predicts 0 logits
    class DummyBinary(eqx.Module):
        l1: eqx.nn.Linear

        def __init__(self, in_features, key):
            self.l1 = eqx.nn.Linear(in_features, 1, key=key)

        def __call__(self, x, key, state):
            y = self.l1(x)
            return y, state

    # 3) Build three identical constructors
    def ctor1(key):
        return DummyBinary(X_train.shape[1], key)

    def ctor2(key):
        return DummyBinary(X_train.shape[1], jr.split(key)[1])

    def ctor3(key):
        return DummyBinary(X_train.shape[1], jr.split(key)[0])

    module_constructors = [ctor1, ctor2, ctor3]

    # 4) Instantiate the recursive ensemble
    ensemble = RecursivePACBayesEnsemble(
        model_constructors=module_constructors,
        task="binary",
        loss_fn=binary_loss,
        optimizer=optax.adam(1e-3),
        bound_type="unexp-bernstein",  # try "plain-kl", "emp-bernstein", "unexp-bernstein", "split-kl"
        delta=0.05,
        T=2,
        seed=42,
        L_max=1.0,
        batch_size=16,
        num_epochs=100,
        patience=2,
    )

    # 5) Fit on (X_train, y_train)
    ensemble.fit(X_train, y_train)

    # 6) Print posteriors and bounds
    print("Recursive PAC-Bayes Posteriors and Bounds:")
    for t, (π, b) in enumerate(
        zip(ensemble.get_posteriors, ensemble.get_bounds), start=1
    ):
        print(f" Stage {t}: π_{t} = {np.round(π, 4)}, B_{t} = {b:.4f}")

    # 7) Predict on hold and test sets
    y_hold_pred = ensemble.predict(X_hold, seed=123)
    hold_err = zero_one_loss(y_hold, y_hold_pred)
    print(f"Hold‐out zero‐one error = {hold_err:.4f}")

    y_test_pred = ensemble.predict(X_test, seed=321)
    test_err = zero_one_loss(y_test, y_test_pred)
    print(f"Test zero‐one error = {test_err:.4f}")

    # 8) Ensure outputs have the right type/shape
    assert isinstance(y_test_pred, np.ndarray) and y_test_pred.shape == (
        X_test.shape[0],
    )
    assert 0.0 <= hold_err <= 1.0 and 0.0 <= test_err <= 1.0

    print("Recursive PAC-Bayes (Equinox) test completed successfully.")
