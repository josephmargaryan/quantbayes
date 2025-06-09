# recursive_pac_bayes_numpyro.py

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import List, Callable, Tuple
from scipy.optimize import minimize
from sklearn.utils import check_random_state

# Import the same PAC-Bayes bound functions as in recursive_pac_bayes.py
from quantbayes.pacbayes.recursive import (
    compute_plain_kl_bound,
    compute_split_kl_bound,
    compute_empirical_bernstein_bound,
    compute_unexpected_bernstein_bound,
)

CRITERIA = {
    "plain-kl",
    "split-kl",
    "emp-bernstein",
    "unexp-bernstein",
}


class RecursivePACBayesEnsembleNumPyro:
    """
    Recursive PAC-Bayes ensemble wrapper for a fixed finite set of NumPyro-based Modules.

    Each "module" returned by a constructor must support:
      - compile(): prepares its inference engine (MCMC/SVI/SteinVI)
      - fit(X, y, rng_key, **kwargs): runs inference on (X,y)
      - predict(X, rng_key, posterior="logits", num_samples): returns samples of logits or predictions
    """

    def __init__(
        self,
        module_constructors: List[Callable[[jax.random.PRNGKey], object]],
        task: str,
        bound_type: str = "split-kl",
        delta: float = 0.05,
        T: int = 2,
        gamma_grid: np.ndarray = None,
        seed: int = 0,
        L_max: float = 1.0,
    ):
        """
        Parameters
        ----------
        module_constructors : list of callables
            Each callable accepts a JAX PRNGKey and returns an uninitialized Module
            (with compile(), fit(), predict() methods).
        task : {"binary", "multiclass", "regression"}
        bound_type : {"plain-kl","split-kl","emp-bernstein","unexp-bernstein"}
        delta : float ∈ (0,1)
            Overall failure probability.
        T : int ≥ 1
            Number of recursive stages.
        gamma_grid : 1D numpy array of candidates in (0,1)
            If None: defaults to np.linspace(0.1, 0.9, 9).
        seed : int
            Global RNG seed for splitting and module RNG.
        L_max : float
            In regression mode, maximum possible loss (for clipping).
        """
        if bound_type not in CRITERIA:
            raise ValueError(f"Unknown bound_type '{bound_type}'.")
        if task not in ("binary", "multiclass", "regression"):
            raise ValueError("`task` must be 'binary', 'multiclass', or 'regression'.")

        self.module_constructors = module_constructors
        self.K = len(module_constructors)
        self.task = task
        self.bound_type = bound_type
        self.delta = delta
        self.T = T
        self.gamma_grid = (
            gamma_grid if gamma_grid is not None else np.linspace(0.1, 0.9, 9)
        )
        self.seed = seed
        self.L_max = L_max

        # Will be populated in .fit()
        self.trained_modules_ = []  # List of trained Module instances
        self.pi_list_ = []  # List of posterior weight arrays [π₁,...,π_T]
        self.bounds_ = []  # List of bound values [B₁,...,B_T]
        self.chunk_indices_ = []  # List of index-arrays for geometric split
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
        modules: List[object],
        X: np.ndarray,
        y: np.ndarray,
        rng: jax.random.PRNGKey,
        num_samples: int,
    ) -> np.ndarray:
        """
        Given a list of K trained modules, compute a (K, n_samples) matrix of 0-1 losses
        for (X,y) on each module. Uses module.predict(..., posterior="logits").
        """
        n = X.shape[0]
        K = len(modules)
        loss_matrix = np.zeros((K, n), dtype=float)
        rngs = jr.split(rng, num=K + 1)[1:]

        for i, module in enumerate(modules):
            preds_samples = module.predict(
                X, rngs[i], posterior="logits", num_samples=num_samples
            )
            # -- Binary case: preds_samples shape (num_samples, n)
            if self.task == "binary":
                # sigmoid over logits, then average
                avg_probs = jax.nn.sigmoid(
                    preds_samples.reshape((num_samples, n))
                ).mean(axis=0)
                pred_labels = np.array(avg_probs >= 0.5, dtype=int)
                loss_matrix[i, :] = (pred_labels != y).astype(float)

            # -- Multiclass case: preds_samples shape (num_samples, n, C)
            elif self.task == "multiclass":
                avg_probs = jax.nn.softmax(preds_samples, axis=-1).mean(
                    axis=0
                )  # shape (n, C)
                pred_labels = np.array(jnp.argmax(avg_probs, axis=-1))
                loss_matrix[i, :] = (pred_labels != y).astype(float)

            # -- Regression case: preds_samples shape (num_samples, n)
            else:  # regression
                avg_preds = preds_samples.reshape((num_samples, n)).mean(axis=0)
                mse = (avg_preds - y) ** 2
                clipped = np.minimum(mse, self.L_max) / self.L_max
                loss_matrix[i, :] = clipped

        return loss_matrix

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        svi_steps: int = 500,
        num_samples: int = 100,
    ) -> "RecursivePACBayesEnsembleNumPyro":
        """
        Fit the Recursive PAC-Bayes ensemble:
          1) Geometric split of data into T chunks.
          2) Compile & train each NumPyro Module once on the entire (X,y).
          3) Build the full (K, n) loss matrix.
          4) Stage 1: plain PAC-Bayes-KL on S₁.
          5) Stages 2..T: chosen bound on Uₜ = Sₜ ∪ ... ∪ S_T.
        """
        X = np.asarray(X)
        y = np.asarray(y).astype(
            int if self.task in ("binary", "multiclass") else float
        )
        n = X.shape[0]

        # 1) Geometric split
        chunks = self._geometric_split_indices(n)
        self.chunk_indices_ = chunks

        # 2) Compile & train each Module on full data
        self.trained_modules_ = []
        rng = jr.PRNGKey(self.seed)
        for ctor in self.module_constructors:
            rng, subkey = jr.split(rng)
            module = ctor(subkey)  # user-provided constructor returns NumPyro Module
            module.compile()  # sets up NUTS/SVI/SteinVI
            module.fit(X, y, subkey, num_steps=svi_steps)
            self.trained_modules_.append(module)

        # 3) Build full loss_matrix_all of shape (K, n)
        rng, loss_key = jr.split(rng)
        loss_matrix_all = self._compute_loss_matrix(
            self.trained_modules_, X, y, loss_key, num_samples
        )

        # ===== Stage 1 (plain PAC-Bayes-KL) =====
        S1_idx = chunks[0]
        loss_S1 = loss_matrix_all[:, S1_idx].mean(axis=1)  # shape (K,)
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
            loss_U_t = loss_matrix_all[:, U_t_idx]  # shape (K, |U_t|)
            m_t = loss_U_t.shape[1]
            delta_t = self.delta / self.T

            # Precompute weighted loss of π_{t-1} on U_t
            weighted_prev_U = pi_prev @ loss_U_t  # shape (|U_t|,)

            def evaluate_for_gamma(gamma: float) -> Tuple[np.ndarray, float]:
                f_gamma_U = loss_U_t - gamma * weighted_prev_U  # shape (K, m_t)

                if self.bound_type == "split-kl":
                    b0 = -gamma
                    b_vals = np.array([b0, 0.0, 1.0 - gamma, 1.0])
                    alpha_vals = np.diff(b_vals)  # [γ, 1-γ, γ]
                    indicator = np.zeros((3, self.K, m_t), dtype=float)
                    for j in range(1, 4):
                        indicator[j - 1] = (f_gamma_U >= b_vals[j]).astype(float)
                    Fhat = indicator.mean(axis=2)  # shape (3, K)

                elif self.bound_type == "plain-kl":
                    f_gamma_means = f_gamma_U.mean(axis=1)  # shape (K,)

                elif self.bound_type == "emp-bernstein":
                    f_gamma_means = f_gamma_U.mean(axis=1)  # shape (K,)
                    var_i = f_gamma_U.var(axis=1, ddof=1)  # sample variance
                    mu_i = f_gamma_means.copy()
                    v_i = var_i.copy()

                elif self.bound_type == "unexp-bernstein":
                    f_gamma_means = f_gamma_U.mean(axis=1)  # shape (K,)
                    s_i = (f_gamma_U * f_gamma_U).mean(axis=1)  # shape (K,)
                    mu_i = f_gamma_means.copy()

                else:
                    raise ValueError(f"Unknown bound_type='{self.bound_type}'")

                def objective_t(v: np.ndarray) -> float:
                    exp_v = np.exp(v - np.max(v))
                    w = exp_v / exp_v.sum()  # candidate π_t
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
                        # — begin λ-clipping —
                        b_val = 1.0 + gamma
                        eps_small = 1e-6
                        max_lam = (1.0 / b_val) - eps_small
                        if max_lam <= 0:
                            lambda_grid_t = np.array([eps_small])
                        else:
                            M = len(self.gamma_grid)
                            lambda_grid_t = np.linspace(eps_small, max_lam, M)

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
                        # — end λ-clipping —
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

            if best_gamma is None:
                mid_idx = len(self.gamma_grid) // 2
                gamma_fallback = float(self.gamma_grid[mid_idx])
                w_fb, B_fb = evaluate_for_gamma(gamma_fallback)

                # --- But for fallback unexp-Bernstein, we must re-clip λ there, too
                if self.bound_type == "unexp-bernstein":

                    def objective_fallback(v: np.ndarray) -> float:
                        exp_v = np.exp(v - np.max(v))
                        w = exp_v / exp_v.sum()
                        KL_div = float(
                            np.sum(w * np.log((w + 1e-12) / (pi_prev + 1e-12)))
                        )

                        # recompute f_gamma_U and moments for fallback gamma
                        f_gamma_U_fb = loss_U_t - gamma_fallback * weighted_prev_U
                        mu_i_fb = f_gamma_U_fb.mean(axis=1)
                        s_i_fb = (f_gamma_U_fb * f_gamma_U_fb).mean(axis=1)
                        # λ-clipping for fallback
                        b_val_fb = 1.0 + gamma_fallback
                        eps_small = 1e-6
                        max_lam_fb = (1.0 / b_val_fb) - eps_small
                        if max_lam_fb <= 0:
                            lambda_grid_fb = np.array([eps_small])
                        else:
                            M = len(self.gamma_grid)
                            lambda_grid_fb = np.linspace(eps_small, max_lam_fb, M)

                        eps_fb = compute_unexpected_bernstein_bound(
                            w=w,
                            mu_i=mu_i_fb,
                            s_i=s_i_fb,
                            KL_div=KL_div,
                            delta_t=delta_t,
                            m=m_t,
                            b=b_val_fb,
                            lambda_grid=lambda_grid_fb,
                        )
                        return eps_fb + gamma_fallback * B_prev

                    v0_fb = np.log(pi_prev + 1e-12)
                    res_fb = minimize(
                        objective_fallback,
                        v0_fb,
                        method="Nelder-Mead",
                        options={"maxiter": 500, "disp": False},
                    )
                    v_opt_fb = res_fb.x
                    exp_vfb = np.exp(v_opt_fb - np.max(v_opt_fb))
                    w_opt_fb = exp_vfb / exp_vfb.sum()
                    best_pi_t = w_opt_fb
                    best_Bt = objective_fallback(v_opt_fb)

                else:
                    best_pi_t = w_fb
                    best_Bt = B_fb

                best_gamma = gamma_fallback

            self.pi_list_.append(best_pi_t)
            self.bounds_.append(best_Bt)
            pi_prev = best_pi_t.copy()
            B_prev = best_Bt

        self.is_fitted = True
        return self

    def predict(
        self, X: np.ndarray, rng_key: jax.random.PRNGKey, num_samples: int = 100
    ) -> np.ndarray:
        """
        Predict labels (or regression values) on X by weighted-vote (final π_T).
        rng_key: a JAX PRNGKey
        num_samples: how many posterior samples to average over
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit(...) before predict(...)")

        X = np.asarray(X)
        n = X.shape[0]
        π_T = self.pi_list_[-1]
        K = self.K
        rngs = jr.split(rng_key, num=K + 1)[1:]

        if self.task == "binary":
            vote_matrix = np.zeros((K, n))
            for i, module in enumerate(self.trained_modules_):
                preds_samples = module.predict(
                    X, rngs[i], posterior="logits", num_samples=num_samples
                )
                avg_probs = jax.nn.sigmoid(
                    preds_samples.reshape((num_samples, n))
                ).mean(axis=0)
                vote_matrix[i, :] = np.where(np.array(avg_probs) >= 0.5, 1, 0)
            weighted_vote = (π_T[:, None] * vote_matrix).sum(axis=0)
            return np.array(weighted_vote >= 0.5, dtype=int)

        elif self.task == "multiclass":
            # We only need to know C; take sample from first module
            sample_logits = self.trained_modules_[0].predict(
                X, rngs[0], posterior="logits", num_samples=num_samples
            )
            C = sample_logits.shape[-1]
            agg_probs = np.zeros((n, C))
            for i, module in enumerate(self.trained_modules_):
                preds_samples = module.predict(
                    X, rngs[i], posterior="logits", num_samples=num_samples
                )
                avg_probs = jax.nn.softmax(preds_samples, axis=-1).mean(
                    axis=0
                )  # shape (n, C)
                agg_probs += π_T[i] * np.array(avg_probs)
            return np.array(np.argmax(agg_probs, axis=1), dtype=int)

        else:  # regression
            agg_preds = np.zeros(n)
            for i, module in enumerate(self.trained_modules_):
                preds_samples = module.predict(
                    X, rngs[i], posterior="logits", num_samples=num_samples
                )
                avg_pred = preds_samples.reshape((num_samples, n)).mean(axis=0)
                agg_preds += π_T[i] * np.array(avg_pred)
            return agg_preds

    def predict_proba(
        self,
        X: np.ndarray,
        rng_key: jax.random.PRNGKey = None,
        num_samples: int = 100,
    ) -> np.ndarray:
        """
        Return weighted‐vote probabilities for classification tasks:
          - Binary:   shape (n_samples, 2) with columns [P(class=0), P(class=1)]
          - Multiclass: shape (n_samples, C), summing to 1
        Regression: returns weighted mean, same as predict().
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit(...) before predict_proba(...)")
        if rng_key is None:
            rng_key = jr.PRNGKey(self.seed)

        X_np = np.asarray(X)
        n = X_np.shape[0]
        π_T = self.pi_list_[-1]
        K = self.K
        rngs = jr.split(rng_key, num=K + 1)[1:]

        # --- Binary ---
        if self.task == "binary":
            # Collect each module’s P(y=1)
            probs_pos = np.zeros((K, n), dtype=float)
            for i, module in enumerate(self.trained_modules_):
                preds = module.predict(
                    X_np, rngs[i], posterior="logits", num_samples=num_samples
                )
                # preds shape (num_samples, n)
                avg_p1 = jax.nn.sigmoid(preds.reshape((num_samples, n))).mean(axis=0)
                probs_pos[i] = np.array(avg_p1)
            # Ensemble P1 = sum_i π_i * p1_i, P0 = 1−P1
            P1 = π_T @ probs_pos
            proba = np.vstack([1.0 - P1, P1]).T
            return proba

        # --- Multiclass ---
        elif self.task == "multiclass":
            # First figure out C
            sample_preds = self.trained_modules_[0].predict(
                X_np, rngs[0], posterior="logits", num_samples=1
            )
            C = sample_preds.shape[-1]
            proba = np.zeros((n, C), dtype=float)

            for i, module in enumerate(self.trained_modules_):
                preds = module.predict(
                    X_np, rngs[i], posterior="logits", num_samples=num_samples
                )
                # preds shape (num_samples, n, C)
                avg_p = jax.nn.softmax(preds, axis=-1).mean(axis=0)  # (n, C)
                proba += π_T[i] * np.array(avg_p)

            return proba

        # --- Regression just return the same as predict() ---
        else:
            return self.predict(X_np, rng_key, num_samples)

    @property
    def get_posteriors(self) -> List[np.ndarray]:
        return self.pi_list_

    @property
    def get_bounds(self) -> List[float]:
        return self.bounds_


if __name__ == "__main__":
    # =============================
    # Minimal sanity check for NumPyro-based RecursivePACBayesEnsembleNumPyro
    # =============================
    import numpy as np
    import jax.random as jr
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import zero_one_loss
    import numpyro
    import numpyro.distributions as dist
    from quantbayes.bnn import Module  # your existing NumPyro Module base class

    # 1) Generate small synthetic binary classification data
    Xc, yc = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=0,
    )
    # Split into train/hold/test (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        Xc, yc, test_size=0.2, random_state=1, stratify=yc
    )
    X_train, X_hold, y_train, y_hold = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=2, stratify=y_temp
    )  # 0.25×0.8=0.2

    # We will “pretend” that the entire dataset is X_all, y_all:
    X_all = np.vstack([X_train, X_hold, X_test])
    y_all = np.concatenate([y_train, y_hold, y_test])

    # Convert to JAX arrays
    X_all = jnp.array(X_all, dtype=jnp.float32)
    y_all = jnp.array(y_all, dtype=jnp.float32)

    # 2) Define three identical Bayesian Logistic Regression Modules
    class BayesLogisticRegression(Module):
        def __init__(self, in_dim, method="svi"):
            super().__init__(method=method)
            self.in_dim = in_dim

        def __call__(self, X, Y=None):
            w = numpyro.sample(
                "w", dist.Normal(jnp.zeros(self.in_dim), jnp.ones(self.in_dim))
            )
            b = numpyro.sample("b", dist.Normal(0.0, 1.0))
            logits = jnp.dot(X, w) + b
            numpyro.deterministic("logits", logits)
            numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=Y)

    in_dim = X_all.shape[1]

    def ctor1(key):
        return BayesLogisticRegression(in_dim, method="svi")

    def ctor2(key):
        return BayesLogisticRegression(in_dim, method="svi")

    def ctor3(key):
        return BayesLogisticRegression(in_dim, method="svi")

    module_constructors = [ctor1, ctor2, ctor3]

    # 3) Build the ensemble
    ensemble = RecursivePACBayesEnsembleNumPyro(
        module_constructors=module_constructors,
        task="binary",
        bound_type="unexp-bernstein",  # try "plain-kl", "emp-bernstein", "unexp-bernstein"
        delta=0.05,
        T=2,
        seed=42,
        L_max=1.0,
    )

    # 4) Fit the ensemble on the entire “all” dataset.
    ensemble.fit(
        X_all,
        y_all,
        svi_steps=200,
        num_samples=50,
    )

    # 5) Print summary of posteriors and bounds (two stages, since T=2)
    print("\nNumPyro Ensemble Posteriors and Bounds (T=2):")
    for t, (π, bound) in enumerate(
        zip(ensemble.get_posteriors, ensemble.get_bounds), start=1
    ):
        print(f"  Stage {t}:  π_{t} = {np.round(π, 4)},  B_{t} = {bound:.4f}")

    # 6) Now evaluate on HOLD and TEST separately
    rng_hold = jr.PRNGKey(999)
    y_hold_preds = ensemble.predict(X_hold, rng_hold, num_samples=50)
    hold_err = zero_one_loss(np.array(y_hold), y_hold_preds)
    print(f"\nHold-out zero-one error = {hold_err:.4f}")

    rng_test = jr.PRNGKey(1001)
    y_test_preds = ensemble.predict(X_test, rng_test, num_samples=50)
    test_err = zero_one_loss(np.array(y_test), y_test_preds)
    print(f"Test zero-one error = {test_err:.4f}")

    # 7) Basic sanity checks
    assert 0.0 <= hold_err <= 1.0 and 0.0 <= test_err <= 1.0
    assert isinstance(y_hold_preds, np.ndarray) and y_hold_preds.shape == (
        X_hold.shape[0],
    )
    assert isinstance(y_test_preds, np.ndarray) and y_test_preds.shape == (
        X_test.shape[0],
    )

    print("NumPyro recursive-PAC-Bayes test completed successfully.")
