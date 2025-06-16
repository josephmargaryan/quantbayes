# recursive_pac_bayes.py

"""
recursive_pac_bayes.py

A production‐level implementation of a finite‐hypothesis Recursive PAC‐Bayes ensemble
that supports both binary and multiclass classification. Users can import the
`RecursivePACBayesEnsemble` class to fit an ensemble over any list of scikit‐learn‐style
classifiers, or run this file as a script to demonstrate usage on the Breast Cancer dataset.

Author: Joseph Margaryan (updated to support multiclass)
Date: 2025-06-05
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Union
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt


def kl_binary(p: float, q: float) -> float:
    """
    Compute binary KL divergence with clipping to avoid numerical issues.
    KL(p || q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def kl_inverse(p: float, c: float, tol: float = 1e-8) -> float:
    """
    Compute kl^{-1^+}(p, c) = sup { q in [p, 1] : KL(p || q) <= c }
    via bisection. If c <= 0, return p.
    """
    if c <= 0:
        return p
    low, high = p, 1.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if kl_binary(p, mid) <= c:
            low = mid
        else:
            high = mid
    return low


def compute_plain_kl_bound(
    p_mean: float, KL_div: float, delta_t: float, n: int
) -> float:
    """
    Plain PAC-Bayes--KL bound inversion:
      B = kl^{-1^+}\Bigl(p_mean,\; (KL_div + ln(2*sqrt(n)/delta_t))/n \Bigr).
    """
    ln_term = np.log(2.0 * np.sqrt(n) / delta_t)
    C = (KL_div + ln_term) / n
    return kl_inverse(p_mean, C)


def compute_split_kl_bound(
    w: np.ndarray,
    Fhat: np.ndarray,
    alpha_vals: np.ndarray,
    b0: float,
    KL_div: float,
    delta: float,
    T: int,
    n: int,
) -> float:
    """
    PAC-Bayes--split-kl bound for a discrete-valued random variable with K_slices slices:
      B = b0 + sum_{j=1}^K_slices alpha_j * kl^{-1^+}( q_j , (KL_div + ln(2*K_slices*T*sqrt(n)/delta)) / n ).
    Here:
      - w:       posterior weights (shape (K,))
      - Fhat:    empirical slice-means (shape (K_slices, K))
      - alpha_vals: length-K_slices array of (b_j - b_{j-1})
      - b0:      the smallest value of the discrete variable
      - KL_div:  KL(π_t || π_{t-1})
      - delta:   overall δ
      - T:       total number of stages (for the union bound)
      - n:       size of the sample U_t
    """
    K_slices = alpha_vals.shape[0]
    ln_term = np.log((2.0 * K_slices * T * np.sqrt(n)) / delta)
    C = (KL_div + ln_term) / n

    eps = b0
    for j in range(K_slices):
        qj = float(w @ Fhat[j])
        eps += alpha_vals[j] * kl_inverse(qj, C)
    return eps


def compute_empirical_bernstein_bound(
    w: np.ndarray,
    mu_i: np.ndarray,
    v_i: np.ndarray,
    KL_div: float,
    delta_t: float,
    m: int,
    b: float,
) -> float:
    """
    Empirical Bernstein bound (second-order, data-dependent).  Here:
      - w:      posterior weights (shape (K,))
      - mu_i:   per-hypothesis empirical means of f_gamma (shape (K,))
      - v_i:    per-hypothesis empirical variances of f_gamma (shape (K,))
      - KL_div: KL(π_t || π_{t-1})
      - delta_t: δ/T
      - m:      size of U_t
      - b:      bound on |f_gamma| (i.e. max(|f_gamma|))
    We compute:
      mu_post = sum_i w_i * mu_i
      var_post ≤ sum_i w_i * v_i + Var_i(mu_i)
      then
      B = mu_post + sqrt( 2*var_post*ln(2/delta_t) / m ) + (7 ln(2/delta_t))/(3m) + KL_div/m
    """
    mu_post = float(np.dot(w, mu_i))
    mu_i = np.asarray(mu_i)
    v_i = np.asarray(v_i)

    mu_sq_post = float(np.dot(w, mu_i * mu_i))
    var_mu = mu_sq_post - mu_post * mu_post
    var_post = float(np.dot(w, v_i) + var_mu)

    ln_term = np.log(2.0 / delta_t)
    sqrt_term = np.sqrt((2.0 * var_post * ln_term) / m)
    linear_term = (7.0 * ln_term) / (3.0 * m)
    return mu_post + sqrt_term + linear_term + (KL_div / m)


def compute_unexpected_bernstein_bound(
    w: np.ndarray,
    mu_i: np.ndarray,
    s_i: np.ndarray,
    KL_div: float,
    delta_t: float,
    m: int,
    b: float,
    lambda_grid: np.ndarray,
) -> float:
    """
    Unexpected Bernstein bound (direct second-moment).  Here:
      - w:        posterior weights, shape (K,)
      - mu_i:     per-hypothesis empirical means of f_gamma, shape (K,)
      - s_i:      per-hypothesis empirical second moments of f_gamma, shape (K,)
      - KL_div:   KL(π_t || π_{t-1})
      - delta_t:  δ/T
      - m:        size of U_t
      - b:        bound on |f_gamma|
      - lambda_grid: 1D numpy array of λ candidates in (0, 1/b)
    We compute:
      mu_post = ∑_i w_i * mu_i
      s_sq_post = ∑_i w_i * s_i
      mu_sq_post = ∑_i w_i * mu_i^2
      var_mu = mu_sq_post - mu_post^2
      s_post = s_sq_post + var_mu
    Then search over λ ∈ λ_grid:
      term(λ) = [ψ(−λb)/(λb^2)] * s_post + [ln(|λ_grid|/δ_t)/(λ m)]
      where ψ(u) = u − ln(1+u)
    Finally,
      B = mu_post + min_{λ} term(λ) + KL_div/m
    """
    mu_post = float(np.dot(w, mu_i))
    mu_i = np.asarray(mu_i)
    s_i = np.asarray(s_i)

    s_sq_post = float(np.dot(w, s_i))
    mu_sq_post = float(np.dot(w, mu_i * mu_i))
    var_mu = mu_sq_post - mu_post * mu_post
    s_post = s_sq_post + var_mu

    best_term = np.inf
    for lam in lambda_grid:
        psi = -lam * b - np.log(1.0 - b * lam)
        term = (psi / (lam * b * b)) * s_post + (
            np.log(len(lambda_grid) / delta_t) / (lam * m)
        )
        if term < best_term:
            best_term = term

    return mu_post + best_term + (KL_div / m)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RecursivePACBayesEnsemble(BaseEstimator, ClassifierMixin):
    """
    A finite‐H Recursive PAC‐Bayes ensemble (scikit‐learn style) with a `bound_type` switch,
    now supporting both binary and multiclass classification.

    Parameters
    ----------
    base_learners : List[Tuple[str, Estimator]]
        List of tuples (name, estimator). These must be scikit‐learn–style classifiers
        implementing fit(X,y), predict(X). They will be cloned internally and fit on all
        training data.
    T : int, default=2
        Number of recursive stages. Must be >= 1.
    delta : float, default=0.05
        Overall failure probability for PAC‐Bayes bounds.
    gamma_grid : Optional[np.ndarray], default=None
        1D array of candidate γ_t values in (0,1). If None, uses np.linspace(0.1,0.9,9).
    bound_type : str, default="split-kl"
        Which PAC‐Bayes bound to use at each stage (excluding Stage 1, which always uses plain KL).
        Options: "plain-kl", "split-kl", "emp-bernstein", "unexp-bernstein".
    lambda_grid : Optional[np.ndarray], default=None
        Grid of λ values in (0,1/bound_max) for "unexp-bernstein". If None, a default grid is used.
    random_state : Optional[Union[int,np.random.RandomState]], default=None
        Controls randomness for data splitting.
    verbose : bool, default=False
        If True, prints intermediate logs.
    task : str, default="binary"
        One of {"binary", "multiclass"}. Determines how predictions are aggregated:
         - "binary": weighted vote for class 1 vs. class 0.
         - "multiclass": weighted vote across all classes in `classes_`.
    """

    def __init__(
        self,
        base_learners: List[Tuple[str, BaseEstimator]],
        T: int = 2,
        delta: float = 0.05,
        gamma_grid: Optional[np.ndarray] = None,
        bound_type: str = "split-kl",
        lambda_grid: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: bool = False,
        task: str = "binary",
    ):
        if task not in ("binary", "multiclass"):
            raise ValueError("`task` must be either 'binary' or 'multiclass'.")

        self.base_learners = base_learners
        self.T = T
        self.delta = delta
        self.gamma_grid = (
            gamma_grid if gamma_grid is not None else np.linspace(0.1, 0.9, 9)
        )
        self.bound_type = bound_type
        self.lambda_grid = (
            lambda_grid
            if lambda_grid is not None
            else np.linspace(1e-4, 0.9999, 200)  # for Unexpected-Bernstein
        )
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self.task = task

        # Will be filled during fit()
        self.trained_classifiers_: List[Tuple[str, BaseEstimator]] = []
        self.pi_list_: List[np.ndarray] = []  # list of posterior weight vectors
        self.bounds_: List[float] = []  # list of B_t bounds
        self.chunk_indices_: List[np.ndarray] = []  # which indices went to which chunk
        self.classes_: np.ndarray = np.array(
            [0, 1]
        )  # placeholder; overwritten in fit()

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)

    def _geometric_split_indices(self, n: int) -> List[np.ndarray]:
        """
        Split {0,...,n-1} into T geometric chunks S1,...,ST so that
        |S1| ~ 1, |S2| ~ 2, |S3| ~ 4, ..., |ST| ~ n/2 (approximately),
        subject to sum = n.  Returns a list of index arrays.
        """
        sizes: List[int] = []
        remaining = n
        for t in range(1, self.T + 1):
            if t == self.T:
                sizes.append(remaining)
            else:
                size_t = min(remaining - (self.T - t), 2 ** (t - 1))
                sizes.append(size_t)
                remaining -= size_t
        total_assigned = sum(sizes)
        if total_assigned != n:
            sizes[-1] += n - total_assigned

        all_idx = np.arange(n)
        self.random_state.shuffle(all_idx)
        chunks: List[np.ndarray] = []
        start = 0
        for s in sizes:
            end = start + s
            chunks.append(all_idx[start:end])
            start = end
        return chunks

    def _compute_loss_matrix(
        self, classifiers: List[BaseEstimator], X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Given a list of trained classifiers (length K), compute a (K, n_samples) matrix
        of 0-1 losses: loss_matrix[i,j] = 1 if clf_i.predict(X[j]) != y[j], else 0.
        Works for both binary and multiclass.
        """
        K = len(classifiers)
        n = X.shape[0]
        loss_matrix = np.zeros((K, n), dtype=float)
        for i, clf in enumerate(classifiers):
            preds = clf.predict(X)
            loss_matrix[i, :] = (preds != y).astype(float)
        return loss_matrix

    def _optimize_stage1(
        self, loss_S1: np.ndarray, n1: int
    ) -> Tuple[np.ndarray, float]:
        """
        Stage 1: plain PAC-Bayes--KL on S1.
        Returns (pi1_weights, B1).
        """
        K = loss_S1.shape[0]
        delta_t = self.delta / self.T

        def objective(v: np.ndarray) -> float:
            exp_v = np.exp(v - np.max(v))
            w = exp_v / exp_v.sum()
            p_mean = float(np.dot(w, loss_S1))
            KL_w = float(np.sum(w * np.log(w * K + 1e-12)))  # KL(w || uniform)
            B1 = compute_plain_kl_bound(p_mean, KL_w, delta_t, n1)
            return B1

        v0 = np.zeros(K)
        res = minimize(
            objective, v0, method="Nelder-Mead", options={"maxiter": 500, "disp": False}
        )
        v_opt = res.x
        exp_vo = np.exp(v_opt - np.max(v_opt))
        w_opt = exp_vo / exp_vo.sum()

        # Recompute B1 exactly
        p_mean_final = float(np.dot(w_opt, loss_S1))
        KL_final = float(np.sum(w_opt * np.log(w_opt * K + 1e-12)))
        B1_value = compute_plain_kl_bound(p_mean_final, KL_final, delta_t, n1)

        return w_opt, B1_value

    def _optimize_stage_t(
        self,
        t: int,
        pi_prev: np.ndarray,
        B_prev: float,
        loss_matrix_all: np.ndarray,
        S_t_idx: np.ndarray,
        U_t_idx: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Stage t ≥ 2: minimize the chosen bound (plain‐KL/excess‐KL/emp‐Bernstein/unexp‐Bernstein).
        - pi_prev: posterior weights from stage t-1 (shape (K,))
        - B_prev: bound from stage t-1
        - loss_matrix_all: (K, total_n) 0-1 loss over the entire dataset
        - S_t_idx: indices for chunk S_t
        - U_t_idx: indices for U_t = S_t ∪ ... ∪ S_T
        Returns (pi_t_weights, B_t, chosen_gamma_t).
        """
        K, _ = loss_matrix_all.shape
        delta_t = self.delta / self.T
        S_t = loss_matrix_all[:, S_t_idx]  # (K, |S_t|)
        U_t = loss_matrix_all[:, U_t_idx]  # (K, |U_t|)
        m_t = U_t.shape[1]

        # Precompute the weighted‐loss of π_{t-1} on U_t
        weighted_loss_prev_U = pi_prev @ U_t  # shape (|U_t|)

        best_gamma = None
        best_pi_t = None
        best_Bt = None

        # For each candidate γ in descending order, attempt to find π_t that lowers the bound
        for gamma_candidate in sorted(self.gamma_grid, reverse=True):
            # 1) Build f_gamma(i, j) = loss_U[i,j] - gamma * weighted_loss_prev_U[j]
            f_gamma_U = U_t - gamma_candidate * weighted_loss_prev_U  # shape (K, m_t)

            # 2) Depending on bound_type, collect empirical statistics
            if self.bound_type == "split-kl":
                # The four possible values of f_gamma: b0=-γ, b1=0, b2=1-γ, b3=1
                b0 = -gamma_candidate
                b_vals = np.array([b0, 0.0, 1.0 - gamma_candidate, 1.0])
                alpha_vals = np.diff(b_vals)  # [γ, 1-γ, γ]
                # Build the 3 indicator slices of shape (3, K, m_t)
                indicator_slices = np.zeros((3, K, m_t), dtype=float)
                for j in range(1, 4):
                    indicator_slices[j - 1] = (f_gamma_U >= b_vals[j]).astype(float)
                # Fhat: shape (3, K) = empirical mean over U_t of each slice
                Fhat = indicator_slices.mean(axis=2)

            elif self.bound_type == "plain-kl":
                # We treat f_gamma as a continuous random variable in [-γ, 1].
                f_gamma_means = f_gamma_U.mean(axis=1)  # shape (K,)

            elif self.bound_type == "emp-bernstein":
                # Compute per-hypothesis empirical mean μ_i and empirical var v_i of f_gamma
                f_gamma_means = f_gamma_U.mean(axis=1)  # (K,)
                var_i = f_gamma_U.var(
                    axis=1, ddof=1
                )  # sample variance with denominator (m_t-1)
                mu_i = f_gamma_means.copy()
                v_i = var_i.copy()  # shape (K,)

            elif self.bound_type == "unexp-bernstein":
                # Compute per-hypothesis empirical mean μ_i and empirical second moment s_i of f_gamma
                f_gamma_means = f_gamma_U.mean(axis=1)  # (K,)
                s_i = (f_gamma_U * f_gamma_U).mean(axis=1)  # (K,)
                mu_i = f_gamma_means.copy()

            else:
                raise ValueError(f"Unknown bound_type='{self.bound_type}'")

            # 3) Define the objective for a fixed γ: v -> ε_t(π_t,γ) + γ * B_prev
            def objective_stage(v: np.ndarray) -> float:
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
                        b=1.0 + gamma_candidate,
                    )
                elif self.bound_type == "unexp-bernstein":
                    # — begin λ‐clipping for unexpected‐Bernstein —
                    b_val = 1.0 + gamma_candidate
                    eps_small = 1e-6
                    max_lam = (1.0 / b_val) - eps_small
                    if max_lam <= 0:
                        lambda_grid_t = np.array([eps_small])
                    else:
                        M = len(self.lambda_grid)
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
                    # — end λ‐clipping —
                else:
                    raise ValueError(f"Unknown bound_type='{self.bound_type}'")

                return eps + gamma_candidate * B_prev

            # Initialize v0 near log(pi_prev)
            v0_t = np.log(pi_prev + 1e-12)
            res_t = minimize(
                objective_stage,
                v0_t,
                method="Nelder-Mead",
                options={"maxiter": 500, "disp": False},
            )
            v_opt_t = res_t.x
            exp_vt = np.exp(v_opt_t - np.max(v_opt_t))
            w_opt_t = exp_vt / exp_vt.sum()

            # Compute B_t candidate
            B_t_candidate = objective_stage(v_opt_t)
            if B_t_candidate < B_prev:
                best_gamma = float(gamma_candidate)
                best_pi_t = w_opt_t
                best_Bt = B_t_candidate
                break

        # Fallback if no gamma lowered the bound
        if best_gamma is None:
            mid_idx = len(self.gamma_grid) // 2
            gamma_fallback = float(self.gamma_grid[mid_idx])

            # Recompute stats for fallback gamma
            f_gamma_U = U_t - gamma_fallback * weighted_loss_prev_U  # (K, m_t)

            if self.bound_type == "split-kl":
                b0 = -gamma_fallback
                b_vals = np.array([b0, 0.0, 1.0 - gamma_fallback, 1.0])
                alpha_vals = np.diff(b_vals)
                indicator_slices = np.zeros((3, K, m_t), dtype=float)
                for j in range(1, 4):
                    indicator_slices[j - 1] = (f_gamma_U >= b_vals[j]).astype(float)
                Fhat = indicator_slices.mean(axis=2)

            elif self.bound_type == "plain-kl":
                f_gamma_means = f_gamma_U.mean(axis=1)

            elif self.bound_type == "emp-bernstein":
                f_gamma_means = f_gamma_U.mean(axis=1)
                var_i = f_gamma_U.var(axis=1, ddof=1)
                mu_i = f_gamma_means.copy()
                v_i = var_i.copy()

            elif self.bound_type == "unexp-bernstein":
                f_gamma_means = f_gamma_U.mean(axis=1)
                s_i = (f_gamma_U * f_gamma_U).mean(axis=1)
                mu_i = f_gamma_means.copy()

            # Redefine objective for fallback
            def objective_fallback(v: np.ndarray) -> float:
                exp_v = np.exp(v - np.max(v))
                w = exp_v / exp_v.sum()
                KL_div = float(np.sum(w * np.log((w + 1e-12) / (pi_prev + 1e-12))))

                if self.bound_type == "split-kl":
                    eps = compute_split_kl_bound(
                        w=w,
                        Fhat=Fhat,
                        alpha_vals=alpha_vals,
                        b0=-gamma_fallback,
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
                        b=1.0 + gamma_fallback,
                    )
                elif self.bound_type == "unexp-bernstein":
                    # — begin λ‐clipping for fallback —
                    b_val = 1.0 + gamma_fallback
                    eps_small = 1e-6
                    max_lam = (1.0 / b_val) - eps_small
                    if max_lam <= 0:
                        lambda_grid_t = np.array([eps_small])
                    else:
                        M = len(self.lambda_grid)
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
                    # — end λ‐clipping —
                else:
                    raise ValueError(f"Unknown bound_type='{self.bound_type}'")

                return eps + gamma_fallback * B_prev

            res_fb = minimize(
                objective_fallback,
                v0_t,
                method="Nelder-Mead",
                options={"maxiter": 500, "disp": False},
            )
            v_opt_fb = res_fb.x
            exp_vfb = np.exp(v_opt_fb - np.max(v_opt_fb))
            w_opt_fb = exp_vfb / exp_vfb.sum()

            best_gamma = gamma_fallback
            best_pi_t = w_opt_fb
            best_Bt = float(objective_fallback(v_opt_fb))

        return best_pi_t, best_Bt, best_gamma

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RecursivePACBayesEnsemble":
        """
        Fit the Recursive PAC‐Bayes ensemble on (X,y). Splits data into T geometric chunks,
        then performs Stage 1 (plain PAC‐Bayes) and Stages 2..T (chosen bound).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_total = X.shape[0]

        # If multiclass, record the set of classes
        if self.task == "multiclass":
            # Assume all classifiers support .classes_ and that they share the same set of classes
            # We'll take classes_ from the first base learner after fitting.
            pass  # Placeholder; we'll set it right after training base learners.

        # 1) Generate T geometric split on indices
        chunks = self._geometric_split_indices(n_total)
        self.chunk_indices_ = chunks

        # 2) Fit each base learner on the FULL training set (to avoid single‐class chunk issues)
        self.trained_classifiers_ = []
        for name, estimator in self.base_learners:
            clf_full = clone(estimator)
            clf_full.fit(X, y)
            self.trained_classifiers_.append((name, clf_full))

        # If multiclass, now store the common classes_ array
        if self.task == "multiclass":
            # Take .classes_ from the first fitted classifier
            self.classes_ = self.trained_classifiers_[0][1].classes_

        # 3) Build loss_matrix_all: shape (K, n_total)
        classifiers_only = [clf for (_, clf) in self.trained_classifiers_]
        loss_matrix_all = self._compute_loss_matrix(classifiers_only, X, y)

        # ============ Stage 1 ============
        S1_idx = chunks[0]
        loss_S1 = loss_matrix_all[:, S1_idx].mean(
            axis=1
        )  # average 0-1 loss per classifier on S1
        pi1, B1 = self._optimize_stage1(loss_S1, len(S1_idx))
        self.pi_list_.append(pi1)
        self.bounds_.append(B1)
        self._log(f"Stage 1 → π₁ = {np.round(pi1,4)}, B₁ = {B1:.4f}")

        # ============ Stages 2..T ============
        pi_prev = pi1.copy()
        B_prev = B1

        # Precompute cumulative union indices for future sets
        future_union_indices: List[np.ndarray] = []
        for t in range(self.T):
            all_future = np.concatenate(chunks[t:], axis=0)
            future_union_indices.append(all_future)

        for t in range(2, self.T + 1):
            S_t_idx = chunks[t - 1]
            U_t_idx = future_union_indices[t - 1]
            pi_t, B_t, gamma_t = self._optimize_stage_t(
                t, pi_prev, B_prev, loss_matrix_all, S_t_idx, U_t_idx
            )
            self.pi_list_.append(pi_t)
            self.bounds_.append(B_t)
            self._log(
                f"Stage {t} → γₜ = {gamma_t:.3f}, πₜ = {np.round(pi_t,4)}, Bₜ = {B_t:.4f}"
            )
            pi_prev, B_prev = pi_t.copy(), B_t

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels on X via weighted‐vote by π_T.  Assumes fit() has been called.
        - If task="binary", returns an array in {0,1}.
        - If task="multiclass", returns an array in the same class labels as .classes_.
        """
        X = np.asarray(X)
        if not self.pi_list_:
            raise ValueError("Model not fitted. Call fit(X,y) first.")
        pi_T = self.pi_list_[-1]
        classifiers_only = [clf for (_, clf) in self.trained_classifiers_]
        K = len(classifiers_only)
        n = X.shape[0]

        preds_matrix = np.zeros((K, n), dtype=int)
        for i, clf in enumerate(classifiers_only):
            preds_matrix[i, :] = clf.predict(X)

        if self.task == "binary":
            # Weighted‐vote for class “1” vs. “0”
            ensemble_preds = np.zeros(n, dtype=int)
            for j in range(n):
                vote_weight_1 = np.sum(pi_T[preds_matrix[:, j] == 1])
                ensemble_preds[j] = 1 if vote_weight_1 >= 0.5 else 0
            return ensemble_preds

        else:  # multiclass
            C = len(self.classes_)
            ensemble_preds = np.zeros(n, dtype=int)
            for j in range(n):
                vote_sums = np.zeros(C, dtype=float)
                for c_idx, c_label in enumerate(self.classes_):
                    vote_sums[c_idx] = np.sum(pi_T[preds_matrix[:, j] == c_label])
                best_idx = np.argmax(vote_sums)
                ensemble_preds[j] = self.classes_[best_idx]
            return ensemble_preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return the weighted‐vote probability for each class.
        - If task="binary", returns shape (n, 2) with columns [P(class=0), P(class=1)].
        - If task="multiclass", returns shape (n, C), summing to 1 across each row.
        """
        X = np.asarray(X)
        if not self.pi_list_:
            raise ValueError("Model not fitted. Call fit(X,y) first.")
        pi_T = self.pi_list_[-1]
        classifiers_only = [clf for (_, clf) in self.trained_classifiers_]
        K = len(classifiers_only)
        n = X.shape[0]

        preds_matrix = np.zeros((K, n), dtype=int)
        for i, clf in enumerate(classifiers_only):
            preds_matrix[i, :] = clf.predict(X)

        if self.task == "binary":
            # Return shape (n, 2): [P(class=0), P(class=1)]
            proba = np.zeros((n, 2), dtype=float)
            for j in range(n):
                weight_for_1 = np.sum(pi_T[preds_matrix[:, j] == 1])
                proba[j, 1] = weight_for_1
                proba[j, 0] = 1.0 - weight_for_1
            return proba

        else:  # multiclass
            C = len(self.classes_)
            proba = np.zeros((n, C), dtype=float)
            for j in range(n):
                for c_idx, c_label in enumerate(self.classes_):
                    proba[j, c_idx] = np.sum(pi_T[preds_matrix[:, j] == c_label])
            return proba  # each row sums to 1 because sum(pi_T) == 1

    def get_posteriors(self) -> List[np.ndarray]:
        """
        Returns the list of posterior weight vectors [π₁, π₂, ..., π_T].
        """
        return self.pi_list_

    def get_bounds(self) -> List[float]:
        """
        Returns the list of stage bounds [B₁, B₂, ..., B_T].
        """
        return self.bounds_


if __name__ == "__main__":
    # =============================
    # Example usage & testing on Breast Cancer dataset (binary)
    # =============================
    import sklearn

    logger.info(f"scikit‐learn version: {sklearn.__version__}")

    # 1. Load data
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 2. Split into trainval (80%) and test (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Define base learners
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    base_learners = [
        ("LogisticRegression", LogisticRegression(max_iter=10000, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("SVM", SVC(kernel="rbf", gamma="scale", probability=True, random_state=42)),
    ]

    # 4. Instantiate the RecursivePACBayesEnsemble for binary
    rpb_binary = RecursivePACBayesEnsemble(
        base_learners=base_learners,
        T=2,
        delta=0.05,
        gamma_grid=np.linspace(0.1, 0.9, 9),
        bound_type="split-kl",  # try "plain-kl", "emp-bernstein", "unexp-bernstein"
        random_state=42,
        verbose=True,
        task="binary",
    )

    # 5. Fit on trainval
    rpb_binary.fit(X_trainval, y_trainval)

    # 6. Print posteriors and bounds
    posteriors = rpb_binary.get_posteriors()
    bounds = rpb_binary.get_bounds()
    for t in range(len(posteriors)):
        logger.info(
            f"π_{t+1} = {np.round(posteriors[t], 4)}, B_{t+1} = {bounds[t]:.4f}"
        )

    # 7. Evaluate on test set
    y_pred = rpb_binary.predict(X_test)
    test_error = np.mean(y_pred != y_test)
    logger.info(f"Ensemble test error (binary): {test_error:.4f}")

    # Compare to individual classifiers
    for name, clf in base_learners:
        clf_clone = clone(clf).fit(X_trainval, y_trainval)
        ind_err = np.mean(clf_clone.predict(X_test) != y_test)
        logger.info(f"Test error ({name}): {ind_err:.4f}")

    # =============================
    # Example usage & testing on a multiclass dataset
    # =============================
    from sklearn.datasets import load_iris

    data2 = load_iris()
    X2, y2 = data2.data, data2.target

    # 80/20 train/test split
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=0, stratify=y2
    )

    # Use the same base learners (they handle multiclass automatically)
    rpb_multi = RecursivePACBayesEnsemble(
        base_learners=base_learners,
        T=2,
        delta=0.05,
        gamma_grid=np.linspace(0.1, 0.9, 9),
        bound_type="unexp-bernstein",
        random_state=0,
        verbose=True,
        task="multiclass",
    )

    # Fit on the multiclass data
    rpb_multi.fit(X2_train, y2_train)

    # Print posteriors and bounds
    posteriors_m = rpb_multi.get_posteriors()
    bounds_m = rpb_multi.get_bounds()
    for t in range(len(posteriors_m)):
        logger.info(
            f"[Multiclass] π_{t+1} = {np.round(posteriors_m[t], 4)}, B_{t+1} = {bounds_m[t]:.4f}"
        )

    # Evaluate on test set
    y2_pred = rpb_multi.predict(X2_test)
    test_error_m = np.mean(y2_pred != y2_test)
    logger.info(f"Ensemble test error (multiclass): {test_error_m:.4f}")

    # Compare to individual classifiers on multiclass
    for name, clf in base_learners:
        clf_clone = clone(clf).fit(X2_train, y2_train)
        ind_err_m = np.mean(clf_clone.predict(X2_test) != y2_test)
        logger.info(f"[Multiclass] Test error ({name}): {ind_err_m:.4f}")

    # 8. (Optional) Visualizations for multiclass are less common for 3‐class; skip if desired.
