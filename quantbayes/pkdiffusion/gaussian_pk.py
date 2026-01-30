# quantbayes/pkdiffusion/gaussian_pk.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GaussianND:
    """Multivariate Gaussian N(mean, cov)."""

    mean: np.ndarray  # (d,)
    cov: np.ndarray  # (d,d)

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])


def _asvec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(-1)


def marginal_1d(g: GaussianND, a: np.ndarray) -> tuple[float, float]:
    """Return mean/var of Y = a^T X when X ~ g."""
    a = _asvec(a)
    mu = float(a @ g.mean)
    var = float(a @ g.cov @ a)
    return mu, var


def pk_update_linear_gaussian(
    prior: GaussianND,
    a: np.ndarray,
    *,
    q_mean: float,
    q_var: float,
) -> GaussianND:
    """
    PK update for linear coarse variable Y=a^T X when prior is Gaussian and
    evidence q is Gaussian N(q_mean, q_var).

    This returns the *exact* updated Gaussian (in this special case).
    Key property: the updated marginal of Y is exactly q.

    Derivation uses conditional decomposition:
      X = E[X|Y] + (X - E[X|Y]), with Cov(X|Y) constant under Gaussian prior,
    and then replace Y~p_Y with Y~q.
    """
    a = _asvec(a)
    mu_y, var_y = marginal_1d(prior, a)
    if q_var <= 0:
        raise ValueError("q_var must be > 0")
    if var_y <= 0:
        raise ValueError("prior var_y must be > 0")

    # b = Cov(X,Y) / Var(Y) = Σ a / (a^T Σ a)
    b = (prior.cov @ a) / var_y  # (d,)

    # Mean update: E_q[E[X|Y]] = mu + b*(E_q[Y]-E_p[Y])
    mu_new = prior.mean + b * (float(q_mean) - mu_y)

    # Cov update:
    #   Cov_new = E_q[Cov(X|Y)] + Cov_q(E[X|Y])
    #          = (Σ - b b^T Var_p(Y)) + b b^T Var_q(Y)
    cov_new = prior.cov + np.outer(b, b) * (float(q_var) - var_y)

    # Symmetrize for numerical cleanliness
    cov_new = 0.5 * (cov_new + cov_new.T)
    return GaussianND(mean=mu_new, cov=cov_new)


def sample_gaussian(g: GaussianND, *, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from N(mean,cov). Returns shape (n,d)."""
    return rng.multivariate_normal(mean=g.mean, cov=g.cov, size=int(n))


def alternating_pk_updates(
    prior: GaussianND,
    a_list: list[np.ndarray],
    q_means: list[float],
    q_vars: list[float],
    *,
    num_iters: int = 25,
) -> list[GaussianND]:
    """
    Alternating PK updates (IPF-style) in the Gaussian-linear setting.

    Because each PK update enforces one marginal exactly, alternating updates
    can converge to a distribution that (approximately) satisfies all constraints
    simultaneously (when compatible).

    Returns the list of intermediate Gaussians, including the start prior.
    """
    if not (len(a_list) == len(q_means) == len(q_vars)):
        raise ValueError("a_list, q_means, q_vars must have same length")
    cur = prior
    hist = [cur]
    for _ in range(int(num_iters)):
        for a, m, v in zip(a_list, q_means, q_vars):
            cur = pk_update_linear_gaussian(cur, a, q_mean=m, q_var=v)
            hist.append(cur)
    return hist
