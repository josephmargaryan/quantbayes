import numpy as np
from scipy.optimize import minimize_scalar
from math import log, ceil, sqrt, exp

# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================


def kl_divergence(q, p):
    """
    Compute the binary KL divergence:
    kl(q || p) = q log(q/p) + (1-q) log((1-q)/(1-p))
    Note: This function is used when q and p are in (0,1).
    """
    # To avoid division-by-zero or log-of-zero, add epsilon.
    eps = 1e-10
    q = np.clip(q, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)
    return q * log(q / p) + (1 - q) * log((1 - q) / (1 - p))


def weighted_kl_divergence(post, prior):
    """
    Compute the KL divergence between two discrete distributions
    post: posterior weights (numpy array summing to 1)
    prior: prior weights (numpy array summing to 1)
    Returns KL(post || prior)
    """
    eps = 1e-10
    post = np.clip(post, eps, 1.0)
    prior = np.clip(prior, eps, 1.0)
    return np.sum(post * np.log(post / prior))


# ============================================================
# 2. PAC-BAYES BOUND FUNCTIONS
# ============================================================


def pac_bayes_kl_bound(empirical_loss, posterior, prior, n, delta):
    """
    Compute the PAC-Bayes-kl bound in its implicit form:
      kl( Empirical Loss || True Loss ) <= ( KL(posterior||prior) + ln((n+1)/delta) ) / n
    where:
      empirical_loss: weighted empirical loss, e.g., E_{h ~ posterior}[ \hat{L}(h) ]
      posterior: numpy array of weights (posterior distribution) over the hypothesis set.
      prior: numpy array of weights (prior distribution) over the hypothesis set.
      n: sample size
      delta: confidence level parameter (e.g., 0.05)

    Since the kl divergence (binary) is defined for scalars, we need to numerically
    invert the inequality to solve for an upper bound on the true loss.
    """
    # Compute the KL divergence between the posterior and the prior.
    kl_p = weighted_kl_divergence(posterior, prior)
    # Define the right-hand-side of the bound inequality:
    rhs = (kl_p + log((n + 1) / delta)) / n

    # We must solve for p in: kl(empirical_loss || p) = rhs.
    # Because the binary kl divergence is monotonic in p (for fixed q in a proper region),
    # we use a numerical root finding procedure.

    def objective(p):
        # Return the difference between kl(empirical_loss || p) and rhs.
        return kl_divergence(empirical_loss, p) - rhs

    # p must be between empirical_loss and 1 (the true loss should be at least the empirical loss)
    # It might be helpful to restrict p to [empirical_loss, 1 - 1e-10]
    lower = empirical_loss
    upper = 1 - 1e-10

    # Use minimize_scalar with bounded region.
    res = minimize_scalar(
        lambda p: abs(objective(p)), bounds=(lower, upper), method="bounded"
    )

    if res.success:
        bound = res.x
    else:
        # As a fallback, use a simple relaxation such as Pinsker's inequality:
        bound = empirical_loss + sqrt((kl_p + log((n + 1) / delta)) / (2 * n))
    return bound


def pac_bayes_empirical_bernstein_bound(
    empirical_loss, empirical_var, posterior, prior, n, delta, c1=1.1, c2=1.1
):
    """
    Compute the PAC-Bayes-Empirical-Bernstein (PB-EB) bound.

    Parameters:
      empirical_loss: weighted empirical loss, e.g., E_{h ~ posterior}[ \hat{L}(h) ]
      empirical_var: weighted empirical variance, e.g., E_{h ~ posterior}[ \hat{V}_n(h) ]
      posterior: numpy array for the posterior distribution over hypotheses.
      prior: numpy array for the prior distribution over hypotheses.
      n: sample size.
      delta: confidence level parameter.
      c1, c2: trade-off parameters (typically just above 1).

    The function returns an upper bound on the true loss L(G_rho) according to Theorem 4.
    """
    kl_p = weighted_kl_divergence(posterior, prior)

    # First compute nu1 and nu2 according to the theorem; note that some constants such as (e-2) are used.
    # For nu1 (from Theorem 2):
    nu1 = ceil((1 / log(c1)) * log(sqrt(((exp(1) - 2) * n) / (4 * log(1 / delta))))) + 1

    # Compute the bound on the average variance using Theorem 3.
    # We define V_bound = empirical_var + additional term
    add_term = (1 + c2) * sqrt(
        empirical_var * (kl_p + log(nu1 / delta)) / (2 * (n - 1))
    ) + (2 * c2 * (kl_p + log(nu1 / delta))) / (n - 1)
    V_bound = empirical_var + add_term
    # Truncate V_bound to at most 1/4 (since loss in [0,1] implies variance <= 1/4)
    V_bound = min(V_bound, 1 / 4)

    # Now apply Theorem 2/PB-Bernstein bound:
    term = (1 + c1) * sqrt(((exp(1) - 2) * V_bound * (kl_p + log(2 * nu1 / delta))) / n)
    bernstein_bound = empirical_loss + term

    # Check if the condition on the posterior is met:
    if sqrt((kl_p + log(2 * nu1 / delta)) / ((exp(1) - 2) * V_bound)) > sqrt(n):
        # Otherwise, use the fallback bound.
        bernstein_bound = empirical_loss + 2 * (kl_p + log(2 * nu1 / delta)) / n

    return bernstein_bound


# ============================================================
# 3. EXAMPLE USAGE ON A SYNTHETIC ENSEMBLE
# ============================================================


def example_synthetic():
    """
    In this example, we assume that we have an ensemble of models (or hypotheses)
    and for each we know:
      - the empirical loss (computed on a sample of size n)
      - the empirical variance (the unbiased variance estimate of the loss)
    We then consider a candidate posterior distribution (e.g. a weighted ensemble)
    and compute both the PAC-Bayes-kl and PB-EB bounds.
    """
    # Suppose we have 5 hypotheses.
    num_hypotheses = 5
    n = 1000  # sample size
    delta = 0.05  # confidence level

    # Example: Empirical losses (each in [0, 1])
    empirical_losses = np.array([0.05, 0.10, 0.15, 0.08, 0.12])
    # Example: Empirical variances for each hypothesis.
    # For binary losses, the variance is L*(1-L) but we allow arbitrary numbers in [0, 1/4].
    empirical_variances = np.array(
        [0.05 * 0.95, 0.10 * 0.90, 0.15 * 0.85, 0.08 * 0.92, 0.12 * 0.88]
    )

    # Candidate posterior: for instance, we choose weights that emphasize lower loss.
    # Here, we use softmax over negative empirical loss.
    posterior = np.exp(-5 * empirical_losses)
    posterior = posterior / np.sum(posterior)

    # Prior distribution: could be uniform or set to reflect our beliefs
    prior = np.ones(num_hypotheses) / num_hypotheses

    # Compute the weighted empirical loss and variance:
    weighted_empirical_loss = np.dot(posterior, empirical_losses)
    weighted_empirical_var = np.dot(posterior, empirical_variances)

    print("Posterior distribution:", posterior)
    print("Weighted Empirical Loss:", weighted_empirical_loss)
    print("Weighted Empirical Variance:", weighted_empirical_var)

    # Compute the PAC-Bayes-kl bound:
    kl_bound = pac_bayes_kl_bound(weighted_empirical_loss, posterior, prior, n, delta)
    print("PAC-Bayes-kl Bound on True Loss: {:.4f}".format(kl_bound))

    # Compute the PAC-Bayes-Empirical-Bernstein bound:
    eb_bound = pac_bayes_empirical_bernstein_bound(
        weighted_empirical_loss,
        weighted_empirical_var,
        posterior,
        prior,
        n,
        delta,
        c1=1.1,
        c2=1.1,
    )
    print("PAC-Bayes-Empirical-Bernstein Bound on True Loss: {:.4f}".format(eb_bound))


if __name__ == "__main__":
    example_synthetic()
