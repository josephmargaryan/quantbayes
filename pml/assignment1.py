import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Helper functions
# ============================================================


def generate_data(N, theta_true, sigma_y2, cov_x):
    """
    Sample inputs x ~ N(0, cov_x) and labels y ~ N(x^T theta_true, sigma_y2)
    """
    d = len(theta_true)
    x = np.random.multivariate_normal(mean=np.zeros(d), cov=cov_x, size=N)
    noise = np.random.normal(loc=0.0, scale=np.sqrt(sigma_y2), size=N)
    y = x @ theta_true + noise
    return x, y


def bayes_linreg_posterior(X, y, sigma_y2, prior_mean, prior_cov):
    """
    Posterior for Bayesian linear regression with Gaussian prior
    theta ~ N(prior_mean, prior_cov), y|x,theta ~ N(x^T theta, sigma_y2).

    Returns posterior mean mu_N and covariance Sigma_N.
    """
    d = X.shape[1]
    prior_prec = np.linalg.inv(prior_cov)  # Λ_0
    Lambda_N = prior_prec + (1.0 / sigma_y2) * X.T @ X  # posterior precision
    Sigma_N = np.linalg.inv(Lambda_N)  # posterior covariance
    # posterior mean: mu_N = Sigma_N (Λ_0 m_0 + sigma^{-2} X^T y)
    mu_N = Sigma_N @ (prior_prec @ prior_mean + (1.0 / sigma_y2) * X.T @ y)
    return mu_N, Sigma_N


def gaussian_pdf_grid(mu, Sigma, theta1_range, theta2_range, num_points=200):
    """
    Compute the 2D Gaussian pdf on a grid over theta in [-3,3]^2 (or other range).
    """
    t1 = np.linspace(theta1_range[0], theta1_range[1], num_points)
    t2 = np.linspace(theta2_range[0], theta2_range[1], num_points)
    T1, T2 = np.meshgrid(t1, t2)
    pos = np.stack([T1, T2], axis=-1)  # shape (num_points, num_points, 2)

    Sigma_inv = np.linalg.inv(Sigma)
    det_Sigma = np.linalg.det(Sigma)
    diff = pos - mu  # broadcasting: (...,2)

    # exponent = -0.5 * (theta - mu)^T Sigma^{-1} (theta - mu)
    exponent = -0.5 * np.einsum("...i,ij,...j->...", diff, Sigma_inv, diff)
    norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_Sigma))
    pdf = norm_const * np.exp(exponent)
    return T1, T2, pdf


def predictive_variance_grid(Sigma_post, sigma_y2, x1_range, x2_range, num_points=200):
    """
    Posterior predictive variance:
    Var[y* | x*, D] = sigma_y2 + x*^T Sigma_post x*
    evaluated on a grid over x in [-3,3]^2 (or other range).
    """
    x1 = np.linspace(x1_range[0], x1_range[1], num_points)
    x2 = np.linspace(x2_range[0], x2_range[1], num_points)
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.stack([X1, X2], axis=-1)  # shape (num_points, num_points, 2)

    var_param = np.einsum("...i,ij,...j->...", pos, Sigma_post, pos)
    var_pred = sigma_y2 + var_param
    return X1, X2, var_pred


# ============================================================
# Main script: A–D
# ============================================================

np.random.seed(0)  # for reproducibility

# True parameters and noise
theta_true = np.array([-1.0, 1.0])
sigma_y2 = 0.1  # observation noise variance

# Prior: theta ~ N(0, I)
prior_mean = np.zeros(2)
prior_cov = np.eye(2)

# Number of data points
N = 20

# ------------------------------------------------------------
# Case 1: x ~ N(0, I_2)
# ------------------------------------------------------------

Sigma_x1 = np.eye(2)  # covariance of x in case 1

# A. Generate dataset
X1, y1 = generate_data(N, theta_true, sigma_y2, Sigma_x1)

# B. Posterior over theta
mu_post1, Sigma_post1 = bayes_linreg_posterior(X1, y1, sigma_y2, prior_mean, prior_cov)
print("Case 1: posterior mean (theta) =", mu_post1)
print("Case 1: posterior covariance (theta) =\n", Sigma_post1)

# Compute posterior pdf for theta in [-3, 3]^2
theta_range = (-3.0, 3.0)
T1_1, T2_1, pdf_theta1 = gaussian_pdf_grid(
    mu_post1, Sigma_post1, theta_range, theta_range
)

# C. Posterior predictive variance on x in [-3,3]^2
x_range = (-3.0, 3.0)
X1_grid_1, X2_grid_1, var_pred1 = predictive_variance_grid(
    Sigma_post1, sigma_y2, x_range, x_range
)


# ------------------------------------------------------------
# Case 2: x ~ N(0, Sigma_x2) with Sigma_x2 = diag(0.1, 1)
# ------------------------------------------------------------

Sigma_x2 = np.array([[0.1, 0.0], [0.0, 1.0]])

# D. Generate second dataset
X2, y2 = generate_data(N, theta_true, sigma_y2, Sigma_x2)

# Posterior over theta for case 2
mu_post2, Sigma_post2 = bayes_linreg_posterior(X2, y2, sigma_y2, prior_mean, prior_cov)
print("\nCase 2: posterior mean (theta) =", mu_post2)
print("Case 2: posterior covariance (theta) =\n", Sigma_post2)

# Posterior pdf for theta in [-3,3]^2
T1_2, T2_2, pdf_theta2 = gaussian_pdf_grid(
    mu_post2, Sigma_post2, theta_range, theta_range
)

# Posterior predictive variance on x in [-3,3]^2
X1_grid_2, X2_grid_2, var_pred2 = predictive_variance_grid(
    Sigma_post2, sigma_y2, x_range, x_range
)


# ============================================================
# Plotting
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Posterior over theta (case 1)
ax = axes[0, 0]
cs = ax.contourf(T1_1, T2_1, pdf_theta1, levels=30)
ax.set_title("Posterior $p(\\theta|D)$ (case 1: $x \\sim N(0, I)$)")
ax.set_xlabel("$\\theta_1$")
ax.set_ylabel("$\\theta_2$")
fig.colorbar(cs, ax=ax)

# Posterior over theta (case 2)
ax = axes[0, 1]
cs = ax.contourf(T1_2, T2_2, pdf_theta2, levels=30)
ax.set_title("Posterior $p(\\theta|D)$ (case 2: $x \\sim N(0, \\Sigma_x)$)")
ax.set_xlabel("$\\theta_1$")
ax.set_ylabel("$\\theta_2$")
fig.colorbar(cs, ax=ax)

# Predictive variance (case 1)
ax = axes[1, 0]
im = ax.imshow(
    var_pred1,
    origin="lower",
    extent=[x_range[0], x_range[1], x_range[0], x_range[1]],
    aspect="auto",
)
ax.set_title("Predictive var $\\mathrm{Var}[y|x,D]$ (case 1)")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
fig.colorbar(im, ax=ax)

# Predictive variance (case 2)
ax = axes[1, 1]
im = ax.imshow(
    var_pred2,
    origin="lower",
    extent=[x_range[0], x_range[1], x_range[0], x_range[1]],
    aspect="auto",
)
ax.set_title("Predictive var $\\mathrm{Var}[y|x,D]$ (case 2)")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

# ============================================================
# (Optional) Experiment: smaller label noise
# Change sigma_y2 = 0.01 and re-run the posterior + predictive parts.
# You should see tighter posterior over theta and lower predictive variance.
# ============================================================

# Example snippet (commented out by default):

# sigma_y2_small = 0.01
# X1_small, y1_small = generate_data(N, theta_true, sigma_y2_small, Sigma_x1)
# mu_post1_small, Sigma_post1_small = bayes_linreg_posterior(
#     X1_small, y1_small, sigma_y2_small, prior_mean, prior_cov
# )
# print("\nWith smaller noise sigma_y^2 = 0.01 (case 1):")
# print("Posterior covariance (theta) =\n", Sigma_post1_small)
#
# T1_1s, T2_1s, pdf_theta1_small = gaussian_pdf_grid(
#     mu_post1_small, Sigma_post1_small, theta_range, theta_range
# )
# X1_grid_1s, X2_grid_1s, var_pred1_small = predictive_variance_grid(
#     Sigma_post1_small, sigma_y2_small, x_range, x_range
# )
#
# # Then re-plot to compare with the sigma_y2 = 0.1 case.


"""
Short conceptual notes for your writeup

Difference between the two input distributions:
In case 2, the first input dimension has much smaller variance (0.1 vs 1). That means the data carry less information about the corresponding weight component $\theta_1$ (since $\sum_n x_{n1}^2$ is small), so the posterior covariance in that direction is larger. You should see more elongated contours in the posterior over $\theta$, and this anisotropy is reflected in the predictive variance surface.

Effect of changing $\sigma_y^2 = 0.1 \to 0.01$:
Smaller observation noise gives a higher effective precision of the likelihood, so the posterior over $\theta$ becomes more concentrated (smaller covariance), and the predictive variance $\mathrm{Var}[y\mid x,D] = \sigma_y^2 + x^\top \Sigma_N x$ decreases both because $\sigma_y^2$ itself is smaller and because $\Sigma_N$ shrinks.
"""
