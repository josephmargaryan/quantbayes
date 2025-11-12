import numpy as np
import matplotlib.pyplot as plt

# ----- Choose spectrum -----
mu = 0.5  # strong convexity (small eigenvalue)
lam = 5.0  # smoothness (large eigenvalue)
alpha = 4.0  # ridge strength (try 0.0, 1.0, 4.0, 20.0)

# Construct a symmetric matrix with eigenvalues [mu, lam]
Q = np.array(
    [[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]]
)  # just a rotation
H = Q @ np.diag([mu, lam]) @ Q.T

# Minimizer and loss
theta_star = np.array([1.0, -1.0])


def f(theta):
    d = theta - theta_star
    return 0.5 * d @ H @ d


def f_reg(theta):
    return f(theta) + 0.5 * alpha * np.dot(theta, theta)


def grad(theta):
    return H @ (theta - theta_star)


def grad_reg(theta):
    return grad(theta) + alpha * theta


# Smoothness/strong-convexity constants
L0, mu0 = lam, mu
L1, mu1 = lam + alpha, mu + alpha
kappa0 = L0 / mu0
kappa1 = L1 / mu1

# Gradient descent step sizes (robust choice 2/(L+mu))
step0 = 2.0 / (L0 + mu0)
step1 = 2.0 / (L1 + mu1)

# Paths
theta0 = np.array([3.0, 3.0])


def run_path(grad_fun, step, n=12):
    th = [theta0]
    for _ in range(n):
        th.append(th[-1] - step * grad_fun(th[-1]))
    return np.array(th)


path0 = run_path(grad, step0)
path1 = run_path(grad_reg, step1)

# Grid for contours
pad = 1.5
mins = np.minimum(path0.min(axis=0), path1.min(axis=0)) - pad
maxs = np.maximum(path0.max(axis=0), path1.max(axis=0)) + pad
xs = np.linspace(mins[0], maxs[0], 500)
ys = np.linspace(mins[1], maxs[1], 500)
X, Y = np.meshgrid(xs, ys)
DX, DY = X - theta_star[0], Y - theta_star[1]
Z0 = 0.5 * (H[0, 0] * DX**2 + 2 * H[0, 1] * DX * DY + H[1, 1] * DY**2)
Z1 = Z0 + 0.5 * alpha * (X**2 + Y**2)

# Plot side-by-side: unregularized vs regularized
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)

ax = axes[0]
cs = ax.contour(X, Y, Z0, levels=20, linewidths=1.0)
ax.plot(path0[:, 0], path0[:, 1], "-o", markersize=3, label="GD path")
for i in range(len(path0) - 1):
    ax.annotate(
        "", xy=path0[i + 1], xytext=path0[i], arrowprops=dict(arrowstyle="->", lw=1.0)
    )
ax.plot(theta_star[0], theta_star[1], "r*", ms=12, label="min")
ax.set_title(f"No L2: κ={kappa0:.2f}")
ax.set_aspect("equal", "box")
ax.legend(loc="upper right")
ax.set_xlabel("θ1")
ax.set_ylabel("θ2")

ax = axes[1]
cs = ax.contour(X, Y, Z1, levels=20, linewidths=1.0)
ax.plot(path1[:, 0], path1[:, 1], "-o", markersize=3, label="GD path")
for i in range(len(path1) - 1):
    ax.annotate(
        "", xy=path1[i + 1], xytext=path1[i], arrowprops=dict(arrowstyle="->", lw=1.0)
    )
ax.plot(theta_star[0], theta_star[1], "r*", ms=12, label="min")
ax.set_title(f"With L2 α={alpha}: κ={(kappa1):.2f}")
ax.set_aspect("equal", "box")
ax.legend(loc="upper right")
ax.set_xlabel("θ1")

plt.tight_layout()
plt.show()
