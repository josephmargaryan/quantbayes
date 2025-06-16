import numpy as np
import matplotlib.pyplot as plt

# reproducibility
np.random.seed(0)

# parameters
n, delta = 100, 0.05
K = 2
p_mid_grid = np.linspace(0, 1, 101)


# KL helpers
def kl_div(p_hat, q):
    first = p_hat * np.log(p_hat / q) if p_hat > 0 else 0.0
    second = (1 - p_hat) * np.log((1 - p_hat) / (1 - q)) if p_hat < 1 else 0.0
    return first + second


def kl_inv(p_hat, eps, tol=1e-10):
    if p_hat >= 1.0:
        return 1.0
    low, high = p_hat, 1.0
    while high - low > tol:
        mid = 0.5 * (low + high)
        if kl_div(p_hat, mid) > eps:
            high = mid
        else:
            low = mid
    return low


# epsilons
epsilon_kl = np.log(1 / delta) / n
epsilon_sp = np.log(K / delta) / n

# storage
kl_diff = np.zeros_like(p_mid_grid)
sp_diff = np.zeros_like(p_mid_grid)

for idx, p_mid in enumerate(p_mid_grid):
    p0 = (1 - p_mid) / 2
    # one sample per grid point
    sample = np.random.choice([0.0, 0.5, 1.0], size=n, p=[p0, p_mid, p0])
    hat = sample.mean()

    # standard KL
    q = kl_inv(hat, epsilon_kl)
    kl_diff[idx] = max(q - hat, 0)

    # split-KL
    h1 = (sample >= 0.5).mean()
    h2 = (sample >= 1.0).mean()
    q1 = kl_inv(h1, epsilon_sp)
    q2 = kl_inv(h2, epsilon_sp)
    sp_diff[idx] = max(0.5 * q1 + 0.5 * q2 - hat, 0)

# plot
plt.figure(figsize=(8, 5))
plt.plot(p_mid_grid, kl_diff, label="KL bound on $p-\\hat p_n$")
plt.plot(p_mid_grid, sp_diff, label="Split-KL bound on $p-\\hat p_n$")
plt.xlabel("$p_{1/2}$")
plt.ylabel("Bound on $p - \\hat p_n$")
plt.legend()
plt.tight_layout()
plt.show()
