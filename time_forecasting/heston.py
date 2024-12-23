import numpy as np
import matplotlib.pyplot as plt


def heston_model(
    S0=100,
    v0=0.04,
    mu=0.05,
    kappa=2.0,
    theta=0.04,
    sigma_v=0.2,
    rho=-0.7,
    T=1,
    N=1000,
    paths=10,
):
    """
    Simulates stock prices using the Heston stochastic volatility model.

    Parameters:
        S0 (float): Initial stock price.
        v0 (float): Initial variance.
        mu (float): Drift rate.
        kappa (float): Speed of mean reversion.
        theta (float): Long-term variance (mean reversion level).
        sigma_v (float): Volatility of variance (vol-of-vol).
        rho (float): Correlation between stock returns and volatility.
        T (float): Time horizon.
        N (int): Number of time steps.
        paths (int): Number of simulated paths.

    Returns:
        S (ndarray): Simulated stock price paths.
        v (ndarray): Simulated variance paths.
    """
    dt = T / N
    t = np.linspace(0, T, N)

    # Initialize arrays
    S = np.zeros((N, paths))
    v = np.zeros((N, paths))
    S[0, :] = S0
    v[0, :] = v0

    # Correlation matrix
    corr_matrix = np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(corr_matrix)  # Cholesky decomposition for correlated normals

    for i in range(1, N):
        # Generate correlated random variables
        Z = np.random.normal(size=(2, paths))  # Independent standard normals
        dW1, dW2 = L @ Z  # Correlated Brownian increments

        # Variance dynamics
        v_prev = np.maximum(v[i - 1, :], 0)  # Ensure non-negative variance
        dv = (
            kappa * (theta - v_prev) * dt
            + sigma_v * np.sqrt(v_prev) * np.sqrt(dt) * dW2
        )
        v[i, :] = v_prev + dv
        v[i, :] = np.maximum(v[i, :], 0)  # Ensure variance remains non-negative

        # Stock price dynamics
        dS = mu * S[i - 1, :] * dt + np.sqrt(v_prev) * S[i - 1, :] * np.sqrt(dt) * dW1
        S[i, :] = S[i - 1, :] + dS

    return S, v, t


# Parameters
S0 = 100
v0 = 0.04
mu = 0.05
kappa = 2.0
theta = 0.04
sigma_v = 0.2
rho = -0.7
T = 1
N = 1000
paths = 5

# Simulate Heston model
S, v, t = heston_model(S0, v0, mu, kappa, theta, sigma_v, rho, T, N, paths)

# Plot stock price paths
plt.figure(figsize=(12, 6))
for i in range(paths):
    plt.plot(t, S[:, i], label=f"Path {i+1}")
plt.title("Heston Model: Simulated Stock Price Paths")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.grid()
plt.legend()
plt.show()

# Plot variance paths
plt.figure(figsize=(12, 6))
for i in range(paths):
    plt.plot(t, v[:, i], label=f"Path {i+1}")
plt.title("Heston Model: Simulated Variance Paths")
plt.xlabel("Time")
plt.ylabel("Variance")
plt.grid()
plt.legend()
plt.show()
