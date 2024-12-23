import numpy as np
import matplotlib.pyplot as plt


def simulate_heston(S0, v0, mu, kappa, theta, eta, rho, T, dt, N):
    """
    Simulates the Heston stochastic volatility model.

    Parameters:
        S0: Initial stock price.
        v0: Initial variance.
        mu: Drift of stock price.
        kappa: Mean-reversion speed of variance.
        theta: Long-term variance mean.
        eta: Volatility of volatility.
        rho: Correlation between Brownian motions.
        T: Time horizon.
        dt: Time step size.
        N: Number of paths to simulate.

    Returns:
        S: Simulated stock prices.
        V: Simulated variances.
    """
    steps = int(T / dt)
    S = np.zeros((steps, N))
    V = np.zeros((steps, N))
    S[0, :] = S0
    V[0, :] = v0

    # Correlated Brownian motions
    Z1 = np.random.normal(size=(steps, N))
    Z2 = np.random.normal(size=(steps, N))
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    for t in range(1, steps):
        # Variance process (square root diffusion)
        V[t, :] = np.abs(
            V[t - 1, :]
            + kappa * (theta - V[t - 1, :]) * dt
            + eta * np.sqrt(V[t - 1, :]) * np.sqrt(dt) * W2[t - 1, :]
        )

        # Stock price process
        S[t, :] = S[t - 1, :] * np.exp(
            (mu - 0.5 * V[t - 1, :]) * dt
            + np.sqrt(V[t - 1, :]) * np.sqrt(dt) * W1[t - 1, :]
        )

    return S, V


# Simulation parameters
S0 = 100
v0 = 0.04
mu = 0.05
kappa = 2.0
theta = 0.04
eta = 0.1
rho = -0.7
T = 1.0
dt = 0.01
N = 1000

# Simulate
S, V = simulate_heston(S0, v0, mu, kappa, theta, eta, rho, T, dt, N)

# Plot results
time = np.linspace(0, T, int(T / dt))
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, S[:, :10])  # Plot first 10 paths
plt.title("Simulated Stock Prices")
plt.xlabel("Time")
plt.ylabel("Price")

plt.subplot(2, 1, 2)
plt.plot(time, V[:, :10])
plt.title("Simulated Variances")
plt.xlabel("Time")
plt.ylabel("Variance")
plt.tight_layout()
plt.show()


from scipy.optimize import minimize


# Log-likelihood function for the Heston model
def heston_log_likelihood(params, S, V, dt):
    mu, kappa, theta, eta, rho = params
    N = len(S)
    log_likelihood = 0

    for t in range(1, N):
        dv = V[t] - V[t - 1]
        dW1 = (np.log(S[t] / S[t - 1]) - (mu - 0.5 * V[t - 1]) * dt) / np.sqrt(
            V[t - 1] * dt
        )
        dW2 = (dv - kappa * (theta - V[t - 1]) * dt) / (eta * np.sqrt(V[t - 1] * dt))
        rho_term = rho * dW1 + np.sqrt(1 - rho**2) * dW2
        log_likelihood += -0.5 * (dW1**2 + rho_term**2)

    return -log_likelihood


# Initial guess for parameters
initial_params = [0.05, 2.0, 0.04, 0.1, -0.7]

# Fit the model
result = minimize(
    heston_log_likelihood,
    initial_params,
    args=(S[:, 0], V[:, 0], dt),
    method="L-BFGS-B",
)
print("Fitted Parameters:", result.x)
