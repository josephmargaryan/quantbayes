import numpy as np


def generate_var2_data(T, K, c, Phi1, Phi2, sigma):
    """
    Generate time series data from a VAR(2) process.
    Args:
        T (int): Number of time steps.
        K (int): Number of variables in the time series.
        c (array): Constants (shape: (K,)).
        Phi1 (array): Coefficients for lag 1 (shape: (K, K)).
        Phi2 (array): Coefficients for lag 2 (shape: (K, K)).
        sigma (array): Covariance matrix for the noise (shape: (K, K)).
    Returns:
        np.ndarray: Generated time series data (shape: (T, K)).
    """
    y = np.zeros((T, K))
    y[:2] = np.random.multivariate_normal(mean=np.zeros(K), cov=sigma, size=2)

    for t in range(2, T):
        y[t] = (
            c
            + Phi1 @ y[t - 1]
            + Phi2 @ y[t - 2]
            + np.random.multivariate_normal(mean=np.zeros(K), cov=sigma)
        )
    return y
