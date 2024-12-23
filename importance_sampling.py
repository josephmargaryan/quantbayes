import numpy as np
import scipy.stats as stats


def p(x, mu=0, sd=1):
    """
    Gaussian distribution with mean=0 and std=1
    """
    return (1 / np.sqrt(2 * np.pi * sd**2)) * np.exp(-((x - mu) ** 2) / (2 * sd**2))


def q(x, mu=5, sd=1):
    """
    Gaussian distribution with mean=5 and std=1
    """
    return (1 / np.sqrt(2 * np.pi * sd**2)) * np.exp(-((x - mu) ** 2) / (2 * sd**2))


def w(x):
    """
    Importance weight
    """
    return p(x) / q(x)


def f(x):
    """
    Function to evaluate
    """
    return (x >= 5).astype(float)


N = 1000
samples = np.random.normal(loc=5, scale=1, size=N)

importance_weight = w(samples)

monte_carlo_estimation = np.mean(f(samples) * importance_weight)

true_value = 1 - stats.norm.cdf(5)

print(f"The true integral of the normal distribution from 5 to infinity: {true_value}")
print(f"The estimated value from our importance sampling: {monte_carlo_estimation}")
