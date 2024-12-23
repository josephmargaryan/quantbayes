import numpy as np


# Define the target distribution
def target_distribution(size):
    return np.random.normal(0, 1, size)


# Generate samples
samples = target_distribution(10000)

# Compute cumulative means
cumulative_means = np.cumsum(samples) / np.arange(1, len(samples) + 1)

import matplotlib.pyplot as plt

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(cumulative_means, label="Cumulative Mean")
plt.axhline(0, color="red", linestyle="--", label="True Mean")
plt.xlabel("Number of Samples")
plt.ylabel("Estimate")
plt.title("Convergence of Monte Carlo Sampling")
plt.legend()
plt.grid()
plt.show()

# Histogram of samples
plt.figure(figsize=(12, 6))

# Convergence
plt.subplot(1, 2, 1)
plt.plot(cumulative_means, label="Cumulative Mean")
plt.axhline(0, color="red", linestyle="--", label="True Mean")
plt.xlabel("Number of Samples")
plt.ylabel("Estimate")
plt.title("Convergence of Monte Carlo Sampling")
plt.legend()
plt.grid()

# Histogram
plt.subplot(1, 2, 2)
plt.hist(
    samples,
    bins=50,
    density=True,
    alpha=0.7,
    color="blue",
    label="Sampled Distribution",
)
x = np.linspace(-4, 4, 100)
plt.plot(
    x,
    1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2),
    color="red",
    label="Theoretical PDF",
)
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Sample Distribution vs. Theoretical Distribution")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Compute standard errors
standard_errors = np.sqrt(
    np.cumsum((samples - cumulative_means) ** 2) / np.arange(1, len(samples) + 1)
)

# Plot with confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(cumulative_means, label="Cumulative Mean")
plt.fill_between(
    range(len(cumulative_means)),
    cumulative_means - 1.96 * standard_errors,
    cumulative_means + 1.96 * standard_errors,
    color="blue",
    alpha=0.2,
    label="95% Confidence Interval",
)
plt.axhline(0, color="red", linestyle="--", label="True Mean")
plt.xlabel("Number of Samples")
plt.ylabel("Estimate")
plt.title("Convergence of Monte Carlo Sampling with Confidence Intervals")
plt.legend()
plt.grid()
plt.show()

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(10, 6))
(line,) = ax.plot([], [], label="Cumulative Mean")
true_mean_line = ax.axhline(0, color="red", linestyle="--", label="True Mean")
ax.set_xlim(0, len(samples))
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Number of Samples")
ax.set_ylabel("Estimate")
ax.set_title("Monte Carlo Sampling Convergence")
ax.legend()
ax.grid()


def update(frame):
    line.set_data(range(frame), cumulative_means[:frame])
    return (line,)


ani = FuncAnimation(fig, update, frames=len(samples), interval=30)
plt.show()
