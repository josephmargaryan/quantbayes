from scipy.stats import boltzmann
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Boltzmann distribution
N = 10  # Number of discrete energy levels
k = 1.38e-23  # Boltzmann constant (arbitrary scale)
temperatures = [1, 2, 5]  # Example temperatures

# Plot Boltzmann PMF for different temperatures
plt.figure(figsize=(12, 6))
energy_levels = np.arange(N)

for T in temperatures:
    lambda_param = 1 / (k * T)  # Inverse temperature
    pmf = boltzmann.pmf(energy_levels, lambda_param, N)  # PMF from scipy
    plt.plot(energy_levels, pmf, marker="o", label=f"T = {T}")

plt.title("Boltzmann Distribution (PMF) using scipy.stats")
plt.xlabel("Energy Levels (E)")
plt.ylabel("Probability P(E)")
plt.xticks(energy_levels)
plt.legend()
plt.grid()
plt.show()
