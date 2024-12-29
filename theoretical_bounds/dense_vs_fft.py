import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 500  # number of samples
delta = 0.05  # confidence level

# Hypothesis space sizes
dense_params = np.arange(10, 200, 10)  # parameter size for dense layers (d * h + h)
fft_params = dense_params / 5  # reduced parameter size for FFT layers


# Generalization bounds
def compute_epsilon(log_hypothesis_size, n, delta):
    return np.sqrt((log_hypothesis_size + np.log(1 / delta)) / (2 * n))


# Compute bounds
dense_log_hypothesis_size = dense_params  # Log size of dense hypothesis space
fft_log_hypothesis_size = fft_params  # Log size of FFT hypothesis space

epsilon_dense = compute_epsilon(dense_log_hypothesis_size, n, delta)
epsilon_fft = compute_epsilon(fft_log_hypothesis_size, n, delta)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(dense_params, epsilon_dense, label="Dense BNN", marker="o")
plt.plot(dense_params, epsilon_fft, label="FFT BNN", marker="x")
plt.xlabel("Number of Parameters (log scale)", fontsize=12)
plt.ylabel("Generalization Bound (ε)", fontsize=12)
plt.title("Generalization Bounds: Dense vs. FFT BNN", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
