##### Normalise RMSE to keep it bounded \in (a, b)
# Compute RMSE normalization
y_range = y_test.max() - y_test.min()
max_rmse = np.sqrt(np.mean((y_test - y_test.mean()) ** 2))
normalized_rmse = rmse / max_rmse

####### Normalize Log loss as well
# Compute Log-Loss normalization
n_classes = len(np.unique(y_test))
max_log_loss = -np.log(1 / n_classes)
normalized_log_loss = log_loss / max_log_loss


# Compute loss for each posterior sample
posterior_samples = predict_regressor(
    mcmc, X_test, regression_model
)  # Shape: (n_samples, n_test_points)

# Compute loss for each posterior sample
loss_samples = []
for i in range(posterior_samples.shape[0]):  # Iterate over posterior samples
    loss = mean_squared_error(y_test, posterior_samples[i])
    loss_samples.append(loss)

loss_samples = np.array(loss_samples)  # Shape: (n_samples,)


# Empirical loss (mean loss)
empirical_loss = np.mean(loss_samples)

# Variance of the loss
loss_variance = np.var(loss_samples, ddof=1)  # Unbiased estimate


# Hoeffding bound
def hoeffding_bound(N, t, a, b):
    return 2 * np.exp(-2 * N * t**2 / (b - a) ** 2)


# Bernstein bound
def bernstein_bound(N, t, variance, a, b):
    return 2 * np.exp(-N * t**2 / (2 * variance + (2 / 3) * (b - a) * t))


# Parameters
a, b = 0, 1  # Assuming loss is scaled or normalized to [0, 1]
N = len(loss_samples)  # Number of posterior samples
t = 0.05  # Margin of deviation

# Compute bounds
hoeffding = hoeffding_bound(N, t, a, b)
bernstein = bernstein_bound(N, t, loss_variance, a, b)

print(f"Hoeffding Bound: {hoeffding}")
print(f"Bernstein Bound: {bernstein}")
