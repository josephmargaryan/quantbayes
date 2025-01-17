import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from gmm import GaussianMixtureModel

# Generate synthetic data
def generate_synthetic_data(num_samples=500, locs=[2.0, 8.0], scales=[1.0, 2.0], weights=[0.4, 0.6]):
    key = random.PRNGKey(42)
    component_ids = random.categorical(key, jnp.log(jnp.array(weights)), shape=(num_samples,))
    data = jnp.array([
        random.normal(key, shape=(1,)) * scales[comp_id] + locs[comp_id]
        for comp_id in component_ids
    ]).flatten()
    return data

# Parameters of the synthetic Gaussian mixture
locs = [2.0, 8.0]   # Means of the components
scales = [1.0, 2.0]  # Standard deviations of the components
weights = [0.4, 0.6] # Mixing proportions

# Generate data
data = generate_synthetic_data(num_samples=500, locs=locs, scales=scales, weights=weights)

# Visualize the generated data
plt.hist(data, bins=30, density=True, alpha=0.6, color="g")
plt.title("Histogram of Synthetic Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Import and initialize the Gaussian Mixture Model
gmm = GaussianMixtureModel(num_components=2)

# Find the best initialization
loss, seed = min((gmm.initialize(data, seed), seed) for seed in range(100))
print(f"Best initialization seed = {seed}, initial loss = {loss}")

# Train the model using SVI
svi_result, gradient_norms = gmm.train_svi(data, learning_rate=0.1, iterations=500)

# Plot the results
gmm.plot_results(data)

# Classify new data
new_data = jnp.linspace(-3, 15, 100)
assignments = gmm.classify(new_data)
plt.scatter(new_data, assignments, c="r", label="Class Assignment")
plt.title("Class Assignment of New Data")
plt.xlabel("Data Value")
plt.ylabel("Class")
plt.legend()
plt.show()
