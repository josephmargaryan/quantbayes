import time
import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal, AutoGuideList
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import cm

# Import your custom layers and guides
from quantbayes import bnn
from quantbayes.fake_data import generate_regression_data
from quantbayes.fourier_guides.guides import SpectralImagGuide, SpectralRealGuide

###############################
# Configuration and Data Setup
###############################

LEARNING_RATE = 0.01
NUM_ITERATIONS = 1000
use_custom_guides = True  # Set to False to use one guide (AutoNormal)

# Set up random keys.
key = jr.key(0)
key, init_key, pred_key, post_key, grad_key, viz_key, interp_key = jr.split(key, 7)

# Generate data.
df = generate_regression_data(n_continuous=16)
X, y = df.drop("target", axis=1), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)

#########################
# Define the Model
#########################
def model(X, y=None):
    """
    Bayesian regression model with a spectral circulant layer and a linear output.
    """
    N, D = X.shape
    # Apply the spectral circulant layer.
    X = bnn.SpectralCirculantLayer(D)(X)
    X = jax.nn.tanh(X)
    
    # Linear weights and bias.
    W = numpyro.sample("W", dist.Normal(0, 1).expand([D, 1]).to_event(2))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([1]).to_event(1))
    
    # Linear combination and squeeze.
    X = jnp.dot(X, W) + b
    mu = jnp.squeeze(X, axis=1)
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    
    # Likelihood.
    with numpyro.plate("data", N):
        numpyro.sample("likelihood", dist.Normal(mu, sigma), obs=y)

##########################
# Set Up Guides and SVI
##########################
optimizer = numpyro.optim.Adam(LEARNING_RATE)

if use_custom_guides:
    print("Using custom guides")
    # For d_in=16, the full half-spectrum is (16//2)+1 = 9.
    K_value = 9
    spectral_real_guide = SpectralImagGuide(model, K=K_value)
    spectral_imag_guide = SpectralRealGuide(model, K=K_value)
    # Hide the Fourier coefficient sites from the default guide.
    other_guide = AutoNormal(numpyro.handlers.block(model, hide=["spectral_circ_jvp_real", "spectral_circ_jvp_imag"]))
    guide = AutoGuideList(model)
    guide.append(spectral_real_guide)
    guide.append(spectral_imag_guide)
    guide.append(other_guide)
else:
    print("Using one guide")
    guide = AutoNormal(model)

svi = SVI(model, guide, optim=optimizer, loss=Trace_ELBO())

#####################
# Training the Model
#####################
print("Training...")
start_time = time.time()
svi_state = svi.init(init_key, X_train, y_train)
losses = []
# Save initial parameters for interpolation diagnostics.
init_params = svi.get_params(svi_state)

for step in range(NUM_ITERATIONS):
    svi_state, loss = svi.update(svi_state, X_train, y_train)
    losses.append(loss)
    if (step + 1) % 100 == 0:
        print(f"Iteration {step+1} Loss: {loss:.3f}")
end_time = time.time()
print(f"Training finished in {end_time - start_time:.3f} seconds.")

# Get final parameters.
params = svi.get_params(svi_state)

# Prediction using SVI model.
predictive = Predictive(model, guide=svi.guide, params=params, num_samples=100)
preds = predictive(pred_key, X_test)["likelihood"]
mean_preds = preds.mean(axis=0)
mae_loss = mean_absolute_error(np.array(y_test), np.array(mean_preds))
print(f"Final MAE Loss: {mae_loss:.4f}")

##################################
# Analysis: Gradient Diagnostics
##################################

def compute_loss(params, model, guide, X, y, rng_key):
    """
    Computes the ELBO loss for given params, model, guide, and data.
    """
    elbo = Trace_ELBO()
    return elbo.loss(rng_key, params, model, guide, X, y)

def measure_gradient_variance(loss_fn, params, model, guide, X, y, num_samples=10):
    """
    Computes gradient norms for num_samples different RNG keys and returns
    the mean and variance of the gradient norms, and the list of all norms.
    """
    grad_fn = jax.grad(loss_fn)
    flat_params, _ = ravel_pytree(params)
    grad_norms = []
    rng_keys = jr.split(jr.PRNGKey(42), num_samples)
    for key in rng_keys:
        grads = grad_fn(params, model, guide, X, y, key)
        flat_grads, _ = ravel_pytree(grads)
        grad_norm = jnp.linalg.norm(flat_grads)
        grad_norms.append(grad_norm)
    grad_norms = jnp.array(grad_norms)
    mean_norm = jnp.mean(grad_norms)
    var_norm = jnp.var(grad_norms)
    return mean_norm, var_norm, grad_norms

mean_grad_norm, grad_norm_variance, grad_norms_all = measure_gradient_variance(
    compute_loss, params, model, guide, X_train, y_train, num_samples=20)
print(f"Gradient Norm Mean: {mean_grad_norm:.3f}")
print(f"Gradient Norm Variance: {grad_norm_variance:.3f}")

# Plot histogram of gradient norms.
plt.figure(figsize=(8, 6))
plt.hist(np.array(grad_norms_all), bins=10, density=True)
plt.title("Histogram of Gradient Norms (Across RNG keys)")
plt.xlabel("Gradient Norm")
plt.ylabel("Density")
plt.show()

######################################
# Analysis: Visualizing the Loss Landscape (2D Contour)
######################################
def visualize_loss_landscape(loss_fn, params, model, guide, X, y, rng_key,
                             grid_range=(-0.1, 0.1), grid_points=21, title="Loss Landscape"):
    """
    Visualizes a 2D slice of the loss landscape by perturbing the flattened parameters
    along two orthonormal directions.
    """
    flat_params, unravel_fn = ravel_pytree(params)
    param_dim = flat_params.shape[0]
    
    # Generate two fixed orthonormal directions.
    key1, key2 = jr.split(rng_key)
    direction1 = jr.normal(key1, (param_dim,))
    direction1 = direction1 / jnp.linalg.norm(direction1)
    direction2 = jr.normal(key2, (param_dim,))
    direction2 = direction2 - jnp.dot(direction2, direction1) * direction1
    direction2 = direction2 / jnp.linalg.norm(direction2)

    scales = np.linspace(grid_range[0], grid_range[1], grid_points)
    loss_grid = np.zeros((grid_points, grid_points))
    print("Loss grid std:", np.std(loss_grid))

    for i, a in enumerate(scales):
        for j, b in enumerate(scales):
            perturbed_params = flat_params + a * direction1 + b * direction2
            new_params = unravel_fn(perturbed_params)
            loss_grid[i, j] = np.array(loss_fn(new_params, model, guide, X, y, rng_key))
    
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(scales, scales, loss_grid, levels=50, cmap='viridis')
    plt.xlabel("Perturbation along direction 1")
    plt.ylabel("Perturbation along direction 2")
    plt.title(title)
    plt.colorbar(cp, label="Loss")
    plt.show()
    
    return loss_grid, scales

loss_grid, scales = visualize_loss_landscape(compute_loss, params, model, guide, X_train, y_train, viz_key,
                                              grid_range=(-0.1, 0.1), grid_points=21,
                                              title="2D Contour of Loss Landscape")

######################################
# Analysis: Visualizing the Loss Landscape (3D Surface)
######################################
def visualize_loss_landscape_3d(loss_grid, scales, title="3D Loss Landscape"):
    """
    Create a 3D surface plot from the loss grid.
    """
    X_grid, Y_grid = np.meshgrid(scales, scales)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_grid, Y_grid, loss_grid, cmap=cm.viridis, edgecolor='none', alpha=0.8)
    ax.set_xlabel("Perturbation along direction 1")
    ax.set_ylabel("Perturbation along direction 2")
    ax.set_zlabel("Loss")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Loss")
    plt.show()

visualize_loss_landscape_3d(loss_grid, scales, title="3D Surface of Loss Landscape")

######################################
# Optional: Interpolation Diagnostic
######################################
def interpolate_loss(loss_fn, init_params, final_params, model, guide, X, y, num_points=21, rng_key=None):
    """
    Interpolates between init_params and final_params and computes the loss along
    the interpolation.
    """
    flat_init, unravel_fn = ravel_pytree(init_params)
    flat_final, _ = ravel_pytree(final_params)
    alphas = np.linspace(0, 1, num_points)
    losses_interp = []
    for alpha in alphas:
        flat_interp = (1 - alpha) * flat_init + alpha * flat_final
        interp_params = unravel_fn(flat_interp)
        # Use a fixed or provided rng_key for consistency in loss evaluation.
        if rng_key is None:
            rng_key = jr.PRNGKey(0)
        loss_val = np.array(loss_fn(interp_params, model, guide, X, y, rng_key))
        losses_interp.append(loss_val)
    return alphas, losses_interp

alphas, interp_losses = interpolate_loss(compute_loss, init_params, params, model, guide, X_train, y_train,
                                          num_points=21, rng_key=viz_key)

plt.figure(figsize=(8, 6))
plt.plot(alphas, interp_losses, marker='o')
plt.title("Interpolation Loss from Initialization to Final Parameters")
plt.xlabel("Interpolation Factor (alpha)")
plt.ylabel("Loss")
plt.show()

######################################
# Additional Diagnostics: Training Loss Statistics
######################################
def smooth_curve(curve, window=50):
    return np.convolve(curve, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(losses, label="Raw Loss")
plt.plot(smooth_curve(losses, window=50), label="Smoothed Loss (window=50)", linewidth=2)
plt.title("Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()

######################################
# Posterior Diagnostics: Examine the posterior of parameter "W"
######################################
plt.figure(figsize=(10, 6))
samples_W = guide.sample_posterior(post_key, params=params)["W"]
plt.hist(np.array(samples_W).flatten(), bins=50, density=True)
plt.title("Posterior Distribution for W")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

######################################
# Additional Diagnostic: Hessian / Curvature Estimation
######################################
def compute_projected_hessian(loss_fn, params, model, guide, X, y, rng_key, directions):
    """
    Compute the 2x2 projected Hessian matrix on the given orthonormal directions.
    directions: list of two directions (each a flattened vector of same shape as flat_params)
    """
    flat_params, unravel_fn = ravel_pytree(params)
    def f(flat_params):
        return loss_fn(unravel_fn(flat_params), model, guide, X, y, rng_key)
    H_full = jax.hessian(f)(flat_params)
    H_proj = np.zeros((len(directions), len(directions)))
    for i, v in enumerate(directions):
        for j, w in enumerate(directions):
            H_proj[i, j] = np.dot(v, np.dot(H_full, w))
    return H_proj

# Use the same projection directions as in the 2D visualization.
flat_params, unravel_fn = ravel_pytree(params)
param_dim = flat_params.shape[0]
key1, key2 = jr.split(viz_key)
direction1 = jr.normal(key1, (param_dim,))
direction1 = direction1 / jnp.linalg.norm(direction1)
direction2 = jr.normal(key2, (param_dim,))
direction2 = direction2 - jnp.dot(direction2, direction1)*direction1
direction2 = direction2 / jnp.linalg.norm(direction2)
# Convert directions to numpy arrays.
dir1_np = np.array(direction1)
dir2_np = np.array(direction2)
directions = [dir1_np, dir2_np]

H_proj = compute_projected_hessian(compute_loss, params, model, guide, X_train, y_train, viz_key, directions)
eigvals = np.linalg.eigvals(H_proj)
print("Projected Hessian (2x2) eigenvalues:", eigvals)

######################################
# Additional Diagnostic: Multiple Projections Across Parameter Space
######################################
def multiple_projections_diagnostics(loss_fn, params, model, guide, X, y, num_projections=5, grid_range=(-0.1, 0.1), grid_points=21):
    flat_params, unravel_fn = ravel_pytree(params)
    opt_points = []
    for seed in range(num_projections):
        rng_key_proj = jr.PRNGKey(seed)
        key1, key2 = jr.split(rng_key_proj)
        # Generate random orthonormal directions.
        v1 = jr.normal(key1, (flat_params.shape[0],))
        v1 = v1 / jnp.linalg.norm(v1)
        v2 = jr.normal(key2, (flat_params.shape[0],))
        v2 = v2 - jnp.dot(v2, v1)*v1
        v2 = v2 / jnp.linalg.norm(v2)
        scales = np.linspace(grid_range[0], grid_range[1], grid_points)
        loss_grid = np.zeros((grid_points, grid_points))
        for i, a in enumerate(scales):
            for j, b in enumerate(scales):
                perturbed = flat_params + a * v1 + b * v2
                params_perturbed = unravel_fn(perturbed)
                loss_grid[i, j] = np.array(loss_fn(params_perturbed, model, guide, X, y, viz_key))
        # Find the minimum loss location in this projection.
        min_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
        a_min = scales[min_idx[0]]
        b_min = scales[min_idx[1]]
        opt_points.append([a_min, b_min])
        # Plot each individual projection.
        plt.figure(figsize=(6,4))
        cp = plt.contourf(scales, scales, loss_grid, levels=50, cmap='viridis')
        plt.plot(a_min, b_min, 'rx', markersize=12, label='Optimum')
        plt.xlabel("Perturbation along v1")
        plt.ylabel("Perturbation along v2")
        plt.title(f"Projection {seed}: Loss Landscape")
        plt.colorbar(cp, label="Loss")
        plt.legend()
        plt.show()
    opt_points = np.array(opt_points)
    print("Optimal coordinates across projections (a, b):")
    print(opt_points)
    print("Mean optimum coordinates:", np.mean(opt_points, axis=0))
    print("Variance of optimum coordinates:", np.var(opt_points, axis=0))
    
multiple_projections_diagnostics(compute_loss, params, model, guide, X_train, y_train, num_projections=5, grid_range=(-0.1, 0.1), grid_points=21)

######################################
# Additional Diagnostic: Enhanced Interpolation Diagnostics with Curvature
######################################
def interpolation_with_curvature(loss_fn, init_params, final_params, model, guide, X, y, num_points=21, rng_key=None):
    flat_init, unravel_fn = ravel_pytree(init_params)
    flat_final, _ = ravel_pytree(final_params)
    alphas = np.linspace(0, 1, num_points)
    losses_interp = []
    for alpha in alphas:
        flat_interp = (1 - alpha)*flat_init + alpha*flat_final
        interp_params = unravel_fn(flat_interp)
        if rng_key is None:
            rng_key = jr.PRNGKey(0)
        loss_val = np.array(loss_fn(interp_params, model, guide, X, y, rng_key))
        losses_interp.append(loss_val)
    losses_interp = np.array(losses_interp)
    
    # Estimate curvature using finite differences.
    first_deriv = np.gradient(losses_interp, alphas)
    second_deriv = np.gradient(first_deriv, alphas)
    
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(alphas, losses_interp, marker='o', label='Loss')
    plt.title("Interpolation Loss from Initialization to Final Parameters")
    plt.xlabel("Interpolation Factor (alpha)")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(alphas, second_deriv, marker='o', label='Estimated Curvature (2nd Derivative)')
    plt.title("Estimated Curvature along Interpolation Path")
    plt.xlabel("Interpolation Factor (alpha)")
    plt.ylabel("Second Derivative")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return alphas, losses_interp, second_deriv

alphas_interp, losses_interp, curvature_interp = interpolation_with_curvature(compute_loss, init_params, params, model, guide, X_train, y_train, num_points=21, rng_key=viz_key)

######################################
# Additional Diagnostic: Comparison of Uncertainty Quantification
######################################
def uncertainty_quantification(predictive, X_test, true_y, num_samples=100):
    predictions = predictive(pred_key, X_test)
    likelihood_samples = predictions["likelihood"]  # shape: (num_samples, N)
    predictive_mean = np.mean(likelihood_samples, axis=0)
    predictive_std = np.std(likelihood_samples, axis=0)
    
    # Plot predictive intervals for a subset of test points.
    num_plot = min(20, X_test.shape[0])
    plt.figure(figsize=(10,6))
    x_axis = np.arange(num_plot)
    plt.errorbar(x_axis, predictive_mean[:num_plot],
                 yerr=1.96*predictive_std[:num_plot],
                 fmt='o', capsize=5, label="95% Predictive Interval")
    plt.scatter(x_axis, true_y[:num_plot], color='red', label="True y")
    plt.title("Predictive Mean with 95% Confidence Intervals")
    plt.xlabel("Test Instance")
    plt.ylabel("Predicted Value")
    plt.legend()
    plt.show()
    
    # Scatter plot: Predictive std vs. absolute error.
    abs_errors = np.abs(predictive_mean - np.array(true_y))
    plt.figure(figsize=(8,6))
    plt.scatter(predictive_std, abs_errors)
    plt.xlabel("Predictive Standard Deviation")
    plt.ylabel("Absolute Prediction Error")
    plt.title("Prediction Uncertainty vs. Error")
    plt.show()
    
    print("Average predictive std:", np.mean(predictive_std))
    print("Average absolute error:", np.mean(abs_errors))
    
    return predictive_mean, predictive_std

predictive_mean, predictive_std = uncertainty_quantification(predictive, X_test, y_test, num_samples=100)
