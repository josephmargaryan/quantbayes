from data import generate_synthetic_data
from model import gmm_model
from guide import initialize_guide
from inference import run_svi, run_mcmc
from plotting import plot_svi_loss, plot_gmm_density, plot_posterior_density, plot_trace
from optimization import hook_optax
import optax
from jax import random


def main():
    # Data
    data = generate_synthetic_data()

    # Model
    K = 2
    guide = initialize_guide(gmm_model, data, K=K)

    # SVI
    optim, gradient_norms = hook_optax(optax.adam(learning_rate=0.1))
    svi_result = run_svi(gmm_model, guide, optim, data, num_iters=int(200))
    plot_svi_loss(svi_result.losses)

    # Parameters
    params = svi_result.params
    weights = params["weights_auto_loc"]
    locs = params["locs_auto_loc"]
    scale = params["scale_auto_loc"]
    print(f"Weights: {weights}, Locs: {locs}, Scale: {scale}")
    plot_gmm_density(data, weights, locs, scale)

    # MCMC
    mcmc = run_mcmc(gmm_model, data)
    mcmc.print_summary()

    # Plot posterior density
    plot_posterior_density(mcmc.get_samples())

    # Plot trace of loc parameters
    plot_trace(mcmc.get_samples())


if __name__ == "__main__":
    main()
