import argparse
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer.hmc import hmc
from numpyro.infer.util import initialize_model
from numpyro.util import fori_collect

matplotlib.use("Agg")  # For non-interactive backends


# Function to generate synthetic data
def generate_synthetic_data(num_steps=1000, seed=42):
    rng_key = random.PRNGKey(seed)

    # Parameters
    true_sigma = 0.05  # Volatility of the Gaussian Random Walk
    true_nu = 5.0  # Degrees of freedom for Student-t distribution

    # Generate latent volatility (Gaussian Random Walk)
    s = jnp.zeros(num_steps)
    for t in range(1, num_steps):
        s = s.at[t].set(s[t - 1] + random.normal(rng_key, shape=()) * true_sigma)

    # Generate returns (Student-t distributed with scale=exp(s))
    returns = jnp.exp(s) * random.t(rng_key, df=true_nu, shape=(num_steps,))

    # Use numerical indices as placeholders for dates
    time_steps = jnp.arange(num_steps)

    return time_steps, returns, s


# Stochastic volatility model
def model(returns):
    step_size = numpyro.sample("sigma", dist.Exponential(50.0))
    s = numpyro.sample(
        "s", dist.GaussianRandomWalk(scale=step_size, num_steps=jnp.shape(returns)[0])
    )
    nu = numpyro.sample("nu", dist.Exponential(0.1))
    return numpyro.sample(
        "r", dist.StudentT(df=nu, loc=0.0, scale=jnp.exp(s)), obs=returns
    )


# Function to print results
def print_results(posterior, time_steps):
    def _print_row(values, row_name=""):
        quantiles = jnp.array([0.2, 0.4, 0.5, 0.6, 0.8])
        row_name_fmt = "{:>8}"
        header_format = row_name_fmt + "{:>12}" * 5
        row_format = row_name_fmt + "{:>12.3f}" * 5
        columns = ["(p{})".format(int(q * 100)) for q in quantiles]
        q_values = jnp.quantile(values, quantiles, axis=0)
        print(header_format.format("", *columns))
        print(row_format.format(row_name, *q_values))
        print("\n")

    print("=" * 20, "sigma", "=" * 20)
    _print_row(posterior["sigma"])
    print("=" * 20, "nu", "=" * 20)
    _print_row(posterior["nu"])
    print("=" * 20, "volatility", "=" * 20)
    for i in range(0, len(time_steps), 180):
        _print_row(jnp.exp(posterior["s"][:, i]), time_steps[i])


# Main function
def main(args):
    # Generate synthetic data
    time_steps, returns, _ = generate_synthetic_data(num_steps=args.num_samples)

    init_rng_key, sample_rng_key = random.split(random.PRNGKey(args.rng_seed))
    model_info = initialize_model(init_rng_key, model, model_args=(returns,))
    init_kernel, sample_kernel = hmc(model_info.potential_fn, algo="NUTS")
    hmc_state = init_kernel(
        model_info.param_info, args.num_warmup, rng_key=sample_rng_key
    )
    hmc_states = fori_collect(
        args.num_warmup,
        args.num_warmup + args.num_samples,
        sample_kernel,
        hmc_state,
        transform=lambda hmc_state: model_info.postprocess_fn(hmc_state.z),
        progbar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    print_results(hmc_states, time_steps)

    volatility_mean = jnp.mean(jnp.exp(hmc_states["s"]), axis=0)
    volatility_lower = jnp.percentile(jnp.exp(hmc_states["s"]), 5, axis=0)
    volatility_upper = jnp.percentile(jnp.exp(hmc_states["s"]), 95, axis=0)

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(time_steps, returns, lw=0.5, label="Returns")
    ax.plot(time_steps, volatility_mean, "red", lw=1.5, label="Volatility (Mean)")
    ax.fill_between(
        time_steps,
        volatility_lower,
        volatility_upper,
        color="r",
        alpha=0.3,
        label="95% CI",
    )
    ax.legend(loc="upper right")
    ax.set(xlabel="Time Step", ylabel="Value", title="Synthetic Stochastic Volatility")
    plt.savefig("stochastic_volatility_plot.pdf")


# Script entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stochastic Volatility Model")
    parser.add_argument("-n", "--num-samples", nargs="?", default=600, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=600, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--rng_seed", default=21, type=int, help="random number generator seed"
    )
    args = parser.parse_args()

    numpyro.set_platform(args.device)

    main(args)
