import argparse
import jax.numpy as jnp
from jax import random
from data import generate_var2_data
from model import var2_scan
from inference import run_inference
from plotting import plot_results


def main(args):
    # Generate synthetic data
    T = args.num_data
    K = 2
    c_true = jnp.array([0.5, -0.3])
    Phi1_true = jnp.array([[0.7, 0.1], [0.2, 0.5]])
    Phi2_true = jnp.array([[0.2, -0.1], [-0.1, 0.2]])
    sigma_true = jnp.array([[0.1, 0.02], [0.02, 0.1]])
    y = generate_var2_data(T, K, c_true, Phi1_true, Phi2_true, sigma_true)

    # Run inference
    rng_key = random.PRNGKey(0)
    samples = run_inference(var2_scan, y, args, rng_key)

    # Plot results
    plot_results(y, samples, T, K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAR(2) example")
    parser.add_argument("--num-data", default=100, type=int)
    parser.add_argument("--num-samples", default=1000, type=int)
    parser.add_argument("--num-warmup", default=1000, type=int)
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    import numpyro

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
