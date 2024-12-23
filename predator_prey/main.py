import argparse
import jax.numpy as jnp
import numpyro
from data import load_predator_prey_data
from model import predator_prey_model
from inference import run_mcmc, predict
from plotting import plot_results


def main(args):
    year, data = load_predator_prey_data()
    mcmc = run_mcmc(
        predator_prey_model, data, args.num_warmup, args.num_samples, args.num_chains
    )
    pop_pred = predict(predator_prey_model, mcmc.get_samples(), data.shape[0])
    pred_mean = jnp.mean(pop_pred, axis=0)
    pred_pi = jnp.percentile(pop_pred, jnp.array([10, 90]), axis=0)
    plot_results(year, data, pred_mean, pred_pi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predator-Prey Model")
    parser.add_argument("-n", "--num-samples", default=1000, type=int)
    parser.add_argument("--num-warmup", default=1000, type=int)
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
