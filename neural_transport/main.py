import argparse
from models import dual_moon_model
from inference import run_vanilla_hmc, run_svi, run_neutra_hmc
from visualization import plot_results
from numpyro.distributions import Normal
import jax.random as random


def main(args):
    print("Running Vanilla HMC...")
    vanilla_samples = run_vanilla_hmc(
        dual_moon_model, args.num_warmup, args.num_samples, args.num_chains
    )

    print("Running SVI...")
    guide, svi_result = run_svi(dual_moon_model, args.num_iters, args.hidden_factor)

    print("Running NeuTra HMC...")
    zs, samples, neutra = run_neutra_hmc(
        dual_moon_model,
        guide,
        svi_result,
        args.num_warmup,
        args.num_samples,
        args.num_chains,
    )

    guide_base_samples = Normal(0, 1).sample(random.PRNGKey(4), (1000, 2))
    guide_trans_samples = neutra.transform_sample(guide_base_samples)["x"]

    print("Plotting results...")
    plot_results(
        svi_result,
        guide.sample_posterior(
            random.PRNGKey(2), svi_result.params, sample_shape=(args.num_samples,)
        )["x"],
        guide_base_samples,
        guide_trans_samples,
        vanilla_samples,
        zs,
        samples["x"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuTra HMC")
    parser.add_argument("-n", "--num-samples", nargs="?", default=4000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--hidden-factor", nargs="?", default=8, type=int)
    parser.add_argument("--num-iters", nargs="?", default=10000, type=int)
    args = parser.parse_args()

    main(args)
