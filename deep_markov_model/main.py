import argparse
from numpyro import set_platform
from train import train_and_predict
from models import model, guide
from data_loader import load_data
from helpers import (
    vis_tune,
    vis_sequence,
    vis_continuous_sequence,
    vis_latent_space,
    vis_predictions,
)
import jax.numpy as jnp
import numpy as np
import jax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample-size", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--num-stein-particles", type=int, default=5)
    parser.add_argument("--num-elbo-particles", type=int, default=5)
    parser.add_argument("--gru-dim", type=int, default=150)
    parser.add_argument("--rng-seed", default=142, type=int)
    parser.add_argument("--device", default="cpu", choices=["gpu", "cpu"])

    args = parser.parse_args()
    set_platform(args.device)

    # Train the model
    results, seqs, rev_seqs, lengths, predictions, latent_variables = train_and_predict(
        args, model, guide, load_data
    )

    # Visualizations
    print("Generating visualizations...")

    # Visualize tunes
    vis_tune(0, seqs, lengths, name="tune_plot.pdf")

    # Visualize sequences
    vis_sequence(0, seqs, lengths, name="sequence_plot.pdf")

    # Visualize continuous sequences
    continuous_data = jnp.sin(jnp.linspace(0, 2 * jnp.pi, lengths[0]))[
        :, None
    ] + np.random.normal(0, 0.1, (lengths[0], 1))
    vis_continuous_sequence(
        0, continuous_data[None, ...], lengths, name="continuous_sequence_plot.pdf"
    )

    # Visualize latent space
    latent_space = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 2))
    vis_latent_space(latent_space, name="latent_space_plot.pdf")

    # Visualize predictions
    ground_truth = np.random.randn(50)
    predicted = ground_truth + np.random.normal(0, 0.1, 50)
    vis_predictions(ground_truth, predicted, name="predictions_plot.pdf")


if __name__ == "__main__":
    main()
