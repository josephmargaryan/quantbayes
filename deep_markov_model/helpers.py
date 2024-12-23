import numpy as np
from jax import nn, numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
import jax


def _normal_init(*shape):
    return lambda rng_key: dist.Normal(scale=0.1).sample(rng_key, shape)


def _reverse_padded(padded, lengths):
    def _reverse_single(p, length):
        new = jnp.zeros_like(p)
        reverse = jnp.roll(p[::-1], length, axis=0)
        return new.at[:].set(reverse)

    return jax.vmap(_reverse_single)(padded, lengths)


def vis_tune(i, tunes, lengths, name="tune_plot.pdf"):
    """Visualizes tunes as a heatmap."""
    tune = tunes[i, : lengths[i]]
    plt.imshow(tune.T, cmap="Greys", aspect="auto")
    plt.ylabel("Data Dimension")
    plt.xlabel("Time Steps")
    plt.title(f"Generated Tune {i}")
    plt.savefig(name)
    plt.show()


def vis_sequence(i, sequences, lengths, name="sequence_plot.pdf"):
    """Visualizes sequences as a heatmap."""
    sequence = sequences[i, : lengths[i]]
    plt.imshow(sequence.T, cmap="viridis", aspect="auto")
    plt.colorbar(label="Value")
    plt.xlabel("Time Steps")
    plt.ylabel("Features")
    plt.title(f"Sequence {i}")
    plt.savefig(name)
    plt.show()


def vis_continuous_sequence(i, sequences, lengths, name="continuous_sequence_plot.pdf"):
    """Visualizes continuous sequences as line plots."""
    sequence = sequences[i, : lengths[i]]
    for feature_idx in range(sequence.shape[1]):
        plt.plot(sequence[:, feature_idx], label=f"Feature {feature_idx}")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title(f"Continuous Sequence {i}")
    plt.legend()
    plt.savefig(name)
    plt.show()


def vis_latent_space(latent_variables, name="latent_space_plot.pdf"):
    """Visualizes latent space variables."""
    plt.scatter(latent_variables[:, 0], latent_variables[:, 1], alpha=0.5)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Latent Space")
    plt.savefig(name)
    plt.show()


def vis_predictions(ground_truth, predictions, name="predictions_plot.pdf"):
    """Visualizes model predictions against ground truth."""
    plt.plot(ground_truth, label="Ground Truth")
    plt.plot(predictions, label="Predictions", linestyle="--")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Model Predictions vs Ground Truth")
    plt.savefig(name)
    plt.show()
