from numpyro.optim import optax_to_numpyro
import argparse
import numpy as np
import jax
from jax import nn, numpy as jnp, random
from optax import adam
import numpyro
from numpyro.contrib.einstein import SteinVI
from numpyro.contrib.einstein.mixture_guide_predictive import MixtureGuidePredictive
from numpyro.contrib.einstein.stein_kernels import RBFKernel
import numpyro.distributions as dist
import matplotlib.pyplot as plt


# 1. Data Generation
def generate_synthetic_data(num_sequences, sequence_length, data_dim, rng_key):
    """Generate synthetic data for training."""
    seqs = random.bernoulli(
        rng_key, p=0.5, shape=(num_sequences, sequence_length, data_dim)
    )
    lengths = jnp.full((num_sequences,), sequence_length)
    return seqs, lengths


# 2. Helper Functions
def _reverse_padded(padded, lengths):
    def _reverse_single(p, length):
        new = jnp.zeros_like(p)
        reverse = jnp.roll(p[::-1], length, axis=0)
        return new.at[:].set(reverse)

    return jax.vmap(_reverse_single)(padded, lengths)


def emitter(x, params):
    """Parameterize the Bernoulli observation likelihood p(x_t | z_t)."""
    l1 = nn.relu(jnp.matmul(x, params["l1"]))
    l2 = nn.relu(jnp.matmul(l1, params["l2"]))
    return jnp.matmul(l2, params["l3"])


def transition(x, params):
    """Parameterize the Gaussian latent transition p(z_t | z_{t-1})."""

    def _gate(x, params):
        l1 = nn.relu(jnp.matmul(x, params["l1"]))
        return nn.sigmoid(jnp.matmul(l1, params["l2"]))

    def _shared(x, params):
        l1 = nn.relu(jnp.matmul(x, params["l1"]))
        return jnp.matmul(l1, params["l2"])

    def _mean(x, params):
        return jnp.matmul(x, params["l1"])

    def _std(x, params):
        l1 = jnp.matmul(nn.relu(x), params["l1"])
        return nn.softplus(l1)

    gt = _gate(x, params["gate"])
    ht = _shared(x, params["shared"])
    loc = (1 - gt) * _mean(x, params["mean"]) + gt * ht
    std = _std(ht, params["std"])
    return loc, std


def plot_predictions(true_vals, pred_vals, sequence_idx=0):
    plt.figure(figsize=(12, 6))
    plt.plot(true_vals[sequence_idx, :, 0], label="True", linestyle="--")
    plt.plot(pred_vals.mean(axis=0)[sequence_idx, :, 0], label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"True vs Predicted for Sequence {sequence_idx}")
    plt.legend()
    plt.show()


# 3. Model and Guide
def model(
    seqs,
    seqs_rev,
    lengths,
    *,
    latent_dim=32,
    emission_dim=100,
    transition_dim=200,
    data_dim=88,
):
    max_seq_length = seqs.shape[1]

    emitter_params = {
        "l1": numpyro.param(
            "emitter_l1", random.normal(random.PRNGKey(0), (latent_dim, emission_dim))
        ),
        "l2": numpyro.param(
            "emitter_l2", random.normal(random.PRNGKey(1), (emission_dim, emission_dim))
        ),
        "l3": numpyro.param(
            "emitter_l3", random.normal(random.PRNGKey(2), (emission_dim, data_dim))
        ),
    }

    trans_params = {
        "gate": {
            "l1": numpyro.param(
                "gate_l1",
                random.normal(random.PRNGKey(3), (latent_dim, transition_dim)),
            ),
            "l2": numpyro.param(
                "gate_l2",
                random.normal(random.PRNGKey(4), (transition_dim, latent_dim)),
            ),
        },
        "shared": {
            "l1": numpyro.param(
                "shared_l1",
                random.normal(random.PRNGKey(5), (latent_dim, transition_dim)),
            ),
            "l2": numpyro.param(
                "shared_l2",
                random.normal(random.PRNGKey(6), (transition_dim, latent_dim)),
            ),
        },
        "mean": {
            "l1": numpyro.param(
                "mean_l1", random.normal(random.PRNGKey(7), (latent_dim, latent_dim))
            )
        },
        "std": {
            "l1": numpyro.param(
                "std_l1", random.normal(random.PRNGKey(8), (latent_dim, latent_dim))
            )
        },
    }

    z0 = numpyro.param("z0", random.normal(random.PRNGKey(9), (latent_dim,)))
    z0 = jnp.broadcast_to(z0, (seqs.shape[0], 1, latent_dim))

    with numpyro.plate("data", seqs.shape[0]):
        masks = jnp.arange(max_seq_length) < lengths[:, None]
        z = numpyro.sample(
            "z",
            dist.Normal(
                jnp.zeros((max_seq_length, latent_dim)),
                jnp.ones((max_seq_length, latent_dim)),
            )
            .mask(False)
            .to_event(2),
        )

        z_shift = jnp.concatenate([z0, z[:, :-1, :]], axis=1)
        z_loc, z_scale = transition(z_shift, trans_params)
        numpyro.sample(
            "z_aux",
            dist.Normal(z_loc, z_scale).mask(masks[..., None]).to_event(2),
            obs=z,
        )

        emission_probs = emitter(z, emitter_params)
        numpyro.sample(
            "tunes",
            dist.Bernoulli(logits=emission_probs).mask(masks[..., None]).to_event(2),
            obs=seqs,
        )


def guide(seqs, seqs_rev, lengths, *, latent_dim=32, gru_dim=150, **kwargs):
    num_sequences, max_seq_length, _ = seqs.shape

    combiner_params = {
        "mean": numpyro.param(
            "combiner_mean",
            jax.random.normal(random.PRNGKey(10), (gru_dim, latent_dim)),
        ),
        "std": numpyro.param(
            "combiner_std", jax.random.normal(random.PRNGKey(11), (gru_dim, latent_dim))
        ),
    }

    with numpyro.plate("data", seqs.shape[0]):
        locs = numpyro.param(
            "locs",
            jax.random.normal(
                random.PRNGKey(12), (num_sequences, max_seq_length, latent_dim)
            ),
        )
        scales = numpyro.param(
            "scales",
            nn.softplus(
                jax.random.normal(
                    random.PRNGKey(13), (num_sequences, max_seq_length, latent_dim)
                )
            ),
        )
        numpyro.sample("z", dist.Normal(locs, scales).to_event(2))


# 4. Evaluation
def evaluate_model(true_vals, pred_vals):
    """Compute evaluation metrics for the model."""
    mse = jnp.mean((true_vals - pred_vals.mean(axis=0)) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")
    return mse


# 5. Main
if __name__ == "__main__":
    # Hyperparameters
    num_sequences = 100
    sequence_length = 50
    data_dim = 88

    rng_key = random.PRNGKey(0)
    seqs, lengths = generate_synthetic_data(
        num_sequences, sequence_length, data_dim, rng_key
    )
    seqs_rev = _reverse_padded(seqs, lengths)

    # Initialize and train SteinVI
    steinvi = SteinVI(
        model,
        guide,
        optax_to_numpyro(adam(1e-3)),
        RBFKernel(),
        num_stein_particles=10,
        num_elbo_particles=5,
    )

    results = steinvi.run(
        rng_key, 10, seqs, seqs_rev, lengths, latent_dim=32, data_dim=data_dim
    )
    params = results.params

    # Generate predictions
    predictive = MixtureGuidePredictive(
        model,
        guide,
        params=params,
        num_samples=10,
        guide_sites=steinvi.guide_sites,
    )

    pred_key = random.PRNGKey(1)
    predictions = predictive(pred_key, seqs, seqs_rev, lengths, latent_dim=32)["tunes"]

    # Evaluate and visualize
    mse = evaluate_model(seqs, predictions)
    plot_predictions(seqs, predictions)
