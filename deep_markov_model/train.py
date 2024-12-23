from numpyro.contrib.einstein import SteinVI
from numpyro.contrib.einstein.mixture_guide_predictive import MixtureGuidePredictive
from numpyro.contrib.einstein.stein_kernels import RBFKernel
from numpyro.optim import optax_to_numpyro
from optax import adam, chain
from jax import random


def train_and_predict(args, model, guide, load_data):
    inf_key, pred_key = random.split(random.PRNGKey(seed=args.rng_seed), 2)

    steinvi = SteinVI(
        model,
        guide,
        optax_to_numpyro(chain(adam(1e-2))),
        RBFKernel(),
        num_elbo_particles=args.num_elbo_particles,
        num_stein_particles=args.num_stein_particles,
    )

    seqs, rev_seqs, lengths = load_data()
    results = steinvi.run(
        inf_key,
        args.max_iter,
        seqs,
        rev_seqs,
        lengths,
        gru_dim=args.gru_dim,
        subsample_size=args.subsample_size,
    )
    pred = MixtureGuidePredictive(
        model,
        guide,
        params=results.params,
        num_samples=1,
        guide_sites=steinvi.guide_sites,
    )
    seqs_valid, rev_seqs_valid, lengths_valid = load_data("valid")
    pred_notes = pred(
        pred_key,
        seqs_valid,
        rev_seqs_valid,
        lengths_valid,
        subsample_size=seqs_valid.shape[0],
        predict=True,
    )["tunes"]

    return results, seqs, rev_seqs, lengths, pred_notes, None
