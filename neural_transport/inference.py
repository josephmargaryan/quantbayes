import os
from jax import random
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro import optim
from numpyro.infer.autoguide import AutoBNAFNormal
from numpyro.infer.reparam import NeuTraReparam


def run_vanilla_hmc(model, num_warmup, num_samples, num_chains):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(random.PRNGKey(0))
    mcmc.print_summary()
    return mcmc.get_samples()["x"].copy()


def run_svi(model, num_iters, hidden_factor):
    guide = AutoBNAFNormal(model, hidden_factors=[hidden_factor, hidden_factor])
    svi = SVI(model, guide, optim.Adam(0.003), Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(1), num_iters)
    return guide, svi_result


def run_neutra_hmc(model, guide, svi_result, num_warmup, num_samples, num_chains):
    neutra = NeuTraReparam(guide, svi_result.params)
    neutra_model = neutra.reparam(model)
    nuts_kernel = NUTS(neutra_model)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(random.PRNGKey(3))
    mcmc.print_summary()
    zs = mcmc.get_samples(group_by_chain=True)["auto_shared_latent"]
    samples = neutra.transform_sample(zs)
    return zs, samples, neutra
