import time
import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap
import numpyro
from numpyro.infer import MCMC, NUTS
from gp_model import kernel
import os


def run_inference(model, args, rng_key, X, Y):
    start = time.time()
    init_strategies = {
        "value": numpyro.infer.init_to_value(
            values={"kernel_var": 1.0, "kernel_noise": 0.05, "kernel_length": 0.5}
        ),
        "median": numpyro.infer.init_to_median(num_samples=10),
        "feasible": numpyro.infer.init_to_feasible(),
        "sample": numpyro.infer.init_to_sample(),
        "uniform": numpyro.infer.init_to_uniform(radius=1),
    }
    kernel = NUTS(model, init_strategy=init_strategies[args.init_strategy])
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thinning,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


def predict(rng_key, X, Y, X_test, var, length, noise, use_cholesky=True):
    k_pp = kernel(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = kernel(X_test, X, var, length, noise, include_noise=False)
    k_XX = kernel(X, X, var, length, noise, include_noise=True)

    if use_cholesky:
        K_xx_cho = jax.scipy.linalg.cho_factor(k_XX)
        K = k_pp - jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, k_pX.T))
        mean = jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, Y))
    else:
        K_xx_inv = jnp.linalg.inv(k_XX)
        K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, k_pX.T))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))

    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), 0.0)) * random.normal(
        rng_key, X_test.shape[:1]
    )
    return mean, mean + sigma_noise
