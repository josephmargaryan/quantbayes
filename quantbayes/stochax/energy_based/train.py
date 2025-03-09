# trainer.py
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from quantbayes.stochax.energy_based.base import BaseEBM


def short_run_langevin_dynamics(
    rng,
    ebm: BaseEBM,
    init_x: jnp.ndarray,
    step_size=0.1,
    n_steps=50,
) -> jnp.ndarray:
    """
    A short-run MCMC sampler using (naive) Langevin updates.
    """

    def mcmc_step(x, rng_input):
        grad_e = jax.grad(lambda z: jnp.mean(ebm.energy(z)))(x)
        noise = jax.random.normal(rng_input, shape=x.shape)
        # One step of SGLD: x = x - step_size/2 * gradE(x) + sqrt(step_size)*noise
        x = x - 0.5 * step_size * grad_e + jnp.sqrt(step_size) * noise
        return x, None

    rngs = jax.random.split(rng, n_steps)
    final_x, _ = jax.lax.scan(mcmc_step, init_x, rngs)
    return final_x


def cd_training_step(
    rng,
    ebm: BaseEBM,
    x_data: jnp.ndarray,
    optimizer_update_fn: Callable,
    opt_state,
    step_size=0.1,
    n_steps=50,
):
    """
    One step of contrastive divergence-like training using short-run MCMC.
    """

    def loss_fn(model, rng):
        # E(real)
        e_real = jnp.mean(model.energy(x_data))

        # Generate negative samples
        init_rng, sampler_rng = jax.random.split(rng, 2)
        init_x = jax.random.normal(init_rng, shape=x_data.shape)  # wide Gaussian init
        x_fake = short_run_langevin_dynamics(
            sampler_rng, model, init_x, step_size, n_steps
        )

        # E(fake)
        e_fake = jnp.mean(model.energy(x_fake))

        loss = e_real - e_fake  # Minimizing this tries to push E(real) < E(fake).
        return loss, (e_real, e_fake)

    # Compute gradients only on the trainable parameters.
    (loss, (e_real, e_fake)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        ebm, rng
    )
    updates, new_opt_state = optimizer_update_fn(grads, opt_state, ebm)
    new_ebm = eqx.apply_updates(ebm, updates)

    return new_ebm, new_opt_state, e_real, e_fake


class EBMTrainer:
    """
    A minimal EBM trainer that uses short-run MCMC (CD-k).
    """

    def __init__(self, ebm: BaseEBM, lr=1e-3):
        self.ebm = ebm
        self.optimizer = optax.adam(lr)
        # Extract the trainable parameters from the model.
        params = eqx.filter(ebm, eqx.is_array)
        self.opt_state = self.optimizer.init(params)

    def train_step(self, rng, x_data):
        def optimizer_update_fn(grads, opt_state, model):
            updates, new_opt_state = self.optimizer.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            return updates, new_opt_state

        new_ebm, new_opt_state, e_real, e_fake = cd_training_step(
            rng, self.ebm, x_data, optimizer_update_fn, self.opt_state
        )
        self.ebm = new_ebm
        self.opt_state = new_opt_state
        return e_real, e_fake
