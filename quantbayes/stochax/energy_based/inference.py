# generation.py
import jax
import jax.numpy as jnp

from quantbayes.stochax.energy_based.base import BaseEBM
from quantbayes.stochax.energy_based.train import short_run_langevin_dynamics


def generate_samples(
    rng, ebm: BaseEBM, n_samples=64, shape=(28, 28), step_size=0.1, n_steps=100
):
    """
    Generate new samples from a trained EBM.
    shape: e.g. for image data, (C, H, W) or (H, W), etc.
    """
    init_x = jax.random.normal(rng, shape=(n_samples,) + shape)
    samples = short_run_langevin_dynamics(
        rng, ebm, init_x, step_size=step_size, n_steps=n_steps
    )
    return samples


def detect_ood(ebm: BaseEBM, x: jnp.ndarray, threshold: float) -> jnp.ndarray:
    """
    Return a boolean mask of which samples are classified as OOD.
    E.g. OOD if energy(x) > threshold.
    """
    energies = ebm.energy(x)
    return energies > threshold
