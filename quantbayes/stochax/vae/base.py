# base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import jax.numpy as jnp


class BaseEncoder(ABC):
    @abstractmethod
    def __call__(
        self, x: jnp.ndarray, *, rng=None, train: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (mu, logvar) of shape (batch, latent_dim)."""
        ...


class BaseDecoder(ABC):
    @abstractmethod
    def __call__(self, z: jnp.ndarray, *, rng=None, train: bool = False) -> jnp.ndarray:
        """Return decoder output. For Bernoulli: logits; for Gaussian: mean."""
        ...


class BaseVAE(ABC):
    @abstractmethod
    def sample_z(self, rng, mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
        """Reparameterization sample z ~ q(z|x)."""
        ...

    @abstractmethod
    def __call__(self, x: jnp.ndarray, rng, *, train: bool = False) -> tuple:
        """
        Forward pass: returns (decoder_out, mu, logvar).
        decoder_out meaning depends on likelihood: logits (Bernoulli) or mean (Gaussian).
        """
        ...
