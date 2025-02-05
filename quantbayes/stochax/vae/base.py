# base_vae.py
from abc import ABC, abstractmethod
import jax.numpy as jnp


class BaseEncoder(ABC):
    @abstractmethod
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Given a batch of inputs x, returns the latent parameters (mu, logvar)
        that parameterize q(z|x) (typically for a Gaussian).

        Parameters:
            x: jnp.ndarray of shape (batch, input_dim)

        Returns:
            A tuple (mu, logvar) each of shape (batch, latent_dim)
        """
        pass


class BaseDecoder(ABC):
    @abstractmethod
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Given a batch of latent variables z, returns a reconstruction of x.

        Parameters:
            z: jnp.ndarray of shape (batch, latent_dim)

        Returns:
            x_recon: jnp.ndarray of shape (batch, output_dim)
        """
        pass


class BaseVAE(ABC):
    @abstractmethod
    def sample_z(self, rng, mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
        """
        Implements the reparameterization trick to sample z ~ q(z|x)
        using the provided mu and logvar.

        Parameters:
            rng: a JAX random key
            mu: jnp.ndarray of shape (batch, latent_dim)
            logvar: jnp.ndarray of shape (batch, latent_dim)

        Returns:
            A sample z of shape (batch, latent_dim)
        """
        pass

    @abstractmethod
    def __call__(self, x: jnp.ndarray, rng) -> tuple:
        """
        The forward pass of the VAE. This should compute the latent parameters,
        sample z, then decode z to produce a reconstruction.

        Parameters:
            x: jnp.ndarray of shape (batch, input_dim)
            rng: a JAX random key

        Returns:
            A tuple (x_recon, mu, logvar)
        """
        pass
