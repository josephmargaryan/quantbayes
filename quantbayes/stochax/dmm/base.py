# base_dmm.py
from abc import ABC, abstractmethod
import jax.numpy as jnp


class BaseTransition(ABC):
    @abstractmethod
    def __call__(self, z_prev: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Given the previous latent state z_prev, returns the parameters (loc, scale)
        of p(z_t | z_{t-1}).

        Parameters:
            z_prev: jnp.ndarray of shape (batch, latent_dim)

        Returns:
            A tuple (loc, scale) each of shape (batch, latent_dim)
        """
        pass


class BaseEmission(ABC):
    @abstractmethod
    def __call__(self, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Given a latent state z, returns the parameters (loc, scale) of p(x_t | z_t).

        Parameters:
            z: jnp.ndarray of shape (batch, latent_dim)

        Returns:
            A tuple (loc, scale) each of shape (batch, observation_dim)
        """
        pass


class BasePosterior(ABC):
    @abstractmethod
    def __call__(self, x_seq: jnp.ndarray) -> jnp.ndarray:
        """
        Given a sequence of observations x_seq, returns a sequence of hidden
        representations or parameters for constructing the variational posterior.

        Parameters:
            x_seq: jnp.ndarray of shape (batch, T, observation_dim)

        Returns:
            A representation (e.g., hidden states) of shape (batch, T, hidden_dim)
        """
        pass


class BaseCombiner(ABC):
    @abstractmethod
    def __call__(
        self, z_prev: jnp.ndarray, h: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Given the previous latent state z_prev and a hidden state h (from the posterior RNN),
        returns the parameters (loc, scale) of q(z_t | z_{t-1}, h_t).

        Parameters:
            z_prev: jnp.ndarray of shape (batch, latent_dim)
            h: jnp.ndarray of shape (batch, hidden_dim)

        Returns:
            A tuple (loc, scale) each of shape (batch, latent_dim)
        """
        pass


class BaseDMM(ABC):
    @abstractmethod
    def reparam_sample(self, rng, loc: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        """
        Implements the reparameterization trick to sample z from a Gaussian
        with given loc and scale.

        Parameters:
            rng: a JAX random key
            loc: jnp.ndarray of shape (batch, latent_dim)
            scale: jnp.ndarray of shape (batch, latent_dim)

        Returns:
            A sample z of shape (batch, latent_dim)
        """
        pass

    @abstractmethod
    def __call__(self, x_seq: jnp.ndarray, rng) -> jnp.ndarray:
        """
        Given a sequence of observations, compute the model’s objective (e.g., the
        negative ELBO) by processing the sequence.

        Parameters:
            x_seq: jnp.ndarray of shape (batch, T, observation_dim)
            rng: a JAX random key

        Returns:
            A scalar representing the negative ELBO (or another objective) averaged over the batch.
        """
        pass
