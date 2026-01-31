# quantbayes/stochax/diffusion/latent/autoencoder_kl.py
from __future__ import annotations
from dataclasses import dataclass
import equinox as eqx
import jax
import jax.numpy as jnp


@dataclass
class AutoencoderKL(eqx.Module):
    """
    Thin wrapper around your existing VAE modules to expose:
      - encode(x, key=None, deterministic=True) -> z
      - decode(z) -> x_recon
      - scale: optional scaling of latents (e.g., 0.18215 for SDXL)
    """

    vae: eqx.Module
    scale: float = 1.0

    def encode(
        self, x: jnp.ndarray, key=None, deterministic: bool = True
    ) -> jnp.ndarray:
        # Prefer to use the encoder directly to avoid sampling noise in latents.
        mu, logvar = self.vae.encoder(x)  # your encoders already exist on vae
        if deterministic or key is None:
            z = mu
        else:
            z = self.vae.sample_z(key, mu, logvar)
        return z / self.scale

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.vae.decoder(z * self.scale)
