from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from ..typing import Array
from .protocols import PKComponents


@dataclass(frozen=True)
class PKPosterior:
    """
    Probability Kinematics (Jeffrey conditioning) posterior in reference-ratio form:

      log p(ω) = log π(ω) + log p(ξ(ω)) - log π(ξ(ω))  + const

    where ξ = coarse_map(ω).

    This is the exact form used in the AF1 reinterpretation and in the toy VRW model (paper Section 5).
    """

    components: PKComponents

    def coarse(self, omega: Array) -> Array:
        return self.components.coarse_map(omega)

    def log_prob(self, omega: Array) -> Array:
        xi = self.coarse(omega)
        return (
            self.components.prior.log_prob(omega)
            + self.components.evidence.log_prob(xi)
            - self.components.reference.log_prob(xi)
        )

    def energy(self, omega: Array) -> Array:
        """Negative log posterior up to additive constant."""
        return -self.log_prob(omega)

    def grad_energy(self, omega: Array) -> Array:
        return jax.grad(self.energy)(omega)

    def batched_log_prob(self, omega_batch: Array) -> Array:
        """Vectorized log_prob over leading batch axis."""
        return jax.vmap(self.log_prob)(omega_batch)

    def batched_energy(self, omega_batch: Array) -> Array:
        return jax.vmap(self.energy)(omega_batch)
