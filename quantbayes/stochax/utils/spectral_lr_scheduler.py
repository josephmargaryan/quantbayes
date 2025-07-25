# asl/schedules.py
"""Learning‑rate & frequency schedulers for Adaptive Spectral Layers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import equinox as eqx


class HasMask(Protocol):
    """Layer interface that exposes a Boolean frequency mask."""

    k_mask: jax.Array  # shape (k_half,) or (H, W)


@dataclass
class ProgressiveK:
    """
    Increases the number of active Fourier bins over training.

    Args
    ----
    k_start : int
        Initial number of frequencies.
    k_final : int
        Final number after `total_steps`.
    total_steps : int
        Number of optimiser steps for the schedule.
    """

    k_start: int
    k_final: int
    total_steps: int

    def __call__(self, step: int, layer: HasMask) -> HasMask:
        """Return a *new* layer with an updated mask (pure functional style)."""
        k_now = int(
            jnp.clip(
                jnp.round(
                    self.k_start
                    + (self.k_final - self.k_start) * step / self.total_steps
                ),
                self.k_start,
                self.k_final,
            )
        )
        new_mask = layer.k_mask.at[:k_now].set(True).at[k_now:].set(False)
        return eqx.tree_at(lambda l: l.k_mask, layer, new_mask)
