from __future__ import annotations

from typing import Any

try:
    from jax import Array as JaxArray
except Exception:  # pragma: no cover
    JaxArray = Any  # fallback for older JAX

Array = JaxArray
PRNGKey = JaxArray
