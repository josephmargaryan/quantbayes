from __future__ import annotations
import equinox as eqx


def clone_module(module: eqx.Module) -> eqx.Module:
    """
    Cheap structural clone of an Equinox module:
    - returns a new dataclass instance
    - shares leaf arrays (JAX arrays are immutable/persistent)
    - any Optax update produces new leaves; the original remains untouched
    """
    params, static = eqx.partition(module, eqx.is_array)
    return eqx.combine(params, static)
