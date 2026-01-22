from __future__ import annotations

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

from quantbayes.pkstruct.utils.angles import wrap_angle
from quantbayes.stochax.pkstruct.samplers import (
    ULAConfig,
    MALAConfig,
    run_ula,
    run_mala_wrapped,
    thin_chain,
)


# =========================
# USER SETTINGS (edit here)
# =========================
SEED = 0
D = 7


def main():
    print("=== stochax sampler sanity demo ===")
    key = jr.PRNGKey(SEED)
    z0 = wrap_angle(jr.normal(key, (D,), dtype=jnp.float64))

    # --- ULA: should stay wrapped ---
    def energy_quadratic(z):
        return 0.5 * jnp.sum(z**2)

    ula_cfg = ULAConfig(step_size=1e-3, num_steps=300, burn_in=50, thin=5)
    zs = run_ula(key, z0, ula_cfg, energy_quadratic)
    zs_thin = thin_chain(zs, burn_in=ula_cfg.burn_in, thin=ula_cfg.thin)
    zs_np = np.array(zs_thin)

    assert np.all(zs_np <= np.pi + 1e-12)
    assert np.all(zs_np > -np.pi - 1e-12)
    print("PASS: ULA maintains wrapping to (-pi, pi].")

    # --- Wrapped MALA: flat energy should accept ~1 ---
    def energy_flat(z):
        return jnp.asarray(0.0, dtype=z.dtype)

    mala_cfg = MALAConfig(step_size=5e-3, num_steps=400, burn_in=50, thin=5, wrap_k=3)
    zs_m, acc = run_mala_wrapped(key, z0, mala_cfg, energy_flat)
    _ = thin_chain(zs_m, burn_in=mala_cfg.burn_in, thin=mala_cfg.thin)

    print(f"wrapped-MALA acceptance (flat energy) = {acc:.3f}")
    assert acc > 0.90
    print("PASS: wrapped-MALA acceptance high for flat energy (proposal symmetry OK).")


if __name__ == "__main__":
    main()
