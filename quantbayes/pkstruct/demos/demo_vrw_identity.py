from __future__ import annotations

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import numpyro.distributions as dist

from quantbayes.pkstruct.core.pk import PKPosterior
from quantbayes.pkstruct.core.protocols import PKComponents
from quantbayes.pkstruct.toy.vrw import vrw_r, stephens_logpdf_r, StephensConfig
from quantbayes.pkstruct.toy.vrw_components import VonMisesIIDPrior


# =========================
# USER SETTINGS (edit here)
# =========================
N = 5
MU = 0.0
KAPPA = 10.0
SEED = 0
DRAWS = 512
TOL = 1e-8


class StephensDensity:
    def __init__(self, cfg: StephensConfig):
        self.cfg = cfg

    def log_prob(self, r):
        return stephens_logpdf_r(r, cfg=self.cfg)


def main():
    print("=== VRW PK identity demo ===")
    prior = VonMisesIIDPrior(N=N, mu=MU, kappa=KAPPA)
    steph_cfg = StephensConfig(kappa=KAPPA, N=N)

    evidence = StephensDensity(steph_cfg)
    reference = StephensDensity(steph_cfg)

    posterior = PKPosterior(
        PKComponents(
            prior=prior,
            coarse_map=vrw_r,
            evidence=evidence,
            reference=reference,
        )
    )

    theta = (
        dist.VonMises(MU, KAPPA)
        .expand([N])
        .to_event(1)
        .sample(jr.PRNGKey(SEED), (DRAWS,))
    )
    theta = jnp.asarray(theta)

    lp_prior = jax.vmap(prior.log_prob)(theta)
    lp_post = jax.vmap(posterior.log_prob)(theta)

    max_abs = float(jnp.max(jnp.abs(lp_post - lp_prior)))
    print(f"Max |logp_post - logp_prior| = {max_abs:.3e}")
    assert max_abs < TOL, f"Identity failed: {max_abs} >= {TOL}"
    print("PASS: evidence==reference cancels (posterior==prior pointwise).")


if __name__ == "__main__":
    main()
