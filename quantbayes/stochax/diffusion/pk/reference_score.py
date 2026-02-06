from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from quantbayes.stochax.diffusion.dataloaders import dataloader as jax_dataloader


class ScoreNet1D(eqx.Module):
    """Tiny 1D score MLP: s(z) ≈ ∂/∂z log π(z)."""

    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    l3: eqx.nn.Linear

    def __init__(self, hidden: int = 128, *, key: jr.PRNGKey):
        k1, k2, k3 = jr.split(key, 3)
        self.l1 = eqx.nn.Linear(1, hidden, key=k1)
        self.l2 = eqx.nn.Linear(hidden, hidden, key=k2)
        self.l3 = eqx.nn.Linear(hidden, 1, key=k3)

    def _single(self, z1: jnp.ndarray) -> jnp.ndarray:
        # z1: (1,)
        h = jnp.tanh(self.l1(z1))
        h = jnp.tanh(self.l2(h))
        out = self.l3(h)  # (1,)
        return out[0]  # scalar

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Accepts z of shape:
          - (B,) or (B,1) or scalar
        Returns:
          - (B,) or scalar
        """
        z = jnp.asarray(z)
        if z.ndim == 0:
            z1 = z.reshape((1,))
            return self._single(z1)

        if z.ndim == 1:
            z2 = z.reshape((-1, 1))
        elif z.ndim == 2 and z.shape[-1] == 1:
            z2 = z
        else:
            raise ValueError(f"ScoreNet1D expects z shape (B,) or (B,1). Got {z.shape}")

        return jax.vmap(self._single)(z2)


@dataclass(frozen=True)
class ScoreNet1DConfig:
    hidden: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    steps: int = 4000
    noise_std: float = 0.08
    seed: int = 0
    print_every: int = 200


def train_or_load_score_net_dsm(
    z_ref: jnp.ndarray,
    score_path: Path,
    *,
    cfg: ScoreNet1DConfig,
) -> ScoreNet1D:
    """
    Train (or load) a 1D score network using denoising score matching on z.

    DSM objective (fixed noise std):
      z_tilde = z + σ ε
      target  = -(z_tilde - z)/σ^2
      min E|| sθ(z_tilde) - target ||^2
    """
    score_path = Path(score_path)
    score_path.parent.mkdir(parents=True, exist_ok=True)

    template = ScoreNet1D(hidden=cfg.hidden, key=jr.PRNGKey(cfg.seed + 888))

    if score_path.exists():
        with open(score_path, "rb") as f:
            net = eqx.tree_deserialise_leaves(f, template)
        return net

    # Ensure z_ref is (N,1)
    z_ref = jnp.asarray(z_ref)
    if z_ref.ndim == 1:
        z_ref = z_ref.reshape((-1, 1))
    elif z_ref.ndim == 2 and z_ref.shape[-1] == 1:
        pass
    else:
        raise ValueError(f"z_ref must be (N,) or (N,1). Got {z_ref.shape}")

    net = template
    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = opt.init(eqx.filter(net, eqx.is_inexact_array))

    loader = jax_dataloader(z_ref, cfg.batch_size, key=jr.PRNGKey(cfg.seed + 999))
    k = jr.PRNGKey(cfg.seed + 1001)

    noise_std = float(cfg.noise_std)

    @eqx.filter_jit
    def step(m: ScoreNet1D, s, z_clean: jnp.ndarray, key: jr.PRNGKey):
        noise = jr.normal(key, z_clean.shape) * noise_std
        z_tilde = z_clean + noise
        target = -((z_tilde - z_clean) / (noise_std**2)).squeeze(-1)  # (B,)

        def loss_fn(mm: ScoreNet1D):
            pred = mm(z_tilde)  # (B,)
            return jnp.mean((pred - target) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        params = eqx.filter(m, eqx.is_inexact_array)
        updates, s = opt.update(grads, s, params)
        m = eqx.apply_updates(m, updates)
        return loss, m, s

    running = 0.0
    for i in range(1, cfg.steps + 1):
        zb = next(loader)
        k, sub = jr.split(k)
        loss, net, opt_state = step(net, opt_state, zb, sub)
        running += float(loss)

        if (cfg.print_every > 0) and (i % cfg.print_every == 0):
            print(f"[score DSM] step {i:5d} | loss {running / cfg.print_every:.6f}")
            running = 0.0

    with open(score_path, "wb") as f:
        eqx.tree_serialise_leaves(f, net)

    return net
