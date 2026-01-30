from __future__ import annotations

from pathlib import Path
import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx

import numpyro.distributions as dist

from quantbayes.pkstruct.toy.vrw import vrw_endpoint
from quantbayes.stochax.diffusion.sde import make_weight_fn
from quantbayes.stochax.diffusion.schedules.vp import make_vp_int_beta
from quantbayes.stochax.diffusion.trainer import train_model
from quantbayes.stochax.diffusion.dataloaders import dataloader

from quantbayes.pkdiffusion.models import ScoreMLP


OUT_DIR = Path("reports/pkdiffusion/vrw_endpoint_score")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # -------------------------
    # Data: VRW prior endpoints
    # -------------------------
    N = 5
    MU = 0.0
    KAPPA = 10.0
    DATASET_SIZE = 120_000

    seed = 0
    key = jr.PRNGKey(seed)
    key_theta, key_model = jr.split(key)

    # Sample theta ~ iid VonMises, then map to endpoint in R^2.
    theta = (
        dist.VonMises(MU, KAPPA)
        .expand([N])
        .to_event(1)
        .sample(key_theta, (DATASET_SIZE,))
    )  # (M,N)
    endpoints = jax.vmap(vrw_endpoint)(theta)  # (M,2)
    endpoints = jnp.asarray(endpoints)

    # -------------------------
    # Diffusion training config
    # -------------------------
    t1 = 10.0
    beta_min = 0.1
    beta_max = 20.0

    lr = 3e-4
    batch_size = 2048
    num_steps = 80_000
    print_every = 1000

    int_beta_fn = make_vp_int_beta(
        "linear", beta_min=beta_min, beta_max=beta_max, t1=t1
    )
    weight_fn = make_weight_fn(int_beta_fn, name="likelihood")

    # -------------------------
    # Model
    # -------------------------
    model = ScoreMLP(dim=2, time_dim=32, width_size=256, depth=3, key=key_model)

    # -------------------------
    # Train (EMA model returned)
    # -------------------------
    ckpt_dir = OUT_DIR / "ckpt"
    ema_model = train_model(
        model=model,
        dataset=endpoints,
        t1=t1,
        lr=lr,
        num_steps=num_steps,
        batch_size=batch_size,
        weight_fn=weight_fn,
        int_beta_fn=int_beta_fn,
        print_every=print_every,
        seed=seed,
        data_loader_func=dataloader,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every=10_000,
        keep_last=3,
    )

    # Save EMA model leaves
    model_path = OUT_DIR / "ema_score_model.eqx"
    with open(model_path, "wb") as f:
        eqx.tree_serialise_leaves(f, ema_model)

    cfg = dict(
        N=N,
        MU=MU,
        KAPPA=KAPPA,
        DATASET_SIZE=DATASET_SIZE,
        t1=t1,
        beta_min=beta_min,
        beta_max=beta_max,
        lr=lr,
        batch_size=batch_size,
        num_steps=num_steps,
        print_every=print_every,
        model_arch=dict(dim=2, time_dim=32, width_size=256, depth=3),
        seed=seed,
        model_path=str(model_path),
    )
    (OUT_DIR / "config.json").write_text(json.dumps(cfg, indent=2))

    print("=== VRW endpoint score training done ===")
    print(f"Saved model: {model_path}")
    print(f"Saved config: {OUT_DIR / 'config.json'}")


if __name__ == "__main__":
    main()
