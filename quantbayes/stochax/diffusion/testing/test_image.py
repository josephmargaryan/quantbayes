from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.diffusion.dataloaders import (
    generate_synthetic_image_dataset,
    dataloader,
)
from quantbayes.stochax.diffusion.trainer import train_model
from quantbayes.stochax.diffusion.edm import edm_batch_loss
from quantbayes.stochax.diffusion import sample_edm, sample_edm_conditional
from quantbayes.stochax.diffusion.generate import generate_with_sampler

from quantbayes.stochax.diffusion.models.unet_2d import UNet
from quantbayes.stochax.diffusion.models.mixer_2d import Mixer2d
from quantbayes.stochax.diffusion.models.adaptive_DiT import DiT
from quantbayes.stochax.diffusion.models.wrappers import (
    UnconditionalWrapper,
    DiTWrapper,
)


def train_edm(model, data, steps=80, seed=0, batch=32):
    return train_model(
        model,
        dataset=data,
        t1=1.0,
        lr=3e-4,
        num_steps=steps,
        batch_size=batch,
        weight_fn=None,
        int_beta_fn=None,
        print_every=max(steps // 2, 1),
        seed=seed,
        data_loader_func=dataloader,
        loss_impl="edm",
        custom_loss=lambda m, b, k: edm_batch_loss(
            m, b, k, sigma_data=0.5, rho_min=-1.2, rho_max=1.2, sample="uniform"
        ),
        checkpoint_dir=None,
    )


def run_one_model(name, factory_fn, data, seed=0):
    key = jr.PRNGKey(seed)
    model_core = factory_fn(key, data.shape[1:])  # (C,H,W)

    if name == "DiT":
        model = DiTWrapper(
            model=model_core,
            num_classes=10,
            time_mode="vp_t",
            null_label_index=None,
            cfg_rescale=0.7,
        )
    else:
        model = UnconditionalWrapper(model=model_core, time_mode="vp_t")

    ema = train_edm(model, data, steps=80, seed=seed, batch=32)

    # Default sampler (DPM++-3M)
    uncond = sample_edm(
        ema_model=ema,
        num_samples=8,
        sample_shape=data.shape[1:],  # (C,H,W)
        key=jr.PRNGKey(seed + 1),
        steps=12,
        sigma_min=0.002,
        sigma_max=1.0,
        sigma_data=0.5,
        rho=7.0,
    )
    print(f"{name} (uncond, dpmpp_3m) samples shape:", uncond.shape)

    if name == "DiT":
        labels = jnp.full((8,), 3, dtype=jnp.int32)
        cond = sample_edm_conditional(
            ema_model=ema,
            label=labels,
            cfg_scale=3.0,
            num_samples=8,
            sample_shape=data.shape[1:],
            key=jr.PRNGKey(seed + 2),
            steps=12,
            sigma_min=0.002,
            sigma_max=1.0,
            sigma_data=0.5,
            rho=7.0,
        )
        print(f"{name} (CFG=3.0, label=3, dpmpp_3m) samples shape:", cond.shape)

    # Unified sampler API demos
    ema_eval = eqx.tree_inference(ema, value=True)

    def denoise_fn(log_sigma, x):
        return ema_eval(log_sigma, x, key=None, train=False)

    uni = generate_with_sampler(
        denoise_fn,
        "unipc",
        data.shape[1:],
        key=jr.PRNGKey(seed + 3),
        num_samples=4,
        sampler_kwargs=dict(
            steps=12, sigma_min=0.002, sigma_max=1.0, sigma_data=0.5, rho=7.0
        ),
    )
    print(f"{name} (unipc) samples shape:", uni.shape)

    ip2 = generate_with_sampler(
        denoise_fn,
        "ipndm",
        data.shape[1:],
        key=jr.PRNGKey(seed + 4),
        num_samples=4,
        sampler_kwargs=dict(
            steps=12, sigma_min=0.002, sigma_max=1.0, sigma_data=0.5, rho=7.0
        ),
    )
    print(f"{name} (ipndm) samples shape:", ip2.shape)

    ip4 = generate_with_sampler(
        denoise_fn,
        "ipndm4",
        data.shape[1:],
        key=jr.PRNGKey(seed + 5),
        num_samples=4,
        sampler_kwargs=dict(
            steps=12, sigma_min=0.002, sigma_max=1.0, sigma_data=0.5, rho=7.0
        ),
    )
    print(f"{name} (ipndm4) samples shape:", ip4.shape)

    v3 = generate_with_sampler(
        denoise_fn,
        "dpmv3",
        data.shape[1:],
        key=jr.PRNGKey(seed + 6),
        num_samples=4,
        sampler_kwargs=dict(
            steps=12, sigma_min=0.002, sigma_max=1.0, sigma_data=0.5, rho=7.0, order=3
        ),
    )
    print(f"{name} (dpmv3-o3) samples shape:", v3.shape)


def make_unet(key, img_shape):
    C, H, W = img_shape
    return UNet(
        data_shape=img_shape,
        is_biggan=False,
        dim_mults=[1, 2],
        hidden_size=32,
        heads=4,
        dim_head=32,
        dropout_rate=0.0,
        num_res_blocks=2,
        attn_resolutions=[H // 2, H, W // 2, W],
        key=key,
    )


def make_mixer(key, img_shape):
    C, H, W = img_shape
    return Mixer2d(
        img_size=img_shape,
        patch_size=4,
        hidden_size=64,
        mix_patch_size=256,
        mix_hidden_size=256,
        num_blocks=2,
        t1=1.0,
        key=key,
    )


def make_dit(key, img_shape):
    C, H, W = img_shape
    return DiT(
        img_size=img_shape,
        patch_size=4,
        in_channels=C,
        embed_dim=192,
        depth=2,
        n_heads=6,
        mlp_ratio=4.0,
        dropout_rate=0.0,
        time_emb_dim=192,
        num_classes=10,
        learn_sigma=False,
        key=key,
    )


def main():
    key = jr.PRNGKey(0)
    data = generate_synthetic_image_dataset(num_samples=256, shape=(3, 32, 32), key=key)

    print("=== UNet ===")
    run_one_model("UNet", make_unet, data, seed=10)

    print("\n=== Mixer ===")
    run_one_model("Mixer", make_mixer, data, seed=20)

    print("\n=== DiT ===")
    run_one_model("DiT", make_dit, data, seed=30)


if __name__ == "__main__":
    main()
