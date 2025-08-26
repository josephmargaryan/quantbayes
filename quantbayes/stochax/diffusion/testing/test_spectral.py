# quantbayes/stochax/diffusion/testing/test_spectral.py
from __future__ import annotations
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from quantbayes.stochax.diffusion.dataloaders import (
    generate_synthetic_image_dataset,
    dataloader,
)
from quantbayes.stochax.diffusion.trainer import train_model
from quantbayes.stochax.diffusion.edm import edm_batch_loss
from quantbayes.stochax.diffusion.generate import generate_with_sampler
from quantbayes.stochax.diffusion.inference import sample_edm, sample_edm_conditional
from quantbayes.stochax.diffusion.models.wrappers import (
    UnconditionalWrapper,
    DiTWrapper,
)

# spectral models
from quantbayes.stochax.diffusion.models.spectral_unet_2d import SpectralUNet2d
from quantbayes.stochax.diffusion.models.spectral_mixer_2d import SpectralMixer2d
from quantbayes.stochax.diffusion.models.spectral_dit import SpectralDiT

# NEW: RFFT-hybrid UNet
from quantbayes.stochax.diffusion.models.rfft_unet_2d import RFFTSpectralUNet2d


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
    core = factory_fn(key, data.shape[1:])  # (C,H,W)

    if name == "SpectralDiT":
        model = DiTWrapper(
            model=core,
            num_classes=10,
            time_mode="vp_t",
            null_label_index=None,
            cfg_rescale=0.7,
        )
    else:
        model = UnconditionalWrapper(model=core, time_mode="vp_t")

    # keep training short for smoke tests
    ema = train_edm(model, data, steps=10, seed=seed, batch=32)

    # Default sampler
    uncond = sample_edm(
        ema_model=ema,
        num_samples=8,
        sample_shape=data.shape[1:],
        key=jr.PRNGKey(seed + 1),
        steps=12,
        sigma_min=0.002,
        sigma_max=1.0,
        sigma_data=0.5,
        rho=7.0,
    )
    print(f"{name} uncond samples:", uncond.shape)

    if name == "SpectralDiT":
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
        print(f"{name} CFG samples:", cond.shape)

    # Unified sampler API demo
    ema_eval = eqx.tree_inference(ema, value=True)

    def denoise_fn(log_sigma, x):
        return ema_eval(log_sigma, x, key=None, train=False)

    xs = generate_with_sampler(
        denoise_fn,
        "unipc",
        data.shape[1:],
        key=jr.PRNGKey(seed + 3),
        num_samples=4,
        sampler_kwargs=dict(
            steps=12, sigma_min=0.002, sigma_max=1.0, sigma_data=0.5, rho=7.0
        ),
    )
    print(f"{name} unipc:", xs.shape)


def make_spectral_unet(key, img_shape):
    C, H, W = img_shape
    return SpectralUNet2d(
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


def make_spectral_mixer(key, img_shape):
    return SpectralMixer2d(
        img_size=img_shape,
        patch_size=4,
        hidden_size=64,
        num_blocks=2,
        token_groups=1,
        key=key,
    )


def make_spectral_dit(key, img_shape):
    C, H, W = img_shape
    return SpectralDiT(
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
        svd_rank=None,  # set to an int to low-rank the spectral dense layers
    )


# NEW: RFFT-hybrid UNet factory
def make_rfft_unet(key, img_shape):
    C, H, W = img_shape
    return RFFTSpectralUNet2d(
        data_shape=img_shape,
        dim_mults=(1, 2),
        hidden_size=32,
        num_res_blocks=2,
        rfft_levels=(0, 1),  # apply time-conditioned RFFT conv in first two levels
        key=key,
    )


def main():
    key = jr.PRNGKey(0)
    data = generate_synthetic_image_dataset(num_samples=256, shape=(3, 32, 32), key=key)

    print("=== Spectral UNet ===")
    run_one_model("SpectralUNet", make_spectral_unet, data, seed=10)

    print("\n=== Spectral Mixer ===")
    run_one_model("SpectralMixer", make_spectral_mixer, data, seed=20)

    print("\n=== Spectral DiT ===")
    run_one_model("SpectralDiT", make_spectral_dit, data, seed=30)

    print("\n=== RFFT UNet (hybrid) ===")
    run_one_model("RFFTUNet", make_rfft_unet, data, seed=40)


if __name__ == "__main__":
    main()
