# test.py
# -----------------------------------------------------------------------------
# End-to-end demonstrations for NumPyro spectral layers + custom guides.
#
# This file shows how to:
#   * Build small toy models with the FFT/circulant and SVD-based spectral layers
#   * Plug in low-rank (or mean-field) guides that match each layer’s sites
#   * Train with SVI and draw posterior/predictive samples
#
# NOTE:
#   - Update the imports under "TRY IMPORTS" to match your repo layout if needed.
#   - Default configs are tiny (quick smoke tests). Increase sizes/steps for real runs.
# -----------------------------------------------------------------------------

from __future__ import annotations
import argparse
import importlib
from dataclasses import dataclass
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoGuideList, AutoNormal
from numpyro import handlers
from numpyro.optim import Adam


# =============================== TRY IMPORTS =================================
# You likely have these in the same directory as this test.py.
# If your module names differ, update the "candidates" lists accordingly.


def _import_layers_module():
    candidates = [
        "spectral_numpyro_layers",
        "spectral_numpyro",
        "numpyro_spectral_layers",
    ]
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception:
            pass
    raise ImportError(
        "Could not import spectral NumPyro layers. "
        "Place them in the same dir and set module name in test.py."
    )


def _import_guides_module():
    candidates = [
        "custom_guides",
        "quantbayes.bnn.layers.custom_guides",
    ]
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception:
            pass
    raise ImportError(
        "Could not import custom guides. "
        "Place custom_guides.py in the same dir or adjust import path."
    )


_layers = _import_layers_module()
_guides = _import_guides_module()

# Pull symbols
SpectralCirculantLayer = _layers.SpectralCirculantLayer
AdaptiveSpectralCirculantLayer = _layers.AdaptiveSpectralCirculantLayer
SpectralCirculantLayer2d = _layers.SpectralCirculantLayer2d
AdaptiveSpectralCirculantLayer2d = _layers.AdaptiveSpectralCirculantLayer2d
SpectralDense = _layers.SpectralDense
AdaptiveSpectralDense = _layers.AdaptiveSpectralDense
SpectralConv2d = _layers.SpectralConv2d
AdaptiveSpectralConv2d = _layers.AdaptiveSpectralConv2d

LowRankFFTGuide = _guides.LowRankFFTGuide
LowRankAdaptiveFFTGuide = _guides.LowRankAdaptiveFFTGuide
LowRankFFTGuide2d = _guides.LowRankFFTGuide2d
LowRankAdaptiveFFTGuide2d = _guides.LowRankAdaptiveFFTGuide2d
LowRankSVDGuide = _guides.LowRankSVDGuide
LowRankAdaptiveSVDGuide = _guides.LowRankAdaptiveSVDGuide
MeanFieldFFTGuide = _guides.MeanFieldFFTGuide
MeanFieldFFTGuide2d = _guides.MeanFieldFFTGuide2d
MeanFieldSVDGuide = _guides.MeanFieldSVDGuide


# ============================= UTILS / DATA ==================================


def set_seed(seed: int):
    numpyro.set_host_device_count(1)
    numpyro.set_platform("cpu")
    jax.config.update("jax_enable_x64", False)
    return jax.random.PRNGKey(seed)


def make_regression_1d(
    n: int, d_in: int, noise: float, key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Synthetic 1D regression data (features + scalar target)."""
    k1, k2, k3 = jax.random.split(key, 3)
    X = jax.random.normal(k1, (n, d_in))
    w_true = jax.random.normal(k2, (d_in,))
    y = X @ w_true + 0.1 * jax.random.normal(k3, (n,))
    if noise > 0:
        y = y + noise * jax.random.normal(k3, (n,))
    return X, y


def make_regression_2d(
    n: int, c_in: int, h: int, w: int, noise: float, key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Synthetic 2D regression data (N, C_in, H, W) + scalar target."""
    k1, k2 = jax.random.split(key)
    X = jax.random.normal(k1, (n, c_in, h, w))
    # Linear readout over mean of pixels (per-channel weights):
    w_true = jax.random.normal(k2, (c_in,))
    pooled = X.mean(axis=(-2, -1))  # (n, c_in)
    y = (pooled @ w_true) + 0.1 * jax.random.normal(k2, (n,))
    if noise > 0:
        y = y + noise * jax.random.normal(k2, (n,))
    return X, y


def orthonormal(n: int) -> jnp.ndarray:
    """Return an n×n orthonormal matrix via QR."""
    q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(n), (n, n)))
    return q


def make_svd_factors(
    in_dim: int, out_dim: int, rank: int, key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (U, V) with shapes (out, r), (in, r) as fixed orthonormal factors."""
    k1, k2 = jax.random.split(key)
    U_full, _ = jnp.linalg.qr(jax.random.normal(k1, (out_dim, out_dim)))
    V_full, _ = jnp.linalg.qr(jax.random.normal(k2, (in_dim, in_dim)))
    U = jax.lax.stop_gradient(U_full[:, :rank])
    V = jax.lax.stop_gradient(V_full[:, :rank])
    return U, V


# ============================== TRAINING CORE ================================


@dataclass
class TrainConfig:
    steps: int = 200
    lr: float = 5e-3
    rank: int = 8
    reparam: bool = True  # if False and tiny dims, you can use dense MVN


def run_svi(model_fn, guide, X, y, cfg: TrainConfig, seed: int = 0):
    rng = set_seed(seed)
    svi = SVI(model_fn, guide, Adam(cfg.lr), loss=Trace_ELBO())
    state = svi.init(rng, X, y)
    for i in range(cfg.steps):
        state, loss = svi.update(state, X, y)
        if (i + 1) % max(1, cfg.steps // 5) == 0:
            print(f"[{i+1:04d}/{cfg.steps}] loss = {float(loss):.3f}")
    params = svi.get_params(state)
    return params, rng


def posterior_predictive(model_fn, guide, params, X, num_samples: int, rng: jax.Array):
    pred = Predictive(model_fn, guide=guide, params=params, num_samples=num_samples)
    samples = pred(rng, X, y=None)
    return samples


# ============================== EXPERIMENTS ==================================


def exp_1d_stationary(seed=0):
    print("\n==> 1D circulant (stationary) with LowRankFFTGuide")
    key = set_seed(seed)
    N, Din = 64, 64
    P, K = 128, 64  # padded_dim, active modes
    X, y = make_regression_1d(N, Din, noise=0.05, key=key)

    # Model
    def model(X, y=None):
        layer = SpectralCirculantLayer(in_features=Din, padded_dim=P, K=K, name="mix1d")
        Feat = layer(X)  # (N, P)
        w = numpyro.sample("readout_w", dist.Normal(0, 1).expand([P]).to_event(1))
        b = numpyro.sample("readout_b", dist.Normal(0, 1))
        f = jnp.einsum("np,p->n", Feat, w) + b
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("y", dist.Normal(f, sigma), obs=y)

    # Guide
    mode = "reparam"
    guide = AutoGuideList(model)
    guide.append(
        LowRankFFTGuide(model, prefix="mix1d", padded_dim=P, K=K, rank=8, mode=mode)
    )
    guide.append(
        AutoNormal(
            handlers.block(model, hide=["mix1d_real", "mix1d_imag"]),
            init_loc_fn=numpyro.infer.init_to_feasible,
        )
    )

    params, rng = run_svi(model, guide, X, y, TrainConfig(), seed=seed)
    post = posterior_predictive(model, guide, params, X, num_samples=32, rng=rng)
    print("Posterior predictive y shape:", post["y"].shape)


def exp_1d_adaptive(seed=0):
    print("\n==> 1D circulant (adaptive) with LowRankAdaptiveFFTGuide")
    key = set_seed(seed)
    N, Din = 64, 64
    P = 128
    X, y = make_regression_1d(N, Din, noise=0.05, key=key)

    def model(X, y=None):
        layer = AdaptiveSpectralCirculantLayer(
            in_features=Din, padded_dim=P, name="mix1d"
        )
        Feat = layer(X)  # (N, P)
        w = numpyro.sample("readout_w", dist.Normal(0, 1).expand([P]).to_event(1))
        b = numpyro.sample("readout_b", dist.Normal(0, 1))
        f = jnp.einsum("np,p->n", Feat, w) + b
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("y", dist.Normal(f, sigma), obs=y)

    mode = "reparam"
    guide = AutoGuideList(model)
    guide.append(
        LowRankAdaptiveFFTGuide(model, prefix="mix1d", padded_dim=P, rank=8, mode=mode)
    )
    guide.append(
        AutoNormal(
            handlers.block(
                model, hide=["mix1d_real", "mix1d_imag", "mix1d_delta_alpha"]
            ),
            init_loc_fn=numpyro.infer.init_to_feasible,
        )
    )

    params, rng = run_svi(model, guide, X, y, TrainConfig(), seed=seed)
    post = posterior_predictive(model, guide, params, X, num_samples=32, rng=rng)
    print("Posterior predictive y shape:", post["y"].shape)


def exp_2d_stationary(seed=0):
    print("\n==> 2D circulant (stationary) with LowRankFFTGuide2d")
    key = set_seed(seed)
    N, Cin, H, W = 32, 3, 16, 16
    Cout = 8
    X, y = make_regression_2d(N, Cin, H, W, noise=0.05, key=key)

    def model(X, y=None):
        layer = SpectralCirculantLayer2d(
            C_in=Cin, C_out=Cout, H_in=H, W_in=W, H_pad=H, W_pad=W, name="mix2d"
        )
        Y = layer(X)  # (N, C_out, H, W)
        pooled = Y.mean(axis=(-2, -1))  # (N, C_out)
        w = numpyro.sample("readout_w", dist.Normal(0, 1).expand([Cout]).to_event(1))
        b = numpyro.sample("readout_b", dist.Normal(0, 1))
        f = jnp.einsum("nc,c->n", pooled, w) + b
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("y", dist.Normal(f, sigma), obs=y)

    mode = "reparam"
    guide = AutoGuideList(model)
    guide.append(
        LowRankFFTGuide2d(
            model,
            prefix="mix2d",
            C_out=Cout,
            C_in=Cin,
            H_pad=H,
            W_pad=W,
            rank=8,
            mode=mode,
        )
    )
    guide.append(
        AutoNormal(
            handlers.block(model, hide=["mix2d_real", "mix2d_imag"]),
            init_loc_fn=numpyro.infer.init_to_feasible,
        )
    )

    params, rng = run_svi(model, guide, X, y, TrainConfig(), seed=seed)
    post = posterior_predictive(model, guide, params, X, num_samples=32, rng=rng)
    print("Posterior predictive y shape:", post["y"].shape)


def exp_2d_adaptive(seed=0):
    print("\n==> 2D circulant (adaptive) with LowRankAdaptiveFFTGuide2d")
    key = set_seed(seed)
    N, Cin, H, W = 32, 3, 16, 16
    Cout = 8
    X, y = make_regression_2d(N, Cin, H, W, noise=0.05, key=key)

    def model(X, y=None):
        layer = AdaptiveSpectralCirculantLayer2d(
            C_in=Cin, C_out=Cout, H_in=H, W_in=W, H_pad=H, W_pad=W, name="mix2d"
        )
        Y = layer(X)  # (N, C_out, H, W)
        pooled = Y.mean(axis=(-2, -1))  # (N, C_out)
        w = numpyro.sample("readout_w", dist.Normal(0, 1).expand([Cout]).to_event(1))
        b = numpyro.sample("readout_b", dist.Normal(0, 1))
        f = jnp.einsum("nc,c->n", pooled, w) + b
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("y", dist.Normal(f, sigma), obs=y)

    mode = "reparam"
    guide = AutoGuideList(model)
    guide.append(
        LowRankAdaptiveFFTGuide2d(
            model,
            prefix="mix2d",
            C_out=Cout,
            C_in=Cin,
            H_pad=H,
            W_pad=W,
            alpha_coarse_shape=(8, 8),
            rank=8,
            mode=mode,
        )
    )
    guide.append(
        AutoNormal(
            handlers.block(
                model, hide=["mix2d_real", "mix2d_imag", "mix2d_delta_alpha"]
            ),
            init_loc_fn=numpyro.infer.init_to_feasible,
        )
    )

    params, rng = run_svi(model, guide, X, y, TrainConfig(), seed=seed)
    post = posterior_predictive(model, guide, params, X, num_samples=32, rng=rng)
    print("Posterior predictive y shape:", post["y"].shape)


def exp_svd_dense_stationary(seed=0):
    print("\n==> SVD Dense (stationary) with LowRankSVDGuide")
    key = set_seed(seed)
    N, Din, Dout, r = 64, 32, 16, 12
    X, y = make_regression_1d(N, Din, noise=0.05, key=key)
    U, V = make_svd_factors(Din, Dout, r, key)

    def model(X, y=None):
        layer = SpectralDense(
            U=V, V=U, alpha=1.0, name="fc_spec"
        )  # NOTE: adjust if your ctor differs
        # If your SpectralDense expects (U, V) such that U: (out,r), V: (in,r),
        # pass them in the correct order. The sample sites should be *_s, *_alpha_z, *_b.
        Y = layer(X)  # (N, Dout)
        w = numpyro.sample("readout_w", dist.Normal(0, 1).expand([Dout]).to_event(1))
        b = numpyro.sample("readout_b", dist.Normal(0, 1))
        f = jnp.einsum("nd,d->n", Y, w) + b
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("y", dist.Normal(f, sigma), obs=y)

    mode = "reparam"
    guide = AutoGuideList(model)
    guide.append(
        LowRankSVDGuide(
            model,
            prefix="fc_spec",
            U=jnp.ones((Dout, r)),
            include_alpha=True,
            rank=8,
            mode=mode,
        )
    )
    guide.append(
        AutoNormal(
            handlers.block(model, hide=["fc_spec_s", "fc_spec_alpha_z"]),
            init_loc_fn=numpyro.infer.init_to_feasible,
        )
    )

    params, rng = run_svi(model, guide, X, y, TrainConfig(), seed=seed)
    post = posterior_predictive(model, guide, params, X, num_samples=32, rng=rng)
    print("Posterior predictive y shape:", post["y"].shape)


def exp_svd_conv_stationary(seed=0):
    print("\n==> SVD Conv2d (stationary) with LowRankSVDGuide")
    key = set_seed(seed)
    N, Cin, H, W = 16, 3, 16, 16
    Cout, Hk, Wk = 6, 3, 3
    strides = (1, 1)
    X, y = make_regression_2d(N, Cin, H, W, noise=0.05, key=key)

    in_dim = Cin * Hk * Wk
    out_dim = Cout
    r = min(in_dim, out_dim) // 2 or 4
    U, V = make_svd_factors(in_dim, out_dim, r, key)  # U: (out, r), V: (in, r)

    def model(X, y=None):
        layer = SpectralConv2d(
            U=U,
            V=V,
            C_in=Cin,
            C_out=Cout,
            H_k=Hk,
            W_k=Wk,
            strides=strides,
            padding="SAME",
            name="spec_conv2d",
        )
        Y = layer(X)  # (N, Cout, H, W)
        pooled = Y.mean(axis=(-2, -1))  # (N, Cout)
        w = numpyro.sample("readout_w", dist.Normal(0, 1).expand([Cout]).to_event(1))
        b = numpyro.sample("readout_b", dist.Normal(0, 1))
        f = jnp.einsum("nc,c->n", pooled, w) + b
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("y", dist.Normal(f, sigma), obs=y)

    mode = "reparam"
    guide = AutoGuideList(model)
    guide.append(
        LowRankSVDGuide(
            model,
            prefix="spec_conv2d",
            U=jnp.ones((Cout, r)),
            include_alpha=True,
            rank=8,
            mode=mode,
        )
    )
    guide.append(
        AutoNormal(
            handlers.block(model, hide=["spec_conv2d_s", "spec_conv2d_alpha_z"]),
            init_loc_fn=numpyro.infer.init_to_feasible,
        )
    )

    params, rng = run_svi(model, guide, X, y, TrainConfig(steps=150), seed=seed)
    post = posterior_predictive(model, guide, params, X, num_samples=16, rng=rng)
    print("Posterior predictive y shape:", post["y"].shape)


def exp_all(seed=0):
    exp_1d_stationary(seed)
    exp_1d_adaptive(seed)
    exp_2d_stationary(seed)
    exp_2d_adaptive(seed)
    exp_svd_conv_stationary(seed)
    # The SVD dense example includes a note about ctor ordering; enable if your API matches
    # exp_svd_dense_stationary(seed)


# ================================== CLI =====================================


def main():
    p = argparse.ArgumentParser(description="Spectral layers + guides demo")
    p.add_argument(
        "--exp",
        type=str,
        default="1d",
        choices=["1d", "1d_adapt", "2d", "2d_adapt", "svd_conv", "svd_dense", "all"],
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.exp == "1d":
        exp_1d_stationary(args.seed)
    elif args.exp == "1d_adapt":
        exp_1d_adaptive(args.seed)
    elif args.exp == "2d":
        exp_2d_stationary(args.seed)
    elif args.exp == "2d_adapt":
        exp_2d_adaptive(args.seed)
    elif args.exp == "svd_conv":
        exp_svd_conv_stationary(args.seed)
    elif args.exp == "svd_dense":
        exp_svd_dense_stationary(args.seed)
    elif args.exp == "all":
        exp_all(args.seed)
    else:
        raise ValueError(f"Unknown exp={args.exp}")


if __name__ == "__main__":
    main()
