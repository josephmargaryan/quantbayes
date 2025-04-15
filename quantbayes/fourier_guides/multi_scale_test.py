import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive, SVI, Trace_ELBO, init_to_sample
import numpyro.optim as optim
from numpyro.infer.autoguide import AutoNormal
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from quantbayes.fourier_guides.guides import MultiScaleSpectralRealGuide, MultiScaleSpectralImagGuide, LowRankMultiScaleSpectralGuide
from quantbayes.fake_data import generate_regression_data

# Import the multi-scale layer defined previously.
# Make sure that MultiScaleSpectralCirculantLayer is either in your PYTHONPATH or adjust the import accordingly.
# from your_module import MultiScaleSpectralCirculantLayer
# For this example, assume the class is defined in this script (see below for its definition).

# -------------------------------
# MultiScaleSpectralCirculantLayer (from previous implementation)
# -------------------------------
import jax.random

@jax.custom_jvp
def spectral_circulant_matmul(x: jnp.ndarray, fft_full: jnp.ndarray) -> jnp.ndarray:
    padded_dim = fft_full.shape[0]
    single_example = x.ndim == 1
    if single_example:
        x = x[None, :]
    d_in = x.shape[-1]
    if d_in < padded_dim:
        pad_len = padded_dim - d_in
        x_pad = jnp.pad(x, ((0, 0), (0, pad_len)))
    elif d_in > padded_dim:
        x_pad = x[..., :padded_dim]
    else:
        x_pad = x
    X_fft = jnp.fft.fft(x_pad, axis=-1)
    y_fft = X_fft * fft_full[None, :]
    y = jnp.fft.ifft(y_fft, axis=-1).real
    if single_example:
        return y[0]
    return y

@spectral_circulant_matmul.defjvp
def spectral_circulant_matmul_jvp(primals, tangents):
    x, fft_full = primals
    dx, dfft = tangents
    padded_dim = fft_full.shape[0]

    single_example = x.ndim == 1
    if single_example:
        x = x[None, :]
        if dx is not None:
            dx = dx[None, :]
    d_in = x.shape[-1]
    if d_in < padded_dim:
        pad_len = padded_dim - d_in
        x_pad = jnp.pad(x, ((0, 0), (0, pad_len)))
        dx_pad = jnp.pad(dx, ((0, 0), (0, pad_len))) if dx is not None else None
    elif d_in > padded_dim:
        x_pad = x[..., :padded_dim]
        dx_pad = dx[..., :padded_dim] if dx is not None else None
    else:
        x_pad = x
        dx_pad = dx

    X_fft = jnp.fft.fft(x_pad, axis=-1)
    primal_y_fft = X_fft * fft_full[None, :]
    primal_y = jnp.fft.ifft(primal_y_fft, axis=-1).real

    if dx_pad is None:
        dX_fft = 0.0
    else:
        dX_fft = jnp.fft.fft(dx_pad, axis=-1)
    if dfft is None:
        term2 = 0.0
    else:
        term2 = X_fft * dfft[None, :]
    dY_fft = dX_fft * fft_full[None, :] + term2
    dY = jnp.fft.ifft(dY_fft, axis=-1).real

    if single_example:
        return primal_y[0], dY[0]
    return primal_y, dY

class MultiScaleSpectralCirculantLayer:
    def __init__(
        self,
        in_features: int,
        padded_dim: int = None,
        coarse_K: int = None,
        fine_K: int = None,
        alpha_coarse: float = None,
        alpha_fine: float = None,
        alpha_coarse_prior=dist.HalfNormal(1.0),
        alpha_fine_prior=dist.HalfNormal(1.0),
        mixture_prior=dist.Beta(1.0, 1.0),
        name: str = "multi_scale_spectral",
        prior_fn=None,
    ):
        self.in_features = in_features
        self.padded_dim = padded_dim if padded_dim is not None else in_features
        # For a real-valued signal, the half-spectrum length is:
        self.k_half = (self.padded_dim // 2) + 1

        # Set default values for number of active frequencies if not provided.
        if coarse_K is None:
            coarse_K = self.k_half // 2
        if fine_K is None:
            fine_K = self.k_half - coarse_K
        self.coarse_K = coarse_K
        self.fine_K = fine_K

        self.alpha_coarse = alpha_coarse
        self.alpha_fine = alpha_fine
        self.alpha_coarse_prior = alpha_coarse_prior
        self.alpha_fine_prior = alpha_fine_prior
        self.mixture_prior = mixture_prior
        self.name = name

        self.prior_fn = prior_fn if prior_fn is not None else (lambda scale: dist.Normal(0.0, scale))
        self._last_fft_full = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.alpha_coarse is None:
            alpha_c = numpyro.sample(f"{self.name}_alpha_coarse", self.alpha_coarse_prior)
        else:
            alpha_c = self.alpha_coarse
        if self.alpha_fine is None:
            alpha_f = numpyro.sample(f"{self.name}_alpha_fine", self.alpha_fine_prior)
        else:
            alpha_f = self.alpha_fine
        mix_weight = numpyro.sample(f"{self.name}_mix_weight", self.mixture_prior)

        # Coarse scale: indices [0, coarse_K)
        coarse_freq_idx = jnp.arange(self.coarse_K)
        prior_std_coarse = 1.0 / jnp.sqrt(1.0 + coarse_freq_idx**alpha_c)
        coarse_real = numpyro.sample(
            f"{self.name}_coarse_real", 
            self.prior_fn(prior_std_coarse).expand([self.coarse_K]).to_event(1)
        )
        coarse_imag = numpyro.sample(
            f"{self.name}_coarse_imag", 
            self.prior_fn(prior_std_coarse).expand([self.coarse_K]).to_event(1)
        )
        coarse_imag = coarse_imag.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.coarse_K > 1):
            coarse_imag = coarse_imag.at[-1].set(0.0)
        coarse_half_complex = coarse_real + 1j * coarse_imag
        coarse_full_half = jnp.concatenate(
            [coarse_half_complex, jnp.zeros((self.k_half - self.coarse_K,), dtype=coarse_half_complex.dtype)]
        )

        # Fine scale: next group of frequencies.
        fine_freq_idx = jnp.arange(self.fine_K)
        prior_std_fine = 1.0 / jnp.sqrt(1.0 + (fine_freq_idx + self.coarse_K)**alpha_f)
        fine_real = numpyro.sample(
            f"{self.name}_fine_real", 
            self.prior_fn(prior_std_fine).expand([self.fine_K]).to_event(1)
        )
        fine_imag = numpyro.sample(
            f"{self.name}_fine_imag", 
            self.prior_fn(prior_std_fine).expand([self.fine_K]).to_event(1)
        )
        fine_half_complex = fine_real + 1j * fine_imag
        fine_full_half = jnp.concatenate(
            [fine_half_complex, jnp.zeros((self.k_half - self.fine_K,), dtype=fine_half_complex.dtype)]
        )

        # Mix the two scales with a convex combination.
        combined_half = mix_weight * coarse_full_half + (1.0 - mix_weight) * fine_full_half

        # Enforce Hermitian symmetry.
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            nyquist = combined_half[-1].real[None]
            fft_full = jnp.concatenate(
                [combined_half[:-1], nyquist, jnp.conjugate(combined_half[1:-1])[::-1]]
            )
        else:
            fft_full = jnp.concatenate(
                [combined_half, jnp.conjugate(combined_half[1:])[::-1]]
            )
        self._last_fft_full = jax.lax.stop_gradient(fft_full)
        bias = numpyro.sample(
            f"{self.name}_bias", 
            dist.Normal(0.0, 1.0).expand([self.padded_dim]).to_event(1)
        )
        out = spectral_circulant_matmul(x, fft_full) + bias
        return out

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self._last_fft_full is None:
            raise ValueError("No Fourier coefficients available; call the layer first.")
        return self._last_fft_full

# -------------------------------
# Regression Model using the Multi-Scale Spectral Layer
# -------------------------------
def multi_scale_regression_model(X, y=None):
    """
    A regression model that uses the MultiScaleSpectralCirculantLayer.
    """
    N, D = X.shape
    # Instantiate the multi-scale spectral layer.
    spectral_layer = MultiScaleSpectralCirculantLayer(
        in_features=D,
        padded_dim=D,
        coarse_K=D//4,     # example setting; adjust as needed
        fine_K=D//4,       # example setting; adjust as needed
        name="multi_scale_spectral"
    )
    # Transform inputs using the spectral layer.
    transformed = spectral_layer(X)
    # Apply a nonlinearity.
    transformed_act = jax.nn.tanh(transformed)
    # Bayesian linear regression on transformed features.
    W = numpyro.sample("W", dist.Normal(0, 1).expand([D, 1]).to_event(2))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([1]).to_event(1))
    preds = jnp.squeeze(jnp.dot(transformed_act, W) + b)
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("obs", dist.Normal(preds, sigma), obs=y)

# ================================
# SVI Inference with Different Guide Configurations
# ================================
def run_svi_inference(model, X, y, num_steps=5000, step_size=0.01,
                      use_custom_guide=True, use_custom_low_rank=False, use_only_autonormal=False,
                      rng_key=jax.random.PRNGKey(0)):
    optimizer = optim.Adam(step_size=step_size)
    
    # Build the guide.
    if use_only_autonormal:
        # Use a generic AutoNormal guide over all latent sites.
        guide = AutoNormal(model, init_loc_fn=init_to_sample)
    else:
        # For the multi-scale spectral layer sites, we build custom guides.
        # Adjust the dimensions; these should match your MultiScale layer settings.
        coarse_K = X.shape[1] // 4   # same as used in the model definition
        fine_K = X.shape[1] // 4
        guide_list = []
        if use_custom_low_rank:
            # For example, use a low-rank guide for the coarse real part.
            guide_list.append(LowRankMultiScaleSpectralGuide(model, K=coarse_K, rank=2, site_name="multi_scale_spectral_coarse_real"))
        else:
            guide_list.append(MultiScaleSpectralRealGuide(model, coarse_K=coarse_K, fine_K=fine_K, name="multi_scale_spectral"))
            guide_list.append(MultiScaleSpectralImagGuide(model, coarse_K=coarse_K, fine_K=fine_K, name="multi_scale_spectral"))
        
        # For all other sites, delegate to AutoNormal.
        guide_list.append(AutoNormal(numpyro.handlers.block(model, hide=[
            "multi_scale_spectral_coarse_real", "multi_scale_spectral_fine_real",
            "multi_scale_spectral_coarse_imag", "multi_scale_spectral_fine_imag"
        ])))
        
        # Combine into an AutoGuideList.
        from numpyro.infer.autoguide import AutoGuideList
        guide = AutoGuideList(model)
        for g in guide_list:
            guide.append(g)

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(rng_key, num_steps=num_steps, X=X, y=y)
    return svi_result, guide

# -------------------------------
# Main test: data loading, model inference, and MAE reporting.
# -------------------------------
if __name__ == '__main__':
    # Data Loading
    df = generate_regression_data(n_continuous=16)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = jnp.array(X_train.values if isinstance(X_train, pd.DataFrame) else X_train)
    X_test = jnp.array(X_test.values if isinstance(X_test, pd.DataFrame) else X_test)
    y_train = jnp.array(y_train.values if isinstance(y_train, pd.Series) else y_train)
    y_test = jnp.array(y_test.values if isinstance(y_test, pd.Series) else y_test)

    # Configure guide selection.
    use_custom_guide = True
    use_custom_low_rank = False
    use_only_autonormal = False

    # Run SVI.
    rng_key = jax.random.PRNGKey(0)
    svi_result, guide = run_svi_inference(multi_scale_regression_model, X_train, y_train,
                                            num_steps=5000, step_size=0.01,
                                            use_custom_guide=use_custom_guide,
                                            use_custom_low_rank=use_custom_low_rank,
                                            use_only_autonormal=use_only_autonormal,
                                            rng_key=rng_key)
    params = svi_result.params

    # Evaluate posterior predictive performance.
    from numpyro.infer import Predictive
    predictive = Predictive(multi_scale_regression_model, guide=guide, params=params, num_samples=500)
    rng_key, rng_key_pp = jax.random.split(rng_key)
    preds = predictive(rng_key_pp, X_test)["obs"]
    pred_mean = jnp.mean(preds, axis=0)
    mae = jnp.mean(jnp.abs(pred_mean - y_test))
    print("Mean Absolute Error on Test Set:", np.array(mae))
