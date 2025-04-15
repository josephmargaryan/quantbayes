import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from quantbayes.fake_data import generate_regression_data


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

class MultiAlphaSpectralCirculantLayer:
    """
    A circulant layer that uses multiple alpha values, each controlling the spectral
    decay in one band of frequencies. The PSD is constructed piecewise.

    Example usage in a model:
        layer = MultiAlphaSpectralCirculantLayer(
            in_features=D,
            padded_dim=32,
            num_bands=3,
            alpha_prior=dist.Exponential(1.0),
            name="multi_alpha_layer"
        )
        X_transformed = layer(X)
        # then pass X_transformed through further layers / linear heads, etc.
    """
    def __init__(
        self,
        in_features: int,
        padded_dim: int = None,
        num_bands: int = 3,
        alpha_prior=dist.Exponential(1.0),
        K: int = None,
        name: str = "spectral_circ_jvp_multi_alpha",
        prior_fn=None,
    ):
        """
        :param in_features: Dimension of the input features.
        :param padded_dim: If provided, the layer will pad (or truncate) inputs to this dimension 
                           for the FFT. Defaults to in_features if None.
        :param num_bands: Number of frequency bands, each having its own alpha.
        :param alpha_prior: Distribution from which each alpha_b is sampled (e.g. Exponential(1.0)).
        :param K: Number of active frequencies to keep in the half-spectrum; if None, we keep all.
        :param name: Base name for the sample sites in NumPyro.
        :param prior_fn: A function mapping a scale to a distribution. By default, Normal(0, scale).
        """
        self.in_features = in_features
        self.padded_dim = padded_dim if padded_dim is not None else in_features
        self.num_bands = num_bands
        self.alpha_prior = alpha_prior
        self.name = name

        # Length of the half spectrum (including DC and possibly Nyquist)
        self.k_half = (self.padded_dim // 2) + 1

        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        self.prior_fn = prior_fn if prior_fn is not None else (lambda scale: dist.Normal(0.0, scale))
        self._last_fft_full = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the spectral circulant transformation to the input x.
        We sample multiple alpha values (one per band), build a piecewise PSD,
        sample real/imag spectral coefficients, impose Hermitian symmetry,
        then multiply via FFT and add bias.
        """
        # --- (1) Sample alpha for each band ---
        alpha_list = []
        for b in range(self.num_bands):
            alpha_b = numpyro.sample(f"{self.name}_alpha_band_{b}", self.alpha_prior)
            alpha_list.append(alpha_b)
        alpha_list = jnp.stack(alpha_list)  # shape = [num_bands]

        # --- (2) Construct frequencies and band assignment ---
        freq_idx = jnp.arange(self.k_half)   # 0,1,2,..., k_half-1
        band_size = self.k_half // self.num_bands
        band_assign = (freq_idx // band_size).clip(0, self.num_bands - 1)

        # --- (3) Build piecewise PSD in a vectorized manner ---
        # For each frequency index, obtain the corresponding alpha from alpha_list,
        # then compute: PSD(f) = 1 / (1 + f^(alpha))
        local_alphas = alpha_list[band_assign]  # shape (k_half,)
        psd = 1.0 / (1.0 + jnp.power(freq_idx, local_alphas))

        # --- (4) Sample spectral coefficients for the first K active frequencies ---
        active_idx = jnp.arange(self.K)
        scales = jnp.sqrt(psd[active_idx])
        real_active = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(scales).expand([self.K]).to_event(1),
        )
        imag_active = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(scales).expand([self.K]).to_event(1),
        )

        # --- (5) Construct full half-spectrum ---
        full_real = jnp.zeros((self.k_half,))
        full_imag = jnp.zeros((self.k_half,))
        full_real = full_real.at[active_idx].set(real_active)
        full_imag = full_imag.at[active_idx].set(imag_active)
        full_imag = full_imag.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            full_imag = full_imag.at[-1].set(0.0)
        half_complex = full_real + 1j * full_imag

        # --- (6) Impose Hermitian symmetry to build full FFT coefficients ---
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            nyquist = half_complex[-1].real[None]
            fft_full = jnp.concatenate(
                [half_complex[:-1], nyquist, jnp.conjugate(half_complex[1:-1])[::-1]]
            )
        else:
            fft_full = jnp.concatenate(
                [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
            )
        self._last_fft_full = jax.lax.stop_gradient(fft_full)

        # --- (7) Sample bias and apply circulant multiplication ---
        bias = numpyro.sample(
            f"{self.name}_bias_spectral",
            dist.Normal(0.0, 1.0).expand([self.padded_dim]).to_event(1),
        )
        out = spectral_circulant_matmul(x, fft_full) + bias
        return out

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self._last_fft_full is None:
            raise ValueError("No Fourier coefficients available; call the layer first.")
        return self._last_fft_full


# ---------------------------------------------------------------------
# Regression Model using MultiAlphaSpectralCirculantLayer
# ---------------------------------------------------------------------
def multi_alpha_regression_model(X, y=None):
    """
    A regression model that uses the MultiAlphaSpectralCirculantLayer.
    """
    N, D = X.shape
    # Create an instance of the spectral layer.
    layer = MultiAlphaSpectralCirculantLayer(
        in_features=D,
        padded_dim=D,       # Using input dimension as FFT dimension
        num_bands=3,
        alpha_prior=dist.Exponential(1.0),
        name="multi_alpha_layer"
    )
    X_transformed = layer(X)
    # Apply nonlinearity.
    X_act = jax.nn.tanh(X_transformed)
    # Bayesian linear layer on transformed features.
    W = numpyro.sample("W", dist.Normal(0, 1).expand([D, 1]).to_event(2))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([1]).to_event(1))
    preds = jnp.squeeze(jnp.dot(X_act, W) + b)
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("obs", dist.Normal(preds, sigma), obs=y)

# ---------------------------------------------------------------------
# Test Script using generate_regression_data(n_continuous=16)
# ---------------------------------------------------------------------
if __name__ == '__main__':
    # Load synthetic regression data.
    df = generate_regression_data(n_continuous=16)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Convert to JAX arrays.
    X_train = jnp.array(X_train.values if isinstance(X_train, pd.DataFrame) else X_train)
    X_test = jnp.array(X_test.values if isinstance(X_test, pd.DataFrame) else X_test)
    y_train = jnp.array(y_train.values if isinstance(y_train, pd.Series) else y_train)
    y_test = jnp.array(y_test.values if isinstance(y_test, pd.Series) else y_test)

    # Use NUTS for inference.
    nuts_kernel = NUTS(multi_alpha_regression_model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000, progress_bar=True)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key, X_train, y_train)
    mcmc.print_summary()

    # Posterior predictive sampling on test set.
    predictive = Predictive(multi_alpha_regression_model, mcmc.get_samples(), num_samples=500)
    rng_key, rng_key_pp = jax.random.split(rng_key)
    preds = predictive(rng_key_pp, X_test)["obs"]
    # Compute mean predictions.
    pred_mean = jnp.mean(preds, axis=0)
    mae = jnp.mean(jnp.abs(pred_mean - y_test))
    print("Mean Absolute Error on Test Set:", np.array(mae))
