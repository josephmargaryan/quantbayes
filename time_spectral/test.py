import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
import optax
from quantbayes.stochax.layers import JVPCirculantProcess

# We'll use your existing utility code for data loading and training:
from quantbayes.stochax import data_loader, train, regression_loss


@jax.custom_jvp
def spectral_circulant_matmul_ts(x: jnp.ndarray, fft_full: jnp.ndarray) -> jnp.ndarray:
    """
    Time-series version of the spectral circulant matmul.
    Always returns a 2D array with shape (batch, padded_dim), even if the input is a single example.

    Args:
      x: either (batch, d_in) or (d_in,). In case of single example, a batch dimension is added.
      fft_full: full Fourier mask, shape (padded_dim,).

    Returns:
      y: the result of IFFT(FFT(x_padded) * fft_full) with an explicit batch dimension.
    """
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
    # Always return y (shape: (batch, padded_dim))
    return y


@spectral_circulant_matmul_ts.defjvp
def spectral_circulant_matmul_ts_jvp(primals, tangents):
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
    # Always return the batched output:
    return primal_y, dY


# ---------------------------------------------------------------
# New TimeSeriesCirculant class that uses the time-series version of spectral matmul.
# ---------------------------------------------------------------
class TimeSeriesCirculant(eqx.Module):
    """
    A spectral circulant layer for time-series inputs.

    This layer accepts a sequence length and number of features, flattens the input,
    applies the spectral circulant transformation (with FFT-based operations), and then adds a bias.

    Attributes:
      seq_len: Length of the time-series.
      d_features: Number of features per time step.
      in_features: Computed as seq_len * d_features.
      padded_dim: Dimension after padding/truncation (default: in_features).
      alpha: Exponent for frequency-dependent scaling.
      K: Number of active frequencies (default: all frequencies).
      k_half: Half the padded dimension plus one (for spectral symmetry).
      w_real, w_imag: The half-spectrum parameters.
      bias: Bias term added after the transformation.
    """

    # Static fields
    seq_len: int = eqx.static_field()
    d_features: int = eqx.static_field()
    k_half: int = eqx.static_field()

    # Dynamic fields
    in_features: int
    padded_dim: int
    alpha: float
    K: int
    w_real: jnp.ndarray  # shape: (k_half,)
    w_imag: jnp.ndarray  # shape: (k_half,)
    bias: jnp.ndarray  # shape: (padded_dim,)

    def __init__(
        self,
        seq_len: int,
        d_features: int,
        padded_dim: int = None,
        alpha: float = 1.0,
        K: int = None,
        *,
        key,
        init_scale: float = 0.1,
        bias_init_scale: float = 0.1,
    ):
        self.seq_len = seq_len
        self.d_features = d_features
        self.in_features = seq_len * d_features
        self.padded_dim = padded_dim if padded_dim is not None else self.in_features
        self.alpha = alpha
        self.k_half = (self.padded_dim // 2) + 1
        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        key_r, key_i, key_b = jr.split(key, 3)
        self.w_real = jr.normal(key_r, (self.k_half,)) * init_scale
        self.w_imag = jr.normal(key_i, (self.k_half,)) * init_scale
        # Enforce DC and Nyquist (if applicable) to be real.
        self.w_imag = self.w_imag.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            self.w_imag = self.w_imag.at[-1].set(0.0)
        self.bias = jr.normal(key_b, (self.padded_dim,)) * bias_init_scale

    def get_fourier_coeffs(self) -> jnp.ndarray:
        freq_mask = jnp.arange(self.k_half) < self.K
        half_complex = (self.w_real * freq_mask) + 1j * (self.w_imag * freq_mask)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            nyquist = half_complex[-1].real[None]
            fft_full = jnp.concatenate(
                [half_complex[:-1], nyquist, jnp.conjugate(half_complex[1:-1])[::-1]]
            )
        else:
            fft_full = jnp.concatenate(
                [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
            )
        return fft_full

    def __call__(self, x: jnp.ndarray, key=None, state=None) -> jnp.ndarray:
        """
        x: Expected to be of shape (batch, in_features), where in_features = seq_len * d_features.
           (For time-series data, flatten each sample first.)
        Returns:
          The transformed output of shape (batch, padded_dim) with bias added.
        """
        fft_full = self.get_fourier_coeffs()
        out = spectral_circulant_matmul_ts(x, fft_full)
        return out + self.bias[None, :], state


class TSForecastModel(eqx.Module):
    spec_layer: TimeSeriesCirculant
    linear_out: eqx.nn.Linear
    seq_len: int = eqx.static_field()
    d_features: int = eqx.static_field()

    def __init__(self, seq_len: int, d_features: int, padded_dim: int, *, key):
        self.seq_len = seq_len
        self.d_features = d_features
        # TimeSeriesCirculant takes the flattened input dimension (seq_len*d_features)
        key_spec, key_lin = jr.split(key)
        self.spec_layer = TimeSeriesCirculant(
            seq_len, d_features, padded_dim=padded_dim, key=key_spec
        )
        # Map from padded_dim to 1 output
        self.linear_out = eqx.nn.Linear(
            in_features=padded_dim, out_features=1, key=key_lin
        )

    def __call__(self, x: jnp.ndarray, key=None, state=None) -> tuple[jnp.ndarray, any]:
        # x expected to be of shape (batch, seq_len*d_features)
        # Get the circulant output (batch, padded_dim)
        out, state = self.spec_layer(x, key, state)
        # Map to a single prediction per sample
        pred = self.linear_out(out)
        return pred, state


# ---------------------------------------------------------------
# Example time-series experiment using the new TimeSeriesCirculant.
# ---------------------------------------------------------------
def generate_sine_mixture_data(n_samples=2000, seq_len=20, d_features=1, rng_key=42):
    """
    Generates synthetic time-series data:
      X[i] in R^(seq_len, d_features), Y[i] = next value in the sequence,
    with multiple sines plus random noise.
    """
    key = jr.PRNGKey(rng_key)
    freqs = jnp.array([0.1, 0.15, 0.3])
    amps = jnp.array([1.0, 0.5, 0.8])
    X_data = []
    Y_data = []
    for i in range(n_samples):
        phase = jr.uniform(key, shape=()) * 2 * jnp.pi
        t = jnp.arange(seq_len + 1, dtype=jnp.float32)
        series = sum(
            amps[f] * jnp.sin(2 * jnp.pi * freqs[f] * (t + phase))
            for f in range(len(freqs))
        )
        series = series + 0.1 * jr.normal(key, shape=(seq_len + 1,))
        X_data.append(series[:seq_len].reshape(seq_len, 1))
        Y_data.append(
            series[seq_len].reshape(
                1,
            )
        )
        key, _ = jr.split(key)
    X_data = jnp.stack(X_data, axis=0)
    Y_data = jnp.stack(Y_data, axis=0)
    return X_data, Y_data


def run_time_series_experiment():
    # 1) Generate data
    X_data, Y_data = generate_sine_mixture_data(n_samples=2000, seq_len=20)
    train_size = 1600
    X_train, X_test = X_data[:train_size], X_data[train_size:]
    Y_train, Y_test = Y_data[:train_size], Y_data[train_size:]
    print(f"X_train has shape: {X_train.shape}")
    print(f"y_train has shape: {Y_train.shape}")

    # 2) Build model:
    # For each time-series sample, flatten it: (seq_len, 1) -> (seq_len,)
    # Our TimeSeriesCirculant expects input shape (batch, in_features) where in_features = seq_len * d_features.
    key = jr.PRNGKey(1234)
    padded_dim = 64
    model = TSForecastModel(seq_len=20, d_features=1, padded_dim=padded_dim, key=key)

    # 3) Optimizer init and training loop (using your existing utilities)
    import optax
    from quantbayes.stochax import train, regression_loss  # assume these are available

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    state = None
    batch_size = 64
    num_epochs = 1000
    patience = 10
    best_model, best_state, train_losses, val_losses = train(
        model=model,
        state=state,
        opt_state=opt_state,
        optimizer=optimizer,
        loss_fn=regression_loss,  # assume this is defined appropriately
        X_train=X_train.reshape(X_train.shape[0], -1),  # flatten each sample
        y_train=Y_train,
        X_val=X_test.reshape(X_test.shape[0], -1),
        y_val=Y_test,
        batch_size=batch_size,
        num_epochs=num_epochs,
        patience=patience,
        key=key,
    )

    # 4) Evaluate and visualize predictions
    import matplotlib.pyplot as plt
    from quantbayes.stochax import predict  # assume predict is defined

    inference_model = eqx.nn.inference_mode(best_model)
    y_preds = predict(
        inference_model, best_state, X_test.reshape(X_test.shape[0], -1), key
    )
    y_preds = jnp.array(y_preds)
    y_preds = jnp.squeeze(y_preds, axis=-1)

    plt.figure(figsize=(8, 4))
    plt.plot(Y_test[:200], label="Ground Truth", marker=".", alpha=0.6)
    plt.plot(y_preds[:200], label="Prediction", marker=".", alpha=0.6)
    plt.title("Time-Series Next-Step Forecast (first 200 test samples)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss over epochs")
    plt.tight_layout()
    plt.legend()
    plt.show()


run_time_series_experiment()
import equinox as eqx
import jax.numpy as jnp


class CirculantMultiHeadAttention(eqx.Module):
    num_heads: int
    d_model: int
    head_dim: int

    # The usual Q, K, V, O transformations
    Wq: eqx.Module  # e.g. JVPCirculantProcess or JVPBlockCirculantProcess
    Wk: eqx.Module
    Wv: eqx.Module
    Wo: eqx.Module

    def __init__(self, d_model, num_heads, key):
        """
        d_model: dimension of the input embeddings
        num_heads: number of attention heads
        """
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        k1, k2, k3, k4 = jax.random.split(key, 4)
        # For example, each transformation is a block-circulant or spectral-circulant layer
        # from d_model -> d_model
        self.Wq = JVPCirculantProcess(in_features=d_model, padded_dim=d_model, key=k1)
        self.Wk = JVPCirculantProcess(in_features=d_model, padded_dim=d_model, key=k2)
        self.Wv = JVPCirculantProcess(in_features=d_model, padded_dim=d_model, key=k3)
        self.Wo = JVPCirculantProcess(in_features=d_model, padded_dim=d_model, key=k4)

    def __call__(self, x, key=None, state=None):
        """
        x shape: (batch_size, seq_len, d_model)
        We produce Q, K, V each of shape (batch_size, seq_len, d_model).
        Then do scaled dot-product attention in multi-head fashion.
        """
        bs, seq_len, dm = x.shape
        # Flatten x for each transform. Then reshape back to 3D
        x2d = x.reshape(bs * seq_len, dm)
        Q2d, _ = self.Wq(x2d)
        K2d, _ = self.Wk(x2d)
        V2d, _ = self.Wv(x2d)

        Q = Q2d.reshape(bs, seq_len, dm)
        K = K2d.reshape(bs, seq_len, dm)
        V = V2d.reshape(bs, seq_len, dm)

        # Split into heads: Q => (bs, seq_len, num_heads, head_dim)
        Q = Q.reshape(bs, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(bs, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(bs, seq_len, self.num_heads, self.head_dim)

        # scaled dot product over the last dimension
        scale = self.head_dim**-0.5

        # attention logits: (bs, num_heads, seq_len, seq_len)
        attn_logits = jnp.einsum("bqhd,bkhd->bhqk", Q, K) * scale
        # softmax
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        # weighted sum
        out_heads = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, V)

        # merge heads
        out_merge = out_heads.reshape(bs, seq_len, dm)

        # final linear (Wo)
        out2d = out_merge.reshape(bs * seq_len, dm)
        out2d, _ = self.Wo(out2d)
        out = out2d.reshape(bs, seq_len, dm)
        return out, state
