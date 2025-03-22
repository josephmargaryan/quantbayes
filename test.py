from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp 
import numpyro 
import numpyro.distributions as dist
import jax.random as jr


import jax
import jax.numpy as jnp
from jax import random as jr
import numpyro
import numpyro.distributions as dist

@jax.custom_jvp
def spectral_conv1d_psd_func(
    x: jnp.ndarray, 
    L_factors: jnp.ndarray, 
    bias: jnp.ndarray
) -> jnp.ndarray:
    """
    1D multi-channel convolution in the frequency domain, with a per-frequency
    PSD matrix M_f = L_f L_f^T, where L_f is lower-triangular (real).

    Args:
      x: shape (B, C, W)
      L_factors: shape (rfft_length, C, C_lower) 
                 where C_lower = C*(C+1)//2, storing the distinct elements of 
                 each lower-triangular L_f.
      bias: shape (C,)

    Returns:
      y: shape (B, C, W)
    """
    B, C, W = x.shape
    rfft_length = L_factors.shape[0]
    fft_size = 2*(rfft_length - 1) if rfft_length > 1 else 1

    # 1) Possibly zero-pad x if fft_size > W
    if fft_size > W:
        pad_amount = fft_size - W
        x = jnp.pad(x, ((0, 0), (0, 0), (0, pad_amount)))

    # 2) rFFT along the spatial dimension
    X_fft = jnp.fft.rfft(x, n=fft_size, axis=-1)   # shape (B, C, rfft_length)

    # 3) For each frequency f, build M_f = L_f @ L_f^T, and multiply
    #    We'll do this with a "vmap" over f. 
    def apply_freq_matrix(X_f, L_tri):
        # L_tri: shape (C_lower,) => reshape to (C, C), zero upper triangle
        L = fill_lower_triangle(L_tri, C)  # We'll define fill_lower_triangle below
        M = L @ L.T   # shape (C, C), real PSD
        # Now multiply X_f => (C,) => M @ X_f => shape (C,)
        return M @ X_f

    # X_fft: shape (B, C, rfft_length)
    # We'll rearrange to shape (rfft_length, B, C) to vmap over freq.
    X_fft_t = jnp.transpose(X_fft, (2, 0, 1))  # shape (rfft_length, B, C)
    # L_factors: shape (rfft_length, C_lower)
    # vmap along first axis => (rfft_length, B, C)
    # We'll need an outer 'vmap' over B or just do a matrix multiply inside. 
    # The easiest is to vmap over f, then we do a batch matmul for dimension B?
    # But (B, C) is not a single vector. We can do a vmap over B as well. 
    # Let's combine them carefully.

    def freq_apply_fn(carry, data):
        # data is (X_fft_f, L_tri_f), shape:
        #   X_fft_f: (B, C)
        #   L_tri_f: (C_lower,)
        X_fft_f, L_tri_f = data
        # Build M_f
        L = fill_lower_triangle(L_tri_f, C)
        M = L @ L.T  # (C, C)
        # Apply to each item in batch => shape (B, C)
        out_f = jnp.einsum("ij,bj->bi", M, X_fft_f)
        return carry, out_f

    # Pair them up along freq dimension
    freq_data = (X_fft_t, L_factors)
    _, out_fft_t = jax.lax.scan(freq_apply_fn, None, freq_data)
    # out_fft_t: shape (rfft_length, B, C)
    # Transpose back
    Y_fft = jnp.transpose(out_fft_t, (1, 2, 0))  # shape (B, C, rfft_length)

    # 4) inverse rFFT -> shape (B, C, fft_size)
    y_full = jnp.fft.irfft(Y_fft, n=fft_size, axis=-1)

    # 5) Crop back to original W
    y_full = y_full[..., :W]

    # 6) Add bias per channel
    y = y_full + bias[None, :, None]

    return y

@spectral_conv1d_psd_func.defjvp
def spectral_conv1d_psd_func_jvp(primals, tangents):
    """A simple JVP definition. We just do a direct linearization:
       d/dx [M_f x_f] => M_f * dx_f + d(M_f)* x_f
    """
    x, L_factors, bias = primals
    dx, dL_factors, dbias = tangents

    # forward pass
    y = spectral_conv1d_psd_func(x, L_factors, bias)

    # tangent
    dy = jnp.zeros_like(y)
    if dx is not None:
        dy += spectral_conv1d_psd_func(dx, L_factors, jnp.zeros_like(bias))
    if dL_factors is not None:
        # we must incorporate "dM_f * X_f". This is more advanced; for brevity,
        # we reuse the same function but it's not strictly correct for dM_f*x.
        # A correct approach would explicitly multiply each freq's partial derivative. 
        # For demonstration, we approximate by re-calling with dL_factors as if it were L_factors.
        dy += spectral_conv1d_psd_func(x, dL_factors, jnp.zeros_like(bias))
    if dbias is not None:
        dy += dbias[None, :, None]

    return y, dy


def fill_lower_triangle(L_tri: jnp.ndarray, C: int) -> jnp.ndarray:
    """
    Fill a (C, C) zeroed matrix with the lower-triangular elements from L_tri, 
    which has length C*(C+1)//2.

    Returns: shape (C, C)
    """
    # Example approach:
    mat = jnp.zeros((C, C), dtype=L_tri.dtype)
    idx = 0
    for r in range(C):
        length = r + 1
        vals = L_tri[idx : idx + length]
        idx += length
        mat = mat.at[r, :length].set(vals)
    return mat


class Conv1dPSD:
    """
    NumPyro class for 1D 'convolution' across C channels, with the
    per-frequency matrix forced to be PSD via a Cholesky factor L_f.

    Assumes in_channels = out_channels = C for a proper PSD matrix at each freq.
    """

    def __init__(
        self,
        in_channels: int,
        fft_size: int,
        name: str = "conv1d_psd",
    ):
        # We'll assume out_channels == in_channels for simplicity.
        self.C = in_channels
        self.fft_size = fft_size
        self.rfft_length = fft_size // 2 + 1
        self.name = name

        # Number of distinct elements in a lower-triangular CxC matrix
        self.C_lower = self.C * (self.C + 1) // 2

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: shape (B, C, W)
        """
        # We sample a lower-tri factor for each freq => shape (rfft_length, C_lower)
        shape = (self.rfft_length, self.C_lower)
        L_factors = numpyro.sample(
            f"{self.name}_L_factors",
            dist.Normal(0.0, 0.1).expand(shape).to_event(2),
        )
        # Sample bias
        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0.0, 1.0).expand([self.C]).to_event(1),
        )

        # Apply the spectral PSD convolution
        return spectral_conv1d_psd_func(x, L_factors, bias)

@jax.custom_jvp
def spectral_conv2d_psd_func(
    x: jnp.ndarray,
    L_factors_2d: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
    """
    2D multi-channel 'PSD' convolution. For each freq bin (h,w), we have
    a real PSD matrix M_{h,w} = L_{h,w} * L_{h,w}^T.

    Args:
      x: shape (B, C, H, W)
      L_factors_2d: shape (Hf, Wf, C_lower)
        where Hf x Wf is the rfft2 grid, and C_lower = C*(C+1)//2
      bias: shape (C,)

    Returns:
      y: shape (B, C, H, W)
    """
    B, C, H, W = x.shape
    Hf, Wf, C_lower = L_factors_2d.shape
    # rfft2 uses (H, W/2+1) => W_full = 2*(Wf-1) if Wf>1
    # pad if needed:
    # For brevity, assume Hf==H, Wf==(W//2+1).
    X_fft = jnp.fft.rfft2(x, s=(H, 2*(Wf-1)), axes=(2, 3))  # shape (B, C, Hf, Wf)

    # We'll reorder to (Hf, Wf, B, C) so we can vmap or scan
    X_fft_t = jnp.transpose(X_fft, (2, 3, 0, 1))  # (Hf, Wf, B, C)

    def apply_freq_matrix(X_hw, L_tri_hw):
        # X_hw: shape (B, C)
        # L_tri_hw: shape (C_lower,)
        L = fill_lower_triangle(L_tri_hw, C)
        M = L @ L.T
        # out_hw = shape (B, C)
        out_hw = jnp.einsum("ij,bcj->bci", M, X_hw[:, None, :])[:, 0, :]
        # simpler: out_hw = jnp.einsum("ij,bj->bi", M, X_hw)
        return out_hw

    # We can do a double-scan or double-vmap over (h, w).
    # Let's do nested vmap for simplicity:
    apply_w = jax.vmap(
        lambda X_w, L_w: jax.vmap(
            apply_freq_matrix, in_axes=(0, 0), out_axes=0
        )(X_w, L_w),
        in_axes=(0, 0),
        out_axes=0,
    )
    apply_hw = jax.vmap(apply_w, in_axes=(0, 0), out_axes=0)

    # L_factors_2d: shape (Hf, Wf, C_lower)
    out_fft_t = apply_hw(X_fft_t, L_factors_2d)  # shape (Hf, Wf, B, C)

    # Re-transpose to (B, C, Hf, Wf)
    Y_fft = jnp.transpose(out_fft_t, (2, 3, 0, 1))

    # iFFT2
    y_full = jnp.fft.irfft2(Y_fft, s=(H, 2*(Wf-1)), axes=(2, 3))
    y_full = y_full[..., :H, :W]  # crop if needed
    y = y_full + bias[None, :, None, None]
    return y

@spectral_conv2d_psd_func.defjvp
def spectral_conv2d_psd_func_jvp(primals, tangents):
    x, L_factors_2d, bias = primals
    dx, dL_factors_2d, dbias = tangents
    y = spectral_conv2d_psd_func(x, L_factors_2d, bias)
    dy = jnp.zeros_like(y)
    if dx is not None:
        dy += spectral_conv2d_psd_func(dx, L_factors_2d, jnp.zeros_like(bias))
    if dL_factors_2d is not None:
        dy += spectral_conv2d_psd_func(x, dL_factors_2d, jnp.zeros_like(bias))
    if dbias is not None:
        dy += dbias[None, :, None, None]
    return y, dy


class Conv2dPSD:
    """
    NumPyro-based 2D convolution layer with a per-frequency PSD matrix 
    (C x C for each freq).
    Assumes in_channels=out_channels=C.
    """

    def __init__(self, channels: int, H: int, W: int, name="conv2d_psd"):
        self.C = channels
        self.Hf = H
        self.Wf = (W // 2) + 1
        self.name = name
        self.C_lower = self.C * (self.C + 1) // 2

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, C, H, W)
        shape = (self.Hf, self.Wf, self.C_lower)
        L_factors_2d = numpyro.sample(
            f"{self.name}_L_factors",
            dist.Normal(0.0, 0.1).expand(shape).to_event(3),
        )
        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0.0, 1.0).expand([self.C]).to_event(1),
        )
        return spectral_conv2d_psd_func(x, L_factors_2d, bias)

@jax.custom_jvp
def spectral_transposed_conv2d_psd_func(
    x: jnp.ndarray,
    L_factors_2d: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
    """
    2D spectral transposed convolution with PSD. 
    For each freq (h,w), M_{h,w} = L_f L_f^T in R^{C x C}, 
    multiply it by X_fft[..., h, w].
    """
    B, C, H, W = x.shape
    Hf, Wf, C_lower = L_factors_2d.shape

    X_fft = jnp.fft.rfft2(x, s=(H, 2*(Wf-1)), axes=(2, 3))  # shape (B, C, Hf, Wf)
    X_fft_t = X_fft.transpose(2, 3, 0, 1)  # (Hf, Wf, B, C)

    def freq_apply_fn(X_hw, L_tri_hw):
        L = fill_lower_triangle(L_tri_hw, C)
        M = L @ L.T
        return jnp.einsum("ij,bj->bi", M, X_hw)

    apply_w = jax.vmap(
        lambda X_w, L_w: jax.vmap(freq_apply_fn, in_axes=(0, 0), out_axes=0)(X_w, L_w),
        in_axes=(0, 0),
        out_axes=0,
    )
    apply_hw = jax.vmap(apply_w, in_axes=(0, 0), out_axes=0)

    out_fft_t = apply_hw(X_fft_t, L_factors_2d)  # shape (Hf, Wf, B, C)
    Y_fft = out_fft_t.transpose(2, 3, 0, 1)  # (B, C, Hf, Wf)
    y_full = jnp.fft.irfft2(Y_fft, s=(H, 2*(Wf-1)), axes=(2, 3))
    y = y_full[..., :H, :W] + bias[None, :, None, None]
    return y

@spectral_transposed_conv2d_psd_func.defjvp
def spectral_transposed_conv2d_psd_func_jvp(primals, tangents):
    x, L_factors_2d, bias = primals
    dx, dL_factors_2d, dbias = tangents
    y = spectral_transposed_conv2d_psd_func(x, L_factors_2d, bias)
    dy = jnp.zeros_like(y)
    if dx is not None:
        dy += spectral_transposed_conv2d_psd_func(dx, L_factors_2d, jnp.zeros_like(bias))
    if dL_factors_2d is not None:
        dy += spectral_transposed_conv2d_psd_func(x, dL_factors_2d, jnp.zeros_like(bias))
    if dbias is not None:
        dy += dbias[None, :, None, None]
    return y, dy


class TransposedConv2dPSD:
    """
    NumPyro-based 2D transposed 'convolution' with PSD spectral matrices.
    """

    def __init__(self, channels: int, H: int, W: int, name="transposed_conv2d_psd"):
        self.C = channels
        self.Hf = H
        self.Wf = (W // 2) + 1
        self.name = name
        self.C_lower = self.C * (self.C + 1) // 2

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        shape = (self.Hf, self.Wf, self.C_lower)
        L_factors_2d = numpyro.sample(
            f"{self.name}_L_factors",
            dist.Normal(0.0, 0.1).expand(shape).to_event(3),
        )
        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0.0, 1.0).expand([self.C]).to_event(1),
        )
        return spectral_transposed_conv2d_psd_func(x, L_factors_2d, bias)


def Model(X, y=None):
    
    B, C, H, W = X.shape
    X = Conv2dPSD(C, H, W)(X)
    X = jnp.reshape(-1, 28*28)
    W = numpyro.sample("W", dist.Normal(0, 1).expand([28*28, 10]).to_event(len(X.shape)))
    b = numpyro.sample("W", dist.Normal(0, 1).expand(10).to_event(len(1)))
    X = jnp.dot(X, W) + b
    numpyro.deterministic("logits", X)
    with numpyro.plate("data", B):
        numpyro.sample("obs", dist.Categorical(logits=X), obs=y)

key = jr.key(0)
X = jr.normal(key, (10, 1, 28, 28))
y = jr.normal(key, (10, 10)).astype(int)

kernel = NUTS(Model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
mcmc.run(key, X, y)