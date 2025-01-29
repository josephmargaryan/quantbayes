import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any

from flax import linen as nn

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
from numpyro.contrib.module import flax_module

import matplotlib.pyplot as plt
import numpy as np

###########################################
#################Utils#####################
###########################################


def extract_module_params(params: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Extract parameters for a specific module by its prefix.

    Args:
        params: Full parameter dictionary.
        prefix: Prefix string (e.g., 'emis', 'trans', 'enc').

    Returns:
        Parameters for the specified module.
    """
    key = f"{prefix}$params"
    if key not in params:
        print(f"[WARNING] No parameters found for prefix '{prefix}'.")
        return {}
    return params[key]


def reconstruct_data(
    X: jnp.ndarray,
    encoder: nn.Module,
    decoder: nn.Module,
    params: Dict[str, Any],
) -> jnp.ndarray:
    """
    Reconstruct data using the trained VAE.

    Args:
        X: Input data of shape (batch, ...).
        encoder: Encoder module.
        decoder: Decoder module.
        params: Dictionary containing trained parameters.

    Returns:
        Reconstructed data of the same shape as the input.
    """
    # Extract parameters for encoder and decoder
    encoder_params = extract_module_params(params, "encoder")
    decoder_params = extract_module_params(params, "decoder")

    # Generate latent variables using the encoder
    def encode_fn(x):
        return encoder.apply({"params": encoder_params}, x)

    mu, log_sigma = encode_fn(X)
    sigma = jnp.exp(log_sigma)

    # Reparameterization trick to sample z
    rng = jax.random.PRNGKey(42)  # Use a fixed key for deterministic behavior
    z = mu + sigma * jax.random.normal(rng, mu.shape)

    # Decode z to reconstruct X
    def decode_fn(z_):
        return decoder.apply({"params": decoder_params}, z_)

    recon_X = decode_fn(z)
    return recon_X


########################################
# 1. Different Encoder Architectures   #
########################################


# --------------------------------------------------------------------
# 1A. Simple Fully-Connected Encoder (for tabular / generic data)
# --------------------------------------------------------------------
class MLPEncoder(nn.Module):
    hidden_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x: (batch_size, input_dim) or any shape you flatten to (N, D).
        Returns: (mu, log_sigma) each (batch_size, latent_dim).
        """
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        # Output layer. We produce 2 * latent_dim => [mu, log_sigma]
        x = nn.Dense(2 * self.latent_dim)(x)
        mu, log_sigma = jnp.split(x, 2, axis=-1)
        return mu, log_sigma


# --------------------------------------------------------------------
# 1B. Convolutional Encoder (for image data)
# --------------------------------------------------------------------
class ConvEncoder(nn.Module):
    """
    Example for images shaped like (batch, H, W, C).
    We'll apply a few convolutions, flatten, and produce mu, log_sigma.
    """

    latent_dim: int
    hidden_channels: int = 32  # number of channels in conv layers

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x: (batch, height, width, channels)
        returns (mu, log_sigma) with shape (batch, latent_dim).
        """
        # Convolution stack
        x = nn.relu(
            nn.Conv(features=self.hidden_channels, kernel_size=(4, 4), strides=(2, 2))(
                x
            )
        )
        x = nn.relu(
            nn.Conv(
                features=self.hidden_channels * 2, kernel_size=(4, 4), strides=(2, 2)
            )(x)
        )
        # Flatten
        x = x.reshape((x.shape[0], -1))
        # Final dense
        x = nn.relu(nn.Dense(256)(x))
        # Output layer
        x = nn.Dense(2 * self.latent_dim)(x)
        mu, log_sigma = jnp.split(x, 2, axis=-1)
        return mu, log_sigma


# --------------------------------------------------------------------
# 1C. LSTM Encoder (for 1D time-series)
# --------------------------------------------------------------------
class LSTMEncoder(nn.Module):
    """
    Example for time-series shaped like (batch, seq_length, features).
    We'll process the entire sequence with LSTM, then produce mu, log_sigma
    from the final hidden state.
    """

    hidden_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x: (batch, time, features)
        returns (mu, log_sigma) (batch, latent_dim).
        """
        # Initialize LSTM cell
        lstm = nn.LSTMCell(self.hidden_dim)
        batch_size = x.shape[0]

        # Initialize carry
        carry = lstm.initialize_carry(
            rng=jax.random.PRNGKey(0),
            input_shape=(batch_size, x.shape[-1]),  # Shape includes batch and features
        )

        # Process the sequence
        for t in range(x.shape[1]):
            carry, h = lstm(carry, x[:, t, :])

        # h is shape (batch_size, hidden_dim)
        # produce mu, log_sigma
        out = nn.relu(nn.Dense(self.hidden_dim)(h))
        out = nn.Dense(2 * self.latent_dim)(out)
        mu, log_sigma = jnp.split(out, 2, axis=-1)
        return mu, log_sigma


# --------------------------------------------------------------------
# 1D. Attention-based Encoder
# --------------------------------------------------------------------
class AttentionEncoder(nn.Module):
    hidden_dim: int
    latent_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x: (batch_size, input_dim) or (batch_size, seq_length, emb_dim).
        For simplicity, we treat x as shape (batch_size, input_dim).
        """
        x_proj = nn.LayerNorm()(nn.relu(nn.Dense(self.hidden_dim)(x)))
        # Self-Attention (treating the single dimension as "length"?
        # If you want multi-token, reshape accordingly.)
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
        )(x_proj, x_proj)

        # Combine and project further
        combined = nn.relu(nn.Dense(self.hidden_dim)(attn_output))
        combined = nn.relu(nn.Dense(self.hidden_dim)(combined))

        # Produce mu, log_sigma
        x_out = nn.Dense(2 * self.latent_dim)(combined)
        mu, log_sigma = jnp.split(x_out, 2, axis=-1)
        return mu, log_sigma


########################################
# 2. Different Decoder Architectures   #
########################################


# --------------------------------------------------------------------
# 2A. Simple Fully-Connected Decoder
# --------------------------------------------------------------------
class MLPDecoder(nn.Module):
    x_dim: int  # dimension of reconstructed data
    hidden_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        z: (batch_size, latent_dim)
        returns recon_x: (batch_size, x_dim)
        """
        z = nn.relu(nn.Dense(self.hidden_dim)(z))
        z = nn.relu(nn.Dense(self.hidden_dim)(z))
        recon_x = nn.Dense(self.x_dim)(z)
        return recon_x


# --------------------------------------------------------------------
# 2B. Convolutional Decoder (for image data)
# --------------------------------------------------------------------
class ConvDecoder(nn.Module):
    """
    Inverse of ConvEncoder. We'll assume final output shape is (batch, H, W, C).
    For demonstration, let's assume we decode 8x8 -> upsample to 16x16 -> 28x28.
    Adjust kernel_size/strides to match your data dimension.
    """

    latent_dim: int
    output_shape: Tuple[int, int, int]  # (H, W, C)
    hidden_channels: int = 32

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        z: (batch_size, latent_dim)
        returns recon_x: (batch_size, H, W, C)
        """
        # Dense -> reshape -> ConvTranspose stack
        x = nn.relu(nn.Dense(7 * 7 * self.hidden_channels)(z))
        x = x.reshape((x.shape[0], 7, 7, self.hidden_channels))  # arbitrary size
        # Upsample to 14x14
        x = nn.relu(
            nn.ConvTranspose(
                features=self.hidden_channels, kernel_size=(4, 4), strides=(2, 2)
            )(x)
        )
        # Upsample to final H, W (e.g. 28x28)
        x = nn.relu(
            nn.ConvTranspose(
                features=self.hidden_channels // 2, kernel_size=(4, 4), strides=(2, 2)
            )(x)
        )
        # Final conv to match output channels
        x = nn.Conv(
            features=self.output_shape[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
        )(x)
        # Here we do not apply a final activation, we let the distribution handle it
        return x


# --------------------------------------------------------------------
# 2C. LSTM Decoder (for time-series)
# --------------------------------------------------------------------
class LSTMDecoder(nn.Module):
    """
    Will decode from z -> repeated hidden state -> LSTM -> output of shape (batch, seq_length, features).
    You can also do a more advanced approach that conditions each step on z, or a teacher forcing approach, etc.
    """

    output_dim: int  # dimension per time-step
    hidden_dim: int
    seq_length: int  # for demonstration, a fixed sequence length

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        z: (batch, latent_dim)
        returns recon_x: (batch, seq_length, output_dim)
        """
        batch_size = z.shape[0]
        # We'll start by projecting z to an initial hidden state
        init_h = nn.relu(nn.Dense(self.hidden_dim)(z))
        init_c = jnp.zeros_like(init_h)  # zero cell state or param-based

        lstm = nn.LSTMCell(self.hidden_dim)
        carry = (init_c, init_h)

        # We'll decode a fixed sequence by feeding zeros or a learned embedding:
        outputs = []
        # Let's use a "learned" token as input at each step:
        learned_token = self.param(
            "learned_token", nn.initializers.normal(stddev=0.1), (self.output_dim,)
        )
        for t in range(self.seq_length):
            carry, h = lstm(carry, jnp.tile(learned_token[None, :], (batch_size, 1)))
            # Project h to output dimension
            dec_out = nn.Dense(self.output_dim)(h)
            outputs.append(dec_out)

        # Stack along time dimension
        recon_x = jnp.stack(outputs, axis=1)  # (batch, seq_length, output_dim)
        return recon_x


# --------------------------------------------------------------------
# 2D. Attention-based Decoder
# --------------------------------------------------------------------
class AttentionDecoder(nn.Module):
    hidden_dim: int
    output_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        z: (batch, latent_dim) or (batch, seq_length, dimension).
        returns: recon_x (batch, output_dim) for simple case
        or (batch, seq_length, dimension) if you want a sequence output.
        """
        # Project z
        z_proj = nn.LayerNorm()(nn.relu(nn.Dense(self.hidden_dim)(z)))
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
        )(z_proj, z_proj)
        combined = nn.relu(nn.Dense(self.hidden_dim)(attn_output))
        recon_x = nn.Dense(self.output_dim)(combined)
        return recon_x


########################################
# 3. VAE model / guide / training      #
########################################


def vae_model(
    X: jnp.ndarray,
    encoder: nn.Module,
    decoder: nn.Module,
    latent_dim: int,
    # Number of dimensions (beyond batch) we treat as event dims.
    # E.g. for (batch, x_dim) => event_dim=1
    #      for (batch, H, W, C) => event_dim=3
    event_ndims: int = 1,
):
    """
    Generic VAE model that registers and calls the given encoder/decoder modules.

    X: Input data of shape (batch_size, ...)
    """
    batch_size = X.shape[0]

    # Register Flax modules in NumPyro
    # We need input_shape so that the modules can be properly initialized
    # by `flax_module`. Adjust the shape accordingly for each encoder/decoder usage.
    encoder_module = flax_module(
        "encoder", encoder, input_shape=X.shape
    )  # shape is (batch, ...)
    decoder_module = flax_module(
        "decoder",
        decoder,
        # For decoder, the input shape is (batch, latent_dim)
        input_shape=(batch_size, latent_dim),
    )

    # Encode
    mu, log_sigma = encoder_module(X)
    sigma = jnp.exp(log_sigma)
    z = numpyro.sample("z", dist.Normal(mu, sigma).to_event(1))

    # Decode
    recon = decoder_module(z)

    # Likelihood (assume Normal with fixed variance=1.0 for demonstration)
    # Use .to_event(event_ndims) so that all dims beyond batch are considered a single event.
    numpyro.sample("obs", dist.Normal(recon, 1.0).to_event(event_ndims), obs=X)


def vae_guide(X: jnp.ndarray, encoder: nn.Module, latent_dim: int):
    """
    VAE guide: same as model but only the encoder -> sample z.
    """
    batch_size = X.shape[0]
    encoder_module = flax_module("encoder", encoder, input_shape=X.shape)

    mu, log_sigma = encoder_module(X)
    sigma = jnp.exp(log_sigma)
    numpyro.sample("z", dist.Normal(mu, sigma).to_event(1))


def train_vae(
    X: jnp.ndarray,
    encoder: nn.Module,
    decoder: nn.Module,
    latent_dim: int,
    event_ndims: int = 1,
    num_steps: int = 1000,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Train the VAE using SVI.
    """

    def model_fn(data):
        return vae_model(data, encoder, decoder, latent_dim, event_ndims)

    def guide_fn(data):
        return vae_guide(data, encoder, latent_dim)

    optimizer = Adam(learning_rate)
    svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
    svi_state = svi.init(jax.random.PRNGKey(seed), X)

    losses = []
    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state, X)
        losses.append(loss)
        if step % max(1, (num_steps // 5)) == 0:
            print(f"[Step {step}] ELBO: {-loss:.4f}")
    return svi.get_params(svi_state), np.array(losses)


########################################
# 4. Example Data / Testing            #
########################################


def generate_random_tabular_data(n=128, d=10, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.array(rng.normal(size=(n, d)))


def generate_random_image_data(n=64, h=28, w=28, c=1, seed=0):
    rng = np.random.default_rng(seed)
    # E.g. values in [0,1], or normal(0,1)
    return jnp.array(rng.normal(size=(n, h, w, c)))


def generate_random_timeseries_data(n=64, t=10, features=2, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.array(rng.normal(size=(n, t, features)))


def plot_losses(losses, title="Training Loss"):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="loss")
    plt.xlabel("iteration")
    plt.ylabel("loss (negative ELBO)")
    plt.title(title)
    plt.legend()
    plt.show()


def main():

    ###################
    # 4A. Tabular Data
    ###################
    print("=== Training MLP VAE on random tabular data ===")
    X_tab = generate_random_tabular_data(n=128, d=10)
    print(f"Tabular data shape: {X_tab.shape}")

    latent_dim = 5
    encoder = MLPEncoder(hidden_dim=32, latent_dim=latent_dim)
    decoder = MLPDecoder(x_dim=X_tab.shape[1], hidden_dim=32)

    params_tab, losses_tab = train_vae(
        X_tab,
        encoder,
        decoder,
        latent_dim=latent_dim,
        event_ndims=1,  # we treat the last dim as the event
        num_steps=200,
        learning_rate=1e-3,
    )
    plot_losses(losses_tab, title="MLP VAE on Tabular Data")

    ###################
    # 4B. Image Data
    ###################
    print("\n=== Training Conv VAE on random image data ===")
    X_img = generate_random_image_data(n=32, h=28, w=28, c=1)
    latent_dim = 10
    # shape is (batch, 28, 28, 1)
    print(f"Image data shape: {X_img.shape}")

    conv_encoder = ConvEncoder(latent_dim=latent_dim, hidden_channels=32)
    # We'll decode back to shape (28, 28, 1)
    conv_decoder = ConvDecoder(
        latent_dim=latent_dim, output_shape=(28, 28, 1), hidden_channels=32
    )

    # For images, we have event_ndims=3 => the last 3 dims (H,W,C) form the "event"
    params_img, losses_img = train_vae(
        X_img,
        conv_encoder,
        conv_decoder,
        latent_dim=latent_dim,
        event_ndims=3,
        num_steps=200,
        learning_rate=1e-3,
    )
    plot_losses(losses_img, title="Conv VAE on Image Data")

    ###################
    # 4C. Time Series
    ###################
    print("\n=== Training LSTM VAE on random time-series data ===")
    X_ts = generate_random_timeseries_data(n=64, t=10, features=2)
    # shape is (batch, 10, 2)
    latent_dim = 6

    lstm_enc = LSTMEncoder(hidden_dim=16, latent_dim=latent_dim)
    # Let's decode a 10-step, 2-feature series:
    lstm_dec = LSTMDecoder(output_dim=2, hidden_dim=16, seq_length=10)

    # event_ndims=2 => (time, features)
    params_ts, losses_ts = train_vae(
        X_ts,
        lstm_enc,
        lstm_dec,
        latent_dim=latent_dim,
        event_ndims=2,
        num_steps=200,
        learning_rate=1e-3,
    )
    plot_losses(losses_ts, title="LSTM VAE on Time Series")

    ###################
    # 4D. Attention
    ###################
    print("\n=== Training Attention VAE on random tabular data ===")
    X_attn = generate_random_tabular_data(n=128, d=12)
    print(f"Data shape: {X_attn.shape}")
    latent_dim = 4
    attn_enc = AttentionEncoder(hidden_dim=32, latent_dim=latent_dim, num_heads=4)
    attn_dec = AttentionDecoder(hidden_dim=32, output_dim=X_attn.shape[1], num_heads=4)

    params_attn, losses_attn = train_vae(
        X_attn,
        attn_enc,
        attn_dec,
        latent_dim=latent_dim,
        event_ndims=1,
        num_steps=200,
        learning_rate=1e-3,
    )
    plot_losses(losses_attn, title="Attention VAE on Tabular Data")

    ###################################
    # 5A. Reconstruct Times Series Data
    ###################################
    print("\n=== Testing Reconstruction for Time Series ===")
    reconstructed_ts = reconstruct_data(X_ts, lstm_enc, lstm_dec, params_ts)

    # Compare original vs reconstructed
    print("\nOriginal Time Series (first sample):")
    print(X_ts[0])
    print("\nReconstructed Time Series (first sample):")
    print(reconstructed_ts[0])

    plt.figure(figsize=(8, 4))
    for feature in range(X_ts.shape[2]):
        plt.plot(X_ts[0, :, feature], label=f"Original Feature {feature+1}")
        plt.plot(
            reconstructed_ts[0, :, feature],
            "--",
            label=f"Reconstructed Feature {feature+1}",
        )
    plt.title("Time Series Reconstruction")
    plt.xlabel("Time Steps")
    plt.legend()
    plt.show()

    ######################################
    # 5B. Reconstruct Attention Based Data
    ######################################
    print("\n=== Testing Reconstruction for Attention-based Tabular Data ===")
    reconstructed_attn = reconstruct_data(X_attn, attn_enc, attn_dec, params_attn)

    # Compare original vs reconstructed
    print("\nOriginal Tabular Data (first 5 samples):")
    print(X_attn[:5])
    print("\nReconstructed Tabular Data (first 5 samples):")
    print(reconstructed_attn[:5])

    plt.figure(figsize=(6, 4))
    plt.scatter(X_attn.flatten(), reconstructed_attn.flatten(), alpha=0.6)
    plt.title("Original vs Reconstructed (Attention-based)")
    plt.xlabel("Original")
    plt.ylabel("Reconstructed")
    plt.show()

    ##################################
    # 5C. Reconstruct Image Data
    ##################################
    print("\n=== Testing Reconstruction for Image Data ===")
    reconstructed_img = reconstruct_data(X_img, conv_encoder, conv_decoder, params_img)

    # Compare original vs reconstructed
    print("\nOriginal Image Data (first sample, reshaped to 2D):")
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(X_img[0].squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    print("\nReconstructed Image Data (first sample, reshaped to 2D):")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img[0].squeeze(), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")
    plt.show()

    ##################################
    # 5D. Reconstruct Tabular Data
    ##################################
    print("\n=== Testing Reconstruction for Tabular Data ===")
    reconstructed_tab = reconstruct_data(X_tab, encoder, decoder, params_tab)

    # Compare original vs reconstructed
    print("\nOriginal Tabular Data (first 5 samples):")
    print(X_tab[:5])
    print("\nReconstructed Tabular Data (first 5 samples):")
    print(reconstructed_tab[:5])

    plt.figure(figsize=(6, 4))
    plt.scatter(X_tab.flatten(), reconstructed_tab.flatten(), alpha=0.6)
    plt.title("Original vs Reconstructed (Tabular)")
    plt.xlabel("Original")
    plt.ylabel("Reconstructed")
    plt.show()


if __name__ == "__main__":
    main()
