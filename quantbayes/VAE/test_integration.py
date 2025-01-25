import jax
import jax.numpy as jnp
from flax import linen as nn
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.optim import Adam
from numpyro.contrib.module import flax_module
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Any, Optional

# Synthetic data for testing
def generate_synthetic_data(batch_size=100, x_dim=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(batch_size, x_dim))
    return jnp.array(X)

#############################################
##################Encoder####################
#############################################

# Encoder with Attention
class AttentionVAEEncoder(nn.Module):
    hidden_dim: int
    latent_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Parameters:
            x: Input data, shape (batch_size, x_dim).
        Returns:
            mu: Latent mean, shape (batch_size, latent_dim).
            log_sigma: Latent log variance, shape (batch_size, latent_dim).
        """
        # Dense projection
        x_proj = nn.relu(nn.Dense(self.hidden_dim)(x))  # Initial projection
        x_proj = nn.LayerNorm()(x_proj)  # Normalize inputs

        # Apply MultiHead Attention
        attention_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim
        )(x_proj, x_proj)  # Self-attention

        # Combine and project further
        combined = nn.relu(nn.Dense(self.hidden_dim)(attention_output))
        combined = nn.relu(nn.Dense(self.hidden_dim)(combined))

        # Compute latent variables
        x_out = nn.Dense(2 * self.latent_dim)(combined)  # Concatenate mu and log_sigma
        mu, log_sigma = jnp.split(x_out, 2, axis=-1)
        return mu, log_sigma


# Encoder Network
class VAEEncoder(nn.Module):
    hidden_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Parameters:
            x: Input data, shape (batch_size, x_dim).
        Returns:
            mu: Latent mean, shape (batch_size, latent_dim).
            log_sigma: Latent log variance, shape (batch_size, latent_dim).
        """
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.Dense(2 * self.latent_dim)(x)  # Concatenate mu and log_sigma
        mu, log_sigma = jnp.split(x, 2, axis=-1)
        return mu, log_sigma


#############################################
##################Decoder####################
#############################################

class AttentionVAEDecoder(nn.Module):
    hidden_dim: int
    x_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters:
            z: Latent variable, shape (batch_size, latent_dim).
        Returns:
            Reconstructed data, shape (batch_size, x_dim).
        """
        # Dense projection
        z_proj = nn.relu(nn.Dense(self.hidden_dim)(z))  # Initial projection
        z_proj = nn.LayerNorm()(z_proj)  # Normalize inputs

        # Apply MultiHead Attention
        attention_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim
        )(z_proj, z_proj)  # Self-attention

        # Reconstruct input
        combined = nn.relu(nn.Dense(self.hidden_dim)(attention_output))
        recon_x = nn.Dense(self.x_dim)(combined)
        return recon_x

# Decoder Network
class VAEDecoder(nn.Module):
    hidden_dim: int
    x_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters:
            z: Latent variable, shape (batch_size, latent_dim).
        Returns:
            Reconstructed data, shape (batch_size, x_dim).
        """
        z = nn.relu(nn.Dense(self.hidden_dim)(z))
        z = nn.relu(nn.Dense(self.hidden_dim)(z))
        recon_x = nn.Dense(self.x_dim)(z)
        return recon_x

# VAE Model
def vae_model(X: jnp.ndarray, encoder: nn.Module, decoder: nn.Module, latent_dim: int):
    """
    VAE model using Flax and NumPyro.
    Parameters:
        X: Input data, shape (batch_size, x_dim).
    """
    batch_size, _ = X.shape

    # Register Flax modules with NumPyro
    encoder_module = flax_module("encoder", encoder, input_shape=(batch_size, X.shape[1]))
    decoder_module = flax_module("decoder", decoder, input_shape=(batch_size, latent_dim))

    # Sample latent variables
    mu, log_sigma = encoder_module(X)
    sigma = jnp.exp(log_sigma)
    z = numpyro.sample("z", dist.Normal(mu, sigma).to_event(1))  # Reparameterization trick

    # Decode latent variables
    recon_x = decoder_module(z)

    # Likelihood of the observed data
    numpyro.sample("obs", dist.Normal(recon_x, 1.0).to_event(1), obs=X)

# VAE Guide
def vae_guide(X: jnp.ndarray, encoder: nn.Module, latent_dim: int):
    """
    Guide for VAE using Flax.
    Parameters:
        X: Input data, shape (batch_size, x_dim).
    """
    batch_size, _ = X.shape

    # Register Flax encoder with NumPyro
    encoder_module = flax_module("encoder", encoder, input_shape=(batch_size, X.shape[1]))

    # Sample latent variables
    mu, log_sigma = encoder_module(X)
    sigma = jnp.exp(log_sigma)
    numpyro.sample("z", dist.Normal(mu, sigma).to_event(1))

# Training Function
def train_vae(X: jnp.ndarray, encoder: nn.Module, decoder: nn.Module, latent_dim: int, num_steps=1000, learning_rate=1e-3):
    """
    Train the VAE using SVI.
    """
    def model_fn(X_):
        return vae_model(X_, encoder, decoder, latent_dim)

    def guide_fn(X_):
        return vae_guide(X_, encoder, latent_dim)

    optimizer = Adam(learning_rate)
    svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
    svi_state = svi.init(jax.random.PRNGKey(0), X)

    losses = []
    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state, X)
        losses.append(loss)
        if step % (num_steps // 5) == 0 or step == num_steps - 1:
            print(f"[VAE] Step {step}, ELBO: {-loss:.4f}")
    return svi.get_params(svi_state), losses

# Visualization
def visualize_reconstructions(X: jnp.ndarray, params: Dict[str, Any], encoder: nn.Module, decoder: nn.Module, latent_dim: int):
    """
    Visualize original vs reconstructed data.
    """
    num_samples = 100  # Number of samples for posterior predictive checks

    # Define Predictive with num_samples
    predictive = Predictive(
        vae_model,
        params=params,
        num_samples=num_samples
    )

    # Generate reconstructions
    reconstructions = predictive(jax.random.PRNGKey(1), X, encoder, decoder, latent_dim)["obs"]

    # Take the mean of the reconstructions across samples
    reconstructions_mean = jnp.mean(reconstructions, axis=0)  # Shape: (batch_size, x_dim)

    # Compare original and reconstructed
    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.plot(X[i], label=f"True (Sample {i+1})", alpha=0.5)
        plt.plot(reconstructions_mean[i], "--", label=f"Reconstructed (Sample {i+1})", alpha=0.7)
    plt.title("True vs Reconstructed Data")
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.legend()
    plt.show()

def visualize_single_reconstruction(
    X: jnp.ndarray,
    params: Dict[str, Any],
    encoder: nn.Module,
    decoder: nn.Module,
    latent_dim: int,
    feature_index: int = 0,
):
    """
    Visualize true vs reconstructed data for a single variable (feature).
    
    Args:
        X: Original data (batch_size, x_dim).
        params: Trained VAE parameters.
        encoder: VAE encoder module.
        decoder: VAE decoder module.
        latent_dim: Dimensionality of the latent space.
        feature_index: Index of the feature to visualize (default: 0).
    """
    num_samples = 100  # Number of posterior predictive samples

    # Define Predictive with num_samples
    predictive = Predictive(
        vae_model,
        params=params,
        num_samples=num_samples
    )

    # Generate reconstructions
    reconstructions = predictive(jax.random.PRNGKey(2), X, encoder, decoder, latent_dim)["obs"]

    # Take the mean of the reconstructions across samples
    reconstructions_mean = jnp.mean(reconstructions, axis=0)  # Shape: (batch_size, x_dim)

    # Extract the feature to visualize
    true_values = X[:, feature_index]
    reconstructed_values = reconstructions_mean[:, feature_index]

    # Plot true vs reconstructed for the selected feature
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label="True", alpha=0.5)
    plt.plot(reconstructed_values, "--", label="Reconstructed", alpha=0.7)
    plt.title(f"True vs Reconstructed Data (Feature {feature_index})")
    plt.xlabel("Samples")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.show()


# Main Function
def main():
    # Data dimensions
    x_dim = 10
    latent_dim = 3
    hidden_dim = 32
    kernel_size = 3  
    num_heads = 4  
    num_layers = 2  

    # Generate synthetic data
    X = generate_synthetic_data(batch_size=100, x_dim=x_dim)

    # Define encoder and decoder

    vae_encoder = VAEEncoder(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )

    vae_decoder = VAEDecoder(
        hidden_dim=hidden_dim,
        x_dim=x_dim,
    )

    vae_att_encoder = AttentionVAEEncoder(
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    num_heads=4
    )

    vae_att_decoder = AttentionVAEDecoder(
        hidden_dim=hidden_dim,
        x_dim=x_dim,
        num_heads=4
    )





    # Train the VAE
    params, losses = train_vae(X, vae_att_encoder, vae_att_decoder, latent_dim, num_steps=100)

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("Training Loss (Negative ELBO)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    # Visualize reconstructions
    visualize_reconstructions(X, params, vae_att_encoder, vae_att_decoder, latent_dim)

    visualize_single_reconstruction(
    X=X,
    params=params,
    encoder=vae_att_encoder,
    decoder=vae_att_decoder,
    latent_dim=latent_dim,
    feature_index=2  # Visualize the 3rd feature (index starts at 0)
    )


if __name__ == "__main__":
    main()
