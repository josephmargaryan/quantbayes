import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Any, Tuple, List
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.optim import Adam
from numpyro.contrib.module import flax_module
import jax.random as random
import numpy as np
import matplotlib.pyplot as plt

###########################################
##############Transitions##################
###########################################

class LSTMTransition(nn.Module):
    """
    LSTM for transitions:
       z_{t-1} -> [mu_z, log_sigma_z]
    """
    hidden_dim: int
    z_dim: int

    @nn.compact
    def __call__(self, z_prev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_prev: (batch_size, z_dim)
        Returns: (mu_z, log_sigma_z), each of shape (batch_size, z_dim)
        """
        lstm_cell = nn.LSTMCell(features=self.hidden_dim)
        carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (z_prev.shape[0], self.hidden_dim))
        carry, h = lstm_cell(carry, z_prev)  # h has shape (batch_size, hidden_dim)

        mu_z = nn.Dense(self.z_dim)(h)
        log_sigma_z = nn.Dense(self.z_dim)(h)
        return mu_z, log_sigma_z


class TransformerTransition(nn.Module):
    """
    Transformer for transitions:
       z_{t-1} -> [mu_z, log_sigma_z]
    """
    num_heads: int
    hidden_dim: int
    z_dim: int

    @nn.compact
    def __call__(self, z_prev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_prev: (batch_size, z_dim)
        Returns: (mu_z, log_sigma_z), each of shape (batch_size, z_dim)
        """
        if self.z_dim % self.num_heads != 0:
            raise ValueError(f"z_dim ({self.z_dim}) must be divisible by num_heads ({self.num_heads}).")

        # Add a dummy time dimension for Transformer compatibility
        z_prev = jnp.expand_dims(z_prev, axis=1)  # Shape: (batch_size, 1, z_dim)
        
        attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(z_prev, z_prev)
        attn = jnp.squeeze(attn, axis=1)  # Remove dummy time dimension, Shape: (batch_size, z_dim)

        x = nn.Dense(self.hidden_dim)(attn)
        x = nn.relu(x)
        mu_z = nn.Dense(self.z_dim)(x)
        log_sigma_z = nn.Dense(self.z_dim)(x)
        return mu_z, log_sigma_z


class ConvTransition(nn.Module):
    hidden_dim: int
    z_dim: int
    kernel_size: int

    @nn.compact
    def __call__(self, z_prev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Reshape for Conv compatibility
        z_prev = z_prev[..., None]  # Add channel dimension -> (batch_size, z_dim, 1)

        # Correctly use kernel_size
        conv = nn.Conv(features=self.hidden_dim, kernel_size=(self.kernel_size,))(z_prev)
        conv = nn.relu(conv)

        # Flatten back to (batch_size, z_dim)
        conv = jnp.mean(conv, axis=-1)  # Average over the "channel" dimension, if needed
        mu_z = nn.Dense(self.z_dim)(conv)
        log_sigma_z = nn.Dense(self.z_dim)(conv)
        return mu_z, log_sigma_z


    
class MLPTransition(nn.Module):
    """
    MLP for transitions:
       z_{t-1} -> [mu_z, log_sigma_z]
    """
    hidden_dim: int
    z_dim: int

    @nn.compact
    def __call__(self, z_prev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_prev: (batch_size, z_dim)
        Returns: (mu_z, log_sigma_z), each of shape (batch_size, z_dim)
        """
        x = nn.relu(nn.Dense(self.hidden_dim)(z_prev))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        # Output dimension is 2*z_dim: (mu_z, log_sigma_z)
        x = nn.Dense(2 * self.z_dim)(x)
        mu_z, log_sigma_z = jnp.split(x, 2, axis=-1)
        return mu_z, log_sigma_z
    
class ConvAttentionTransition(nn.Module):
    """
    Hybrid transition combining Convolution and Attention:
       z_{t-1} -> [mu_z, log_sigma_z]
    """
    hidden_dim: int
    z_dim: int
    kernel_size: int = 3
    num_heads: int = 4

    @nn.compact
    def __call__(self, z_prev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_prev: (batch_size, z_dim)
        Returns: (mu_z, log_sigma_z), each of shape (batch_size, z_dim)
        """
        # Reshape for Conv1D
        z_prev = jnp.expand_dims(z_prev, axis=-1)  # Shape: (batch_size, z_dim, 1)

        # Apply convolution
        conv = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.kernel_size,),
            padding="SAME"
        )(z_prev)  # Shape: (batch_size, z_dim, hidden_dim)

        # Flatten for attention
        conv_flat = jnp.mean(conv, axis=-1)  # Shape: (batch_size, z_dim)

        # Apply Multi-Head Attention
        z_proj = nn.Dense(features=self.hidden_dim)(conv_flat)  # Shape: (batch_size, hidden_dim)
        z_proj = jnp.expand_dims(z_proj, axis=1)  # Add sequence dimension for attention
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            out_features=self.hidden_dim
        )(z_proj, z_proj, z_proj)  # Shape: (batch_size, 1, hidden_dim)

        attn = jnp.squeeze(attn, axis=1)  # Remove sequence dimension

        # Output layers for [mu_z, log_sigma_z]
        x = nn.relu(nn.Dense(self.hidden_dim)(attn))
        x = nn.Dense(2 * self.z_dim)(x)  # Shape: (batch_size, 2 * z_dim)
        mu_z, log_sigma_z = jnp.split(x, 2, axis=-1)
        return mu_z, log_sigma_z

class TransformerLSTMTransition(nn.Module):
    """
    Hybrid transition combining Transformer and LSTM:
       z_{t-1} -> [mu_z, log_sigma_z]
    """
    hidden_dim: int
    z_dim: int
    num_heads: int = 4

    @nn.compact
    def __call__(self, z_prev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_prev: (batch_size, z_dim)
        Returns: (mu_z, log_sigma_z), each of shape (batch_size, z_dim)
        """
        # Project to higher dimensional space for attention
        z_proj = nn.Dense(features=self.hidden_dim)(z_prev)  # Shape: (batch_size, hidden_dim)
        z_proj = jnp.expand_dims(z_proj, axis=1)  # Add sequence dimension for attention

        # Apply Multi-Head Attention
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            out_features=self.hidden_dim
        )(z_proj, z_proj, z_proj)  # Shape: (batch_size, 1, hidden_dim)

        attn = jnp.squeeze(attn, axis=1)  # Remove sequence dimension

        # Process with LSTM
        lstm_cell = nn.LSTMCell(features=self.hidden_dim)
        carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (attn.shape[0], self.hidden_dim))
        carry, h = lstm_cell(carry, attn)  # h: (batch_size, hidden_dim)

        # Output layers for [mu_z, log_sigma_z]
        x = nn.relu(nn.Dense(self.hidden_dim)(h))
        x = nn.Dense(2 * self.z_dim)(x)  # Shape: (batch_size, 2 * z_dim)
        mu_z, log_sigma_z = jnp.split(x, 2, axis=-1)
        return mu_z, log_sigma_z

class ConvLSTMTransition(nn.Module):
    """
    Hybrid transition combining Convolution and LSTM:
       z_{t-1} -> [mu_z, log_sigma_z]
    """
    hidden_dim: int
    z_dim: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, z_prev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_prev: (batch_size, z_dim)
        Returns: (mu_z, log_sigma_z), each of shape (batch_size, z_dim)
        """
        # Reshape for Conv1D
        z_prev = jnp.expand_dims(z_prev, axis=-1)  # Shape: (batch_size, z_dim, 1)

        # Apply convolution
        conv = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.kernel_size,),
            padding="SAME"
        )(z_prev)  # Shape: (batch_size, z_dim, hidden_dim)

        # Flatten for LSTM
        conv_flat = jnp.mean(conv, axis=-1)  # Shape: (batch_size, z_dim)

        # Process with LSTM
        lstm_cell = nn.LSTMCell(features=self.hidden_dim)
        carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (conv_flat.shape[0], self.hidden_dim))
        carry, h = lstm_cell(carry, conv_flat)  # h: (batch_size, hidden_dim)

        # Output layers for [mu_z, log_sigma_z]
        x = nn.relu(nn.Dense(self.hidden_dim)(h))
        x = nn.Dense(2 * self.z_dim)(x)  # Shape: (batch_size, 2 * z_dim)
        mu_z, log_sigma_z = jnp.split(x, 2, axis=-1)
        return mu_z, log_sigma_z



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




#############################################
##################Emission###################
#############################################



class MLPEmission(nn.Module):
    """
    MLP for emissions:
       z_t -> [mu_x, log_sigma_x]
    """
    hidden_dim: int
    x_dim: int
    z_dim: int

    @nn.compact
    def __call__(self, z_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_t: (batch_size, z_dim)
        Returns: (mu_x, log_sigma_x), each of shape (batch_size, x_dim)
        """
        x = nn.relu(nn.Dense(self.hidden_dim)(z_t))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        # Output dimension is 2*x_dim: (mu_x, log_sigma_x)
        x = nn.Dense(2 * self.x_dim)(x)
        mu_x, log_sigma_x = jnp.split(x, 2, axis=-1)
        return mu_x, log_sigma_x
    
class ConvEmission(nn.Module):
    """
    Convolution-based emission:
       z_t -> [mu_x, log_sigma_x]
    """
    hidden_dim: int
    x_dim: int
    z_dim: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, z_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_t: (batch_size, z_dim)
        Returns: (mu_x, log_sigma_x), each of shape (batch_size, x_dim)
        """
        # Reshape z_t to (batch_size, z_dim, 1) for Conv1D
        z_t = jnp.expand_dims(z_t, axis=-1)

        # Apply convolution
        conv = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.kernel_size,),
            padding="SAME"
        )(z_t)  # Shape: (batch_size, z_dim, hidden_dim)

        conv = nn.relu(conv)

        # Flatten back to (batch_size, z_dim)
        conv = jnp.mean(conv, axis=-1)  # Reduce over channel dimension

        # Final dense layers to produce [mu_x, log_sigma_x]
        x = nn.relu(nn.Dense(self.hidden_dim)(conv))
        out = nn.Dense(2 * self.x_dim)(x)  # Shape: (batch_size, 2 * x_dim)
        mu_x, log_sigma_x = jnp.split(out, 2, axis=-1)
        return mu_x, log_sigma_x

class TransformerEmission(nn.Module):
    """
    Transformer-based emission:
       z_t -> [mu_x, log_sigma_x]
    """
    hidden_dim: int
    x_dim: int
    z_dim: int
    num_heads: int = 4

    @nn.compact
    def __call__(self, z_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_t: (batch_size, z_dim)
        Returns: (mu_x, log_sigma_x), each of shape (batch_size, x_dim)
        """
        # Project z_t to a higher dimensional space divisible by num_heads
        z_proj = nn.Dense(features=self.hidden_dim)(z_t)  # Shape: (batch_size, hidden_dim)

        # Add dummy sequence dimension for compatibility with attention
        z_proj = jnp.expand_dims(z_proj, axis=1)  # Shape: (batch_size, 1, hidden_dim)

        # Self-attention layer
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            out_features=self.hidden_dim
        )(z_proj, z_proj, z_proj)  # Shape: (batch_size, 1, hidden_dim)

        # Remove sequence dimension
        attn = jnp.squeeze(attn, axis=1)  # Shape: (batch_size, hidden_dim)

        # Map to the output space [mu_x, log_sigma_x]
        x = nn.relu(nn.Dense(self.hidden_dim)(attn))  # Shape: (batch_size, hidden_dim)
        out = nn.Dense(2 * self.x_dim)(x)  # Shape: (batch_size, 2 * x_dim)
        mu_x, log_sigma_x = jnp.split(out, 2, axis=-1)
        return mu_x, log_sigma_x

class LSTMEmission(nn.Module):
    """
    LSTM-based emission:
       z_t -> [mu_x, log_sigma_x]
    """
    hidden_dim: int
    x_dim: int
    z_dim: int

    @nn.compact
    def __call__(self, z_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_t: (batch_size, z_dim)
        Returns: (mu_x, log_sigma_x), each of shape (batch_size, x_dim)
        """
        lstm_cell = nn.LSTMCell(features=self.hidden_dim)
        carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (z_t.shape[0], self.hidden_dim))

        # Process latent variable with LSTM
        carry, h = lstm_cell(carry, z_t)  # h: (batch_size, hidden_dim)

        # Final dense layers to produce [mu_x, log_sigma_x]
        x = nn.relu(nn.Dense(self.hidden_dim)(h))
        out = nn.Dense(2 * self.x_dim)(x)  # Shape: (batch_size, 2 * x_dim)
        mu_x, log_sigma_x = jnp.split(out, 2, axis=-1)
        return mu_x, log_sigma_x


class ConvAttentionEmission(nn.Module):
    """
    Hybrid emission combining Convolution and Attention:
       z_t -> [mu_x, log_sigma_x]
    """
    hidden_dim: int
    x_dim: int
    z_dim: int
    kernel_size: int = 3
    num_heads: int = 4

    @nn.compact
    def __call__(self, z_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_t: (batch_size, z_dim)
        Returns: (mu_x, log_sigma_x), each of shape (batch_size, x_dim)
        """
        # Reshape z_t for Conv1D
        z_t = jnp.expand_dims(z_t, axis=-1)  # Shape: (batch_size, z_dim, 1)

        # Apply convolution
        conv = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.kernel_size,),
            padding="SAME"
        )(z_t)  # Shape: (batch_size, z_dim, hidden_dim)

        # Flatten for attention
        conv_flat = jnp.mean(conv, axis=-1)  # Shape: (batch_size, z_dim)

        # Project to higher dimensional space for attention
        z_proj = nn.Dense(features=self.hidden_dim)(conv_flat)  # Shape: (batch_size, hidden_dim)
        z_proj = jnp.expand_dims(z_proj, axis=1)  # Add sequence dimension for attention

        # Apply Multi-Head Attention
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            out_features=self.hidden_dim
        )(z_proj, z_proj, z_proj)  # Shape: (batch_size, 1, hidden_dim)

        attn = jnp.squeeze(attn, axis=1)  # Remove sequence dimension

        # Final dense layers to produce [mu_x, log_sigma_x]
        x = nn.relu(nn.Dense(self.hidden_dim)(attn))  # Shape: (batch_size, hidden_dim)
        out = nn.Dense(2 * self.x_dim)(x)  # Shape: (batch_size, 2 * x_dim)
        mu_x, log_sigma_x = jnp.split(out, 2, axis=-1)
        return mu_x, log_sigma_x
    
class TransformerLSTMEmission(nn.Module):
    """
    Hybrid emission combining Transformer and LSTM:
       z_t -> [mu_x, log_sigma_x]
    """
    hidden_dim: int
    x_dim: int
    z_dim: int
    num_heads: int = 4

    @nn.compact
    def __call__(self, z_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_t: (batch_size, z_dim)
        Returns: (mu_x, log_sigma_x), each of shape (batch_size, x_dim)
        """
        # Project to higher dimensional space for attention
        z_proj = nn.Dense(features=self.hidden_dim)(z_t)  # Shape: (batch_size, hidden_dim)
        z_proj = jnp.expand_dims(z_proj, axis=1)  # Add sequence dimension for attention

        # Apply Transformer Attention
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            out_features=self.hidden_dim
        )(z_proj, z_proj, z_proj)  # Shape: (batch_size, 1, hidden_dim)

        attn = jnp.squeeze(attn, axis=1)  # Remove sequence dimension

        # Process with LSTM
        lstm_cell = nn.LSTMCell(features=self.hidden_dim)
        carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (attn.shape[0], self.hidden_dim))
        carry, h = lstm_cell(carry, attn)  # h: (batch_size, hidden_dim)

        # Final dense layers to produce [mu_x, log_sigma_x]
        x = nn.relu(nn.Dense(self.hidden_dim)(h))  # Shape: (batch_size, hidden_dim)
        out = nn.Dense(2 * self.x_dim)(x)  # Shape: (batch_size, 2 * x_dim)
        mu_x, log_sigma_x = jnp.split(out, 2, axis=-1)
        return mu_x, log_sigma_x

class ConvLSTMEmission(nn.Module):
    """
    Hybrid emission combining Convolution and LSTM:
       z_t -> [mu_x, log_sigma_x]
    """
    hidden_dim: int
    x_dim: int
    z_dim: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, z_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_t: (batch_size, z_dim)
        Returns: (mu_x, log_sigma_x), each of shape (batch_size, x_dim)
        """
        # Reshape z_t for Conv1D
        z_t = jnp.expand_dims(z_t, axis=-1)  # Shape: (batch_size, z_dim, 1)

        # Apply convolution
        conv = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.kernel_size,),
            padding="SAME"
        )(z_t)  # Shape: (batch_size, z_dim, hidden_dim)

        # Flatten for LSTM
        conv_flat = jnp.mean(conv, axis=-1)  # Shape: (batch_size, z_dim)

        # Process with LSTM
        lstm_cell = nn.LSTMCell(features=self.hidden_dim)
        carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (conv_flat.shape[0], self.hidden_dim))
        carry, h = lstm_cell(carry, conv_flat)  # h: (batch_size, hidden_dim)

        # Final dense layers to produce [mu_x, log_sigma_x]
        x = nn.relu(nn.Dense(self.hidden_dim)(h))  # Shape: (batch_size, hidden_dim)
        out = nn.Dense(2 * self.x_dim)(x)  # Shape: (batch_size, 2 * x_dim)
        mu_x, log_sigma_x = jnp.split(out, 2, axis=-1)
        return mu_x, log_sigma_x



######################################################
#####################Encoder##########################
######################################################



class LSTMEncoder(nn.Module):
    """
    Amortized encoder using LSTM:
      Reads the entire sequence of observations and outputs parameters for q(z_t | X)
    """
    hidden_dim: int
    x_dim: int
    z_dim: int

    @nn.compact
    def __call__(self, x_seq: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x_seq: shape (batch, T, x_dim)
        Returns:
          mu_seq, log_sigma_seq: each shape (batch, T, z_dim)
        """
        batch_size, T, _ = x_seq.shape

        # Define submodules
        lstm_cell = nn.LSTMCell(features=self.hidden_dim)
        dense1 = nn.Dense(features=self.hidden_dim)
        dense2 = nn.Dense(features=2 * self.z_dim)

        # Initialize carry
        carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, self.hidden_dim))

        mu_list = []
        log_sigma_list = []

        # Unroll LSTM over time
        for t in range(T):
            x_t = x_seq[:, t, :]  # shape (batch_size, x_dim)
            carry, h = lstm_cell(carry, x_t)  # h shape => (batch, hidden_dim)

            # Produce (mu_t, log_sigma_t) from h using Dense layers
            x = nn.relu(dense1(h))
            out = dense2(x)
            mu_t, log_sigma_t = jnp.split(out, 2, axis=-1)

            mu_list.append(mu_t)             # each shape (batch, z_dim)
            log_sigma_list.append(log_sigma_t)

        # Stack across time => shape (batch, T, z_dim)
        mu_seq        = jnp.stack(mu_list, axis=1)        # (batch, T, z_dim)
        log_sigma_seq = jnp.stack(log_sigma_list, axis=1) # (batch, T, z_dim)
        return mu_seq, log_sigma_seq
    
class AttentionEncoder(nn.Module):
    """
    Encoder with Multi-Head Dot-Product Attention to parameterize q(z_t | X).
    """
    hidden_dim: int
    x_dim: int
    z_dim: int
    num_heads: int = 4  # Number of attention heads

    @nn.compact
    def __call__(self, x_seq: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x_seq: shape (batch, T, x_dim)
        Returns:
          mu_seq, log_sigma_seq: each shape (batch, T, z_dim)
        """
        batch_size, T, _ = x_seq.shape

        # Linear projection to query, key, value spaces
        q = nn.Dense(features=self.hidden_dim)(x_seq)  # Query
        k = nn.Dense(features=self.hidden_dim)(x_seq)  # Key
        v = nn.Dense(features=self.hidden_dim)(x_seq)  # Value

        # Apply multi-head attention
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            out_features=self.hidden_dim
        )(q, k, v)  # Shape: (batch, T, hidden_dim)

        # Transform to latent space dimension
        dense1 = nn.Dense(features=self.hidden_dim)
        dense2 = nn.Dense(features=2 * self.z_dim)

        # Apply Dense layers to produce mu and log_sigma
        x = nn.relu(dense1(attention))  # (batch, T, hidden_dim)
        out = dense2(x)                # (batch, T, 2 * z_dim)
        mu_seq, log_sigma_seq = jnp.split(out, 2, axis=-1)
        return mu_seq, log_sigma_seq

class ConvEncoder(nn.Module):
    """
    Convolutional Encoder to parameterize q(z_t | X).
    """
    hidden_dim: int
    x_dim: int
    z_dim: int
    kernel_size: int = 3  # Size of the convolutional kernel
    strides: int = 1      # Stride for the convolution

    @nn.compact
    def __call__(self, x_seq: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x_seq: shape (batch, T, x_dim)
        Returns:
          mu_seq, log_sigma_seq: each shape (batch, T, z_dim)
        """
        # Apply 1D convolution over the time axis
        conv = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.kernel_size,),
            strides=(self.strides,),
            padding="SAME",
        )(x_seq)  # (batch, T, hidden_dim)

        # Transform to latent space dimension
        dense1 = nn.Dense(features=self.hidden_dim)
        dense2 = nn.Dense(features=2 * self.z_dim)

        # Apply Dense layers to produce mu and log_sigma
        x = nn.relu(dense1(conv))     # (batch, T, hidden_dim)
        out = dense2(x)              # (batch, T, 2 * z_dim)
        mu_seq, log_sigma_seq = jnp.split(out, 2, axis=-1)
        return mu_seq, log_sigma_seq
    
class ConvLSTMEncoder(nn.Module):
    """
    Hybrid Encoder combining Convolution and LSTM for q(z_t | X).
    """
    hidden_dim: int
    x_dim: int
    z_dim: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x_seq: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x_seq: shape (batch, T, x_dim)
        Returns:
          mu_seq, log_sigma_seq: each shape (batch, T, z_dim)
        """
        # Apply 1D Convolution
        conv = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.kernel_size,),
            strides=(1,),
            padding="SAME"
        )(x_seq)  # Shape: (batch, T, hidden_dim)

        # LSTM processing
        lstm_cell = nn.LSTMCell(features=self.hidden_dim)
        carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (conv.shape[0], self.hidden_dim))

        mu_list, log_sigma_list = [], []
        for t in range(conv.shape[1]):
            carry, h = lstm_cell(carry, conv[:, t, :])  # LSTM step
            x = nn.relu(nn.Dense(features=self.hidden_dim)(h))
            out = nn.Dense(features=2 * self.z_dim)(x)  # Shape: (batch, 2 * z_dim)
            mu, log_sigma = jnp.split(out, 2, axis=-1)
            mu_list.append(mu)
            log_sigma_list.append(log_sigma)

        # Stack across time
        mu_seq = jnp.stack(mu_list, axis=1)  # Shape: (batch, T, z_dim)
        log_sigma_seq = jnp.stack(log_sigma_list, axis=1)
        return mu_seq, log_sigma_seq

class MLPEncoder(nn.Module):
    """
    MLP Encoder to parameterize q(z_t | X).
    """
    hidden_dim: int
    x_dim: int
    z_dim: int

    @nn.compact
    def __call__(self, x_seq: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x_seq: shape (batch, T, x_dim)
        Returns:
          mu_seq, log_sigma_seq: each shape (batch, T, z_dim)
        """
        batch_size, T, x_dim = x_seq.shape

        # Flatten the sequence to process all time steps jointly
        x_flat = x_seq.reshape((batch_size * T, x_dim))  # (batch * T, x_dim)

        # Fully connected MLP layers
        x = nn.relu(nn.Dense(self.hidden_dim)(x_flat))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        out = nn.Dense(2 * self.z_dim)(x)  # (batch * T, 2 * z_dim)

        # Reshape back to sequence form
        out = out.reshape((batch_size, T, 2 * self.z_dim))  # (batch, T, 2 * z_dim)
        mu_seq, log_sigma_seq = jnp.split(out, 2, axis=-1)
        return mu_seq, log_sigma_seq
    
class ConvAttentionEncoder(nn.Module):
    """
    Hybrid Encoder combining Convolution and Multi-Head Attention for q(z_t | X).
    """
    hidden_dim: int
    x_dim: int
    z_dim: int
    kernel_size: int = 3
    num_heads: int = 4

    @nn.compact
    def __call__(self, x_seq: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x_seq: shape (batch, T, x_dim)
        Returns:
          mu_seq, log_sigma_seq: each shape (batch, T, z_dim)
        """
        # Apply 1D Convolution
        conv = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.kernel_size,),
            strides=(1,),
            padding="SAME"
        )(x_seq)  # Shape: (batch, T, hidden_dim)

        # Multi-Head Attention
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            out_features=self.hidden_dim
        )(conv, conv, conv)  # Self-attention

        # Dense transformations to produce mu and log_sigma
        x = nn.relu(nn.Dense(features=self.hidden_dim)(attention))
        out = nn.Dense(features=2 * self.z_dim)(x)  # Shape: (batch, T, 2 * z_dim)
        mu_seq, log_sigma_seq = jnp.split(out, 2, axis=-1)
        return mu_seq, log_sigma_seq


class TransformerLSTMEncoder(nn.Module):
    """
    Hybrid Encoder combining Transformer and LSTM for q(z_t | X).
    """
    hidden_dim: int
    x_dim: int
    z_dim: int
    num_heads: int = 4  # Keep the number of heads

    @nn.compact
    def __call__(self, x_seq: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x_seq: shape (batch, T, x_dim)
        Returns:
          mu_seq, log_sigma_seq: each shape (batch, T, z_dim)
        """
        # Project input to a higher dimensional space divisible by num_heads
        x_proj = nn.Dense(features=self.hidden_dim)(x_seq)  # Shape: (batch, T, hidden_dim)

        # Multi-Head Attention for global context
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            out_features=self.hidden_dim
        )(x_proj, x_proj, x_proj)  # Self-attention

        # LSTM processing
        lstm_cell = nn.LSTMCell(features=self.hidden_dim)
        carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (attention.shape[0], self.hidden_dim))

        mu_list, log_sigma_list = [], []
        for t in range(attention.shape[1]):
            carry, h = lstm_cell(carry, attention[:, t, :])  # LSTM step
            x = nn.relu(nn.Dense(features=self.hidden_dim)(h))
            out = nn.Dense(features=2 * self.z_dim)(x)  # Shape: (batch, 2 * z_dim)
            mu, log_sigma = jnp.split(out, 2, axis=-1)
            mu_list.append(mu)
            log_sigma_list.append(log_sigma)

        # Stack across time
        mu_seq = jnp.stack(mu_list, axis=1)  # Shape: (batch, T, z_dim)
        log_sigma_seq = jnp.stack(log_sigma_list, axis=1)
        return mu_seq, log_sigma_seq



def transformer_lstm_guide(X: jnp.ndarray, z_dim: int, hidden_dim: int, x_dim: int):
    """
    Guide using Transformer-LSTM Encoder to parameterize q(z_t | X).
    """
    batch_size, T, _ = X.shape

    # Register TransformerLSTM encoder module with NumPyro
    enc_module = flax_module(
        "transformer_lstm_enc", 
        TransformerLSTMEncoder(hidden_dim=hidden_dim, x_dim=x_dim, z_dim=z_dim),
        input_shape=(batch_size, T, x_dim)
    )

    # Run the Transformer-LSTM encoder
    mu_seq, log_sigma_seq = enc_module(X)
    # Shape: (batch, T, z_dim)

    # Sample z_t
    for t in range(T):
        z_t = numpyro.sample(
            f"z_{t}",
            dist.Normal(mu_seq[:, t], jnp.exp(log_sigma_seq[:, t]))
                .to_event(1)  # Treat z_dim as event dimension
        )


def conv_attention_guide(X: jnp.ndarray, z_dim: int, hidden_dim: int, x_dim: int):
    """
    Guide using Conv-Attention Encoder to parameterize q(z_t | X).
    """
    batch_size, T, _ = X.shape

    # Register ConvAttention encoder module with NumPyro
    enc_module = flax_module(
        "conv_attention_enc", 
        ConvAttentionEncoder(hidden_dim=hidden_dim, x_dim=x_dim, z_dim=z_dim),
        input_shape=(batch_size, T, x_dim)
    )

    # Run the ConvAttention encoder
    mu_seq, log_sigma_seq = enc_module(X)
    # Shape: (batch, T, z_dim)

    # Sample z_t
    for t in range(T):
        z_t = numpyro.sample(
            f"z_{t}",
            dist.Normal(mu_seq[:, t], jnp.exp(log_sigma_seq[:, t]))
                .to_event(1)  # Treat z_dim as event dimension
        )


def conv_lstm_guide(X: jnp.ndarray, z_dim: int, hidden_dim: int, x_dim: int):
    """
    Guide using Conv-LSTM Encoder to parameterize q(z_t | X).
    """
    batch_size, T, _ = X.shape

    # Register ConvLSTM encoder module with NumPyro
    enc_module = flax_module(
        "conv_lstm_enc", 
        ConvLSTMEncoder(hidden_dim=hidden_dim, x_dim=x_dim, z_dim=z_dim),
        input_shape=(batch_size, T, x_dim)
    )

    # Run the ConvLSTM encoder
    mu_seq, log_sigma_seq = enc_module(X)
    # Shape: (batch, T, z_dim)

    # Sample z_t
    for t in range(T):
        z_t = numpyro.sample(
            f"z_{t}",
            dist.Normal(mu_seq[:, t], jnp.exp(log_sigma_seq[:, t]))
                .to_event(1)  # Treat z_dim as event dimension
        )



def conv_guide(X: jnp.ndarray, z_dim: int, hidden_dim: int, x_dim: int):
    """
    Guide using a Convolutional Encoder to parameterize q(z_t | X).
    """
    batch_size, T, _ = X.shape

    # Register encoder module with NumPyro
    enc_module = flax_module(
        "conv_enc", 
        ConvEncoder(hidden_dim=hidden_dim, x_dim=x_dim, z_dim=z_dim),
        input_shape=(batch_size, T, x_dim)
    )

    # Run the Convolutional encoder
    mu_seq, log_sigma_seq = enc_module(X)
    # shape: (batch, T, z_dim)

    # Sample z_t
    for t in range(T):
        z_t = numpyro.sample(
            f"z_{t}",
            dist.Normal(mu_seq[:, t], jnp.exp(log_sigma_seq[:, t]))
                .to_event(1)  # Treat z_dim as event dimension
        )


def attention_guide(X: jnp.ndarray, z_dim: int, hidden_dim: int, x_dim: int):
    """
    Guide using an Attention Encoder to parameterize q(z_t | X).
    """
    batch_size, T, _ = X.shape

    # Register encoder module with NumPyro
    enc_module = flax_module(
        "attention_enc", 
        AttentionEncoder(hidden_dim=hidden_dim, x_dim=x_dim, z_dim=z_dim),
        input_shape=(batch_size, T, x_dim)
    )

    # Run the Attention encoder
    mu_seq, log_sigma_seq = enc_module(X)
    # shape: (batch, T, z_dim)

    # Sample z_t
    for t in range(T):
        z_t = numpyro.sample(
            f"z_{t}",
            dist.Normal(mu_seq[:, t], jnp.exp(log_sigma_seq[:, t]))
                .to_event(1)  # Treat z_dim as event dimension
        )

def lstm_guide(
    X: jnp.ndarray,
    z_dim: int = 2,
    hidden_dim: int = 16,
    x_dim: int = 3
):
    """
    Guide using an LSTM encoder to parameterize q(z_t | X).
    """
    batch_size, T, _ = X.shape

    # Register encoder module with NumPyro using flax_module
    enc_module = flax_module("enc", LSTMEncoder(hidden_dim=hidden_dim, x_dim=x_dim, z_dim=z_dim), input_shape=(batch_size, T, x_dim))

    # Run the LSTM encoder
    mu_seq, log_sigma_seq = enc_module(X)
    # shape: (batch, T, z_dim)

    # Sample z_0..z_{T-1}
    for t in range(T):
        z_t = numpyro.sample(
            f"z_{t}",
            dist.Normal(mu_seq[:, t], jnp.exp(log_sigma_seq[:, t]))
                .to_event(1)  # Treat z_dim as event dimension
        )



def mlp_guide(X: jnp.ndarray, z_dim: int, hidden_dim: int, x_dim: int):
    """
    Guide using an MLP Encoder to parameterize q(z_t | X).
    """
    batch_size, T, _ = X.shape

    # Register MLP encoder module with NumPyro
    enc_module = flax_module(
        "mlp_enc", 
        MLPEncoder(hidden_dim=hidden_dim, x_dim=x_dim, z_dim=z_dim),
        input_shape=(batch_size, T, x_dim)
    )

    # Run the MLP encoder
    mu_seq, log_sigma_seq = enc_module(X)
    # shape: (batch, T, z_dim)

    # Sample z_t
    for t in range(T):
        z_t = numpyro.sample(
            f"z_{t}",
            dist.Normal(mu_seq[:, t], jnp.exp(log_sigma_seq[:, t]))
                .to_event(1)  # Treat z_dim as event dimension
        )




def apply_mlp(nested_params: Dict[str, Any], module: nn.Module, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Manually apply a Flax module 'module' to input x,
    using the nested parameters in 'nested_params'.
    """
    return module.apply({"params": nested_params}, x)


##############################################################################
# 2) Model & Guide
##############################################################################

def dmm_model(
    X: jnp.ndarray,
    transition: nn.Module,
    emission: nn.Module,
    z_dim: int = 2,
    hidden_dim: int = 16,
    x_dim: int = 3
):
    """
    Deep Markov Model in NumPyro, vectorized over the batch dimension.

    X: (batch_size, T, x_dim)
    """
    batch_size, T, _ = X.shape

    # Register transition and emission modules with NumPyro using flax_module
    trans_module = flax_module("trans", transition, input_shape=(batch_size, z_dim))
    emis_module = flax_module("emis", emission, input_shape=(batch_size, z_dim))

    # Sample z_0 for each item in the batch
    z_0 = numpyro.sample(
        "z_0",
        dist.Normal(jnp.zeros((batch_size, z_dim)),
                    jnp.ones((batch_size, z_dim)))
            .to_event(1)  # Treat z_dim as event dimension
    )

    z_prev = z_0
    for t in range(T):
        if t > 0:
            # Transition: z_t given z_{t-1}
            mu_z, log_sigma_z = trans_module(z_prev)
            sigma_z = jnp.exp(log_sigma_z)
            z_t = numpyro.sample(
                f"z_{t}",
                dist.Normal(mu_z, sigma_z).to_event(1)  # Treat z_dim as event dimension
            )
        else:
            z_t = z_prev  # For t=0, z_t = z_0

        # Emission: x_t given z_t
        mu_x, log_sigma_x = emis_module(z_t)
        sigma_x = jnp.exp(log_sigma_x)

        # Observed data
        numpyro.sample(
            f"x_{t}",
            dist.Normal(mu_x, sigma_x).to_event(1),  # x_dim as event dimension
            obs=X[:, t]
        )

        z_prev = z_t



##############################################################################
# 3) Training Function
##############################################################################

def train_dmm(
    X: jnp.ndarray,
    transition: nn.Module, 
    emission: nn.Module,
    z_dim: int = 2,
    hidden_dim: int = 16,
    num_steps: int = 1000,
    learning_rate: float = 1e-3,
    x_dim: int = 3,
    guide = lstm_guide
) -> Tuple[Dict[str, Any], list]:
    """
    Train the Deep Markov Model using SVI.

    Returns:
      params: Trained parameters.
      losses: List of loss values over training steps.
    """
    rng = random.PRNGKey(0)
    rng_svi = rng  # Use the same RNG for simplicity

    def model_fn(X_):
        return dmm_model(X_, transition=transition, emission=emission,  z_dim=z_dim, hidden_dim=hidden_dim, x_dim=x_dim)

    def guide_fn(X_):
        return guide(X_, z_dim=z_dim, hidden_dim=hidden_dim, x_dim=x_dim)

    # Setup optimizer and SVI
    optimizer = Adam(learning_rate)
    svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
    svi_state = svi.init(rng_svi, X)

    losses = []
    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state, X)
        losses.append(loss)
        if step % (num_steps // 5) == 0 or step == num_steps - 1:
            print(f"[DMM] step={step}, ELBO={-loss:.2f}")
    params = svi.get_params(svi_state)
    
    # Debug: Print all parameter keys
    print("Parameter Keys:", params.keys())
    
    return params, losses


##############################################################################
# 4) Synthetic Data Generation
##############################################################################

def synthetic_data_dmm(batch_size=20, T=15, z_dim=2, x_dim=3, seed=0):
    """
    Generate synthetic linear-Gaussian data for DMM demonstration.
    """
    rng = np.random.default_rng(seed)

    A = 0.8 * np.eye(z_dim)  # Transition matrix
    B = rng.normal(scale=0.5, size=(x_dim, z_dim))  # Emission matrix

    Z = np.zeros((batch_size, T, z_dim))
    X = np.zeros((batch_size, T, x_dim))

    for b in range(batch_size):
        z_prev = rng.normal(size=(z_dim,))
        for t in range(T):
            if t == 0:
                Z[b, t, :] = z_prev
            else:
                Z[b, t, :] = A @ Z[b, t - 1, :] + 0.1 * rng.normal(size=(z_dim,))
            X[b, t, :] = B @ Z[b, t, :] + 0.1 * rng.normal(size=(x_dim,))
    return jnp.array(X), jnp.array(Z)


##############################################################################
# 5) Visualization Functions
##############################################################################

def plot_loss_curve(losses: list):
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("Training Loss (Negative ELBO)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()


def visualize_latent_space(X: jnp.ndarray, Z_true: jnp.ndarray, params: Dict[str, Any], 
                           guide: nn.Module,
                          z_dim: int = 2, plot_dims: Tuple[int, int] = (0, 1)):
    """
    Compare true latent variables with inferred latents from the guide.
    Parameters:
        plot_dims: Tuple of two integers specifying which dimensions to plot (default: (0, 1)).
    """
    # Generate a new PRNG key
    rng_key = random.PRNGKey(1)

    # Use Predictive to sample from the guide
    predictive = Predictive(guide, params=params, num_samples=100)
    guide_samples = predictive(rng_key, X, z_dim=z_dim, hidden_dim=16, x_dim=3)

    # Collect all z_t samples
    z_keys = [k for k in guide_samples.keys() if k.startswith('z_')]
    z_samples = [guide_samples[k] for k in z_keys]  # Each: (num_samples, batch_size, z_dim)

    # Stack and compute mean over samples
    z_samples = jnp.stack(z_samples, axis=0)  # (T, num_samples, batch_size, z_dim)
    z_mean = jnp.mean(z_samples, axis=1)      # (T, batch_size, z_dim)
    z_mean = jnp.transpose(z_mean, (1, 0, 2)) # (batch_size, T, z_dim)

    # Plot true vs inferred latent trajectories
    batch_size, T, _ = Z_true.shape
    dim_x, dim_y = plot_dims  # Selected dimensions for plotting
    fig, axes = plt.subplots(4, batch_size // 4, figsize=(12, 6), constrained_layout=True)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for i in range(batch_size):
        axes[i].plot(Z_true[i, :, dim_x], Z_true[i, :, dim_y], label='True Z', marker='o')
        axes[i].plot(z_mean[i, :, dim_x], z_mean[i, :, dim_y], label='Inferred Z', marker='x')
        axes[i].set_title(f"Batch {i+1}")
        axes[i].set_xlabel(f"z_dim {dim_x}")
        axes[i].set_ylabel(f"z_dim {dim_y}")

    # Add a global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2)

    plt.show()



def visualize_reconstructions(X: jnp.ndarray, params: Dict[str, Any],
                              emission: nn.Module,
                              guide: nn.Module,
                              z_dim: int = 2, hidden_dim: int = 16, x_dim: int = 3, 
                              plot_dims: List[int] = [0]):
    """
    Reconstruct observations from inferred latent variables and compare to true observations.
    Parameters:
        plot_dims: List of integers specifying which observation dimensions to plot (default: [0]).
    """
    print("[DEBUG] Starting visualize_reconstructions...")

    # Define a predictive function using the guide
    predictive = Predictive(guide, params=params, num_samples=100)

    # Generate samples from the guide
    rng_key = random.PRNGKey(2)
    guide_samples = predictive(rng_key, X, z_dim=z_dim, hidden_dim=hidden_dim, x_dim=x_dim)

    # Collect all z_t samples
    z_keys = [k for k in guide_samples.keys() if k.startswith('z_')]
    z_samples = [guide_samples[k] for k in z_keys]  # Each: (num_samples, batch_size, z_dim)

    # Stack and compute mean over samples
    z_samples = jnp.stack(z_samples, axis=0)  # (T, num_samples, batch_size, z_dim)
    z_mean = jnp.mean(z_samples, axis=1)      # (T, batch_size, z_dim)
    z_mean = jnp.transpose(z_mean, (1, 0, 2)) # (batch_size, T, z_dim)

    # Extract emission parameters with 'emis.' prefix
    emis_params_nested = extract_module_params(params, 'emis')
    print("All Parameter Keys:", params.keys())

    # Create the emission module
    emis_module = emission

    # Apply emission module with extracted parameters
    reconstructed_X = []
    for t in range(z_mean.shape[1]):
        z_t = z_mean[:, t, :]  # (batch_size, z_dim)
        try:
            # Apply the emission module with extracted parameters
            mu_x, _ = emis_module.apply({"params": emis_params_nested}, z_t)  # (batch_size, x_dim)
        except Exception as e:
            print(f"[ERROR] Emission module failed at time step {t}: {e}")
            raise
        reconstructed_X.append(mu_x)

    reconstructed_X = jnp.stack(reconstructed_X, axis=1)  # (batch_size, T, x_dim)

    # Plot selected observation dimensions
    for dim in plot_dims:
        plt.figure(figsize=(12, 6))
        for i in range(X.shape[0]):
            plt.plot(X[i, :, dim], label=f'True X (dim={dim})' if i == 0 else "", alpha=0.5)
            plt.plot(reconstructed_X[i, :, dim], '--', label=f'Reconstructed X (dim={dim})' if i == 0 else "", alpha=0.7)
        plt.title(f"True vs Reconstructed Observations (dim={dim})")
        plt.xlabel("Time step")
        plt.ylabel(f"x_dim {dim}")
        plt.legend()
        plt.show()



def demo_dmm():
    lstm_transition = LSTMTransition(hidden_dim=16, z_dim=2)
    transformer_transition = TransformerTransition(num_heads=2, hidden_dim=16, z_dim=2)
    conv_transition = ConvTransition(hidden_dim=16, z_dim=2, kernel_size=3)
    mlp_transition = MLPTransition(hidden_dim=16, z_dim=2)

    conv_attention_transition = ConvAttentionTransition(
    hidden_dim=16, z_dim=2, kernel_size=3, num_heads=4
    )
    transformer_lstm_transition = TransformerLSTMTransition(
        hidden_dim=16, z_dim=2, num_heads=4
    )
    conv_lstm_transition = ConvLSTMTransition(
        hidden_dim=16, z_dim=2, kernel_size=3
    )


    mlp_emission = MLPEmission(hidden_dim=16, x_dim=3, z_dim=2)
    conv_emission = ConvEmission(hidden_dim=16, x_dim=3, z_dim=2, kernel_size=3)
    transformer_emission = TransformerEmission(hidden_dim=16, x_dim=3, z_dim=2, num_heads=4)
    lstm_emission = LSTMEmission(hidden_dim=16, x_dim=3, z_dim=2)
    conv_attention_emission = ConvAttentionEmission(
        hidden_dim=16,  # Number of hidden dimensions
        x_dim=3,        # Observation dimensionality
        z_dim=2,        # Latent state dimensionality
        kernel_size=3,  # Kernel size for convolution
        num_heads=4     # Number of attention heads
    )

    conv_lstm_emission = ConvLSTMEmission(
        hidden_dim=16,  # Number of hidden dimensions
        x_dim=3,        # Observation dimensionality
        z_dim=2,        # Latent state dimensionality
        kernel_size=3   # Kernel size for convolution
    )

    transformer_lstm_emission = TransformerLSTMEmission(
        hidden_dim=16,  # Number of hidden dimensions
        x_dim=3,        # Observation dimensionality
        z_dim=2,        # Latent state dimensionality
        num_heads=4     # Number of attention heads
    )

    # Generate synthetic data
    X, Z = synthetic_data_dmm(batch_size=20, T=15, z_dim=2, x_dim=3)
    print("Synthetic data shape:", X.shape, "(batch_size, T, x_dim)")

    # Train DMM
    params, losses = train_dmm(X, 
                               transition=transformer_lstm_transition,
                               emission=transformer_lstm_emission, 
                               z_dim=2, 
                               hidden_dim=16, 
                               num_steps=10, 
                               guide=transformer_lstm_guide
                               )

    print("Final DMM loss (Negative ELBO):", losses[-1])

    # Plot training loss
    plot_loss_curve(losses)

    # Visualize latent space
    visualize_latent_space(X=X,
                           Z_true=Z,
                           params=params,
                           guide=transformer_lstm_guide,
                           z_dim=2,
                           plot_dims=(0, 2))

    # Visualize reconstructions
    visualize_reconstructions(X=X,
                              params=params,
                              emission=transformer_lstm_emission,
                              guide=transformer_lstm_guide,
                              z_dim=2,
                              hidden_dim=16,
                              x_dim=3,
                              plot_dims=[0, 1])

if __name__ == "__main__":
    demo_dmm()