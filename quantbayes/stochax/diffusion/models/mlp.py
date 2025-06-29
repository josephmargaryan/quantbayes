import math
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


##############################################
# Sinusoidal Time Embedding
##############################################
class SinusoidalTimeEmb(eqx.Module):
    emb: jnp.ndarray

    def __init__(self, dim: int):
        half_dim = dim // 2
        # Compute frequencies in log space.
        emb = math.log(10000.0) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: scalar diffusion time (or 0D array)
        emb = x * self.emb
        out = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return out


##############################################
# MLP-based Diffusion Model for Tabular Data
##############################################
class DiffusionMLP(eqx.Module):
    # Time embedding modules.
    time_emb: SinusoidalTimeEmb
    time_proj: eqx.nn.MLP

    # Data embedding and processing.
    embed: eqx.nn.Linear
    mlp: eqx.nn.MLP
    out_proj: eqx.nn.Linear

    input_dim: int
    hidden_dim: int

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        time_emb_dim: int,
        dropout_rate: float = 0.0,
        *,
        key: jax.random.PRNGKey,
    ):
        """
        Args:
            input_dim: Dimension of the input tabular data features.
            hidden_dim: Internal hidden dimension.
            num_layers: Depth of the core MLP (number of layers).
            time_emb_dim: Dimension for the sinusoidal time embedding.
            dropout_rate: Dropout probability applied inside the MLP (if desired).
            key: PRNG key.
        """
        # Split keys for submodules.
        keys = jr.split(key, 4)
        k_time, k_embed, k_mlp, k_out = keys

        # Initialize time embedding modules.
        self.time_emb = SinusoidalTimeEmb(time_emb_dim)
        self.time_proj = eqx.nn.MLP(
            in_size=time_emb_dim,
            out_size=hidden_dim,
            width_size=2 * hidden_dim,
            depth=2,
            key=k_time,
        )

        # Embed the input feature vector into the hidden dimension.
        self.embed = eqx.nn.Linear(input_dim, hidden_dim, key=k_embed)

        # Core MLP processing. This MLP processes the combined (embedded input + time)
        # representation. You can adjust the depth (num_layers) as needed.
        self.mlp = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=num_layers,
            key=k_mlp,
        )

        # Project back to the original feature dimension.
        self.out_proj = eqx.nn.Linear(hidden_dim, input_dim, key=k_out)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def _forward(
        self, t: float, x: jnp.ndarray, *, key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Forward pass for a single sample.

        Args:
            t: Diffusion time (scalar).
            x: Input feature vector of shape (input_dim,).
            key: Optional PRNG key (not used here but included for API consistency).

        Returns:
            Output feature vector of shape (input_dim,).
        """
        # Compute time embedding and project to hidden dimension.
        t_emb = self.time_emb(jnp.array(t))  # shape (time_emb_dim,)
        t_emb = self.time_proj(t_emb)  # shape (hidden_dim,)

        # Embed the input data.
        h = self.embed(x)  # shape (hidden_dim,)
        # Combine input embedding with time conditioning.
        h = h + t_emb

        # Process through core MLP.
        h = self.mlp(h)
        # Project back to original feature space.
        out = self.out_proj(h)
        return out

    def __call__(
        self, t: float, x: jnp.ndarray, *, key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Supports both batched and single-sample input.

        Args:
            t: Diffusion time (scalar).
            x: Input features. Can be shape (input_dim,) or batched (N, input_dim).
            key: Optional PRNG key.

        Returns:
            Output features with the same shape as input.
        """
        if x.ndim == 2:
            # Batched input: apply vmap over the batch dimension.
            if key is not None:
                keys = jr.split(key, x.shape[0])
                return jax.vmap(lambda sample, k: self._forward(t, sample, key=k))(
                    x, keys
                )
            else:
                return jax.vmap(lambda sample: self._forward(t, sample, key=None))(x)
        else:
            # Single sample.
            return self._forward(t, x, key=key)


##############################################
# Test function for DiffusionMLP
##############################################
def test_diffusion_mlp():
    key = jr.PRNGKey(0)
    input_dim = 10  # For example, 10 tabular features.
    hidden_dim = 64
    num_layers = 3
    time_emb_dim = 32
    dropout_rate = 0.1

    model = DiffusionMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        time_emb_dim=time_emb_dim,
        dropout_rate=dropout_rate,
        key=key,
    )

    # Create a dummy input feature vector.
    dummy_input = jnp.ones(input_dim)
    t = 0.5  # Example diffusion time.
    out = model(t, dummy_input, key=jr.PRNGKey(42))
    assert out.shape == (
        input_dim,
    ), f"Expected output shape {(input_dim,)}, got {out.shape}"
    print("DiffusionMLP forward pass succeeded with output shape:", out.shape)


if __name__ == "__main__":
    test_diffusion_mlp()
