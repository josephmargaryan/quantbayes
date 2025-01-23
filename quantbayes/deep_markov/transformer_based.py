from mlp_based import synthetic_data_dmm
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
import numpy as np
import jax.random as random
import jax.numpy as jnp

def naive_guide(X, transition_init, emission_init, z_dim=2):
    batch_size, T, x_dim = X.shape
    mu_0 = numpyro.param("q_mu_0", jnp.zeros(z_dim))
    log_sigma_0 = numpyro.param("q_log_sigma_0", jnp.zeros(z_dim))
    z_0 = numpyro.sample(
        "z_0",
        dist.Normal(mu_0[None, :], jnp.exp(log_sigma_0)[None, :])
            .expand((batch_size, z_dim))
            .to_event(1)
    )
    z_prev = z_0
    for t in range(T):
        if t == 0: 
            continue
        mu_t = numpyro.param(f"q_mu_{t}", jnp.zeros(z_dim))
        log_sigma_t = numpyro.param(f"q_log_sigma_{t}", jnp.zeros(z_dim))
        z_t = numpyro.sample(
            f"z_{t}",
            dist.Normal(mu_t[None, :], jnp.exp(log_sigma_t)[None, :])
                .expand((batch_size, z_dim))
                .to_event(1)
        )
        z_prev = z_t

class SimpleTransformerEncoder(nn.Module):
    hidden_dim: int
    num_heads: int
    z_dim: int
    
    @nn.compact
    def __call__(self, z_sequence: jnp.ndarray) -> jnp.ndarray:
        """
        z_sequence: shape (batch_size, seq_len, z_dim)
        We'll apply a single self-attention block, then MLP, then return the final hidden.
        """
        # 1) Self-attention
        x = nn.LayerNorm()(z_sequence)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=0.0,  # for simplicity
            deterministic=True
        )(x, x)  # self-attention
        x = z_sequence + x

        # 2) MLP
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.hidden_dim)(y)
        y = nn.relu(y)
        y = nn.Dense(self.z_dim)(y)  # output dim = z_dim if we want
        out = x + y  # residual
        return out  # shape (batch_size, seq_len, z_dim)


class TransformerTransition(nn.Module):
    hidden_dim: int
    z_dim: int
    num_heads: int = 2

    @nn.compact
    def __call__(self, z_sequence: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_sequence: (batch_size, seq_len, z_dim) -> we output (mu, log_sigma) for the *last* time step
        """
        # Run a small transformer encoder
        x = SimpleTransformerEncoder(self.hidden_dim, self.num_heads, self.z_dim)(z_sequence)
        # We'll take the final time step's hidden vector
        final = x[:, -1, :]  # shape (batch_size, z_dim)

        # Then produce (mu, log_sigma)
        out = nn.Dense(2 * self.z_dim)(final)
        mu, log_sigma = jnp.split(out, 2, axis=-1)
        return mu, log_sigma

def transformer_dmm_model(
    X: jnp.ndarray,
    transition_init,
    emission_init,
    z_dim=2
):
    batch_size, T, x_dim = X.shape

    # Retrieve parameters
    trans_params = {k: numpyro.param(f"trans_{k}", v)
                    for k,v in transition_init["params"].items()}
    trans_module = transition_init["module"]

    emis_params = {k: numpyro.param(f"emis_{k}", v)
                   for k,v in emission_init["params"].items()}
    emis_module = emission_init["module"]

    # sample z_0
    z_0 = numpyro.sample(
        "z_0",
        dist.Normal(jnp.zeros((batch_size, z_dim)),
                    jnp.ones((batch_size, z_dim))).to_event(1)
    )
    z_prev = z_0

    # We'll keep a buffer of [z_0, z_1, ..., z_{t-1}]
    z_buffer = jnp.expand_dims(z_0, axis=1)  # shape (batch_size, 1, z_dim)

    for t in range(T):
        if t == 0:
            z_t = z_prev
        else:
            # run transformer on z_buffer
            mu_z, log_sigma_z = trans_module.apply({"params": trans_params}, z_buffer)
            sigma_z = jnp.exp(log_sigma_z)
            z_t = numpyro.sample(f"z_{t}",
                                 dist.Normal(mu_z, sigma_z).to_event(1))

        # Emission
        mu_x, log_sigma_x = emis_module.apply({"params": emis_params}, z_t)
        sigma_x = jnp.exp(log_sigma_x)
        numpyro.sample(f"x_{t}",
                       dist.Normal(mu_x, sigma_x).to_event(1),
                       obs=X[:, t])

        # Update buffer
        if t > 0:  # we only append new z after the first step
            z_buffer = jnp.concatenate([z_buffer, jnp.expand_dims(z_t, 1)], axis=1)
        z_prev = z_t

# training & demo
def make_transformer_inits(rng, hidden_dim=16, z_dim=2, num_heads=2):
    module = TransformerTransition(hidden_dim=hidden_dim, z_dim=z_dim, num_heads=num_heads)
    # We must pass a dummy sequence, e.g. shape (batch=1, seq_len=2, z_dim=2)
    dummy_input = jnp.zeros((1, 2, z_dim))
    params = module.init(rng, dummy_input)
    return {"params": params["params"], "module": module}

def train_transformer_dmm(X, z_dim=2, hidden_dim=16, num_heads=2, num_steps=1000):
    rng = random.PRNGKey(0)
    rt, re, rsvi = random.split(rng, 3)

    batch_size, T, x_dim = X.shape
    transition_init = make_transformer_inits(rt, hidden_dim, z_dim, num_heads)
    
    # Reuse an MLP emission for x|z
    class TransformerEmission(nn.Module):
        hidden_dim: int
        x_dim: int
        z_dim: int
        @nn.compact
        def __call__(self, z_t):
            x = nn.relu(nn.Dense(self.hidden_dim)(z_t))
            x = nn.relu(nn.Dense(self.hidden_dim)(x))
            out = nn.Dense(2 * self.x_dim)(x)
            mu, log_sigma = jnp.split(out, 2, axis=-1)
            return mu, log_sigma

    emis_module = TransformerEmission(hidden_dim=hidden_dim, x_dim=x_dim, z_dim=z_dim)
    dummy_z = jnp.zeros((1, z_dim))
    emis_params = emis_module.init(re, dummy_z)
    emission_init = {"params": emis_params["params"], "module": emis_module}

    def model_fn(X_):
        return transformer_dmm_model(X_, transition_init, emission_init, z_dim=z_dim)
    def guide_fn(X_):
        return naive_guide(X_, transition_init, emission_init, z_dim)

    optimizer = Adam(1e-3)
    svi = SVI(model_fn, guide_fn, optimizer, Trace_ELBO())
    svi_state = svi.init(rsvi, X)

    losses = []
    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state, X)
        losses.append(loss)
        if step % 200 == 0:
            print(f"[Transformer-DMM] step={step}, ELBO={-loss:.2f}")
    return losses

def demo_transformer_dmm():
    X, _ = synthetic_data_dmm(batch_size=10, T=8, z_dim=2, x_dim=3)
    losses = train_transformer_dmm(X, z_dim=2, hidden_dim=16, num_heads=2, num_steps=600)
    print("Final Transformer-DMM ELBO:", -losses[-1])

if __name__ == "__main__":
    demo_transformer_dmm()
