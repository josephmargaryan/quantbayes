# lstm_dmm.py

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


class TransitionLSTMCell(nn.Module):
    """
    A single-step LSTM-based transition.
    We'll store:
      - self.lstm: an LSTMCell
      - self.out: a Dense layer to produce (mu_z, log_sigma_z)
    We'll call it with (carry, z_prev), returning (new_carry, (mu, log_sigma)).
    """
    hidden_dim: int
    z_dim: int

    @nn.compact
    def __call__(self,
                 carry: Tuple[jnp.ndarray, jnp.ndarray],
                 z_prev: jnp.ndarray
                 ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray],
                            Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        carry: (c, h) for LSTM state
        z_prev: shape (batch_size, z_dim) or (z_dim,)
        Returns:
          new_carry, (mu, log_sigma)
        """
        lstm_cell = nn.LSTMCell(features=self.hidden_dim)
        new_carry, h = lstm_cell(carry, z_prev)  # h is shape (batch, hidden_dim)
        
        # Now produce mu_z, log_sigma_z from h
        x = nn.Dense(self.hidden_dim)(h)
        x = nn.relu(x)
        x = nn.Dense(2 * self.z_dim)(x)
        mu_z, log_sigma_z = jnp.split(x, 2, axis=-1)

        return new_carry, (mu_z, log_sigma_z)

    @staticmethod
    def initialize_carry(batch_size, hidden_dim):
        """
        Utility to create the initial LSTM state (c,h).
        """
        return (jnp.zeros((batch_size, hidden_dim)),
                jnp.zeros((batch_size, hidden_dim)))


# We'll reuse an Emission MLP for x_t|z_t
class LSTMEmission(nn.Module):
    hidden_dim: int
    x_dim: int
    z_dim: int
    @nn.compact
    def __call__(self, z_t):
        # same idea as MLP
        x = nn.relu(nn.Dense(self.hidden_dim)(z_t))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.Dense(2 * self.x_dim)(x)
        mu, log_sigma = jnp.split(x, 2, axis=-1)
        return mu, log_sigma


def lstm_dmm_model(
    X: jnp.ndarray,
    transition_init,
    emission_init,
    z_dim=2,
    hidden_dim=16,
):
    batch_size, T, x_dim = X.shape

    # Register transition-lstm params
    transition_params = {
        k: numpyro.param(f"lstm_trans_{k}", v)
        for k, v in transition_init["params"].items()
    }
    transition_module = transition_init["module"]

    # Register emission-lstm params
    emission_params = {
        k: numpyro.param(f"lstm_emis_{k}", v)
        for k, v in emission_init["params"].items()
    }
    emission_module = emission_init["module"]

    # Sample z_0
    z_0 = numpyro.sample(
        "z_0",
        dist.Normal(jnp.zeros((batch_size, z_dim)),
                    jnp.ones((batch_size, z_dim))).to_event(1)
    )
    z_prev = z_0

    # Initialize carry for LSTM
    carry = transition_module.initialize_carry(batch_size, hidden_dim)

    for t in range(T):
        if t == 0:
            z_t = z_prev
        else:
            # one LSTM step
            new_carry, (mu_z, log_sigma_z) = transition_module.apply(
                {"params": transition_params}, carry, z_prev
            )
            carry = new_carry
            sigma_z = jnp.exp(log_sigma_z)
            z_t = numpyro.sample(f"z_{t}", dist.Normal(mu_z, sigma_z).to_event(1))

        # Emission
        mu_x, log_sigma_x = emission_module.apply({"params": emission_params}, z_t)
        sigma_x = jnp.exp(log_sigma_x)
        numpyro.sample(f"x_{t}",
                       dist.Normal(mu_x, sigma_x).to_event(1),
                       obs=X[:, t])
        z_prev = z_t

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

def make_lstm_inits(rng, z_dim=2, hidden_dim=16):
    """
    Initialize the TransitionLSTMCell, plus an LSTMEmission for x|z.
    """
    from flax.core import freeze, unfreeze
    trans_module = TransitionLSTMCell(hidden_dim=hidden_dim, z_dim=z_dim)
    # We must call init with a dummy carry + dummy z_prev
    batch_size = 1
    carry = trans_module.initialize_carry(batch_size, hidden_dim)
    dummy_zprev = jnp.zeros((batch_size, z_dim))
    params_trans = trans_module.init(rng, carry, dummy_zprev)

    return {
        "params": params_trans["params"],
        "module": trans_module
    }

def make_emis_inits(rng, x_dim=3, z_dim=2, hidden_dim=16):
    emis_module = LSTMEmission(hidden_dim=hidden_dim, x_dim=x_dim, z_dim=z_dim)
    dummy_z = jnp.zeros((1, z_dim))
    params_emis = emis_module.init(rng, dummy_z)

    return {
        "params": params_emis["params"],
        "module": emis_module
    }

def train_lstm_dmm(X, z_dim=2, hidden_dim=16, num_steps=1000):
    batch_size, T, x_dim = X.shape
    rng = random.PRNGKey(0)
    rng1, rng2, rng3 = random.split(rng, 3)

    transition_init = make_lstm_inits(rng1, z_dim=z_dim, hidden_dim=hidden_dim)
    emission_init   = make_emis_inits(rng2, x_dim=x_dim, z_dim=z_dim, hidden_dim=hidden_dim)

    def model_fn(X_):
        return lstm_dmm_model(X_, transition_init, emission_init, z_dim, hidden_dim)
    def guide_fn(X_):
        return naive_guide(X_, transition_init, emission_init, z_dim)

    optimizer = Adam(1e-3)
    svi = SVI(model_fn, guide_fn, optimizer, Trace_ELBO())
    svi_state = svi.init(rng3, X)

    losses = []
    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state, X)
        losses.append(loss)
        if step % 200 == 0:
            print(f"[LSTM-DMM] step={step}, ELBO={-loss:.2f}")
    return losses

def demo_lstm_dmm():
    X, _ = synthetic_data_dmm(batch_size=20, T=15, z_dim=2, x_dim=3)
    losses = train_lstm_dmm(X, z_dim=2, hidden_dim=16, num_steps=1000)
    print("Final LSTM-DMM ELBO:", -losses[-1])

if __name__ == "__main__":
    demo_lstm_dmm()
