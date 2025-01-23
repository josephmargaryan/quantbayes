import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
import jax.random as random
import numpy as np

#######################################
# 1) Reuse a naive guide
#######################################
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


#######################################
# 2) CNN Transition
#######################################
class CNNTransition(nn.Module):
    """
    1D CNN-based transition:
    We'll take a window of size K (the most recent K states),
    shape: (batch_size, K, z_dim).
    We'll do a 1D conv over length=K, with channels=z_dim.
    Then produce (mu, log_sigma).
    """
    hidden_channels: int
    z_dim: int
    kernel_size: int = 3
    K: int = 5

    @nn.compact
    def __call__(self, z_window: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_window: (batch_size, K, z_dim)
          - 'K' is the 'spatial' dimension
          - 'z_dim' is the 'channel' dimension (channels-last)
        """
        # 1) Apply 1D Convolution. By default, Flax expects
        #    shape [batch, length, in_channels].
        #    Our shape is (batch, K, z_dim) => length=K, channels=z_dim
        x = nn.Conv(
            features=self.hidden_channels,
            kernel_size=(self.kernel_size,),
            strides=(1,),
            padding='VALID'
        )(z_window)  
        # Now x.shape: (batch, new_length, hidden_channels)
        # where new_length = K - kernel_size + 1 (if padding='VALID')

        x = nn.relu(x)

        # 2) Flatten
        x = x.reshape(x.shape[0], -1)  # shape (batch, new_length * hidden_channels)

        # 3) Produce (mu, log_sigma)
        x = nn.Dense(2 * self.z_dim)(x)
        mu, log_sigma = jnp.split(x, 2, axis=-1)
        return mu, log_sigma


#######################################
# 3) Model
#######################################
def cnn_dmm_model(
    X: jnp.ndarray,
    transition_init,
    emission_init,
    z_dim=2,
    K=5
):
    """
    A DMM that uses a CNN-based transition.
    We keep a buffer of the last K z-states, shape (batch, K, z_dim).
    """
    batch_size, T, x_dim = X.shape

    # Register transition + emission
    trans_params = {k: numpyro.param(f"cnn_trans_{k}", v)
                    for k, v in transition_init["params"].items()}
    trans_module = transition_init["module"]

    emis_params = {k: numpyro.param(f"cnn_emis_{k}", v)
                   for k, v in emission_init["params"].items()}
    emis_module = emission_init["module"]

    # Sample z_0
    z_0 = numpyro.sample(
        "z_0",
        dist.Normal(jnp.zeros((batch_size, z_dim)),
                    jnp.ones((batch_size, z_dim)))
            .to_event(1)
    )

    # We'll keep a buffer of shape (batch_size, K, z_dim).
    # Initialize it by repeating z_0 along the "K" dimension.
    z_buffer = jnp.tile(jnp.expand_dims(z_0, axis=1), (1, K, 1))

    z_prev = z_0
    for t in range(T):
        if t == 0:
            z_t = z_prev
        else:
            # CNN transition
            mu_z, log_sigma_z = trans_module.apply({"params": trans_params}, z_buffer)
            sigma_z = jnp.exp(log_sigma_z)
            z_t = numpyro.sample(f"z_{t}", dist.Normal(mu_z, sigma_z).to_event(1))

        # Emission
        mu_x, log_sigma_x = emis_module.apply({"params": emis_params}, z_t)
        sigma_x = jnp.exp(log_sigma_x)
        numpyro.sample(f"x_{t}",
                       dist.Normal(mu_x, sigma_x).to_event(1),
                       obs=X[:, t])

        # Update buffer: shift left, append new z
        # drop the first element in dimension=1, then add z_t at the end
        z_buffer = jnp.concatenate([z_buffer[:, 1:, :], z_t[:, None, :]], axis=1)
        z_prev = z_t


#######################################
# 4) Setup & Training
#######################################
def make_cnn_inits(rng, z_dim=2, hidden_channels=16, kernel_size=3, K=5):
    module = CNNTransition(hidden_channels=hidden_channels,
                           z_dim=z_dim,
                           kernel_size=kernel_size,
                           K=K)
    # Dummy input shape: (batch=1, K=5, z_dim=2)
    dummy_in = jnp.zeros((1, K, z_dim))
    params = module.init(rng, dummy_in)
    return {"params": params["params"], "module": module}


def train_cnn_dmm(X, z_dim=2, K=5, hidden_channels=16, num_steps=1000):
    rng = random.PRNGKey(0)
    rt, re, rsvi = random.split(rng, 3)
    batch_size, T, x_dim = X.shape

    # Transition init
    transition_init = make_cnn_inits(rt, z_dim=z_dim,
                                     hidden_channels=hidden_channels,
                                     kernel_size=3, K=K)

    # Emission: reuse a simple MLP
    class CNNEmission(nn.Module):
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

    emis_module = CNNEmission(hidden_dim=16, x_dim=x_dim, z_dim=z_dim)
    dummy_z = jnp.zeros((1, z_dim))
    emis_params = emis_module.init(re, dummy_z)
    emission_init = {"params": emis_params["params"], "module": emis_module}

    def model_fn(X_):
        return cnn_dmm_model(X_, transition_init, emission_init, z_dim=z_dim, K=K)

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
            print(f"[CNN-DMM] step={step}, ELBO={-loss:.2f}")
    return losses


#######################################
# 5) Synthetic Data + Demo
#######################################
def synthetic_data_dmm(batch_size=20, T=15, z_dim=2, x_dim=3, seed=0):
    rng = np.random.default_rng(seed)
    A = 0.8 * np.eye(z_dim)
    B = rng.normal(size=(x_dim, z_dim)) * 0.5

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


def demo_cnn_dmm():
    X, _ = synthetic_data_dmm(batch_size=20, T=15, z_dim=2, x_dim=3)
    print("CNN-DMM synthetic data shape:", X.shape)
    losses = train_cnn_dmm(X, z_dim=2, K=5, hidden_channels=16, num_steps=600)
    print("Final CNN-DMM ELBO:", -losses[-1])


if __name__ == "__main__":
    demo_cnn_dmm()
