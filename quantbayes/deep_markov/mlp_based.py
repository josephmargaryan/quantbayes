import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Any, Tuple
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
import jax.random as random
import numpy as np

############################################
# 1) Flax Modules
############################################

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
        z_prev: (batch_size, z_dim) or (z_dim,) if called on single sample
        Returns: (mu, log_sigma) each shape (batch_size, z_dim)
        """
        x = nn.relu(nn.Dense(self.hidden_dim)(z_prev))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        # Output dimension is 2*z_dim: (mu, log_sigma)
        x = nn.Dense(2 * self.z_dim)(x)
        mu, log_sigma = jnp.split(x, 2, axis=-1)
        return mu, log_sigma

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
        z_t: (batch_size, z_dim) or (z_dim,)
        Returns: (mu_x, log_sigma_x) each shape (batch_size, x_dim)
        """
        x = nn.relu(nn.Dense(self.hidden_dim)(z_t))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        # Output dimension is 2*x_dim: (mu_x, log_sigma_x)
        x = nn.Dense(2 * self.x_dim)(x)
        mu, log_sigma = jnp.split(x, 2, axis=-1)
        return mu, log_sigma


def apply_mlp(params: Dict, module: nn.Module, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Manually apply a Flax module 'module' to input x,
    using the parameters in 'params' PyTree.
    """
    return module.apply({"params": params}, x)


############################################
# 2) Model & Guide
############################################

def dmm_model(
    X: jnp.ndarray,
    transition_init: Dict[str, Any],
    emission_init: Dict[str, Any],
    z_dim: int = 2
):
    """
    Deep Markov Model in NumPyro, vectorized over the batch dimension.

    X: (batch_size, T, x_dim)
    We sample a latent z_t for each of the batch items at once.
    """
    batch_size, T, x_dim = X.shape
    
    # 1) Register and retrieve transition MLP parameters
    transition_params = {}
    for k, v in transition_init["params"].items():
        transition_params[k] = numpyro.param(f"transition_{k}", v)
    transition_module = transition_init["module"]
    
    # 2) Register and retrieve emission MLP parameters
    emission_params = {}
    for k, v in emission_init["params"].items():
        emission_params[k] = numpyro.param(f"emission_{k}", v)
    emission_module = emission_init["module"]

    # 3) Sample z_0 for *each* item in the batch at once
    # shape: (batch_size, z_dim)
    z_0 = numpyro.sample(
        "z_0",
        dist.Normal(jnp.zeros((batch_size, z_dim)),
                    jnp.ones((batch_size, z_dim)))
            .to_event(1)  # <--- make z_dim an event dimension
    )

    z_prev = z_0
    for t in range(T):
        if t == 0:
            z_t = z_prev
        else:
            # Transition: z_t given z_{t-1}
            mu_z, log_sigma_z = apply_mlp(transition_params, transition_module, z_prev)
            sigma_z = jnp.exp(log_sigma_z)
            z_t = numpyro.sample(
                f"z_{t}",
                dist.Normal(mu_z, sigma_z).to_event(1)  # ensure z_dim is event dimension
            )
        
        # Emission: x_t given z_t
        mu_x, log_sigma_x = apply_mlp(emission_params, emission_module, z_t)
        sigma_x = jnp.exp(log_sigma_x)
        
        # Observed data
        numpyro.sample(
            f"x_{t}",
            dist.Normal(mu_x, sigma_x).to_event(1),  # x_dim as event dimension
            obs=X[:, t]
        )
        
        z_prev = z_t


def dmm_guide(
    X: jnp.ndarray,
    transition_init: Dict[str, Any],
    emission_init: Dict[str, Any],
    z_dim: int = 2
):
    """
    Very naive guide: factorized Normal for each z_t, ignoring X.
    """
    batch_size, T, x_dim = X.shape

    # We'll define global parameters for each z_t across the entire batch
    # in a naive way.
    
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
            z_t = z_prev
        else:
            mu_t = numpyro.param(f"q_mu_{t}", jnp.zeros(z_dim))
            log_sigma_t = numpyro.param(f"q_log_sigma_{t}", jnp.zeros(z_dim))
            z_t = numpyro.sample(
                f"z_{t}",
                dist.Normal(mu_t[None, :], jnp.exp(log_sigma_t)[None, :])
                    .expand((batch_size, z_dim))
                    .to_event(1)
            )
        z_prev = z_t


############################################
# 3) Initialize & Train
############################################

def make_mlp_inits(rng, mlp_class, hidden_dim, z_dim=None, x_dim=None):
    """
    Initialize a Flax MLP module and return { 'params': ..., 'module': ... }.
    We'll pass a dummy input to get shape-based initialization.
    """
    init_module = mlp_class(hidden_dim=hidden_dim, z_dim=z_dim, x_dim=x_dim) \
                  if x_dim is not None else mlp_class(hidden_dim=hidden_dim, z_dim=z_dim)
    dummy_in = jnp.zeros((1, z_dim))  # shape (batch_size=1, z_dim)
    params = init_module.init(rng, dummy_in)
    return {
        "params": params["params"],
        "module": init_module
    }

def train_dmm(X, z_dim=2, hidden_dim=16, num_steps=1000):
    """
    X: (batch_size, T, x_dim)
    Returns: (params, losses)
    """
    rng = random.PRNGKey(0)
    rng_trans, rng_emis, rng_svi = random.split(rng, 3)
    
    batch_size, T, x_dim = X.shape
    transition_init = make_mlp_inits(rng_trans, MLPTransition, hidden_dim, z_dim=z_dim)
    emission_init   = make_mlp_inits(rng_emis, MLPEmission,   hidden_dim, z_dim=z_dim, x_dim=x_dim)
    
    def model_fn(X_):
        return dmm_model(X_, transition_init, emission_init, z_dim=z_dim)
    def guide_fn(X_):
        return dmm_guide(X_, transition_init, emission_init, z_dim=z_dim)
    
    optimizer = Adam(1e-3)
    svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
    svi_state = svi.init(rng_svi, X)
    
    losses = []
    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state, X)
        losses.append(loss)
        if step % 200 == 0:
            print(f"[DMM] step={step}, ELBO={-loss:.2f}")
    params = svi.get_params(svi_state)
    return params, losses


############################################
# 4) Synthetic data + Demo
############################################

def synthetic_data_dmm(batch_size=20, T=15, z_dim=2, x_dim=3, seed=0):
    """
    Very naive linear-Gaussian synthetic data generator for demonstration.
    """
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
                Z[b, t, :] = A @ Z[b, t-1, :] + 0.1 * rng.normal(size=(z_dim,))
            X[b, t, :] = B @ Z[b, t, :] + 0.1 * rng.normal(size=(x_dim,))
    return jnp.array(X), jnp.array(Z)


import matplotlib.pyplot as plt

def visualize_data_and_model(X, Z, params, transition_init, emission_init):
    """
    Visualizes the synthetic data, latent states, and model reconstruction.
    Args:
        X: (batch_size, T, x_dim) - Observed data
        Z: (batch_size, T, z_dim) - True latent states
        params: Trained parameters of the guide
        transition_init, emission_init: MLP initialization for transition and emission
    """
    batch_size, T, x_dim = X.shape

    # Retrieve guide parameters for latent states
    inferred_z = []
    for t in range(T):
        mu_t = params[f"q_mu_{t}"]
        inferred_z.append(mu_t)
    inferred_z = jnp.stack(inferred_z, axis=1)  # (z_dim, T)
    
    # Emission model reconstruction
    emission_params = {k.replace("emission_", ""): v for k, v in params.items() if "emission_" in k}
    reconstructed_X = []
    emission_module = emission_init["module"]

    for t in range(T):
        mu_x, _ = apply_mlp(emission_params, emission_module, inferred_z[:, t, :])
        reconstructed_X.append(mu_x)
    reconstructed_X = jnp.stack(reconstructed_X, axis=1)  # (batch_size, T, x_dim)

    # Plot original data
    plt.figure(figsize=(16, 8))
    for i in range(min(batch_size, 5)):  # Plot up to 5 samples
        plt.subplot(3, 1, 1)
        plt.plot(X[i, :, 0], label=f"Sample {i}")
        plt.title("Observed Data (X)")
        plt.xlabel("Time Steps")
        plt.ylabel("X values")
        plt.legend()

        # Plot true latent states
        plt.subplot(3, 1, 2)
        plt.plot(Z[i, :, 0], label=f"True Z Sample {i}")
        plt.title("True Latent States (Z)")
        plt.xlabel("Time Steps")
        plt.ylabel("Latent Z values")
        plt.legend()

        # Plot reconstructed data
        plt.subplot(3, 1, 3)
        plt.plot(reconstructed_X[i, :, 0], label=f"Reconstructed Sample {i}")
        plt.title("Reconstructed Data")
        plt.xlabel("Time Steps")
        plt.ylabel("X values")
        plt.legend()

    plt.tight_layout()
    plt.show()

def demo_dmm():
    # Generate synthetic data
    X, Z = synthetic_data_dmm(batch_size=20, T=15, z_dim=2, x_dim=3)
    print("DMM synthetic data shape:", X.shape, "(batch_size, T, x_dim)")
    
    params, losses = train_dmm(X, z_dim=2, hidden_dim=16, num_steps=1000)
    print("DMM final loss (negative ELBO) ~", losses[-1])

    visualize_data_and_model(X, Z, params, make_mlp_inits(random.PRNGKey(0), MLPTransition, 16, z_dim=2), 
                         make_mlp_inits(random.PRNGKey(1), MLPEmission, 16, z_dim=2, x_dim=3))
    
    return params, losses

if __name__ == "__main__":
    dmm_params, dmm_losses = demo_dmm()
