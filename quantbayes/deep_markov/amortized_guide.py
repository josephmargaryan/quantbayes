import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from flax import linen as nn
from typing import Any, Dict, Tuple
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam

##############################################################################
# 1) MODEL (SAME AS BEFORE)
##############################################################################

class MLPTransition(nn.Module):
    hidden_dim: int
    z_dim: int
    
    @nn.compact
    def __call__(self, z_prev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.relu(nn.Dense(self.hidden_dim)(z_prev))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        out = nn.Dense(2 * self.z_dim)(x)
        mu, log_sigma = jnp.split(out, 2, axis=-1)
        return mu, log_sigma

class MLPEmission(nn.Module):
    hidden_dim: int
    x_dim: int
    z_dim: int

    @nn.compact
    def __call__(self, z_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.relu(nn.Dense(self.hidden_dim)(z_t))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        out = nn.Dense(2 * self.x_dim)(x)
        mu, log_sigma = jnp.split(out, 2, axis=-1)
        return mu, log_sigma

def apply_mlp(params: Dict, module: nn.Module, x: jnp.ndarray):
    return module.apply({"params": params}, x)

def dmm_model(
    X: jnp.ndarray,
    transition_init: Dict[str, Any],
    emission_init: Dict[str, Any],
    z_dim: int
):
    """
    MLP-based DMM:
      - At time t, z_t ~ N(mu_z, sigma_z), 
        where (mu_z, sigma_z) = TransitionMLP(z_{t-1}).
      - x_t ~ N(mu_x, sigma_x), 
        where (mu_x, sigma_x) = EmissionMLP(z_t).
    """
    batch_size, T, x_dim = X.shape

    # Register
    trans_params = {k: numpyro.param("trans_"+k, v) 
                    for k,v in transition_init["params"].items()}
    trans_module = transition_init["module"]
    emis_params  = {k: numpyro.param("emis_"+k, v)
                    for k,v in emission_init["params"].items()}
    emis_module  = emission_init["module"]
    
    # z_0
    z_0 = numpyro.sample(
        "z_0",
        dist.Normal(jnp.zeros((batch_size, z_dim)),
                    jnp.ones((batch_size, z_dim)))
            .to_event(1)
    )
    z_prev = z_0

    for t in range(T):
        if t == 0:
            z_t = z_prev
        else:
            mu_z, log_sigma_z = apply_mlp(trans_params, trans_module, z_prev)
            sigma_z = jnp.exp(log_sigma_z)
            z_t = numpyro.sample(f"z_{t}", dist.Normal(mu_z, sigma_z).to_event(1))
        
        mu_x, log_sigma_x = apply_mlp(emis_params, emis_module, z_t)
        sigma_x = jnp.exp(log_sigma_x)
        numpyro.sample(f"x_{t}",
                       dist.Normal(mu_x, sigma_x).to_event(1),
                       obs=X[:, t])
        
        z_prev = z_t

##############################################################################
# 2) AMORTIZED GUIDE: LSTM Encoder
##############################################################################

class LSTMEncoder(nn.Module):     ## <-- Im going to try and use a transformer instead of a LSTM in the future
    """
    Amortized encoder:
      LSTM reads x_{1..T}, 
      at each step outputs mu_t, log_sigma_t for z_t.
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
        
        # Define submodules once
        lstm_cell = nn.LSTMCell(features=self.hidden_dim)
        dense1 = nn.Dense(features=self.hidden_dim)
        dense2 = nn.Dense(features=2 * self.z_dim)
        
        # Initialize carry
        carry = lstm_cell.initialize_carry(random.PRNGKey(0), (batch_size, self.hidden_dim))
        
        mu_list = []
        log_sigma_list = []
        
        # Unroll LSTM over time
        for t in range(T):
            x_t = x_seq[:, t, :]  # shape (batch_size, x_dim)
            carry, h = lstm_cell(carry, x_t)  # h shape => (batch, hidden_dim)
            
            # produce (mu_t, log_sigma_t) from h using defined Dense layers
            x = nn.relu(dense1(h))
            out = dense2(x)
            mu_t, log_sigma_t = jnp.split(out, 2, axis=-1)
            
            mu_list.append(mu_t)             # each shape (batch, z_dim)
            log_sigma_list.append(log_sigma_t)
        
        # Stack across time => shape (batch, T, z_dim)
        mu_seq        = jnp.stack(mu_list, axis=1)        # (batch, T, z_dim)
        log_sigma_seq = jnp.stack(log_sigma_list, axis=1) # (batch, T, z_dim)
        return mu_seq, log_sigma_seq

def amortized_guide(
    X: jnp.ndarray,
    transition_init: Dict[str, Any],
    emission_init: Dict[str, Any],
    encoder_init: Dict[str, Any],
    z_dim: int
):
    """
    A guide that uses an LSTMEncoder over X to produce 
    q(z_t | x_{1..T}) = Normal(mu_{t}, sigma_{t}).
    """
    batch_size, T, x_dim = X.shape
    
    # Register encoder parameters
    encoder_params = {
        k: numpyro.param("encoder_"+k, v)
        for k, v in encoder_init["params"].items()
    }
    encoder_module = encoder_init["module"]
    
    # Run the LSTM encoder
    mu_seq, log_sigma_seq = encoder_module.apply({"params": encoder_params}, X)
    # shape: (batch, T, z_dim)

    # Sample z_0.. z_{T-1}
    z_0 = numpyro.sample(
        "z_0",
        dist.Normal(mu_seq[:, 0], jnp.exp(log_sigma_seq[:, 0]))
            .to_event(1)  # treat z_dim as event dim
    )
    
    for t in range(1, T):
        z_t = numpyro.sample(
            f"z_{t}",
            dist.Normal(mu_seq[:, t], jnp.exp(log_sigma_seq[:, t]))
                .to_event(1)
        )


def make_encoder_inits(rng, hidden_dim, x_dim, z_dim):
    encoder = LSTMEncoder(hidden_dim=hidden_dim, x_dim=x_dim, z_dim=z_dim)
    # We need a dummy input shape for (batch=1, T=2, x_dim)
    dummy_x = jnp.zeros((1, 2, x_dim))
    params = encoder.init(rng, dummy_x)
    return {
        "params": params["params"],
        "module": encoder
    }

def train_amortized_dmm(
    X: jnp.ndarray,
    z_dim: int = 2,
    hidden_dim: int = 16,
    num_steps: int = 1000
):
    """
    X: (batch_size, T, x_dim)

    We'll keep the same MLP-based transition/emission, 
    but the guide is an LSTM encoder over X.
    """
    rng = random.PRNGKey(0)
    rt, re, renc, rsvi = random.split(rng, 4)
    batch_size, T, x_dim = X.shape

    # 1) Initialize MLP-based transition + emission (for the model)
    trans_module = MLPTransition(hidden_dim=hidden_dim, z_dim=z_dim)
    dummy_z      = jnp.zeros((1, z_dim))
    trans_params = trans_module.init(rt, dummy_z)

    emis_module = MLPEmission(hidden_dim=hidden_dim, x_dim=x_dim, z_dim=z_dim)
    emis_params = emis_module.init(re, dummy_z)

    transition_init = {"params": trans_params["params"], "module": trans_module}
    emission_init   = {"params": emis_params["params"], "module": emis_module}

    # 2) Initialize LSTM encoder for the guide
    encoder_init = make_encoder_inits(renc, hidden_dim, x_dim, z_dim)

    # 3) Define partial model / guide
    def model_fn(X_):
        return dmm_model(X_, transition_init, emission_init, z_dim)

    def guide_fn(X_):
        return amortized_guide(X_, transition_init, emission_init, encoder_init, z_dim)

    # 4) SVI
    optimizer = Adam(1e-3)
    svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
    svi_state = svi.init(rsvi, X)

    losses = []
    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state, X)
        losses.append(loss)
        if step % 200 == 0:
            print(f"[Amortized-DMM] step={step}, ELBO={-loss:.2f}")
    return losses

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

def demo_amortized_dmm():
    X, Z = synthetic_data_dmm(batch_size=20, T=15, x_dim=3, z_dim=2)
    print("Data shape:", X.shape)
    losses = train_amortized_dmm(X, z_dim=2, hidden_dim=16, num_steps=1000)
    print("Final Amortized-DMM ELBO:", -losses[-1])

if __name__ == "__main__":
    demo_amortized_dmm()
