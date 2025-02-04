import jax
import jax.numpy as jnp
import equinox as eqx
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
import optax
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Define a Deep Autoregressive Model using Equinox
# =============================================================================
class DeepAR(eqx.Module):
    """
    A simple deep autoregressive network.
    
    This module uses an LSTMCell to update a hidden state from the previous
    observation and then uses a linear layer to predict the mean of the current
    observation.
    
    For simplicity we assume that each observation is a scalar.
    """
    lstm: eqx.nn.LSTMCell
    fc: eqx.nn.Linear
    hidden_size: int = eqx.static_field()

    def __init__(self, input_size: int, hidden_size: int, *, key):
        # For an LSTMCell, input_size is the dimensionality of the input (here 1)
        # and hidden_size is the size of the hidden state.
        self.hidden_size = hidden_size
        k1, k2 = jax.random.split(key)
        self.lstm = eqx.nn.LSTMCell(input_size, hidden_size, key=k1)
        self.fc = eqx.nn.Linear(hidden_size, 1, key=k2)

    def __call__(self, x: jnp.ndarray, state: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Given an input scalar x (shape (1,)) and state (h, c) (each of shape (hidden_size,)),
        update the state and produce a prediction (a scalar) for the current observation.
        """
        h, c = self.lstm(x, state)  # h, c: (hidden_size,)
        mu = self.fc(h)             # shape (1,)
        return mu, (h, c)

# =============================================================================
# 2. Define the NumPyro Model Using the DeepAR Network
# =============================================================================
def deep_ar_model(x_seq: jnp.ndarray, nn_module: DeepAR):
    """
    A time-series model that uses a deep autoregressive network.
    
    Args:
      x_seq: a 1D array of observations of shape (T,). (For simplicity, we assume
             a single time series of scalar observations.)
      nn_module: an Equinox module (DeepAR) that given the previous observation
                 and hidden state produces the predicted mean.
                 
    The model assumes that
      x_t ~ Normal(mu_t, sigma)
    where mu_t is given by nn_module (autoregressively).
    """
    T = x_seq.shape[0]
    # Learnable noise standard deviation (must be positive)
    sigma = numpyro.param("sigma", 1.0, constraint=dist.constraints.positive)
    # Initialize the LSTM state to zeros.
    h0 = jnp.zeros((nn_module.hidden_size,))
    c0 = jnp.zeros((nn_module.hidden_size,))
    state = (h0, c0)

    # For t=0, we use the observed x_0 as the "previous" input.
    x_prev = x_seq[0:1]  # shape (1,)
    for t in range(T):
        mu_t, state = nn_module(x_prev, state)
        # Observe x_t ~ Normal(mu_t, sigma)
        numpyro.sample(f"obs_{t}", dist.Normal(mu_t.squeeze(), sigma), obs=x_seq[t])
        # Autoregressively use the current observation as input for the next step.
        x_prev = x_seq[t:t+1]

def deep_ar_guide(x_seq: jnp.ndarray, nn_module: DeepAR):
    return AutoDelta(deep_ar_model)(x_seq, nn_module)

# =============================================================================
# 3. Synthetic Data Generation
# =============================================================================
def generate_synthetic_timeseries(T: int, noise_std: float = 0.5, seed: int = 0):
    """
    Generate a simple autoregressive time series.
    x_t = 0.8 * x_{t-1} + noise, with x_0 = 0.
    """
    rng = np.random.default_rng(seed)
    x = [0.0]
    for t in range(1, T):
        x.append(0.8 * x[-1] + rng.normal(scale=noise_std))
    return jnp.array(x)

# =============================================================================
# 4. Training with NumPyro SVI and Forecasting
# =============================================================================
def main():
    # Set a random seed.
    rng_key = jax.random.PRNGKey(0)
    
    # Generate synthetic time-series data.
    T = 50
    x_seq = generate_synthetic_timeseries(T, noise_std=0.5, seed=42)
    
    # Initialize our deep autoregressive network.
    # Here input_size = 1 (scalar observations), hidden_size chosen arbitrarily.
    hidden_size = 16
    nn_key, rng_key = jax.random.split(rng_key)
    deep_ar_net = DeepAR(input_size=1, hidden_size=hidden_size, key=nn_key)
    
    # Setup SVI with the model and an AutoDelta guide.
    optimizer = optax.adam(1e-2)
    svi = SVI(lambda x_seq: deep_ar_model(x_seq, deep_ar_net),
              lambda x_seq: deep_ar_guide(x_seq, deep_ar_net),
              optimizer,
              loss=Trace_ELBO())
    
    # Run SVI
    svi_state = svi.init(rng_key, x_seq)
    num_steps = 100
    losses = []
    for step in range(num_steps):
        rng_key, subkey = jax.random.split(rng_key)
        svi_state, loss = svi.update(svi_state, x_seq)
        losses.append(loss)
        if (step + 1) % 10 == 0:
            print(f"Step {step+1} loss: {loss:.3f}")
    
    params = svi.get_params(svi_state)
    
    # Plot the training loss.
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()
    
    # Plot the observed time series.
    plt.figure(figsize=(8, 4))
    plt.plot(x_seq, label="Observed")
    plt.xlabel("Time step")
    plt.ylabel("x")
    plt.title("Synthetic Time Series (Observed)")
    plt.legend()
    plt.show()
    
    # =============================================================================
    # Forecasting: Produce predictions using the trained network.
    # =============================================================================
    # Extract the learned noise level from the parameters.
    sigma = params["sigma"]
    print("Learned sigma:", sigma)
    
    # For forecasting we use the trained deep_ar_net.
    # We initialize the LSTM state with zeros and feed the first observation,
    # then for each subsequent step we feed back the predicted mean.
    state = (jnp.zeros((deep_ar_net.hidden_size,)), jnp.zeros((deep_ar_net.hidden_size,)))
    x_prev = x_seq[0:1]  # Start with the first observed value.
    forecast = []
    for t in range(T):
        mu_t, state = deep_ar_net(x_prev, state)
        forecast.append(mu_t.squeeze())
        # For forecasting we feed the predicted mean as the next input.
        x_prev = mu_t  
    forecast = jnp.array(forecast)
    
    # Plot the observed series with the forecast overlaid.
    plt.figure(figsize=(8, 4))
    plt.plot(x_seq, label="Observed")
    plt.plot(forecast, label="Forecast", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("x")
    plt.title("Observed Time Series and Forecast")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
