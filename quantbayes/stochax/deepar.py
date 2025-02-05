import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
# 1. DeepAR Equinox Module
# --------------------------------------------------------------------------------
class DeepAR(eqx.Module):
    """
    DeepAR module with a single-layer LSTM cell and a Linear head mapping
    hidden state -> (mu, log_sigma).
    """
    lstm: eqx.nn.LSTMCell
    fc: eqx.nn.Linear
    hidden_size: int = eqx.static_field()

    def __init__(self, input_size: int, hidden_size: int, *, key: jnp.ndarray):
        self.hidden_size = hidden_size
        k1, k2 = jr.split(key)
        self.lstm = eqx.nn.LSTMCell(input_size, hidden_size, key=k1)
        # Outputs [mu, log_sigma]
        self.fc = eqx.nn.Linear(hidden_size, 2, key=k2)

    def __call__(self, x: jnp.ndarray, state: tuple[jnp.ndarray, jnp.ndarray]):
        """
        Single-step forward pass.
          x: shape (input_size,)
          state: (h, c) each shape (hidden_size,)
        returns: (mu, log_sigma), (new_h, new_c)
        """
        h, c = self.lstm(x, state)
        out = self.fc(h)  # shape (2,)
        return out, (h, c)

# --------------------------------------------------------------------------------
# 2. Synthetic Data Generation
# --------------------------------------------------------------------------------
def generate_synthetic_timeseries(T: int = 50, noise_std: float = 0.2, seed: int = 0):
    """
    A simple AR(1) synthetic time series:
        x_t = 0.8 * x_{t-1} + noise
    """
    rng = np.random.default_rng(seed)
    x = [0.0]
    for _ in range(1, T):
        x.append(0.8 * x[-1] + rng.normal(scale=noise_std))
    return jnp.array(x)

# --------------------------------------------------------------------------------
# 3. NumPyro Model: Place Priors on LSTM + Linear Weights, then Teacher-Force
# --------------------------------------------------------------------------------
def bayesian_deepar_model(x_seq=None, net=None):
    """
    A fully Bayesian DeepAR:
      - x_seq: shape (T,), the observed time series.
      - net: an *initialized* Equinox DeepAR instance (only used for shapes).

    We:
      1) Sample the LSTM and Linear parameters from Normal(0,prior_scale).
      2) Replace net's parameters with these random draws.
      3) Teacher-force through the time series, sampling x_t ~ Normal(mu_t, sigma_t).
    """
    # Just a small prior scale for demonstration.
    prior_scale = 0.1

    # --------------------------------------------------------------------------
    # A) Manually sample all LSTM parameters from a Normal prior
    #    (weight_ih, weight_hh, bias, etc.)
    # --------------------------------------------------------------------------
    # LSTM
    w_ih = numpyro.sample(
        "lstm_weight_ih",
        dist.Normal(0, prior_scale).expand(net.lstm.weight_ih.shape).to_event(2),
    )
    w_hh = numpyro.sample(
        "lstm_weight_hh",
        dist.Normal(0, prior_scale).expand(net.lstm.weight_hh.shape).to_event(2),
    )
    if net.lstm.use_bias and net.lstm.bias is not None:
        b_lstm = numpyro.sample(
            "lstm_bias",
            dist.Normal(0, prior_scale).expand(net.lstm.bias.shape).to_event(1),
        )
    else:
        b_lstm = None

    # --------------------------------------------------------------------------
    # B) Manually sample all Linear layer parameters from a Normal prior
    # --------------------------------------------------------------------------
    fc_weight = numpyro.sample(
        "fc_weight",
        dist.Normal(0, prior_scale).expand(net.fc.weight.shape).to_event(2),
    )
    if net.fc.use_bias and net.fc.bias is not None:
        fc_bias = numpyro.sample(
            "fc_bias",
            dist.Normal(0, prior_scale).expand(net.fc.bias.shape).to_event(1),
        )
    else:
        fc_bias = None

    # --------------------------------------------------------------------------
    # C) Replace them inside a copy of `net` to get a random-draw network
    # --------------------------------------------------------------------------
    # To do this elegantly, we can use `eqx.tree_at`.
    # We want to fix net.lstm.weight_ih, net.lstm.weight_hh, net.lstm.bias,
    # and similarly net.fc.weight, net.fc.bias.
    bayesian_net = eqx.tree_at(
        lambda m: (m.lstm.weight_ih, m.lstm.weight_hh, m.lstm.bias),
        net,
        (w_ih, w_hh, b_lstm),
    )
    bayesian_net = eqx.tree_at(
        lambda m: (m.fc.weight, m.fc.bias),
        bayesian_net,
        (fc_weight, fc_bias),
    )

    # --------------------------------------------------------------------------
    # D) Forward pass with teacher forcing
    #    x_seq[t] ~ Normal(mu_t, sigma_t)
    # --------------------------------------------------------------------------
    T = x_seq.shape[0] if x_seq is not None else 0
    h = jnp.zeros(net.hidden_size)
    c = jnp.zeros(net.hidden_size)
    x_prev = jnp.atleast_1d(x_seq[0]) if x_seq is not None else jnp.array([0.0])

     # mus = []  # We'll store predicted means for convenience (to visualize)
    # For each time step:
    for t in range(T):
        # One-step forward
        out, (h, c) = bayesian_net(x_prev, (h, c))
        mu_t, log_sigma_t = out
        sigma_t = jnp.exp(log_sigma_t)

        # Record the predicted mean (so that Predictive can retrieve it)
        numpyro.deterministic(f"mu_{t}", mu_t)

        # Sample the observation
        numpyro.sample(f"obs_{t}", dist.Normal(mu_t, sigma_t), obs=x_seq[t])

        # Teacher-forcing: feed the *true* x_t as the next input
        x_prev = jnp.atleast_1d(x_seq[t])

    # No explicit return needed. The samples & deterministics are in the NumPyro trace.
    # If you want to gather them into an array, see "Predictive(...)" usage below.

# --------------------------------------------------------------------------------
# 4. Main: MCMC Inference + Posterior Predictive
# --------------------------------------------------------------------------------
def main():
    # Generate data
    T = 50
    x_seq = generate_synthetic_timeseries(T, noise_std=0.2, seed=42)

    # Initialize an Equinox DeepAR (just for shape definitions).
    # We place a prior on these shapes, but the actual weight values
    # here do NOT matter; they'll be replaced by random samples in the model.
    rng_key = jr.PRNGKey(0)
    hidden_size = 16
    net = DeepAR(input_size=1, hidden_size=hidden_size, key=rng_key)

    # Set up MCMC
    nuts_kernel = NUTS(bayesian_deepar_model, target_accept_prob=0.8)
    mcmc = MCMC(nuts_kernel, num_warmup=300, num_samples=300, num_chains=1)
    mcmc.run(rng_key, x_seq=x_seq, net=net)
    samples = mcmc.get_samples()

    print("Finished MCMC. Posterior sample shapes:")
    for k, v in samples.items():
        print(k, v.shape)

    # --------------------------------------------------------------------------------
    # Posterior Predictive on the training range:
    # We'll get:
    #   "obs_{t}"  => the sampled draws of x_t
    #   "mu_{t}"   => the predicted means
    # from each posterior sample.
    # --------------------------------------------------------------------------------
    predictive = Predictive(
        model=bayesian_deepar_model,
        posterior_samples=samples,
        return_sites=[f"obs_{t}" for t in range(T)] + [f"mu_{t}" for t in range(T)],
    )
    pred_dict = predictive(jr.PRNGKey(1), x_seq=x_seq, net=net)
    # pred_dict is a dict like {"obs_0": array(...), "obs_1": ..., "mu_0": ..., ...}.

    # Each pred_dict["obs_t"] has shape (n_samples,). Each pred_dict["mu_t"] has the same shape.
    # We'll stack them along T for easier plotting.
    # shape => (n_samples, T)
    obs_samples = jnp.column_stack([pred_dict[f"obs_{t}"] for t in range(T)])
    mu_samples = jnp.column_stack([pred_dict[f"mu_{t}"] for t in range(T)])

    # Let's get means + intervals across the posterior draws
    obs_mean = obs_samples.mean(axis=0)
    obs_std = obs_samples.std(axis=0) # noqa

    mu_mean = mu_samples.mean(axis=0)
    mu_std = mu_samples.std(axis=0)

    # --------------------------------------------------------------------------------
    # Plot the results
    # --------------------------------------------------------------------------------
    time_axis = np.arange(T)

    plt.figure(figsize=(10,5))
    plt.plot(time_axis, x_seq, label="True Observations", color="blue")

    # Posterior predictive means
    plt.plot(time_axis, mu_mean, label="Posterior Mean (mu)", color="red")
    # 2-sigma intervals around mu
    plt.fill_between(
        time_axis,
        mu_mean - 2*mu_std,
        mu_mean + 2*mu_std,
        color="red", alpha=0.2,
        label="±2σ around mu"
    )

    # We can also plot the "obs_samples" mean, which includes noise draws.
    # It's often close to mu_mean if the model is well-fit.
    plt.plot(time_axis, obs_mean, label="Posterior Mean of obs", color="green", linestyle="--")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Fully Bayesian DeepAR Fit on Synthetic AR(1) Data")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
