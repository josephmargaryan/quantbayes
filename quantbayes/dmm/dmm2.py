import jax
import jax.numpy as jnp
from jax import random

import flax.linen as nn
import flax.linen.initializers as init

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
from numpyro.contrib.module import flax_module

import matplotlib.pyplot as plt
import numpy as np


##############################################
# 1. Flax Modules for the Deep Markov Model #
##############################################

class GatedTransition(nn.Module):
    """
    Parameterizes p(z_t | z_{t-1}) as a diagonal Gaussian:
      z_t ~ Normal(loc, scale).

    A gating mechanism mixes identity(z_{t-1}) with a learned 'proposed_mean'.
    """
    z_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, z_prev: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_prev: (batch_size, z_dim)
        Returns: (loc, scale) each shape (batch_size, z_dim)
        """
        relu = nn.relu
        sigmoid = nn.sigmoid
        softplus = nn.softplus

        gate_hidden = relu(nn.Dense(self.hidden_dim)(z_prev))
        gate = sigmoid(nn.Dense(self.z_dim)(gate_hidden))

        prop_hidden = relu(nn.Dense(self.hidden_dim)(z_prev))
        proposed_mean = nn.Dense(self.z_dim)(prop_hidden)

        loc = (1.0 - gate) * z_prev + gate * proposed_mean

        scale_hidden = relu(proposed_mean)
        scale = softplus(nn.Dense(self.z_dim)(scale_hidden)) + 1e-4
        return loc, scale


class Emitter(nn.Module):
    """
    Parameterizes p(x_t | z_t) as a diagonal Gaussian:
      x_t ~ Normal(loc, scale).
    """
    z_dim: int
    emission_dim: int
    x_dim: int

    @nn.compact
    def __call__(self, z_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        z_t: (batch_size, z_dim)
        Returns: (loc, scale) each (batch_size, x_dim)
        """
        relu = nn.relu
        softplus = nn.softplus

        hidden = relu(nn.Dense(self.emission_dim)(z_t))
        hidden = relu(nn.Dense(self.emission_dim)(hidden))
        out = nn.Dense(2 * self.x_dim)(hidden)
        loc, log_scale = jnp.split(out, 2, axis=-1)
        scale = softplus(log_scale) + 1e-4
        return loc, scale


class RNNEncoder(nn.Module):
    """
    Single-layer LSTM that processes x_{1:T} and returns hidden states at each step.
    """
    rnn_dim: int
    x_dim: int

    @nn.compact
    def __call__(self, x_seq: jnp.ndarray) -> jnp.ndarray:
        """
        x_seq: (batch, T, x_dim)
        returns rnn_outputs: (batch, T, rnn_dim)
        """
        batch_size, T, _ = x_seq.shape

        # Define a scanned LSTMCell
        lstm_scan = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
        )(features=self.rnn_dim)

        # Correctly initialize carry (c, h) with shape (batch_size, rnn_dim)
        carry = lstm_scan.initialize_carry(
            jax.random.PRNGKey(0),
            (batch_size,)  # Correct batch_dims
        )

        outputs = []
        for time_idx in range(T):
            carry, h_t = lstm_scan(carry, x_seq[:, time_idx, :])
            outputs.append(h_t)

        rnn_outputs = jnp.stack(outputs, axis=1)  # shape (batch, T, rnn_dim)
        return rnn_outputs


class Combiner(nn.Module):
    """
    Parameterizes q(z_t | [z_{t-1}, h_t]) => (loc, scale).
    We'll pass a single array x = concat(z_prev, h_t).
    """
    z_dim: int
    rnn_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        x: shape (batch, z_dim + rnn_dim)
        returns: (loc, scale) each shape (batch, z_dim)
        """
        tanh = nn.tanh
        softplus = nn.softplus

        hidden = nn.Dense(self.rnn_dim)(x)
        hidden = tanh(hidden)
        loc = nn.Dense(self.z_dim)(hidden)
        scale = softplus(nn.Dense(self.z_dim)(hidden)) + 1e-4
        return loc, scale


#########################################################
# 2. The Model & Guide for the DMM + Synthetic Data Gen #
#########################################################

def model(
    x_seq: jnp.ndarray,
    z_dim: int,
    transition_dim: int,
    emission_dim: int,
    rnn_dim: int
):
    """
    p(z_{1:T}, x_{1:T}):
      - z_1 ~ Normal(z_0_loc, z_0_scale)
      - z_t ~ GatedTransition(z_{t-1})
      - x_t ~ Emitter(z_t)
    """
    batch_size, T, x_dim = x_seq.shape

    # Register submodules with flax_module
    trans_mod = flax_module("trans",
                            GatedTransition(z_dim=z_dim, hidden_dim=transition_dim),
                            input_shape=(1, z_dim))
    emit_mod  = flax_module("emit",
                            Emitter(z_dim=z_dim, emission_dim=emission_dim, x_dim=x_dim),
                            input_shape=(1, z_dim))

    # Trainable initial-latent distribution
    z_0_loc = numpyro.param("z_0_loc", jnp.zeros(z_dim))
    z_0_scale = numpyro.param("z_0_scale",
                              0.1 * jnp.ones(z_dim),
                              constraint=dist.constraints.positive)

    with numpyro.plate("sequences", batch_size):
        # z_1
        z_prev = numpyro.sample(
            "z_1",
            dist.Normal(z_0_loc, z_0_scale).to_event(1)
        )
        # x_1
        x_loc_1, x_scale_1 = emit_mod(z_prev)
        numpyro.sample(
            "x_1",
            dist.Normal(x_loc_1, x_scale_1).to_event(1),
            obs=x_seq[:, 0, :]
        )

        # unroll from step=1..T-1 => produce z_{2..T}, x_{2..T}
        for step in range(1, T):
            loc_z, scale_z = trans_mod(z_prev)
            z_t = numpyro.sample(
                f"z_{step+1}",
                dist.Normal(loc_z, scale_z).to_event(1)
            )

            x_loc, x_scale = emit_mod(z_t)
            numpyro.sample(
                f"x_{step+1}",
                dist.Normal(x_loc, x_scale).to_event(1),
                obs=x_seq[:, step, :]
            )
            z_prev = z_t


def guide(
    x_seq: jnp.ndarray,
    z_dim: int,
    transition_dim: int,
    emission_dim: int,
    rnn_dim: int
):
    """
    q(z_{1:T} | x_{1:T}):
      - RNNEncoder => h_{t}
      - z_1 ~ Normal(z_q0_loc, z_q0_scale)
      - z_t ~ Normal(Combiner(concat(z_{t-1}, h_t)))
    """
    batch_size, T, x_dim = x_seq.shape

    enc_mod = flax_module("enc",
                          RNNEncoder(rnn_dim=rnn_dim, x_dim=x_dim),
                          input_shape=(1, T, x_dim))
    comb_mod = flax_module("comb",
                           Combiner(z_dim=z_dim, rnn_dim=rnn_dim),
                           input_shape=(1, z_dim + rnn_dim))

    # Trainable init q(z_1)
    z_q0_loc = numpyro.param("z_q0_loc", jnp.zeros(z_dim))
    z_q0_scale = numpyro.param("z_q0_scale",
                               0.1 * jnp.ones(z_dim),
                               constraint=dist.constraints.positive)

    # RNN => h_{t}
    rnn_out = enc_mod(x_seq)  # shape (batch, T, rnn_dim)

    with numpyro.plate("sequences", batch_size):
        # z_1
        z_prev = numpyro.sample(
            "z_1",
            dist.Normal(z_q0_loc, z_q0_scale).to_event(1)
        )

        # unroll from step=1..T-1 => produce z_{2..T}
        for step in range(1, T):
            h_t = rnn_out[:, step, :]  # shape (batch, rnn_dim)
            concat_z_h = jnp.concatenate([z_prev, h_t], axis=-1)  # (batch, z_dim+rnn_dim)
            loc_q, scale_q = comb_mod(concat_z_h)

            z_t = numpyro.sample(
                f"z_{step+1}",
                dist.Normal(loc_q, scale_q).to_event(1)
            )
            z_prev = z_t


########################################
# 3. Synthetic Data + SVI Training     #
########################################

def generate_synthetic_data(rng_key, n_sequences=32, T=10, x_dim=2, z_dim=2):
    """
    Generates synthetic data with linear transitions and emissions.
    """
    # Define true parameters for transitions and emissions
    true_trans_loc = jnp.array([0.5, -0.5])  # Bias for z_t
    true_trans_scale = jnp.array([0.8, 0.8])  # Scaling for z_{t-1} to z_t
    true_emit_loc = jnp.array([1.0, -1.0])    # Bias for x_t
    true_emit_scale = jnp.array([1.2, 1.2])   # Scaling for z_t to x_t

    x_list = []
    z_list = []

    for seq_idx in range(n_sequences):
        rng_key, subkey = random.split(rng_key)
        z_prev = random.normal(subkey, shape=(z_dim,))  # Initialize z_1
        seq_x = []
        seq_z = [z_prev]

        for t in range(T):
            rng_key, subkey = random.split(rng_key)
            z_t = true_trans_loc + true_trans_scale * z_prev + 0.1 * random.normal(subkey, shape=(z_dim,))
            rng_key, subkey = random.split(rng_key)
            x_t = true_emit_loc + true_emit_scale * z_t + 0.1 * random.normal(subkey, shape=(x_dim,))
            seq_z.append(z_t)
            seq_x.append(x_t)
            z_prev = z_t

        z_list.append(jnp.stack(seq_z[:-1]))  # Shape: (T, z_dim)
        x_list.append(jnp.stack(seq_x))        # Shape: (T, x_dim)

    x_data = jnp.stack(x_list, axis=0)  # Shape: (n_sequences, T, x_dim)
    z_data_true = jnp.stack(z_list, axis=0)  # Shape: (n_sequences, T, z_dim)
    return x_data, z_data_true



def train_dmm(
    x_data,
    z_dim=2,
    transition_dim=8,
    emission_dim=8,
    rnn_dim=8,
    num_steps=1000,
    lr=1e-3,
    seed=0
):
    """
    Runs SVI with model(...) / guide(...).
    Returns (trained_params, losses).
    """
    def model_fn(x):
        return model(x, z_dim, transition_dim, emission_dim, rnn_dim)

    def guide_fn(x):
        return guide(x, z_dim, transition_dim, emission_dim, rnn_dim)

    optimizer = Adam(lr)
    svi = SVI(model_fn, guide_fn, optimizer, Trace_ELBO())

    rng_key = random.PRNGKey(seed)
    svi_state = svi.init(rng_key, x_data)

    losses = []
    for step in range(num_steps):
        svi_state, loss_val = svi.update(svi_state, x_data)
        losses.append(loss_val)

        if step % max(1, num_steps // 5) == 0:
            print(f"[Step {step}]  ELBO = {-loss_val:.4f}")

    trained_params = svi.get_params(svi_state)
    return trained_params, np.array(losses)


#####################################
# 4. Posterior Means & Plot Helpers #
#####################################

def posterior_means(
    trained_params,
    x_data: jnp.ndarray,
    z_dim: int,
    transition_dim: int,
    emission_dim: int,
    rnn_dim: int
):
    """
    We'll do a manual pass through the guide to get q(z_t) means:
      1) RNN => h_t
      2) z_1 = z_q0_loc
      3) z_t = combiner([z_{t-1}, h_t]).loc
    """
    enc_def = RNNEncoder(rnn_dim=rnn_dim, x_dim=x_data.shape[-1])
    comb_def = Combiner(z_dim=z_dim, rnn_dim=rnn_dim)

    enc_params = trained_params["enc$params"]
    comb_params = trained_params["comb$params"]

    # q(z_1)
    z_q0_loc = trained_params["z_q0_loc"]
    # ignoring z_q0_scale for the posterior mean => z_1 is just z_q0_loc

    batch_size, T, _ = x_data.shape

    # 1) RNN
    rnn_out = enc_def.apply({"params": enc_params}, x_data)

    z_means = []

    # 2) z_1
    z_1_mean = jnp.tile(z_q0_loc[None, :], (batch_size, 1))
    z_prev = z_1_mean
    z_means.append(z_1_mean)

    # unroll for t=2..T
    for step in range(1, T):
        h_t = rnn_out[:, step, :]  # (batch, rnn_dim)
        concat_z_h = jnp.concatenate([z_prev, h_t], axis=-1)
        loc_q, scale_q = comb_def.apply({"params": comb_params}, concat_z_h)
        z_t_mean = loc_q
        z_means.append(z_t_mean)
        z_prev = z_t_mean

    # shape => (T, batch, z_dim) => (batch, T, z_dim)
    z_means = jnp.stack(z_means, axis=0).transpose((1,0,2))
    return z_means


#####################################
# 5. Main Demo
#####################################

def main():
    rng_key = random.PRNGKey(0)

    # Hyperparameters
    n_sequences = 32
    T = 10
    x_dim = 2
    z_dim = 2
    trans_dim = 8
    emis_dim = 8
    rnn_dim = 8
    n_steps = 100

    # 1) Generate data
    x_data, z_data_true = generate_synthetic_data(
        rng_key
    )
    print(f"x_data shape: {x_data.shape}, z_data_true shape: {z_data_true.shape}")

    # 2) Train
    trained_params, losses = train_dmm(
        x_data,
        z_dim=z_dim,
        transition_dim=trans_dim,
        emission_dim=emis_dim,
        rnn_dim=rnn_dim,
        num_steps=n_steps,
        lr=1e-3,
        seed=42
    )

    # 3) Plot training curve
    plt.figure(figsize=(6,4))
    plt.plot(losses, label="Negative ELBO")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("DMM Training Curve")
    plt.legend()
    plt.show()

    # 4) Posterior means
    z_post_means = posterior_means(trained_params, x_data,
                                   z_dim=z_dim,
                                   transition_dim=trans_dim,
                                   emission_dim=emis_dim,
                                   rnn_dim=rnn_dim)

    # Compare single sequence latent
    seq_idx = 0
    plt.figure(figsize=(8,4))
    for dim_i in range(z_dim):
        plt.plot(z_data_true[seq_idx, :, dim_i], label=f"True z[{dim_i}]")
        plt.plot(z_post_means[seq_idx, :, dim_i], "--", label=f"Posterior z[{dim_i}]")
    plt.title(f"Latent states (true vs. posterior mean), seq={seq_idx}")
    plt.legend()
    plt.show()

    # 5) Compare reconstructions of x
    emit_params = trained_params["emit$params"]
    emitter_def = Emitter(z_dim=z_dim, emission_dim=emis_dim, x_dim=x_dim)

    recons = []
    for t in range(T):
        loc_x, scale_x = emitter_def.apply({"params": emit_params}, z_post_means[:, t, :])
        recons.append(loc_x)
    recons = jnp.stack(recons, axis=1)  # (batch, T, x_dim)

    fig, axes = plt.subplots(1, x_dim, figsize=(12,4))
    if x_dim == 1:
        axes = [axes]
    for dim_i in range(x_dim):
        ax = axes[dim_i]
        ax.plot(x_data[seq_idx, :, dim_i], label=f"x true dim={dim_i}")
        ax.plot(recons[seq_idx, :, dim_i], '--', label=f"x recon dim={dim_i}")
        ax.set_title(f"Dimension {dim_i}")
        ax.legend()
    plt.suptitle("True vs. Reconstructed Observations (sequence 0)")
    plt.show()

    def visualize_latents_all(z_true, z_post):
        # Flatten the batch and time dimensions
        z_true_flat = z_true.reshape(-1, z_true.shape[-1])    # Shape: (n_sequences*T, z_dim)
        z_post_flat = z_post.reshape(-1, z_post.shape[-1])    # Shape: (n_sequences*T, z_dim)
        
        plt.figure(figsize=(8,8))
        plt.scatter(z_true_flat[:,0], z_true_flat[:,1], label='True Latent', alpha=0.5, color='blue')
        plt.scatter(z_post_flat[:,0], z_post_flat[:,1], label='Posterior Mean', alpha=0.5, color='red')
        plt.legend()
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.title("Latent Space Comparison Across All Sequences")
        plt.show()

    visualize_latents_all(z_data_true, z_post_means)


if __name__ == "__main__":
    main()
