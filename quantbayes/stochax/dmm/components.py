# dmm_components.py
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt 
import numpy as np
import optax

from quantbayes.stochax.dmm.base import (BaseCombiner, BaseDMM, BaseEmission,
                                         BasePosterior, BaseTransition)

#######################################
# 1. TRANSITION COMPONENTS
#######################################

class LinearTransition(eqx.Module, BaseTransition):
    """
    A simple linear transition for p(z_t | z_{t-1}).
    It applies a linear mapping to z_prev and adds a learned bias.
    """
    weight: jnp.ndarray  # shape: (latent_dim, latent_dim)
    bias: jnp.ndarray    # shape: (latent_dim,)
    log_scale: jnp.ndarray  # shape: (latent_dim,) constant (learned)

    def __init__(self, latent_dim: int, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.weight = jax.random.normal(k1, (latent_dim, latent_dim)) * 0.1
        self.bias = jax.random.normal(k2, (latent_dim,)) * 0.1
        # Initialize log_scale near zero (i.e. scale ~1)
        self.log_scale = jax.random.normal(k3, (latent_dim,)) * 0.01

    def __call__(self, z_prev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # z_prev: (batch, latent_dim)
        loc = jnp.dot(z_prev, self.weight) + self.bias
        # Use the same scale for each example (broadcast along batch)
        scale = jnp.exp(self.log_scale)
        # Return (loc, scale) with shape (batch, latent_dim)
        return loc, jnp.broadcast_to(scale, loc.shape)


class MLPTransition(eqx.Module, BaseTransition):
    """
    An MLP-based transition. It takes z_prev and maps it to 2*latent_dim values,
    which are interpreted as the mean and log-scale.
    """
    mlp: eqx.nn.MLP
    latent_dim: int

    def __init__(self, latent_dim: int, hidden_dim: int, *, key):
        self.latent_dim = latent_dim
        self.mlp = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=2 * latent_dim,
            width_size=hidden_dim,
            depth=2,
            key=key
        )

    def __call__(self, z_prev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # z_prev: (batch, latent_dim)
        # Apply the MLP on each example
        out = jax.vmap(self.mlp)(z_prev)  # (batch, 2*latent_dim)
        loc, log_scale = jnp.split(out, 2, axis=-1)
        scale = jnp.exp(log_scale)
        return loc, scale

#######################################
# 2. EMISSION COMPONENTS
#######################################

class MLPEmission(eqx.Module, BaseEmission):
    """
    An MLP-based emission that maps a latent variable z
    to parameters for p(x_t | z_t). It outputs 2*observation_dim values,
    interpreted as (loc, log_scale).
    """
    mlp: eqx.nn.MLP
    observation_dim: int

    def __init__(self, latent_dim: int, hidden_dim: int, observation_dim: int, *, key):
        self.observation_dim = observation_dim
        self.mlp = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=2 * observation_dim,
            width_size=hidden_dim,
            depth=2,
            key=key
        )

    def __call__(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # z: (batch, latent_dim)
        out = jax.vmap(self.mlp)(z)  # (batch, 2*observation_dim)
        loc, log_scale = jnp.split(out, 2, axis=-1)
        scale = jnp.exp(log_scale)
        return loc, scale

#######################################
# 3. POSTERIOR COMPONENTS
#######################################

class LSTMPosterior(eqx.Module, BasePosterior):
    """
    A simple LSTM-based posterior network.
    It processes a sequence of observations (batch, T, observation_dim)
    and returns hidden states (batch, T, hidden_dim).
    """
    lstm: eqx.nn.LSTMCell
    hidden_dim: int
    observation_dim: int

    def __init__(self, observation_dim: int, hidden_dim: int, *, key):
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.lstm = eqx.nn.LSTMCell(input_size=observation_dim, hidden_size=hidden_dim, key=key)

    def __call__(self, x_seq: jnp.ndarray) -> jnp.ndarray:
        """
        x_seq: (batch, T, observation_dim)
        Returns: hidden states h_seq of shape (batch, T, hidden_dim)
        """
        batch, T, _ = x_seq.shape
        # Initialize hidden and cell states with zeros
        h = jnp.zeros((batch, self.hidden_dim))
        c = jnp.zeros((batch, self.hidden_dim))
        h_list = []
        # Loop over time steps (use jax.lax.scan in production for speed)
        for t in range(T):
            x_t = x_seq[:, t, :]  # (batch, observation_dim)
            # Use vmap over the batch dimension
            h, c = jax.vmap(self.lstm)(x_t, (h, c))
            h_list.append(h)
        # Stack h_list along time axis: (T, batch, hidden_dim) then transpose to (batch, T, hidden_dim)
        h_seq = jnp.stack(h_list, axis=1)
        return h_seq

#######################################
# 4. COMBINER COMPONENTS
#######################################

class MLPCombiner(eqx.Module, BaseCombiner):
    """
    An MLP-based combiner that fuses the previous latent state and the current hidden state
    from the posterior RNN to yield the parameters for q(z_t | z_{t-1}, h_t).
    """
    mlp: eqx.nn.MLP
    latent_dim: int
    hidden_dim: int

    def __init__(self, latent_dim: int, hidden_dim: int, *, key):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        # Input is concatenation of z_prev and h (size: latent_dim + hidden_dim)
        # Output is 2*latent_dim (for loc and log_scale)
        self.mlp = eqx.nn.MLP(
            in_size=latent_dim + hidden_dim,
            out_size=2 * latent_dim,
            width_size=hidden_dim,
            depth=2,
            key=key
        )

    def __call__(self, z_prev: jnp.ndarray, h: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Both z_prev and h are (batch, …)
        inp = jnp.concatenate([z_prev, h], axis=-1)  # (batch, latent_dim + hidden_dim)
        out = jax.vmap(self.mlp)(inp)  # (batch, 2*latent_dim)
        loc, log_scale = jnp.split(out, 2, axis=-1)
        scale = jnp.exp(log_scale)
        return loc, scale





class DMM(eqx.Module, BaseDMM):
    """
    A simple Deep Markov Model that uses:
      - A learned prior for z₁.
      - A transition network for p(z_t | zₜ₋₁).
      - An emission network for p(x_t | z_t).
      - A posterior RNN to get hidden states from x_seq.
      - A combiner to yield q(z_t | zₜ₋₁, h_t).
    """
    transition: eqx.Module  # e.g. LinearTransition or MLPTransition
    emission: eqx.Module    # e.g. MLPEmission
    posterior: eqx.Module   # e.g. LSTMPosterior
    combiner: eqx.Module    # e.g. MLPCombiner
    latent_dim: int
    # Learned prior parameters for z₁:
    z1_loc: jnp.ndarray
    z1_logscale: jnp.ndarray

    def __init__(
        self,
        observation_dim: int,
        latent_dim: int,
        hidden_dim: int,
        transition_type: str = "mlp",  # or "linear"
        *,
        key
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.latent_dim = latent_dim
        if transition_type == "linear":
            self.transition = LinearTransition(latent_dim, key=k1)
        else:
            self.transition = MLPTransition(latent_dim, hidden_dim, key=k1)
        self.emission = MLPEmission(latent_dim, hidden_dim, observation_dim, key=k2)
        self.posterior = LSTMPosterior(observation_dim, hidden_dim, key=k3)
        self.combiner = MLPCombiner(latent_dim, hidden_dim, key=k4)
        # Initialize learned prior for z₁
        self.z1_loc = jnp.zeros((latent_dim,))
        self.z1_logscale = jnp.zeros((latent_dim,))  # so that scale = exp(0) = 1

    def reparam_sample(self, rng, loc: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        eps = jax.random.normal(rng, shape=loc.shape)
        return loc + scale * eps

    def __call__(self, x_seq: jnp.ndarray, rng) -> jnp.ndarray:
        """
        x_seq: (batch, T, observation_dim)
        Returns: negative ELBO (averaged over batch)
        """
        batch, T, _ = x_seq.shape
        # Obtain posterior hidden states via the posterior network:
        h_seq = self.posterior(x_seq)  # shape (batch, T, hidden_dim)
        rngs = jax.random.split(rng, T)
        # Process time step 1:
        # Use a fixed "dummy" previous latent of zeros for t=1:
        z0 = jnp.zeros((batch, self.latent_dim))
        q1_loc, q1_scale = self.combiner(z0, h_seq[:, 0, :])
        z1 = self.reparam_sample(rngs[0], q1_loc, q1_scale)
        # Compute log probabilities for z1:
        # Prior: p(z1) ~ N(z1_loc (learned), exp(z1_logscale))
        prior_scale = jnp.exp(self.z1_logscale)
        lp_z1 = -0.5 * jnp.sum(((z1 - self.z1_loc) / prior_scale) ** 2 + 2 * jnp.log(prior_scale) + jnp.log(2 * jnp.pi), axis=-1)
        lq_z1 = -0.5 * jnp.sum(((z1 - q1_loc) / q1_scale) ** 2 + 2 * jnp.log(q1_scale) + jnp.log(2 * jnp.pi), axis=-1)
        log_w = jnp.sum(lp_z1 - lq_z1)
        # Emission log prob at t=1:
        x1 = x_seq[:, 0, :]
        e_loc, e_scale = self.emission(z1)
        lp_x1 = -0.5 * jnp.sum(((x1 - e_loc) / e_scale) ** 2 + 2 * jnp.log(e_scale) + jnp.log(2 * jnp.pi), axis=-1)
        log_w += jnp.sum(lp_x1)
        z_prev = z1
        # Process time steps t=2,...,T:
        for t in range(1, T):
            # Get combiner output from previous latent and h_t:
            qt_loc, qt_scale = self.combiner(z_prev, h_seq[:, t, :])
            z_t = self.reparam_sample(rngs[t], qt_loc, qt_scale)
            # Transition: p(z_t | z_prev)
            pt_loc, pt_scale = self.transition(z_prev)
            lp_zt = -0.5 * jnp.sum(((z_t - pt_loc) / pt_scale) ** 2 + 2 * jnp.log(pt_scale) + jnp.log(2 * jnp.pi), axis=-1)
            lq_zt = -0.5 * jnp.sum(((z_t - qt_loc) / qt_scale) ** 2 + 2 * jnp.log(qt_scale) + jnp.log(2 * jnp.pi), axis=-1)
            log_w += jnp.sum(lp_zt - lq_zt)
            # Emission: p(x_t | z_t)
            x_t = x_seq[:, t, :]
            e_loc, e_scale = self.emission(z_t)
            lp_xt = -0.5 * jnp.sum(((x_t - e_loc) / e_scale) ** 2 + 2 * jnp.log(e_scale) + jnp.log(2 * jnp.pi), axis=-1)
            log_w += jnp.sum(lp_xt)
            z_prev = z_t
        neg_elbo = -log_w / batch
        return neg_elbo


class DeepMarkovModel(DMM):
    """
    A Deep Markov Model that encapsulates its own training and reconstruction logic.
    
    Inherits from DMM (which implements __call__ to compute the negative ELBO)
    and adds:
      - fit(): train the model given a dataset of sequences.
      - reconstruct(): given a sequence, produce the latent path and reconstructions.
    """
    
    def fit(
        self,
        x_data: jnp.ndarray,
        n_epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> "DeepMarkovModel":
        """
        Trains the DMM on the provided sequence data.
        
        Parameters:
            x_data: jnp.ndarray of shape (N, T, observation_dim)
            n_epochs: Number of training epochs.
            batch_size: Batch size.
            lr: Learning rate.
            seed: RNG seed.
        
        Returns:
            A new, trained DeepMarkovModel instance.
        """
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(self, eqx.is_array))
        N = x_data.shape[0]
        n_batches = int(np.ceil(N / batch_size))
        rng = jax.random.PRNGKey(seed)
        model = self  # start from the current model

        @eqx.filter_jit
        def loss_fn(model: DeepMarkovModel, x_seq, rng):
            # The DMM's __call__ returns the negative ELBO (averaged over batch)
            return model(x_seq, rng)

        @eqx.filter_jit
        def make_step(model: DeepMarkovModel, x_seq, optimizer, opt_state, rng):
            loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x_seq, rng)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_val

        for epoch in range(n_epochs):
            perm = np.random.permutation(N)
            x_shuf = x_data[perm]
            epoch_loss = 0.0
            for i in range(n_batches):
                batch = x_shuf[i * batch_size : (i + 1) * batch_size]
                rng, step_key = jax.random.split(rng)
                model, opt_state, loss_val = make_step(model, batch, optimizer, opt_state, step_key)
                epoch_loss += loss_val
            epoch_loss /= n_batches
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, -ELBO: {epoch_loss:.4f}")
        return model

    def reconstruct(
        self, x_seq: jnp.ndarray, rng_key, plot: bool = True
    ) -> jnp.ndarray:
        """
        Given an input sequence x_seq (of shape (1, T, observation_dim)),
        use the posterior network and the emission network to generate
        a reconstruction sequence. In this example, the reconstruction
        is obtained by sampling a latent trajectory using the combiner and
        the transition networks, then mapping each latent to an observation.
        
        Parameters:
            x_seq: jnp.ndarray of shape (1, T, observation_dim)
            rng_key: a JAX random key
            plot: if True and the observation is 1D or 2D, plot the original
                  vs. reconstruction.
                  
        Returns:
            recon_seq: jnp.ndarray of shape (T, observation_dim)
        """
        # Here we follow similar steps as in your sample_forward demonstration.
        # Assume that x_seq has batch size 1.
        B, T, obs_dim = x_seq.shape
        # Obtain hidden states from the posterior network.
        h_seq = self.posterior(x_seq)  # shape: (1, T, hidden_dim)
        # Split the rng key into T subkeys.
        rngs = jax.random.split(rng_key, T)
        # For t = 1, we use a dummy previous latent (zeros) and combine with h_1.
        z0 = jnp.zeros((B, self.latent_dim))
        q1_loc, q1_scale = self.combiner(z0, h_seq[:, 0, :])
        z1 = self.reparam_sample(rngs[0], q1_loc, q1_scale)
        # Get the emission parameters and reconstruction for time 1.
        e_loc, _ = self.emission(z1)
        recons = [e_loc]  # list to store each time step reconstruction
        z_prev = z1

        # Process subsequent time steps.
        for t in range(1, T):
            qt_loc, qt_scale = self.combiner(z_prev, h_seq[:, t, :])
            z_t = self.reparam_sample(rngs[t], qt_loc, qt_scale)
            e_loc, _ = self.emission(z_t)
            recons.append(e_loc)
            z_prev = z_t

        # Concatenate along time: result shape (T, observation_dim)
        recon_seq = jnp.concatenate(recons, axis=0)
        
        if plot:
            # If observation dimension is 1 or 2, attempt a simple visualization.
            recon_np = np.array(recon_seq)
            x_np = np.array(x_seq[0])
            plt.figure(figsize=(8, 4))
            if obs_dim == 1:
                plt.plot(x_np[:, 0], "b-o", label="Original")
                plt.plot(recon_np[:, 0], "r--o", label="Reconstruction")
                plt.title("DMM Reconstruction (1D)")
            elif obs_dim == 2:
                plt.scatter(x_np[:, 0], x_np[:, 1], c="b", label="Original")
                plt.scatter(recon_np[:, 0], recon_np[:, 1], c="r", label="Reconstruction")
                plt.title("DMM Reconstruction (2D)")
            else:
                print("Observation dim > 2: skipping plot")
                return recon_seq
            plt.legend()
            plt.show()

        return recon_seq