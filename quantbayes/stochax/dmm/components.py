# dmm_components.py
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from quantbayes.stochax.dmm.base import (
    BaseCombiner,
    BaseDMM,
    BaseEmission,
    BasePosterior,
    BaseTransition,
)

__all__ = ["DeepMarkovModel"]

#######################################
# 1. TRANSITION COMPONENTS
#######################################


class LinearTransition(eqx.Module, BaseTransition):
    """
    A simple linear transition for p(z_t | z_{t-1}).
    It applies a linear mapping to z_prev and adds a learned bias.
    """

    weight: jnp.ndarray  # shape: (latent_dim, latent_dim)
    bias: jnp.ndarray  # shape: (latent_dim,)
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
            key=key,
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
            key=key,
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
        self.lstm = eqx.nn.LSTMCell(
            input_size=observation_dim, hidden_size=hidden_dim, key=key
        )

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
            key=key,
        )

    def __call__(
        self, z_prev: jnp.ndarray, h: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    emission: eqx.Module  # e.g. MLPEmission
    posterior: eqx.Module  # e.g. LSTMPosterior
    combiner: eqx.Module  # e.g. MLPCombiner
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
        key,
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

    def __call__(self, x_seq: jnp.ndarray, rng, kl_weight: float = 1.0) -> jnp.ndarray:
        batch, T, _ = x_seq.shape
        h_seq = self.posterior(x_seq)  # (batch, T, hidden_dim)
        rngs = jax.random.split(rng, T)

        # Set free bits per dimension
        free_bits_per_dim = 0.1  # adjust as needed
        free_bits_total = free_bits_per_dim * self.latent_dim

        # Process t = 1
        z0 = jnp.zeros((batch, self.latent_dim))
        q1_loc, q1_scale = self.combiner(z0, h_seq[:, 0, :])
        z1 = self.reparam_sample(rngs[0], q1_loc, q1_scale)

        prior_scale = jnp.exp(self.z1_logscale)
        lp_z1 = -0.5 * jnp.sum(
            ((z1 - self.z1_loc) / prior_scale) ** 2
            + 2 * jnp.log(prior_scale)
            + jnp.log(2 * jnp.pi),
            axis=-1,
        )
        lq_z1 = -0.5 * jnp.sum(
            ((z1 - q1_loc) / q1_scale) ** 2
            + 2 * jnp.log(q1_scale)
            + jnp.log(2 * jnp.pi),
            axis=-1,
        )
        kl_1 = jnp.maximum(lq_z1 - lp_z1, free_bits_total)

        x1 = x_seq[:, 0, :]
        e_loc, e_scale = self.emission(z1)
        lp_x1 = -0.5 * jnp.sum(
            ((x1 - e_loc) / e_scale) ** 2 + 2 * jnp.log(e_scale) + jnp.log(2 * jnp.pi),
            axis=-1,
        )
        reconstruction = lp_x1
        kl_term = kl_1
        z_prev = z1

        for t in range(1, T):
            qt_loc, qt_scale = self.combiner(z_prev, h_seq[:, t, :])
            z_t = self.reparam_sample(rngs[t], qt_loc, qt_scale)
            pt_loc, pt_scale = self.transition(z_prev)
            lp_zt = -0.5 * jnp.sum(
                ((z_t - pt_loc) / pt_scale) ** 2
                + 2 * jnp.log(pt_scale)
                + jnp.log(2 * jnp.pi),
                axis=-1,
            )
            lq_zt = -0.5 * jnp.sum(
                ((z_t - qt_loc) / qt_scale) ** 2
                + 2 * jnp.log(qt_scale)
                + jnp.log(2 * jnp.pi),
                axis=-1,
            )
            kl_t = jnp.maximum(lq_zt - lp_zt, free_bits_total)
            kl_term += kl_t

            x_t = x_seq[:, t, :]
            e_loc, e_scale = self.emission(z_t)
            lp_xt = -0.5 * jnp.sum(
                ((x_t - e_loc) / e_scale) ** 2
                + 2 * jnp.log(e_scale)
                + jnp.log(2 * jnp.pi),
                axis=-1,
            )
            reconstruction += lp_xt

            z_prev = z_t

        neg_elbo = -(reconstruction - kl_weight * kl_term)
        return jnp.mean(neg_elbo)


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
        validation_split: float = 0.2,
        patience: int = 10,
        lambda_forecast: float = 1.0,  # weight for the forecasting error term
        T_obs: int = 10,  # number of observed steps in each sequence
    ) -> "DeepMarkovModel":
        @eqx.filter_jit
        def loss_fn(model: DeepMarkovModel, x_seq, rng, kl_weight: float):
            return model(x_seq, rng, kl_weight=kl_weight)

        @eqx.filter_jit
        def make_step(
            model: DeepMarkovModel, x_seq, optimizer, opt_state, rng, kl_weight: float
        ):
            loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
                model, x_seq, rng, kl_weight
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_val

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(self, eqx.is_array))

        # Split the data into training and validation sets.
        N = x_data.shape[0]
        n_val = int(N * validation_split)
        n_train = N - n_val
        perm = np.random.permutation(N)
        train_data = x_data[perm[:n_train]]
        val_data = x_data[perm[n_train:]]
        n_train_batches = int(np.ceil(n_train / batch_size))
        n_val_batches = int(np.ceil(n_val / batch_size))

        # Assume each sequence has total length T.
        T = x_data.shape[1]
        T_pred = T - T_obs  # forecast horizon

        # Annealing schedule: ramp KL weight over 80% of epochs.
        annealing_epochs = int(n_epochs * 0.8)
        rng_local = jax.random.PRNGKey(seed)

        # Use a separate variable for the current model.
        cur_model = self

        best_composite_metric = float("inf")
        best_model = None
        patience_counter = 0

        for epoch in range(n_epochs):
            current_kl_weight = min(1.0, (epoch + 1) / annealing_epochs)

            # Training loop.
            perm_train = np.random.permutation(n_train)
            train_data_shuffled = train_data[perm_train]
            train_loss_epoch = 0.0
            for i in range(n_train_batches):
                batch = train_data_shuffled[i * batch_size : (i + 1) * batch_size]
                rng_local, step_key = jax.random.split(rng_local)
                cur_model, opt_state, loss_val = make_step(
                    cur_model, batch, optimizer, opt_state, step_key, current_kl_weight
                )
                train_loss_epoch += loss_val
            train_loss_epoch /= n_train_batches

            # Compute validation loss (ELBO) over the validation set.
            val_loss_epoch = 0.0
            for i in range(n_val_batches):
                batch = val_data[i * batch_size : (i + 1) * batch_size]
                rng_local, val_key = jax.random.split(rng_local)
                loss_val = loss_fn(cur_model, batch, val_key, current_kl_weight)
                val_loss_epoch += loss_val
            val_loss_epoch /= n_val_batches

            # Compute forecasting error on the validation set.
            # We loop over each validation sample (you can also batch this if needed).
            forecast_error_sum = 0.0
            sample_count = 0
            for batch in np.array(
                val_data
            ):  # converting to NumPy for easier Python loop
                # Ensure the sample has shape (1, T, observation_dim)
                sample = jnp.array(batch)[None, ...]  # shape: (1, T, observation_dim)
                observed_seq = sample[:, :T_obs, :]
                true_future = sample[:, T_obs:, :]
                rng_local, pred_key = jax.random.split(rng_local)
                # Use n_samples=1 for error computation.
                predicted_future = cur_model.predict(
                    observed_seq,
                    n_steps=T_pred,
                    rng_key=pred_key,
                    n_samples=1,
                    plot=False,
                )
                # predicted_future: shape (1, T_pred, observation_dim)
                error = jnp.mean((predicted_future[0] - true_future[0]) ** 2)
                forecast_error_sum += error
                sample_count += 1
            avg_forecast_error = forecast_error_sum / sample_count

            # Define a composite metric. (Adjust lambda_forecast so that the scales match.)
            composite_metric = val_loss_epoch + lambda_forecast * avg_forecast_error

            print(
                f"Epoch {epoch+1}/{n_epochs}, Train -ELBO: {train_loss_epoch:.4f}, "
                f"Val -ELBO: {val_loss_epoch:.4f}, Forecast MSE: {avg_forecast_error:.4f}, "
                f"Composite: {composite_metric:.4f}, KL weight: {current_kl_weight:.4f}"
            )

            # Early stopping based on the composite metric.
            if composite_metric < best_composite_metric:
                best_composite_metric = composite_metric
                best_model = cur_model  # store the best model
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch+1} due to no improvement for {patience} epochs."
                )
                break

        return best_model if best_model is not None else cur_model

    def reconstruct(
        self, x_seq: jnp.ndarray, rng_key, n_samples: int = 5, plot: bool = True
    ) -> jnp.ndarray:
        """
        Given an input sequence x_seq (of shape (1, T, observation_dim)),
        generate n_samples reconstructions.
        Returns: an array of shape (n_samples, T, observation_dim)
        """
        B, T, obs_dim = x_seq.shape
        assert B == 1, "Reconstruction is implemented for batch size 1."

        # Obtain the posterior hidden states.
        h_seq = self.posterior(x_seq)  # shape: (1, T, hidden_dim)

        # Split rng_key into T*n_samples keys.
        rngs = jax.random.split(rng_key, T * n_samples).reshape(T, n_samples, -1)

        # --- For time step t=1 ---
        z0 = jnp.zeros((1, self.latent_dim))
        q1_loc, q1_scale = self.combiner(z0, h_seq[:, 0, :])  # shape: (1, latent_dim)

        # Tile the q1 parameters so we can sample n_samples times.
        q1_loc_tiled = jnp.repeat(q1_loc, n_samples, axis=0)  # (n_samples, latent_dim)
        q1_scale_tiled = jnp.repeat(
            q1_scale, n_samples, axis=0
        )  # (n_samples, latent_dim)

        def sample_t1(key, loc, scale):
            return self.reparam_sample(key, loc, scale)

        # Vmap over keys, loc, and scale. This produces shape (n_samples, latent_dim).
        z1_samples = jax.vmap(sample_t1)(rngs[0], q1_loc_tiled, q1_scale_tiled)

        e_loc, _ = self.emission(z1_samples)  # e_loc shape: (n_samples, obs_dim)
        recons = [e_loc]  # list of reconstructions for t=1
        z_prev = z1_samples  # (n_samples, latent_dim)

        # --- For time steps t=2,...,T ---
        for t in range(1, T):
            # For each sample, compute the combiner output.
            # h_seq[:, t, :] is of shape (1, hidden_dim); repeat to (n_samples, hidden_dim)
            h_t_tiled = h_seq[:, t, :].repeat(n_samples, axis=0)
            qt_loc, qt_scale = self.combiner(
                z_prev, h_t_tiled
            )  # (n_samples, latent_dim) each

            def sample_t(key, loc, scale):
                return self.reparam_sample(key, loc, scale)

            # Vmap over keys and corresponding parameters.
            z_t_samples = jax.vmap(sample_t)(
                rngs[t], qt_loc, qt_scale
            )  # (n_samples, latent_dim)

            e_loc, _ = self.emission(z_t_samples)  # (n_samples, obs_dim)
            recons.append(e_loc)
            z_prev = z_t_samples

        recon_seq = jnp.stack(recons, axis=1)  # shape: (n_samples, T, obs_dim)

        if plot:
            import matplotlib.pyplot as plt

            orig = np.array(x_seq[0])
            plt.figure(figsize=(10, 4))
            for i in range(min(n_samples, 4)):
                plt.plot(np.array(recon_seq[i, :, 0]), "--", label=f"Recon {i+1}")
            plt.plot(orig[:, 0], "k", label="Original", linewidth=2)
            plt.legend()
            plt.title("Multiple Reconstructions")
            plt.show()

        return recon_seq

    def predict(
        self,
        x_seq_observed: jnp.ndarray,
        n_steps: int,
        rng_key,
        n_samples: int = 1,
        plot: bool = True,
    ) -> jnp.ndarray:
        """
        Given an observed sequence x_seq_observed (shape: (1, T, observation_dim)),
        predict n_steps into the future by generating n_samples trajectories.

        Parameters:
            x_seq_observed: jnp.ndarray of shape (1, T, observation_dim)
            n_steps: Number of future time steps to predict.
            rng_key: JAX random key.
            n_samples: Number of prediction trajectories to sample.
            plot: If True, plot the observed sequence and the multiple predicted trajectories.

        Returns:
            predictions: jnp.ndarray of shape (n_samples, n_steps, observation_dim)
        """
        # 1. Infer the latent state from the observed sequence.
        h_seq = self.posterior(x_seq_observed)  # shape: (1, T, hidden_dim)
        # Use the last time-step's hidden state to compute the latent for time T.
        z0 = jnp.zeros((1, self.latent_dim))
        q_T_loc, q_T_scale = self.combiner(z0, h_seq[:, -1, :])

        # Sample n_samples initial latent states at time T.
        rng_key, subkey = jax.random.split(rng_key)
        sample_keys = jax.random.split(subkey, n_samples)
        z_T_samples = jax.vmap(
            lambda key: self.reparam_sample(key, q_T_loc, q_T_scale)
        )(sample_keys)
        # z_T_samples shape: (n_samples, latent_dim)

        predictions_list = []  # List to collect predictions for each future time step.
        z_samples = z_T_samples  # shape: (n_samples, latent_dim)

        for _ in range(n_steps):
            # For each sample, compute the next latent state using the transition network.
            pt_loc, pt_scale = jax.vmap(self.transition)(
                z_samples
            )  # each of shape (n_samples, latent_dim)
            rng_key, subkey = jax.random.split(rng_key)
            sample_keys = jax.random.split(subkey, n_samples)
            z_next = jax.vmap(
                lambda key, loc, scale: self.reparam_sample(key, loc, scale)
            )(
                sample_keys, pt_loc, pt_scale
            )  # shape: (n_samples, latent_dim)
            # Decode each latent state using the emission network.
            e_loc, _ = jax.vmap(self.emission)(
                z_next
            )  # expected shape: (n_samples, observation_dim)
            predictions_list.append(e_loc)
            z_samples = z_next  # update latent state for next step

        # Stack predictions along the time dimension.
        predictions = jnp.stack(predictions_list, axis=0)
        # Expected shape if each e_loc is (n_samples, observation_dim):
        #   (n_steps, n_samples, observation_dim)
        #
        # However, if there is an extra singleton dimension, e.g. shape is (n_steps, n_samples, observation_dim, 1),
        # we squeeze it out.
        if predictions.ndim == 4:
            predictions = jnp.squeeze(
                predictions, axis=-1
            )  # Now shape: (n_steps, n_samples, observation_dim)

        # Transpose to shape (n_samples, n_steps, observation_dim)
        predictions = jnp.transpose(predictions, (1, 0, 2))

        if plot:
            import matplotlib.pyplot as plt

            # Convert to numpy arrays for plotting.
            observed_np = np.array(x_seq_observed)[0, :, 0]  # shape: (T_obs,)
            T_obs = x_seq_observed.shape[1]
            plt.figure(figsize=(10, 4))
            # Plot observed data.
            plt.plot(np.arange(T_obs), observed_np, "bo-", label="Observed")
            # For each sample, draw a connecting line and plot its trajectory.
            for i in range(n_samples):
                traj = np.array(predictions)[i, :, 0]  # shape: (n_steps,)
                # Connect the last observed point with the first predicted point.
                plt.plot([T_obs - 1, T_obs], [observed_np[-1], traj[0]], "k-")
                # Plot the predicted trajectory.
                plt.plot(
                    np.arange(T_obs, T_obs + n_steps),
                    traj,
                    "r--o",
                    label=f"Predicted {i+1}" if i == 0 else None,
                )
            plt.xlabel("Time Step")
            plt.ylabel("Observation")
            plt.legend()
            plt.title(
                "Forecasting: Observed Data vs. Multiple Predicted Future Trajectories"
            )
            plt.show()

        return predictions
