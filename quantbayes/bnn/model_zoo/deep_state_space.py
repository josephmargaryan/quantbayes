import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from quantbayes.bnn.layers.base import Module

class DeepStateSpace(Module):
    """
    A Deep Markov Model for 1D time-series, with global neural network parameters:

      z_1 ~ Normal(0,1)
      z_t ~ Normal(mu_t, sigma_t),   (mu_t, sigma_t) = f_trans(z_{t-1})
      x_t ~ Normal(decoder(z_t), emission_sigma)
    """

    def __init__(self, hidden_dim=32, method="svi"):
        super().__init__(method=method, task_type="regression")
        self.hidden_dim = hidden_dim

    def _init_mlp_params(self, prefix, in_dim, out_dim):
        """
        Helper to define an MLP layer with shape [in_dim, hidden_dim] -> [hidden_dim, out_dim].
        We'll store them as named numpyro.param(...) so they are global.
        """
        # e.g. W1 shape: (in_dim, hidden_dim)
        W1 = numpyro.param(f"{prefix}_W1", 0.01 * jnp.ones((in_dim, self.hidden_dim)))
        b1 = numpyro.param(f"{prefix}_b1", jnp.zeros((self.hidden_dim,)))
        
        # e.g. W2 shape: (hidden_dim, out_dim)
        W2 = numpyro.param(f"{prefix}_W2", 0.01 * jnp.ones((self.hidden_dim, out_dim)))
        b2 = numpyro.param(f"{prefix}_b2", jnp.zeros((out_dim,)))
        
        return (W1, b1, W2, b2)

    def _forward_mlp(self, x, params, activation=jax.nn.tanh):
        """
        A 2-layer MLP with shape: x -> (W1,b1)-> hidden -> (W2,b2)-> out
        """
        (W1, b1, W2, b2) = params
        h = activation(jnp.dot(x, W1) + b1)
        out = jnp.dot(h, W2) + b2
        return out

    def __call__(self, X, y=None):
        """
        X: shape (batch_size, seq_len). We'll do 1D time-series for each row in X.
        We define a global transition-MLP and emission-MLP. 
        Then for each data row we unroll z_t.

        This calls:
          trans_params = param(...)   # global
          emit_params  = param(...)   # global
        Then we do with plate("batch", batch_size): sample z_t and x_t.

        Returns None (the log-likelihood is added via the Normal(..., obs=...)).
        """
        batch_size, seq_len = X.shape

        # 1) Define global MLP parameters for transition net: z_{t-1} -> (mu, log_sigma).
        trans_mlp_params = self._init_mlp_params(
            prefix="transition", 
            in_dim=1,      # input is z_{t-1} shape (batch_size,1)
            out_dim=2      # output is (mu, log_sigma)
        )

        # 2) Define global MLP parameters for emission net: z_t -> (mu_x, log_sigma_x).
        emit_mlp_params = self._init_mlp_params(
            prefix="emission",
            in_dim=1,
            out_dim=2
        )

        # 3) We unroll over batch dimension
        with numpyro.plate("batch", batch_size):
            # Sample initial latent z_0. We'll just do z_1 ~ Normal(0,1).
            z_prev = numpyro.sample("z_init", dist.Normal(0.0, 1.0))   # shape: (batch_size,)
            z_prev = z_prev[:, None]  # shape (batch_size, 1)

            for t in range(seq_len):
                # 3a) Transition step: z_t ~ Normal(...).
                # pass z_prev -> transition MLP -> (mu, log_sigma).
                trans_out = self._forward_mlp(z_prev, trans_mlp_params)
                mu_z, log_sigma_z = jnp.split(trans_out, 2, axis=-1)  # shape (batch_size,1) each
                sigma_z = jnp.exp(log_sigma_z)

                z_t = numpyro.sample(
                    f"z_{t}",
                    dist.Normal(mu_z, sigma_z).to_event(1),
                )  # shape: (batch_size,1)

                # 3b) Emission step: x_t ~ Normal(...).
                emit_out = self._forward_mlp(z_t, emit_mlp_params)
                mu_x, log_sigma_x = jnp.split(emit_out, 2, axis=-1)
                sigma_x = jnp.exp(log_sigma_x)

                numpyro.sample(
                    f"x_{t}",
                    dist.Normal(mu_x.squeeze(-1), sigma_x.squeeze(-1)),
                    obs=X[:, t],  # shape (batch_size,)
                )

                z_prev = z_t

        return None

def generate_synthetic_dssm(num_samples=50, seq_len=10):
    rng = jax.random.PRNGKey(0)
    z = jnp.zeros((num_samples, seq_len))
    x = jnp.zeros((num_samples, seq_len))

    for i in range(num_samples):
        zt = 0.1 * jax.random.normal(rng, (1,))
        xt = zt + 0.05 * jax.random.normal(rng, (1,))
        z_i = [zt]
        x_i = [xt]
        for t in range(1, seq_len):
            zt = 0.9 * z_i[-1] + 0.1 * jax.random.normal(rng, (1,))
            xt = zt + 0.05 * jax.random.normal(rng, (1,))
            z_i.append(zt)
            x_i.append(xt)
        z = z.at[i].set(jnp.concatenate(z_i))
        x = x.at[i].set(jnp.concatenate(x_i))
    return x  # ignoring the latent z for now

def test_deep_state_space():
    X = generate_synthetic_dssm(num_samples=50, seq_len=10)

    model = DeepStateSpace(hidden_dim=16, method="svi")
    # We'll compile with some basic SVI config
    model.compile(num_steps=1000, learning_rate=1e-2)
    
    rng_key = jax.random.PRNGKey(42)
    model.fit(X, y_train=None, rng_key=rng_key)
    print("Deep State Space Model fitted with SVI!")

if __name__ == "__main__":
    test_deep_state_space()