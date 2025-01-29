from quantbayes.bnn.layers import MultiHeadSelfAttention, Linear, Module
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


class BayesianTransformer(Module):
    """
    A single-block Bayesian Transformer for regression on a sequence:
      - MultiHeadSelfAttention
      - FeedForward
      - Output dist for y
    """

    def __init__(self, embed_dim, num_heads=2, method="svi"):
        super().__init__(method=method, task_type="regression")
        self.mha = MultiHeadSelfAttention(embed_dim, num_heads, name="bayes_mha")
        self.ff1 = Linear(embed_dim, embed_dim, name="bayes_tf_ff1")
        self.out = Linear(embed_dim, 1, name="bayes_tf_out")


def __call__(self, X, y=None):
    """
    X: shape (batch_size, seq_len)
    """
    batch_size, seq_len = X.shape

    # Sample global initial latent state z_0 (shared across all batches)
    z_0 = numpyro.sample("z_0_global", dist.Normal(0, 1))

    # Process each batch
    with numpyro.plate("batch", batch_size):
        z_prev = jnp.repeat(z_0[None], batch_size, axis=0)[:, None]  # (batch_size, 1)

        for t in range(seq_len):
            # Transition step
            mu_z, sigma_z = self.transition(z_prev)
            z_t = numpyro.sample(f"z_{t}", dist.Normal(mu_z, sigma_z).to_event(1))

            # Emission step
            mu_x, sigma_x = self.emission(z_t)
            numpyro.sample(
                f"x_{t}",
                dist.Normal(mu_x.squeeze(-1), sigma_x.squeeze(-1)),
                obs=X[:, t],
            )

            z_prev = z_t

    return None


def test_bayesian_transformer():
    rng = jax.random.PRNGKey(0)
    batch_size, seq_len, embed_dim = 32, 10, 4
    X = jax.random.normal(rng, (batch_size, seq_len, embed_dim))
    # Let's define a trivial "true" model: y = average of last token of X
    y = jnp.mean(X[:, -1, :], axis=-1) + 0.05 * jax.random.normal(rng, (batch_size,))

    model = BayesianTransformer(embed_dim=embed_dim, num_heads=1, method="svi")
    model.compile(num_steps=2000, learning_rate=0.001)
    model.fit(X, y, rng_key=rng)
    print("BayesianTransformer training done!")


if __name__ == "__main__":
    test_bayesian_transformer()
