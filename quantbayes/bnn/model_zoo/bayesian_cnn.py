from quantbayes.bnn.layers import Conv1d, Linear, Module
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro


class BayesianCNN(Module):
    """
    A simple Bayesian 1D CNN regressor:
      - One convolutional layer
      - Flatten
      - One linear output
    """

    def __init__(self, in_channels, out_channels=8, kernel_size=3, method="nuts"):
        """
        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the conv layer.
        :param kernel_size: Convolution kernel size.
        :param method: 'svi', 'nuts', or 'steinvi'.
        """
        super().__init__(method=method, task_type="regression")
        self.conv1d = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            name="bayes_conv1d",
        )
        # The output dimension after conv depends on input length & stride/padding
        # For simplicity, let's just flatten everything, then produce 1 scalar:
        # We'll figure out the dimension dynamically in `__call__`.
        self.out_layer = Linear(out_channels, 1, name="bayes_conv1d_out")

    def __call__(self, X, y=None):
        """
        X: shape (batch_size, in_channels, width)
        y: shape (batch_size,)
        """
        # 1) Convolution
        convolved = self.conv1d(X)  # shape = (batch_size, out_channels, new_width)

        # 2) We'll just do a naive global average pooling across width:
        #    shape -> (batch_size, out_channels)
        pooled = jnp.mean(convolved, axis=-1)

        # 3) Dense output -> scalar
        logits = self.out_layer(pooled).squeeze(-1)  # (batch_size,)

        # 4) Sample noise
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))

        # 5) Likelihood
        numpyro.sample("y", dist.Normal(logits, sigma), obs=y)

        return logits


def test_bayesian_cnn():
    rng = jax.random.PRNGKey(0)
    batch_size, in_channels, width = 50, 2, 20
    X = jax.random.normal(rng, (batch_size, in_channels, width))
    true_weights = jax.random.normal(rng, (in_channels,))
    y = jnp.sum(
        X[..., -1].transpose(1, 0) * true_weights[:, None], axis=0
    ) + 0.1 * jax.random.normal(rng, (batch_size,))

    model = BayesianCNN(in_channels=2, out_channels=4, kernel_size=3, method="nuts")
    model.compile(num_warmup=200, num_samples=500)  # MCMC parameters
    model.fit(X, y, rng_key=rng)
    samples = model.get_samples
    print("BayesianCNN MCMC done! Posterior samples keys:", samples.keys())


if __name__ == "__main__":
    test_bayesian_cnn()
