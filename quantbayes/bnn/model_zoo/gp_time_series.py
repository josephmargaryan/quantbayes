from quantbayes.bnn import Module
from quantbayes.bnn.layers import GaussianProcessLayer
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


class GaussianProcessTimeSeries(Module):
    """
    A simple Gaussian Process regressor for time series, using your GaussianProcessLayer.
    We'll do y ~ MVN(0, K), where K = gp_layer(X).
    """

    def __init__(self, input_dim, method="svi"):
        super().__init__(method=method, task_type="regression")
        self.gp_layer = GaussianProcessLayer(input_dim=input_dim, name="gp")

    def __call__(self, X, y=None):
        """
        X: shape (N, input_dim)
        y: shape (N,) or (N,1)
        We'll treat X as the entire dataset at once (no plate needed if we do a single MVN).
        """
        N, _ = X.shape

        # build NxN kernel
        K = self.gp_layer(X)  # shape (N, N)

        # zero-mean for simplicity, or define a param for the mean
        mean = jnp.zeros((N,))

        # sample y
        numpyro.sample(
            "y", dist.MultivariateNormal(loc=mean, covariance_matrix=K), obs=y
        )

        return None  # or return mean or K


def generate_synthetic_gp_data(N=50, input_dim=1):
    # Let's generate 1D time points X= [0,1,2,...], then sine wave + noise
    rng = jax.random.PRNGKey(0)
    X = jnp.linspace(0, 1, N)
    X = X[:, None]  # shape (N,1)
    y = jnp.sin(2 * jnp.pi * X.squeeze(-1)) + 0.1 * jax.random.normal(rng, (N,))
    return X, y


def test_gaussian_process_ts():
    X, y = generate_synthetic_gp_data(N=50)
    model = GaussianProcessTimeSeries(input_dim=1, method="svi")
    model.compile(num_steps=2000, learning_rate=0.01)
    rng_key = jax.random.PRNGKey(42)
    model.fit(X, y, rng_key=rng_key)
    print("Gaussian Process Time Series fitted!")


if __name__ == "__main__":
    test_gaussian_process_ts()
