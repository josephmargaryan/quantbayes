import jax 
import jax.random as jr 
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import PRNGKeyArray

from quantbayes.fake_data import generate_regression_data 
from quantbayes import bnn
from quantbayes.bnn.utils import evaluate_mcmc

class Spectral(bnn.Module):
    def __init__(self):
        super().__init__()
        self.fft_layer = bnn.JVPCirculantProcess(1)
    def __call__(self, X, y=None):
        N, D = X.shape
        X = self.fft_layer(X)
        X = bnn.Linear(1, 1)(X)
        mu  = X.squeeze()
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("data", N):
            numpyro.sample("likelihood", dist.Normal(mu, sigma), obs=y)

class Circ(bnn.Module):
    def __init__(self):
        super().__init__()
        self.fft_layer = bnn.JVPCirculant(1)
    def __call__(self, X, y=None):
        N, D = X.shape
        X = self.fft_layer(X)
        X = bnn.Linear(1, 1)(X)
        mu  = X.squeeze()
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("data", N):
            numpyro.sample("likelihood", dist.Normal(mu, sigma), obs=y)

spectral_model = Spectral()
circ_model = Circ()

def eval(
        model: bnn.Module, 
        X: jax.numpy.ndarray, 
        y: jax.numpy.ndarray, 
        key: PRNGKeyArray
        ):
    model.compile(num_samples=500, num_chains=2)
    model.fit(X, y, key)
    results = evaluate_mcmc(model)
    for key, value in results.items():
        print(key)
        print(value)

if __name__ == "__main__":
    df = generate_regression_data()
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    key = jr.key(0)
    eval(circ_model, X, y, key)

    """
    Spectral:
    total_params: 6

    Circ: 
    total_params: 5
    """