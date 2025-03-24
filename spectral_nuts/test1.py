import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import tree_flatten

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.hmc_util import euclidean_kinetic_energy

from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------
# Import your CirculantProcess and data generator.
from quantbayes.bnn import CirculantProcess
from quantbayes.fake_data import generate_regression_data


################################################################################
# BaseLinea
################################################################################

# ------------------------------------------------------------------------------
# Define a spectral model that uses the CirculantProcess.
def spectral_model(X, y=None):
    D = X.shape[1]
    cp_layer = CirculantProcess(in_features=D, use_bias=False)
    X_transformed = cp_layer(X)
    # Store Fourier coefficients.
    fourier = cp_layer.get_fourier_coeffs()
    numpyro.deterministic("spectral", fourier)
    
    X_transformed = jax.nn.tanh(X_transformed)
    
    W = numpyro.sample("W", dist.Normal(jnp.zeros((D, 1)), jnp.ones((D, 1))).to_event(2))
    b = numpyro.sample("b", dist.Normal(0, 1))
    mu = jnp.dot(X_transformed, W) + b
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("obs", dist.Normal(mu.squeeze(), sigma), obs=y)

# ------------------------------------------------------------------------------
# Test code: Run adaptive spectral NUTS on regression data.
def main():
    df = generate_regression_data(n_continuous=12, random_seed=24)
    X, y = df.drop("target", axis=1), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    
    rng_key = random.PRNGKey(24)
    
    # For a padded dimension equal to D (here 24), the half-spectrum length is (24//2)+1 = 13.
    D = X_train.shape[1]
    padded_dim = D
    k_half = (padded_dim // 2) + 1
    init_spectral_mass = jnp.ones(k_half)
    
    adaptive_nuts_kernel = NUTS(
        spectral_model
    )
    
    mcmc = MCMC(adaptive_nuts_kernel, num_warmup=500, num_samples=1000)
    mcmc.run(rng_key, X=X_train, y=y_train)
    samples = mcmc.get_samples()
    
    predictive = Predictive(spectral_model, samples)
    preds = predictive(rng_key, X=X_test)
    mean_preds = preds["obs"].mean(axis=0)
    
    loss = mean_squared_error(np.array(y_test), np.array(mean_preds))
    print("Test MSE:", loss)

if __name__ == "__main__":
    main()
