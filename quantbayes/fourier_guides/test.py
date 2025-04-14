import jax
import time
import numpy as np
import jax.random as jr
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal, AutoLowRankMultivariateNormal, AutoGuideList
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from quantbayes import bnn
from quantbayes.fake_data import generate_regression_data
from quantbayes.fourier_guides.guides import SpectralImagGuide, SpectralRealGuide

# Configuration parameters
LEARNING_RATE = 0.01
NUM_ITERATIONS = 1000

# Split the key for different usages.
key = jr.key(0)
key, init_key, pred_key, post_key = jr.split(key, 4)

# Data generation and preparation
df = generate_regression_data(n_continuous=16)
X, y = df.drop("target", axis=1), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)

def model(X, y=None):
    """
    Bayesian regression model with a spectral circulant layer and a linear output.
    """
    N, D = X.shape
    # Transform input using a spectral circulant layer.
    X = bnn.SpectralCirculantLayer(D)(X)
    X = jax.nn.tanh(X)
    
    # Linear layer weights and biases
    W = numpyro.sample("W", dist.Normal(0, 1).expand([D, 1]).to_event(2))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([1]).to_event(1))
    
    # Compute the linear combination.
    X = jnp.dot(X, W) + b
    # Squeeze the singleton dimension from the output.
    mu = jnp.squeeze(X, axis=1)
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    
    # Likelihood over observed data.
    with numpyro.plate("data", N):
        numpyro.sample("likelihood", dist.Normal(mu, sigma), obs=y)


# Set up the optimizer and SVI.
optimizer = numpyro.optim.Adam(LEARNING_RATE)
use_custom_guides = False
# Set up the optimizer and SVI.
optimizer = numpyro.optim.Adam(LEARNING_RATE)
if use_custom_guides:
    print("Using custom guides")
    K_value = 9
    spectral_real_guide = SpectralRealGuide(model, K=K_value)
    spectral_imag_guide = SpectralImagGuide(model, K=K_value)
    other_guide = AutoNormal(numpyro.handlers.block(model, hide=["spectral_circ_jvp_real", "spectral_circ_jvp_imag"]))
    guide = AutoGuideList(model)
    guide.append(spectral_real_guide)
    guide.append(spectral_imag_guide)
    guide.append(other_guide)
else:
    print("Using one guide")
    guide = AutoNormal(model)

svi = SVI(model, guide, optim=optimizer, loss=Trace_ELBO())

# Initialize the SVI state.
svi_state = svi.init(init_key, X_train, y_train)

# Training loop.
tak = time.time()
losses = []
for step in range(NUM_ITERATIONS):
    svi_state, loss = svi.update(svi_state, X_train, y_train)
    losses.append(loss)
    if (step + 1) % 100 == 0:
        print(f"Iteration {step+1} Loss: {loss:.3f}")

tik = time.time()
# Get the learned variational parameters.
params = svi.get_params(svi_state)

# Prediction using the learned model.
predictive = Predictive(
    model,
    guide=svi.guide,
    params=params,
    num_samples=100
)
preds = predictive(pred_key, X_test)["likelihood"]
mean_preds = preds.mean(axis=0)
mae_loss = mean_absolute_error(np.array(y_test), np.array(mean_preds))
print(f"Final MAE Loss: {mae_loss:.4f}")
print(f"Time taken: {tik-tak:.3f}")

# Sample the approximate posterior for "W".
samples_W = guide.sample_posterior(post_key, params=params)["W"]

# Plot the loss over iterations.
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title("Loss over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()  # display first plot

# Plot the approximate posterior distribution for "W".
plt.figure(figsize=(10, 6))
plt.hist(samples_W.flatten(), bins=50, density=True)
plt.title("Approximate Posterior Distribution for W")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()  # display second plot
