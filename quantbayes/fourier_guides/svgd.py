import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpyro
import numpyro.distributions as dist
from numpyro.optim import Adagrad
from numpyro.infer import Predictive
from numpyro.contrib.einstein import ASVGD, RBFKernel, MixtureGuidePredictive
from numpyro.infer.autoguide import AutoNormal
from quantbayes import bnn
from quantbayes.fake_data import generate_binary_classification_data

df = generate_binary_classification_data(n_continuous=16)
X, y = df.drop("target", axis=1), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)

print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)
print("y_train shape", y_train.shape)
print("y_test shape", y_test.shape)

LEARNING_RATE = 0.01
NUM_ITERATIONS = 1000
predictive = True

key = jr.PRNGKey(0)
key, init_key, pred_key, grad_key, viz_key = jr.split(key, 5)


def model(X, y=None):
    """
    Bayesian binary classification model with a Spectral Circulant Layer and a linear output.
    """
    N, D = X.shape
    X_out = bnn.SpectralCirculantLayer(D)(X)
    X_out = jax.nn.tanh(X_out)
    W = numpyro.sample("W", dist.Normal(0, 1).expand([D, 1]).to_event(2))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([1]).to_event(1))

    logits = jnp.squeeze(jnp.dot(X_out, W) + b)
    numpyro.deterministic("logits", logits)

    with numpyro.plate("data", N):
        numpyro.sample("likelihood", dist.Bernoulli(logits=logits), obs=y)


inference = ASVGD(
    model,
    optim=Adagrad(LEARNING_RATE),
    kernel_fn=RBFKernel(),
    num_stein_particles=10,
    num_cycles=10,
    transition_speed=10,
)

print("Training with ASVGD...")
start_time = time.time()
svgd_result = inference.run(
    init_key, NUM_ITERATIONS, X_train, y_train, progress_bar=True
)
end_time = time.time()
print(f"Training finished in {end_time - start_time:.3f} seconds.")

params = inference.get_params(svgd_result.state)

if predictive:
    predictive = Predictive(
        model,
        params=params,
        guide=inference.guide,
        num_samples=100,
        return_sites=None,
        batch_ndims=1,
    )
else:
    predictive = MixtureGuidePredictive(
        model=model,
        guide=inference.guide,
        params=params,
        guide_sites=inference.guide_sites,
    )
predictions = predictive(pred_key, X_test)["logits"]
mean_preds = predictions.mean(axis=0)
print(mean_preds.shape)
mean_probs = jax.nn.sigmoid(mean_preds)
print(mean_preds.shape)
final_loss = log_loss(np.array(y_test), np.array(mean_probs))
print(f"Final Log Loss: {final_loss:.4f}")
