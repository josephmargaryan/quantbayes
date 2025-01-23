import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from quantbayes.bnn.utils.generalization_bound import BayesianAnalysis
from quantbayes import bnn
from quantbayes.fake_data import *


class Test(bnn.Module):
    def __init__(self):
        super().__init__(method="steinvi", task_type="regression")

    def __call__(self, X, y=None):
        in_features = X.shape[
            -1
        ]  # No particle dimension here, handled internally by ParticleLinear
        fcl = bnn.FFTParticleLinear(in_features, "particle_lcl1")
        fcl = jax.nn.silu(fcl(X))
        out = bnn.ParticleLinear(in_features, 1, "particle_out", aggregation="max")
        logits = out(fcl).squeeze()
        numpyro.deterministic("logits", logits)
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("y", dist.Normal(logits, sigma), obs=y)


df = generate_regression_data()
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)

# Instantiate and train the model
model = Test()
rng_key = jax.random.PRNGKey(0)

# Compile with NUTS
model.compile()

# Fit the model
model.fit(X_train, y_train, rng_key, num_steps=100000)

# Make predictions
predictions = model.predict(X_test, rng_key, "y")
# probs = jax.nn.softmax(predictions, axis=-1)

model.visualize(X_test, y_test)
bound = BayesianAnalysis(
    len(X_train), 0.05, "regression", "steinvi", model.get_stein_result
)
bound.compute_pac_bayesian_bound(predictions, y_test)
