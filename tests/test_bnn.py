import jax
import numpyro
import jax.random as jr
import jax.numpy as jnp
import numpyro.distributions as dist

from quantbayes.fake_data import (
    generate_regression_data,
    generate_binary_classification_data,
    generate_multiclass_classification_data,
)
from sklearn.model_selection import train_test_split
from quantbayes import bnn

##############################################################
# REGRESSION
##############################################################

df = generate_regression_data(n_categorical=1, n_continuous=1)
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)


class MyModel(bnn.Module):
    def __init__(self):
        super().__init__(method="nuts", task_type="regression")

    def __call__(self, X, y=None):
        N, in_features = X.shape

        X = bnn.FFTLinear(in_features=in_features, name="fft layer 1")(X)
        X = jax.nn.tanh(X)
        X = bnn.Linear(in_features=in_features, out_features=1, name="linear layer")(X)
        logits = X.squeeze()
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("data", N):
            numpyro.sample("likelihood", dist.Normal(logits, sigma), obs=y)


train_key, val_key = jr.split(jr.key(12), 2)
model = MyModel()
model.compile(num_warmup=10, num_samples=10)
model.fit(X_train, y_train, train_key)
model.visualize(X_test, y_test, posterior="likelihood")

##############################################################
# BINARY
##############################################################

df = generate_binary_classification_data(n_categorical=1, n_continuous=2)
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)


class MyModel(bnn.Module):
    def __init__(self):
        super().__init__(method="nuts", task_type="binary")

    def __call__(self, X, y=None):
        N, in_features = X.shape

        X = bnn.FFTLinear(in_features=in_features, name="fft layer 1")(X)
        X = jax.nn.tanh(X)
        X = bnn.Linear(in_features=in_features, out_features=1, name="linear layer")(X)
        logits = X.squeeze()
        numpyro.deterministic("logits", logits)
        with numpyro.plate("data", N):
            numpyro.sample("likelihood", dist.Bernoulli(logits=logits), obs=y)


train_key, val_key = jr.split(jr.key(12), 2)
model = MyModel()
model.compile(num_warmup=10, num_samples=10)
model.fit(X_train, y_train, train_key)
model.visualize(X_test, y_test, posterior="logits", features=(1, 2))

##############################################################
# MULTICLASS
##############################################################

df = generate_multiclass_classification_data(
    n_categorical=1, n_continuous=2, n_classes=3
)
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)


class MyModel(bnn.Module):
    def __init__(self):
        super().__init__(method="nuts", task_type="multiclass")

    def __call__(self, X, y=None):
        N, in_features = X.shape

        X = bnn.FFTLinear(in_features=in_features, name="fft layer 1")(X)
        X = jax.nn.tanh(X)
        X = bnn.Linear(in_features=in_features, out_features=3, name="linear layer")(X)
        numpyro.deterministic("logits", X)
        with numpyro.plate("data", N):
            numpyro.sample("likelihood", dist.Categorical(logits=X), obs=y)


train_key, val_key = jr.split(jr.key(12), 2)
model = MyModel()
model.compile(num_warmup=10, num_samples=10)
model.fit(X_train, y_train, train_key)
model.visualize(X_test, y_test, posterior="logits", num_classes=3)
