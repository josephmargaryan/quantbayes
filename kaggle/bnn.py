import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
from kaggle.test import X_test, X_train, y_test, y_train
from sklearn.metrics import log_loss

from quantbayes import bnn
from quantbayes.bnn.utils import (
    expected_calibration_error,
    plot_calibration_curve,
    plot_roc_curve,
)

X_train, X_test, y_train, y_test = (
    jnp.array(X_train),
    jnp.array(X_test),
    jnp.array(y_train),
    jnp.array(y_test),
)


class Spectral(bnn.Module):
    def __init__(self, method):
        super().__init__(method=method)

    def __call__(self, X, y=None):
        N, D = X.shape
        X = bnn.JVPCirculantProcess(D)(X)
        X = jax.nn.tanh(X)
        X = bnn.Linear(D, 1)(X)
        logits = X.squeeze()
        numpyro.deterministic("logits", logits)
        with numpyro.plate("Data", N):
            numpyro.sample("likelihood", dist.Bernoulli(logits=logits), obs=y)


class Dense(bnn.Module):
    def __init__(self, method):
        super().__init__(method=method)

    def __call__(self, X, y=None):
        N, D = X.shape
        X = bnn.Linear(D, 1)(X)
        logits = X.squeeze()
        numpyro.deterministic("logits", logits)
        with numpyro.plate("Data", N):
            numpyro.sample("likelihood", dist.Bernoulli(logits=logits), obs=y)


if __name__ == "__main__":
    k = jr.key(0)
    model = Dense()
    model.compile(num_chains=2)
    model.fit(X_train, y_train, k)
    preds = model.predict(X_test, k)
    preds = jax.nn.sigmoid(preds).mean(axis=0)

    loss = log_loss(np.array(y_test), np.array(preds))
    plot_calibration_curve(y_test, preds)
    plot_roc_curve(y_test.ravel(), preds)
    ece = expected_calibration_error(y_test, preds)

    print(f"Ensemble loss: {loss:.3f}")
    print(f"Ensemble ECE: {ece:.3f}")
