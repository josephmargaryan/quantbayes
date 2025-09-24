import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from sklearn.metrics import accuracy_score

from quantbayes import bnn


def model(X, y=None):
    N, _, H, W = X.shape
    X = bnn.RFFTCirculant2D(C_in=1, C_out=1, H_in=28, W_in=28)(X)
    X = jnp.reshape(X, (N, -1))
    X = jax.nn.tanh(X)
    X = bnn.Linear(H * W, 10)(X)
    numpyro.deterministic("logits", X)
    with numpyro.plate("data", N):
        numpyro.sample("obs", dist.Categorical(logits=X), obs=y)


clf = bnn.NumpyroClassifier(
    model=model,
    method="nuts",
    logits_site="logits",  # or set proba_site="proba"
    n_posterior_samples=200,
    random_state=0,
)

X = jax.random.normal(jax.random.PRNGKey(0), (10, 1, 28, 28))
y = jax.random.randint(jax.random.PRNGKey(1), (10,), 0, 10)
X_train, X_test = X[:8], X[8:]
y_train, y_test = y[:8], y[8:]

clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)  # averaged posterior probabilities
y_hat = clf.predict(X_test)  # argmax over averaged probabilities


acc = accuracy_score(np.array(y_hat), np.array(y_test))
print(f"Test accuracy: {acc:.3f}")
