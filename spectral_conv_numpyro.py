import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


from quantbayes import bnn
import jax.random as jr
import equinox as eqx
from quantbayes.stochax.utils import bayesianize, prior_fn


class Simple(eqx.Module):
    conv: eqx.nn.Conv2d
    l: eqx.nn.Linear

    def __init__(self, key):
        k1, k2 = jr.split(key, 2)
        self.conv = eqx.nn.Conv2d(3, 64, 3, (1, 1), (1, 1), key=k1)
        self.l = eqx.nn.Linear(64 * 24 * 24, 1, key=k2)

    def __call__(self, x):
        x = self.conv(x)
        x = jnp.reshape(x, (-1,))
        x = self.l(x)
        return x


class Test2(bnn.Module):
    def __init__(self):
        super().__init__(method="svi")

    def __call__(self, X, y=None):
        N, C, H, W = X.shape
        net = Simple(jr.key(0))
        net = bayesianize(net, prior_fn=prior_fn)
        X = jax.vmap(net)(X)
        logits = X.squeeze()
        numpyro.deterministic("logits", logits)
        with numpyro.plate("data", N):
            numpyro.sample("likelihood", dist.Bernoulli(logits=logits), obs=y)


class Test(bnn.Module):
    def __init__(self):
        super().__init__(method="svi")

    def __call__(self, X, y=None):
        N, C, H, W = X.shape
        X = bnn.SpectralConv2d(3, 64, (24, 24))(X)
        X = X.reshape(X.shape[0], 64 * 24 * 24)
        X = bnn.Linear(64 * 24 * 24, 1)(X)
        logits = X.squeeze()
        numpyro.deterministic("logits", logits)
        with numpyro.plate("data", N):
            numpyro.sample("likelihood", dist.Bernoulli(logits=logits), obs=y)


N, C, H, W = 10, 3, 24, 24
key = jr.key(0)
X = jr.randint(key, (N, C, H, W), minval=0, maxval=256)
X = X.astype(jnp.float32)
y = jr.bernoulli(key, shape=(N,)).astype(int)


import time

tak = time.time()
model = Test2()
model.compile()
model.fit(X, y, key)
tik = time.time()
preds = model.predict(X, key)
losses = model.get_losses

print(f"Time: {tik-tak:.3f}")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses)
plt.ylabel("Loss")
plt.xlabel("Step")
plt.title("Loss over steps")
plt.tight_layout()
plt.grid(True)
plt.show()
