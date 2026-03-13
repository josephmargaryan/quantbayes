import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpyro
from jax import tree_util as jtu
import numpyro.distributions as dist

__all__ = ["bayesianize", "prior_fn"]


def _path_to_name(path):
    parts = []
    for k in path:
        if isinstance(k, jtu.GetAttrKey):
            parts.append(k.name)
        elif isinstance(k, jtu.DictKey):
            parts.append(str(k.key))
        elif isinstance(k, jtu.SequenceKey):
            parts.append(str(k.idx))
        else:
            parts.append(str(k))
    return "params." + ".".join(parts)


def bayesianize(module: eqx.Module, prior_fn):
    def replace(path, leaf):
        if eqx.is_array(leaf):  # robust array check
            name = _path_to_name(path)
            return numpyro.sample(name, prior_fn(leaf.shape))
        return leaf

    return jtu.tree_map_with_path(replace, module)


def prior_fn(shape, dist_cls=dist.Normal, **dist_kwargs):
    """Instantiate a distribution with keyword args, then expand to `shape`."""
    base = dist_cls(**dist_kwargs)
    return base.expand(shape).to_event(len(shape))


# examples:
# lambda s: prior_fn(s, dist_cls=dist.Normal, loc=0.0, scale=1.0)
# lambda s: prior_fn(s, dist_cls=dist.Laplace, loc=0.0, scale=0.5)
# lambda s: prior_fn(s, dist_cls=dist.Uniform, low=-0.1, high=0.1)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    import numpy as np
    import equinox as eqx
    import jax
    import jax.random as jr

    from quantbayes import bnn

    rng = np.random.RandomState(0)
    N, C, H, W, NUM_CLASSES = 2048, 1, 28, 28, 10
    X_np = rng.rand(N, C, H, W).astype("float32")  # [N, C, H, W]
    y_np = rng.randint(0, NUM_CLASSES, size=(N,)).astype("int32")

    # train / val split
    split = int(0.8 * N)
    X_train, X_val = X_np[:split], X_np[split:]
    y_train, y_val = y_np[:split], y_np[split:]

    class SimpleCNN(eqx.Module):
        conv1: eqx.nn.Conv2d
        fc1: eqx.nn.Linear

        def __init__(self, key):
            k1, k2, k3, k4 = jr.split(key, 4)
            self.conv1 = eqx.nn.Conv2d(1, 8, kernel_size=3, padding=1, key=k1)
            self.fc1 = eqx.nn.Linear(28 * 28 * 8, 10, key=k2)

        def __call__(self, x):
            x = self.conv1(x)
            x = jax.nn.tanh(x)
            x = jax.numpy.reshape(x, shape=(8 * 28 * 28))
            x = self.fc1(x)
            return x

    def bayesian_cnn(X, y=None):
        model = bayesianize(SimpleCNN(jr.key(0)), prior_fn=prior_fn)
        X = jax.vmap(model)(X)
        numpyro.deterministic("logits", X)
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("obs", dist.Categorical(logits=X), obs=y)

    clf = bnn.NumpyroClassifier(
        model=bayesian_cnn,
        method="svi",
        guide=None,
        logits_site="logits",
        n_posterior_samples=200,
        num_steps=100,
        random_state=2,
    )

    tak = time.time()
    clf.fit(X_train, y_train)
    tik = time.time()

    y_proba = clf.predict_proba(X_val)  # averaged posterior probabilities
    y_hat = clf.predict(X_val)  # argmax over averaged probabilities

    acc = accuracy_score(np.array(y_hat), np.array(y_val))
    print(f"Test accuracy: {acc:.3f}")
    print(f"Time taken: {tik-tak:.3f}")

    plt.plot(clf.losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
    plt.title("SVI Training Loss")
    plt.show()
