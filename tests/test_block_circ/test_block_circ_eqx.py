import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist

from quantbayes import bnn
from quantbayes.bnn.utils import plot_hdi
from quantbayes.fake_data import generate_regression_data
from quantbayes.stochax.utils import BlockCirculantLinear, bayesianize, prior_fn


class MyBlockCirculantNet(eqx.Module):
    bc_layer: BlockCirculantLinear  # our big block-circulant layer
    final_layer: eqx.nn.Linear

    def __init__(self, in_features, hidden_dim, *, key):
        key1, key2, key3 = jr.split(key, 3)
        self.bc_layer = BlockCirculantLinear(
            in_features=in_features,
            out_features=hidden_dim,
            block_size=4,
            key=key1,
            init_scale=0.01,
            use_bernoulli_diag=True,
        )
        self.final_layer = eqx.nn.Linear(hidden_dim, 1, key=key2)

    def __call__(self, x):
        h = self.bc_layer(x)
        h = jax.nn.tanh(h)
        return jax.vmap(self.final_layer)(h)


class MyBayes(bnn.Module):
    def __init__(self):
        super().__init__(task_type="regression", method="nuts")

    def __call__(self, X, y=None):
        N, in_features = X.shape
        net = MyBlockCirculantNet(
            in_features=in_features, hidden_dim=16, key=jr.key(123)
        )
        net = bayesianize(net, prior_fn)
        X = jax.vmap(net)(X)
        logits = X.squeeze()
        numpyro.deterministic("logits", logits)
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("data", N):
            numpyro.sample("likelihood", dist.Normal(logits, sigma), obs=y)


def test_bayesianize_block_circ():
    df = generate_regression_data(n_categorical=0, n_continuous=1)
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    train_key, val_key = jr.split(jr.key(321), 2)
    model = MyBayes()
    model.compile(num_warmup=500, num_samples=1000)
    model.fit(X, y, train_key)
    model.visualize(X, y, posterior="likelihood")
    preds = model.predict(X, val_key, posterior="likelihood")
    plot_hdi(preds, X)


def test_my_block_circulant_net():
    key = jr.PRNGKey(0)
    in_features = 64  # Example input dimensionality
    hidden_dim = 128  # Example hidden dimension
    batch_size = 4

    # Instantiate the network
    model = MyBlockCirculantNet(in_features, hidden_dim, key=key)

    # Create a random batch of input data
    key, subkey = jr.split(key)
    x = jr.normal(subkey, (batch_size, in_features))

    # Run a forward pass
    y = model(x)

    # Check that the output shape is as expected
    assert y.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size,1)}, got {y.shape}"

    # Optionally, test differentiability by computing the gradient of a simple loss
    loss_fn = lambda m, x: jnp.mean(m(x))
    grads = jax.grad(loss_fn)(model, x)

    print("MyBlockCirculantNet test passed!")
    print("Output shape:", y.shape)
    # Optionally print part of the gradient information
    print("Gradient for final layer weights (sample):", grads.final_layer.weight[0])


if __name__ == "__main__":
    test_my_block_circulant_net()
    test_bayesianize_block_circ()
