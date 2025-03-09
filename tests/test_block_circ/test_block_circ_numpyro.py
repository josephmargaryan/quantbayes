import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist

from quantbayes import bnn
from quantbayes.bnn.utils import plot_hdi
from quantbayes.fake_data import generate_binary_classification_data


class MyCircBlock(bnn.Module):
    def __init__(self):
        super().__init__(task_type="regression", method="nuts")

    def __call__(self, X, y=None):
        N, in_features = X.shape
        block_layer = bnn.BlockCirculantLayer(
            in_features=in_features,
            out_features=16,
            block_size=4,
            name="blockcirc_1",
            W_prior_fn=lambda shape: dist.Normal(0, 0.1).expand(shape),
            diag_prior_fn=lambda shape: dist.RelaxedBernoulli(
                temperature=0.1, probs=0.5
            )
            .expand(shape)
            .to_event(1),
            bias_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape),
        )
        X = block_layer(X)
        X = jax.nn.tanh(X)
        X = bnn.Linear(in_features=16, out_features=1, name="out")(X)
        logits = X.squeeze()
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        logits = numpyro.deterministic("logits", logits)
        with numpyro.plate("data", N):
            numpyro.sample("likelihood", dist.Normal(logits, sigma), obs=y)


train_key, val_key = jr.split(jr.key(34), 2)
df = generate_binary_classification_data(n_categorical=0, n_continuous=1)
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
model = MyCircBlock()
model.compile(num_warmup=500, num_samples=1000)
model.fit(X, y, train_key)
model.visualize(X, y, posterior="likelihood")
preds = model.predict(X, val_key)
plot_hdi(preds, X)
