import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from sklearn.model_selection import train_test_split

from quantbayes import bnn, fake_data


# Assume your MyNet model is defined as follows:
class FFT(bnn.Module):
    def __init__(self, in_features):
        super().__init__(method="nuts", task_type="regression")
        # Instantiate layers once.
        self.fft_layer = bnn.SmoothTruncCirculantLayer(
            in_features=in_features, alpha=1, K=7, name="fft_layer"
        )
        self.out_layer = bnn.Linear(in_features=in_features, out_features=1, name="out")

    def __call__(self, X, y=None):
        N, in_features = X.shape
        # Use the pre-instantiated FFT layer.
        X_pre = self.fft_layer(X)
        # Apply nonlinearity.
        X_nl = jax.nn.tanh(X_pre)
        # Final linear mapping.
        X_out = self.out_layer(X_nl)
        logits = X_out.squeeze()
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("data", N):
            numpyro.sample("likelihood", dist.Normal(logits, sigma), obs=y)
        return logits

    def get_preactivations(self, X):
        """
        Compute and return the pre-activations from the FFT layer.
        We use jax.lax.stop_gradient to ensure that no tracer is leaked.
        """
        X_pre = self.fft_layer(X)
        return jax.lax.stop_gradient(X_pre)


tkey, vkey = jr.split(jr.key(12), 2)

df = fake_data.generate_regression_data(n_categorical=0, n_continuous=13)
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = FFT(13)
model.compile(num_warmup=10, num_samples=10)
model.fit(X, y, tkey)
