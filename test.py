from quantbayes import bnn

import quantbayes.bnn as bnn
import numpyro.distributions as dist
import numpyro
import jax


class Test(bnn.Module):
    def __init__(self):
        super().__init__(method="nuts", task_type="regression")

    def forward(self, X, y=None):
        _, in_features = X.shape
        fcl = bnn.FFTLinear(in_features=in_features, name="fft layer")
        fcl = jax.nn.tanh(fcl(X))
        out = bnn.Linear(out_features=1, name="output layer")(fcl)
        logits = numpyro.deterministic("logits", out.squeeze())
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("y", dist.Normal(logits, sigma), obs=y)
