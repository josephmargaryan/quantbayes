import jax.numpy as jnp
from jax import random, pure_callback
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from numpyro import sample, handlers
from numpyro.infer import SVI, TraceEnum_ELBO, init_to_value, MCMC, NUTS
from numpyro.contrib.funsor import config_enumerate, infer_discrete
from numpyro.infer.autoguide import AutoDelta
from numpyro.distributions import Dirichlet, LogNormal, Normal, Categorical, constraints
import optax
import scipy.stats


class GaussianMixtureModel:
    def __init__(self, num_components=2):
        self.num_components = num_components
        self.model = self._build_model()
        self.global_guide = None
        self.trained_params = None

    def _build_model(self):
        @config_enumerate
        def model(data):
            # Global variables.
            weights = sample("weights", Dirichlet(0.5 * jnp.ones(self.num_components)))
            scale = sample("scale", LogNormal(0.0, 2.0))
            with handlers.plate("components", self.num_components):
                locs = sample("locs", Normal(0.0, 10.0))

            with handlers.plate("data", len(data)):
                # Local variables.
                assignment = sample("assignment", Categorical(weights))
                sample("obs", Normal(locs[assignment], scale), obs=data)

        return model

    def initialize(self, data, seed):
        init_values = {
            "weights": jnp.ones(self.num_components) / self.num_components,
            "scale": jnp.sqrt(data.var() / 2),
            "locs": data[
                random.categorical(
                    random.PRNGKey(seed),
                    jnp.ones(len(data)) / len(data),
                    shape=(self.num_components,),
                )
            ],
        }

        global_model = handlers.block(
            handlers.seed(self.model, random.PRNGKey(0)),
            hide_fn=lambda site: site["name"]
            not in ["weights", "scale", "locs", "components"],
        )
        self.global_guide = AutoDelta(
            global_model, init_loc_fn=init_to_value(values=init_values)
        )
        handlers.seed(self.global_guide, random.PRNGKey(0))(data)  # Warm up the guide

        elbo = TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, self.model, self.global_guide, data)

    def train_svi(self, data, learning_rate=0.1, iterations=200):
        elbo = TraceEnum_ELBO()

        def hook_optax(optimizer):
            gradient_norms = defaultdict(list)

            def append_grad(grad):
                for name, g in grad.items():
                    gradient_norms[name].append(float(jnp.linalg.norm(g)))
                return grad

            def update_fn(grads, state, params=None):
                grads = pure_callback(append_grad, grads, grads)
                return optimizer.update(grads, state, params=params)

            return (
                optax.GradientTransformation(optimizer.init, update_fn),
                gradient_norms,
            )

        optim, gradient_norms = hook_optax(optax.adam(learning_rate=learning_rate))
        global_svi = SVI(self.model, self.global_guide, optim, loss=elbo)
        svi_result = global_svi.run(random.PRNGKey(0), iterations, data)

        self.trained_params = svi_result.params
        return svi_result, gradient_norms

    def train_nuts(self, data, num_samples=250, num_warmup=50):
        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(random.PRNGKey(2), data)
        mcmc.print_summary()
        self.trained_params = mcmc.get_samples()

    def classify(self, data, temperature=0, rng_key=None):
        inferred_model = infer_discrete(
            handlers.replay(self.model, trace=self._get_guide_trace(data)),
            temperature=temperature,
            first_available_dim=-2,
            rng_key=rng_key,
        )
        seeded_model = handlers.seed(inferred_model, random.PRNGKey(0))
        trace = handlers.trace(seeded_model).get_trace(data)
        return trace["assignment"]["value"]

    def _get_guide_trace(self, data):
        trained_global_guide = handlers.substitute(
            self.global_guide, self.trained_params
        )
        return handlers.trace(trained_global_guide).get_trace(data)

    def plot_results(self, data):
        weights = self.trained_params["weights_auto_loc"]
        locs = self.trained_params["locs_auto_loc"]
        scale = self.trained_params["scale_auto_loc"]

        X = jnp.arange(-3, 15, 0.1)
        Y = sum(
            weights[k] * scipy.stats.norm.pdf((X - locs[k]) / scale)
            for k in range(self.num_components)
        )

        plt.figure(figsize=(10, 4), dpi=100).set_facecolor("white")
        plt.plot(X, Y, "k--")
        plt.plot(data, jnp.zeros(len(data)), "k*")
        plt.title("Density of Mixture Model")
        plt.ylabel("Probability Density")
        plt.show()


if __name__ == "__main__":

    data = jnp.array([...])
    gmm = GaussianMixtureModel(num_components=2)
    loss, _ = min((gmm.initialize(data, seed), seed) for seed in range(100))
    gmm.train_svi(data)
    gmm.plot_results(data)
