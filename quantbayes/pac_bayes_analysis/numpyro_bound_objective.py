"""
numpyro/infer/pacbayes_extensions.py

Extensions to NumPyro's SVI ELBO objective to include PAC-Bayes bounds:
  - McAllester (PacBayesBound)
  - Split-KL (SplitKLPacBayesBound)
  - Empirical Bernstein (EmpiricalBernsteinPacBayesBound)

Usage:
    from numpyro.infer import SVI, Trace_ELBO
    from pacbayes_extensions import PacBayesBound, SplitKLPacBayesBound, EmpiricalBernsteinPacBayesBound

    svi = SVI(model, guide, optimizer, loss=PacBayesBound(n_data, delta))
"""

import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.kl import kl_divergence
from numpyro.infer import SVI
from numpyro.infer.elbo import ELBO
from numpyro.infer.util import compute_log_probs
from numpyro.handlers import seed, replay
from numpyro.util import _validate_model, check_model_guide_match


class _PACBase(ELBO):
    """Base class for PAC-Bayes SVI objectives."""

    def __init__(
        self,
        n_data,
        delta=0.05,
        risk_fn=None,
        num_particles=1,
        vectorize_particles=True,
    ):
        if n_data < 2:
            raise ValueError("n_data must be >= 2")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0,1)")
        super().__init__(
            num_particles=num_particles, vectorize_particles=vectorize_particles
        )
        self.n_data = n_data
        self.delta = delta
        self.risk_fn = risk_fn or self._default_risk

    @staticmethod
    def _default_risk(m_trace, m_logp, n_data):
        total_ll = 0.0
        for name, lp in m_logp.items():
            site = m_trace[name]
            if site["type"] == "sample" and site.get("is_observed", False):
                total_ll += jnp.sum(lp)
        raw = -total_ll / (n_data * jnp.log(2.0))
        return jnp.clip(raw, 0.0, 1.0)

    def loss_with_mutable_state(self, rng_key, params, model, guide, *args, **kwargs):
        keys = random.split(rng_key, self.num_particles)

        def _particle(key):
            g_logp, g_trace = compute_log_probs(seed(guide, key), args, kwargs, params)
            m_logp, m_trace = compute_log_probs(
                replay(seed(model, key), g_trace), args, kwargs, params
            )
            check_model_guide_match(m_trace, g_trace)
            _validate_model(m_trace, plate_warning="loose")
            kl = sum(
                kl_divergence(g_trace[n]["fn"], m_trace[n]["fn"])
                for n in g_trace
                if n in m_trace
            )
            risk = self.risk_fn(m_trace, m_logp, self.n_data)
            extras = self._extras(m_trace, m_logp)
            return (kl, risk) + extras

        vals = self.vectorize_particles_fn(_particle, keys)
        kl_mean = jnp.mean(vals[0])
        risk_mean = jnp.mean(vals[1])
        extras_mean = tuple(jnp.mean(v) for v in vals[2:])
        loss = self._bound_formula(kl_mean, risk_mean, *extras_mean)
        return {"loss": loss, "mutable_state": None}

    def _extras(self, m_trace, m_logp):
        return ()

    def _bound_formula(self, kl, risk, *extras):
        raise NotImplementedError


class PacBayesBound(_PACBase):
    """McAllester-style PAC-Bayes bound."""

    def _bound_formula(self, kl, risk, *extras):
        n, δ = self.n_data, self.delta
        term = kl + jnp.log(2.0 * jnp.sqrt(n) / δ)
        return risk + jnp.sqrt(term / (2.0 * (n - 1)))


class SplitKLPacBayesBound(_PACBase):
    """Split-KL PAC-Bayes bound (Seldin et al.)."""

    def _bound_formula(self, kl, risk, *extras):
        n, δ = self.n_data, self.delta
        half_kl = 0.5 * kl
        logt = jnp.log(2.0 * jnp.sqrt(n) / δ)
        return (
            risk
            + jnp.sqrt((half_kl + logt) / (2.0 * (n - 1)))
            + (half_kl + logt) / (n - 1)
        )


class EmpiricalBernsteinPacBayesBound(_PACBase):
    """Two-stage empirical-Bernstein PAC-Bayes bound."""

    def __init__(
        self,
        n_data,
        delta=0.05,
        c1=1.1,
        c2=1.1,
        num_particles=1,
        vectorize_particles=True,
    ):
        super().__init__(
            n_data,
            delta=delta,
            num_particles=num_particles,
            vectorize_particles=vectorize_particles,
        )
        self.c1, self.c2 = c1, c2

    def _extras(self, m_trace, m_logp):
        losses = []
        for name, lp in m_logp.items():
            site = m_trace[name]
            if site["type"] == "sample" and site.get("is_observed", False):
                pe = jnp.sum(lp, axis=tuple(range(1, lp.ndim)))
                losses.append(jnp.clip(-pe / jnp.log(2.0), 0.0, 1.0))
        all_l = jnp.concatenate([jnp.atleast_1d(l) for l in losses], axis=0)
        return (jnp.var(all_l, ddof=1),)

    def _bound_formula(self, kl, risk, var):
        n, δ = self.n_data, self.delta
        δ2, δ1 = δ / 2, δ / 2
        nu2 = jnp.maximum(
            jnp.log(0.5 * jnp.sqrt((n - 1) / jnp.log(1 / δ2) + 1) + 0.5)
            / jnp.log(self.c2),
            1.0,
        )
        Vb = var + (1 + self.c2) * jnp.sqrt(
            (var * (kl + jnp.log(nu2 / δ2))) / (2 * (n - 1))
        )
        Vb += 2 * self.c2 * (kl + jnp.log(nu2 / δ2)) / (n - 1)
        V_hat = jnp.clip(Vb, 0.0, 0.25)
        nu1 = jnp.maximum(
            jnp.log(jnp.sqrt((jnp.e - 2) * n / (4 * jnp.log(1 / δ1))))
            / jnp.log(self.c1),
            1.0,
        )
        KL_term = kl + jnp.log(2 * nu1 / δ1)
        cond = KL_term / ((jnp.e - 2) * V_hat) <= n
        bern = risk + (1 + self.c1) * jnp.sqrt(((jnp.e - 2) * V_hat * KL_term) / n)
        fallback = risk + 2 * KL_term / n
        return jnp.where(jnp.isnan(bern), fallback, jnp.where(cond, bern, fallback))


# -----------------------------------------------------------------------------
# Demonstration on Iris classification with Bayesian logistic regression
# -----------------------------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from numpyro.infer import Trace_ELBO
import numpy as np


def iris_model(X, y=None):
    D = X.shape[1]
    num_classes = int(jnp.max(y)) + 1 if y is not None else 3
    w = numpyro.sample("w", dist.Normal(0, 1).expand([num_classes, D]).to_event(2))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([num_classes]).to_event(1))
    logits = jnp.dot(X, w.T) + b
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("obs", dist.Categorical(logits=logits), obs=y)


def iris_guide(X, y=None):
    D = X.shape[1]
    num_classes = int(jnp.max(y)) + 1 if y is not None else 3
    w_loc = numpyro.param("w_loc", jnp.zeros((num_classes, D)))
    w_scale = numpyro.param(
        "w_scale", jnp.ones((num_classes, D)), constraint=dist.constraints.positive
    )
    b_loc = numpyro.param("b_loc", jnp.zeros((num_classes,)))
    b_scale = numpyro.param(
        "b_scale", jnp.ones((num_classes,)), constraint=dist.constraints.positive
    )
    numpyro.sample("w", dist.Normal(w_loc, w_scale).to_event(2))
    numpyro.sample("b", dist.Normal(b_loc, b_scale).to_event(1))


def run_iris_test(test_size=0.3, steps=2000, print_every=500):
    # Load & split
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=0
    )
    X_train, X_test = jnp.array(X_train), jnp.array(X_test)
    y_train, y_test = jnp.array(y_train), jnp.array(y_test)

    # Choose your objectives
    methods = {
        "ELBO": Trace_ELBO(num_particles=5),
        "McAllester": PacBayesBound(len(X_train), delta=0.1, num_particles=5),
        "SplitKL": SplitKLPacBayesBound(len(X_train), delta=0.1, num_particles=5),
        "Bernstein": EmpiricalBernsteinPacBayesBound(
            len(X_train), delta=0.1, num_particles=5
        ),
    }

    for name, loss_obj in methods.items():
        print(f"\n=== {name} on Iris ===")
        svi = SVI(iris_model, iris_guide, numpyro.optim.Adam(1e-2), loss=loss_obj)
        state = svi.init(random.PRNGKey(0), X_train, y_train)
        for i in range(1, steps + 1):
            state, _ = svi.update(state, X_train, y_train)
            if i % print_every == 0:
                # Training loss
                loss = svi.evaluate(state, X_train, y_train)
                # Posterior means
                params = svi.get_params(state)
                w_mean = params["w_loc"]
                b_mean = params["b_loc"]
                # Test accuracy
                logits = jnp.dot(X_test, w_mean.T) + b_mean
                preds = jnp.argmax(logits, axis=-1)
                acc = float(jnp.mean(preds == y_test) * 100)
                print(f"step {i:4d} | loss={loss:.4f} | test_acc={acc:.1f}%")


if __name__ == "__main__":
    run_iris_test()
