# numpyro_sklearn.py
# -----------------------------------------------------------------------------
# A scikit-learn compatible wrapper around arbitrary NumPyro models.
#
# Features
# --------
# - Works with three inference backends:
#       * MCMC (NUTS)
#       * SVI (AutoGuide or custom guides)
#       * SteinVI (with MixtureGuidePredictive, if available)
# - sklearn API: __init__, fit, predict, predict_proba, get_params, set_params,
#   __sklearn_is_fitted__, _more_tags, score (via mixins), plus posterior utilities.
# - Regression and Classification frontends that pick sensible defaults.
#
# Usage (regression)
# ------------------
# def model(X, y=None):
#     # build your NumPyro model; typically sample y with obs=y
#     ...
#
# est = NumpyroRegressor(
#     model=model,
#     method="svi",
#     guide=my_guide,                    # or None -> AutoNormal
#     learning_rate=1e-2,
#     num_steps=2000,
#     n_posterior_samples=200,
#     random_state=0,
# )
# est.fit(X_train, y_train)
# y_pred = est.predict(X_test)          # posterior predictive mean of site='y'
# post = est.predict_posterior(X_test)  # dict of predictive samples (default sites)
#
# Usage (classification)
# ----------------------
# def model(X, y=None):
#     # produce either 'proba' site (probabilities) OR 'logits' site (pre-softmax)
#     ...
#
# clf = NumpyroClassifier(
#     model=model,
#     method="nuts",
#     logits_site="logits",              # or set proba_site="proba"
#     n_posterior_samples=200,
#     random_state=0,
# )
# clf.fit(X_train, y_train)
# y_proba = clf.predict_proba(X_test)   # averaged posterior probabilities
# y_hat = clf.predict(X_test)           # argmax over averaged probabilities
#
# Notes
# -----
# - Your `model` must accept (X, y=None) and may emit any named sites.
# - For regression, we default to predictive site "y".
# - For classification, set either `proba_site` or `logits_site`.
#   If neither is set, we try to use "proba" or "logits" automatically.
# - For SteinVI, numpyro.contrib.einstein must be available.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam, Adagrad

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

# Optional imports for SteinVI & mixture predictive
_HAS_EINSTEIN = False
try:
    from numpyro.contrib.einstein import RBFKernel, SteinVI, MixtureGuidePredictive

    _HAS_EINSTEIN = True
except Exception:
    pass


# ------------------------------ helpers -------------------------------------


def _as_prng_key(
    random_state: Optional[Union[int, np.random.RandomState]],
) -> jax.Array:
    if random_state is None:
        return jax.random.PRNGKey(0)
    if isinstance(random_state, (int, np.integer)):
        return jax.random.PRNGKey(int(random_state))
    return jax.random.PRNGKey(np.random.SeedSequence().entropy)


def _split_key(key: jax.Array) -> Tuple[jax.Array, jax.Array]:
    return jax.random.split(key, 2)


def _softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    x = x - jnp.max(x, axis=axis, keepdims=True)
    e = jnp.exp(x)
    return e / jnp.sum(e, axis=axis, keepdims=True)


def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))


def _to_device_array(x) -> jnp.ndarray:
    if isinstance(x, jnp.ndarray):
        return x
    return jnp.asarray(x)


@dataclass
class TrainConfig:
    # SVI/SteinVI shared
    learning_rate: float = 1e-2
    num_steps: int = 1000
    progress_bar: bool = True
    # MCMC
    num_warmup: int = 500
    num_samples: int = 1000
    num_chains: int = 1
    # SteinVI
    num_stein_particles: int = 16
    num_elbo_particles: int = 1
    repulsion_temperature: float = 1.0
    rbf_lengthscale: Optional[float] = None


# ----------------------- Base NumPyro Estimator ------------------------------


class _NumpyroBase(BaseEstimator):
    """Shared machinery for sklearn-compatible NumPyro estimators."""

    def __init__(
        self,
        model: Callable,
        *,
        method: str = "svi",  # 'svi' | 'nuts' | 'steinvi'
        guide: Optional[object] = None,
        optimizer: str = "adam",  # 'adam' | 'adagrad'
        learning_rate: float = 1e-2,
        num_steps: int = 1000,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        num_elbo_particles: int = 1,
        num_stein_particles: int = 16,
        repulsion_temperature: float = 1.0,
        rbf_lengthscale: Optional[float] = None,
        n_posterior_samples: int = 200,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        progress_bar: bool = True,
        predict_site: Optional[str] = None,  # default depends on subclass
        logits_site: Optional[str] = None,
        proba_site: Optional[str] = None,
    ):
        self.model = model
        self.method = method
        self.guide = guide
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_elbo_particles = num_elbo_particles
        self.num_stein_particles = num_stein_particles
        self.repulsion_temperature = repulsion_temperature
        self.rbf_lengthscale = rbf_lengthscale
        self.n_posterior_samples = n_posterior_samples
        self.random_state = random_state
        self.progress_bar = progress_bar
        self.predict_site = predict_site
        self.logits_site = logits_site
        self.proba_site = proba_site

    # sklearn plumbing --------------------------------------------------------

    def _more_tags(self) -> Dict[str, object]:
        # Back-compat for sklearn <=1.4; harmless for >=1.5
        return {
            "non_deterministic": True,
            "requires_y": True,
            "X_types": ["2darray", "3darray", "4darray", "numeric"],
            "allow_nan": False,
        }

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "fitted_", False)

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        return super().get_params(deep=deep)

    # Internal fitted-check (robust across sklearn versions)
    def _ensure_fitted(self, attributes: Optional[Sequence[str]] = None) -> None:
        if not getattr(self, "fitted_", False):
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted yet. Call fit() first."
            )
        if attributes:
            missing = [a for a in attributes if not hasattr(self, a)]
            if missing:
                raise NotFittedError(
                    f"{self.__class__.__name__} is missing fitted attribute(s): {missing}. "
                    "Did you call fit()?"
                )

    # core fit/predict utils --------------------------------------------------

    def _compile(self):
        """Instantiate the chosen inference object."""
        m = self.method.lower()
        if m == "nuts":
            kernel = NUTS(self.model)
            self._inference_ = MCMC(
                kernel,
                num_warmup=self.num_warmup,
                num_samples=self.num_samples,
                num_chains=self.num_chains,
                progress_bar=self.progress_bar,
            )
            self._backend_ = "mcmc"
        elif m == "svi":
            guide = self.guide or AutoNormal(self.model)
            if self.optimizer.lower() == "adam":
                optim = Adam(self.learning_rate)
            elif self.optimizer.lower() == "adagrad":
                optim = Adagrad(self.learning_rate)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer}")
            self._inference_ = SVI(
                self.model,
                guide,
                optim,
                loss=Trace_ELBO(num_particles=self.num_elbo_particles),
            )
            self._backend_ = "svi"
        elif m == "steinvi":
            if not _HAS_EINSTEIN:
                raise ImportError(
                    "SteinVI requested but numpyro.contrib.einstein is not available."
                )
            guide = self.guide or AutoNormal(self.model)
            kernel_fn = (
                RBFKernel(self.rbf_lengthscale)
                if self.rbf_lengthscale is not None
                else RBFKernel()
            )
            self._inference_ = SteinVI(
                model=self.model,
                guide=guide,
                optim=Adagrad(self.learning_rate),
                kernel_fn=kernel_fn,
                repulsion_temperature=self.repulsion_temperature,
                num_stein_particles=self.num_stein_particles,
                num_elbo_particles=self.num_elbo_particles,
            )
            self._backend_ = "steinvi"
        else:
            raise ValueError(f"Unknown inference method: {self.method}")

    def _init_rng(self):
        key = _as_prng_key(self.random_state)
        self._rng_, self._rng_pred_ = _split_key(key)

    # public API --------------------------------------------------------------

    def fit(self, X, y=None):
        """Fit the estimator to (X, y)."""
        X = _to_device_array(X)
        y = None if y is None else _to_device_array(y)

        self._compile()
        self._init_rng()

        if self._backend_ == "mcmc":
            self._inference_.run(self._rng_, X, y)
            self.posterior_samples_ = self._inference_.get_samples()
            self.params_ = None
        elif self._backend_ == "svi":
            svi_state = self._inference_.init(self._rng_, X, y)
            self.losses = []
            for i in range(int(self.num_steps)):
                svi_state, loss = self._inference_.update(svi_state, X, y)
                self.losses.append(loss)
                print(f"\rSVI step {i+1}/{self.num_steps}", end="", flush=True)
            print()
            self.params_ = self._inference_.get_params(svi_state)
            self.posterior_samples_ = None
        elif self._backend_ == "steinvi":
            result = self._inference_.run(
                self._rng_, int(self.num_steps), X, y, progress_bar=self.progress_bar
            )
            self._stein_result_ = result
            self.params_ = self._inference_.get_params(result.state)
            self.posterior_samples_ = None
        else:
            raise RuntimeError("Unexpected backend.")

        self.fitted_ = True
        return self

    # prediction core: returns Predictive-like callable
    def _make_predictive(self, n_samples: Optional[int] = None):
        self._ensure_fitted()
        n = int(n_samples or self.n_posterior_samples)
        if self._backend_ == "mcmc":
            return Predictive(self.model, posterior_samples=self.posterior_samples_)
        elif self._backend_ == "svi":
            return Predictive(
                self.model,
                guide=self._inference_.guide,
                params=self.params_,
                num_samples=n,
            )
        elif self._backend_ == "steinvi":
            return MixtureGuidePredictive(
                model=self.model,
                guide=self._inference_.guide,
                params=self.params_,
                num_samples=n,
                guide_sites=self._inference_.guide_sites,
            )
        else:
            raise RuntimeError("Unexpected backend.")

    def predict_posterior(
        self,
        X,
        *,
        sites: Optional[Sequence[str]] = None,
        num_samples: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Return raw posterior predictive draws for given `sites`.
        If `sites` is None, returns all sites produced by Predictive.
        Shapes are (S, ...) with S=num_samples.
        """
        X = _to_device_array(X)
        pred = self._make_predictive(num_samples)
        key = (
            _as_prng_key(random_state) if random_state is not None else self._rng_pred_
        )
        self._rng_pred_, key = _split_key(key)
        samples = pred(key, X, y=None)
        if sites is None:
            return samples
        return {k: samples[k] for k in sites if k in samples}


# ----------------------- Regressor and Classifier ----------------------------


class NumpyroRegressor(_NumpyroBase, RegressorMixin):
    """
    sklearn-style regressor that wraps a NumPyro model.
    """

    def __init__(
        self,
        model: Callable,
        *,
        predict_site: str = "y",
        method: str = "svi",
        guide: Optional[object] = None,
        optimizer: str = "adam",
        learning_rate: float = 1e-2,
        num_steps: int = 1000,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        num_elbo_particles: int = 1,
        num_stein_particles: int = 16,
        repulsion_temperature: float = 1.0,
        rbf_lengthscale: Optional[float] = None,
        n_posterior_samples: int = 200,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        progress_bar: bool = True,
        logits_site: Optional[str] = None,  # unused for regressor
        proba_site: Optional[str] = None,  # unused for regressor
    ):
        super().__init__(
            model=model,
            method=method,
            guide=guide,
            optimizer=optimizer,
            learning_rate=learning_rate,
            num_steps=num_steps,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            num_elbo_particles=num_elbo_particles,
            num_stein_particles=num_stein_particles,
            repulsion_temperature=repulsion_temperature,
            rbf_lengthscale=rbf_lengthscale,
            n_posterior_samples=n_posterior_samples,
            random_state=random_state,
            progress_bar=progress_bar,
            predict_site=predict_site,
            logits_site=logits_site,
            proba_site=proba_site,
        )

    def predict(self, X):
        self._ensure_fitted()
        site = self.predict_site or "y"
        draws = self.predict_posterior(X, sites=[site])[site]  # (S, N, ...)
        mean = np.array(jnp.mean(draws, axis=0))
        return mean.squeeze()


class NumpyroClassifier(_NumpyroBase, ClassifierMixin):
    """
    sklearn-style classifier that wraps a NumPyro model.
    """

    def __init__(
        self,
        model: Callable,
        *,
        proba_site: Optional[str] = None,
        logits_site: Optional[str] = None,
        method: str = "svi",
        guide: Optional[object] = None,
        optimizer: str = "adam",
        learning_rate: float = 1e-2,
        num_steps: int = 1000,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        num_elbo_particles: int = 1,
        num_stein_particles: int = 16,
        repulsion_temperature: float = 1.0,
        rbf_lengthscale: Optional[float] = None,
        n_posterior_samples: int = 200,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        progress_bar: bool = True,
        predict_site: Optional[str] = None,  # unused for classifier
    ):
        super().__init__(
            model=model,
            method=method,
            guide=guide,
            optimizer=optimizer,
            learning_rate=learning_rate,
            num_steps=num_steps,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            num_elbo_particles=num_elbo_particles,
            num_stein_particles=num_stein_particles,
            repulsion_temperature=repulsion_temperature,
            rbf_lengthscale=rbf_lengthscale,
            n_posterior_samples=n_posterior_samples,
            random_state=random_state,
            progress_bar=progress_bar,
            predict_site=predict_site,
            logits_site=logits_site,
            proba_site=proba_site,
        )

    def fit(self, X, y):
        """Fit and memorize label encoding."""
        self._label_encoder_ = LabelEncoder()
        y_enc = self._label_encoder_.fit_transform(np.asarray(y))
        self.classes_ = self._label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        super().fit(X, y_enc)
        return self

    def _posterior_proba(self, X) -> np.ndarray:
        self._ensure_fitted(attributes=["_label_encoder_"])
        X = _to_device_array(X)

        # If the model exposes probabilities directly, prefer them.
        if self.proba_site is not None:
            samples = self.predict_posterior(X, sites=[self.proba_site])
            draws = samples.get(self.proba_site)
            if draws is None:
                raise KeyError(f"Requested proba_site='{self.proba_site}' not found.")
            probs = jnp.mean(draws, axis=0)  # (N,) or (N,C?) or (N,1)
            if probs.ndim == 1:
                probs = jnp.stack([1.0 - probs, probs], axis=-1)
            elif probs.shape[-1] == 1:
                p1 = probs[..., 0]
                probs = jnp.stack([1.0 - p1, p1], axis=-1)
            return np.asarray(probs)

        # Otherwise consume logits (default name 'logits'); average *probabilities* per draw.
        site = self.logits_site or "logits"
        samples = self.predict_posterior(X, sites=[site])
        draws = samples.get(site)

        # Optional auto-detection: if 'logits' missing, try 'proba' once (keeps the promise in Notes)
        if draws is None:
            probe = self.predict_posterior(X, num_samples=1)  # cheap introspection
            if "proba" in probe:
                draws = self.predict_posterior(X, sites=["proba"])["proba"]
                probs = jnp.mean(draws, axis=0)
                if probs.ndim == 1:
                    probs = jnp.stack([1.0 - probs, probs], axis=-1)
                elif probs.shape[-1] == 1:
                    p1 = probs[..., 0]
                    probs = jnp.stack([1.0 - p1, p1], axis=-1)
                return np.asarray(probs)
            available = list(probe.keys())
            raise KeyError(
                f"Neither site '{site}' nor 'proba' found. Available sites: {available}"
            )

        # Binary: (S, N) or (S, N, 1) -> average sigmoid
        if draws.ndim == 2 or (draws.ndim == 3 and draws.shape[-1] == 1):
            logits = draws if draws.ndim == 2 else draws[..., 0]
            p1 = jax.nn.sigmoid(logits)  # (S, N)
            p1 = jnp.mean(p1, axis=0)  # (N,)
            probs = jnp.stack([1.0 - p1, p1], axis=-1)
            return np.asarray(probs)

        # Multiclass: (S, N, C) -> average softmax
        probs = jnp.mean(jax.nn.softmax(draws, axis=-1), axis=0)  # (N, C)
        return np.asarray(probs)

    def predict_proba(self, X):
        self._ensure_fitted(attributes=["_label_encoder_"])
        return self._posterior_proba(X)

    def predict(self, X):
        probs = self.predict_proba(X)  # (N, C)
        idx = np.argmax(probs, axis=-1)
        y_hat = self._label_encoder_.inverse_transform(idx)
        return y_hat

    def decision_function(self, X):
        """
        Return averaged logits (multiclass) or log-odds (binary).
        Shape: (N,) for binary, (N, C) for multiclass.
        """
        self._ensure_fitted(attributes=["_label_encoder_"])
        X = _to_device_array(X)

        if self.logits_site is not None:
            draws = self.predict_posterior(X, sites=[self.logits_site])[
                self.logits_site
            ]
            logits = jnp.mean(draws, axis=0)
            logits = (
                logits.squeeze(-1)
                if logits.ndim == 2 and logits.shape[-1] == 1
                else logits
            )
            return np.array(logits)

        if self.proba_site is None:
            probs = self.predict_proba(X)
        else:
            draws = self.predict_posterior(X, sites=[self.proba_site])[self.proba_site]
            probs = jnp.mean(draws, axis=0)

        probs = probs if probs.ndim > 1 else jnp.stack([1 - probs, probs], axis=-1)
        eps = 1e-7
        probs = jnp.clip(probs, eps, 1 - eps)

        if probs.shape[-1] == 2:
            logit = jnp.log(probs[..., 1] / probs[..., 0])
            return np.array(logit)
        else:
            return np.array(jnp.log(probs))
