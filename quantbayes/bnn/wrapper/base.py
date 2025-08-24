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
from sklearn.utils.validation import check_is_fitted


# Optional imports for SteinVI & mixture predictive
_HAS_EINSTEIN = False
try:
    from numpyro.contrib.einstein import RBFKernel, SteinVI, MixtureGuidePredictive

    _HAS_EINSTEIN = True
except Exception:
    # We allow usage without SteinVI; if method="steinvi" we'll raise a helpful error.
    pass


# ------------------------------ helpers -------------------------------------


def _as_prng_key(
    random_state: Optional[Union[int, np.random.RandomState]],
) -> jax.Array:
    if random_state is None:
        return jax.random.PRNGKey(0)
    if isinstance(random_state, (int, np.integer)):
        return jax.random.PRNGKey(int(random_state))
    # Fallback: hash the state
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
    # Accept numpy / jax arrays; pass through shape
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
        return {
            "non_deterministic": True,
            "requires_y": True,
            "X_types": ["2darray", "3darray", "4darray", "numeric"],
            "allow_nan": False,
        }

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "fitted_", False)

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        # BaseEstimator would do this automatically, but we keep it explicit.
        return super().get_params(deep=deep)

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
            for _ in range(int(self.num_steps)):
                svi_state, _ = self._inference_.update(svi_state, X, y)
            self.params_ = self._inference_.get_params(svi_state)
            self.posterior_samples_ = None
        elif self._backend_ == "steinvi":
            result = self._inference_.run(
                self._rng_, int(self.num_steps), X, y, progress_bar=self.progress_bar
            )
            # Keep the whole result; we’ll read params via API:
            self._stein_result_ = result
            self.params_ = self._inference_.get_params(result.state)
            self.posterior_samples_ = None
        else:
            raise RuntimeError("Unexpected backend.")

        self.fitted_ = True
        return self

    # prediction core: returns Predictive-like callable
    def _make_predictive(self, n_samples: Optional[int] = None):
        check_is_fitted(self, attributes=["fitted_"])
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
            # Build a mixture across stein particles
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

    Conventions:
    - Uses `predict_site` (default "y") as the numeric response.
    - `predict` returns the posterior predictive MEAN over samples.
    - Use `predict_posterior` for full draws and uncertainty.
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
        """Posterior predictive mean for the chosen `predict_site` (default: 'y')."""
        site = self.predict_site or "y"
        draws = self.predict_posterior(X, sites=[site])[site]  # (S, N, ...)
        mean = np.array(jnp.mean(draws, axis=0))
        return mean.squeeze()

    # RegressorMixin provides score() = R^2 by default.


class NumpyroClassifier(_NumpyroBase, ClassifierMixin):
    """
    sklearn-style classifier that wraps a NumPyro model.

    Conventions:
    - The model should expose either:
        * probabilities in `proba_site` (e.g., Bernoulli probs or categorical probs), OR
        * logits in `logits_site` (pre-sigmoid for binary or pre-softmax for multiclass).
    - `predict_proba` returns averaged probabilities across posterior samples.
    - `predict` returns argmax over averaged probabilities mapped back to original labels.
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
        # Decide which site to use: proba_site -> probabilities; else logits_site.
        if self.proba_site is not None:
            site = self.proba_site
            draws = self.predict_posterior(X, sites=[site])[
                site
            ]  # (S, N, C?) or (S, N)
            probs = jnp.mean(draws, axis=0)  # (N, C?) or (N,)
            probs = probs if probs.ndim > 1 else jnp.stack([1 - probs, probs], axis=-1)
            return np.array(probs)
        # else use logits
        site = self.logits_site or "logits"
        draws = self.predict_posterior(X, sites=[site])[site]  # (S, N, C?) or (S, N)
        logits = jnp.mean(draws, axis=0)  # (N, C?) or (N,)
        if logits.ndim == 1 or (self.n_classes_ == 2 and logits.shape[-1] == 1):
            p1 = _sigmoid(logits if logits.ndim == 1 else logits[..., 0])
            probs = jnp.stack([1.0 - p1, p1], axis=-1)
        else:
            probs = _softmax(logits, axis=-1)
        return np.array(probs)

    def predict_proba(self, X):
        """Average posterior probabilities across samples."""
        check_is_fitted(self, attributes=["fitted_", "_label_encoder_"])
        return self._posterior_proba(X)

    def predict(self, X):
        """Argmax over averaged probabilities; returns original class labels."""
        probs = self.predict_proba(X)  # (N, C)
        idx = np.argmax(probs, axis=-1)
        y_hat = self._label_encoder_.inverse_transform(idx)
        return y_hat

    # --- add inside NumpyroClassifier --------------------------------------------

    def decision_function(self, X):
        """
        Return averaged logits (multiclass) or log-odds (binary).
        If only probabilities are available (proba_site), we convert to log-odds/logits.
        Shape: (N,) for binary, (N, C) for multiclass.
        """
        check_is_fitted(self, attributes=["fitted_", "_label_encoder_"])
        X = _to_device_array(X)

        if self.logits_site is not None:
            draws = self.predict_posterior(X, sites=[self.logits_site])[
                self.logits_site
            ]
            logits = jnp.mean(draws, axis=0)
            # Binary logits can be shape (N,) or (N,1); squeeze if needed
            logits = (
                logits.squeeze(-1)
                if logits.ndim == 2 and logits.shape[-1] == 1
                else logits
            )
            return np.array(logits)

        # fall back to proba -> logits
        if self.proba_site is None:
            # last resort: reuse predict_proba then log/softmax inverse
            probs = self.predict_proba(X)
        else:
            draws = self.predict_posterior(X, sites=[self.proba_site])[self.proba_site]
            probs = jnp.mean(draws, axis=0)

        probs = probs if probs.ndim > 1 else jnp.stack([1 - probs, probs], axis=-1)
        eps = 1e-7
        probs = jnp.clip(probs, eps, 1 - eps)

        if probs.shape[-1] == 2:
            # binary log-odds
            logit = jnp.log(probs[..., 1] / probs[..., 0])
            return np.array(logit)
        else:
            # multiclass logits up to affine constant: log p (good enough for ranking)
            return np.array(jnp.log(probs))

    def predict_log_proba(self, X):
        """
        Return log-probabilities averaged over posterior samples (log of mean probs).
        Shape: (N, C); for binary ensure two columns.
        """
        probs = self.predict_proba(X)
        eps = 1e-9
        return np.log(np.clip(probs, eps, 1.0))

    # ClassifierMixin provides score() = accuracy by default.


# ----------------------------- Extra utilities -------------------------------


class NumpyroPosteriorMixin:
    """Optional mixin with convenience posterior utilities."""

    def predict_mean_std(
        self,
        X,
        site: Optional[str] = None,
        num_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return posterior predictive mean and std for a numeric site.
        Defaults to estimator's `predict_site` if available, else 'y'.
        """
        if site is None:
            site = getattr(self, "predict_site", None) or "y"
        draws = self.predict_posterior(X, sites=[site], num_samples=num_samples)[site]
        mean = np.array(jnp.mean(draws, axis=0))
        std = np.array(jnp.std(draws, axis=0))
        return mean, std

    def predict_quantiles(
        self,
        X,
        site: Optional[str] = None,
        q: Sequence[float] = (0.05, 0.5, 0.95),
        num_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return posterior predictive quantiles for a numeric site.
        Shape: (len(q), ...) ordered by q.
        """
        if site is None:
            site = getattr(self, "predict_site", None) or "y"
        draws = self.predict_posterior(X, sites=[site], num_samples=num_samples)[site]
        qs = np.quantile(np.array(draws), q, axis=0)
        return qs


# If you want, you can export convenience classes with the mixin included:
class NumpyroRegressorPlus(NumpyroPosteriorMixin, NumpyroRegressor):
    """Regressor with posterior convenience methods included."""

    pass


class NumpyroClassifierPlus(NumpyroPosteriorMixin, NumpyroClassifier):
    """Classifier with posterior convenience methods included."""

    pass
