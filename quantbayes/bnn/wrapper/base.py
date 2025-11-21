# quantbayes/bnn/wrapper/base.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import warnings

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
    """Shared machinery for sklearn-compatible NumPyro estimators.

    New (Lipschitz logging) options:
        log_lipschitz: bool
            If True, estimate Lipschitz-related deterministic sites during SVI
            training and store a per-step history in `lip_history_`.
            Requires your model to define deterministic sites with those names
            (e.g. 'Lip_network').

        lip_sites: sequence of str
            Names of deterministic sites to log. Defaults to ('Lip_network',).

        lip_num_samples: int
            Number of Predictive samples used when estimating per-step
            Lipschitz statistics. Typically 1–5 is enough since this is just
            a diagnostic during training.

        lip_log_interval: int
            How often (in SVI steps) to log Lipschitz metrics.

        lip_batch_size: int
            How many examples from X to pass when evaluating Lipschitz sites.
            Since Lipschitz deterministics depend only on weights, 1 is fine.
    """

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
        # ---- NEW: Lipschitz logging options --------------------------------
        log_lipschitz: bool = False,
        lip_sites: Optional[Sequence[str]] = None,
        lip_num_samples: int = 1,
        lip_log_interval: int = 10,
        lip_batch_size: int = 1,
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

        # Lipschitz logging config
        self.log_lipschitz = bool(log_lipschitz)
        self.lip_sites = tuple(lip_sites) if lip_sites is not None else ("Lip_network",)
        self.lip_num_samples = int(lip_num_samples)
        self.lip_log_interval = int(lip_log_interval)
        self.lip_batch_size = int(lip_batch_size)

        # Will be populated during fit() if logging is enabled
        self.lip_history_: Optional[Dict[str, object]] = None

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
            try:
                from numpyro.contrib.einstein import RBFKernel, SteinVI
            except Exception as e:
                raise ImportError(
                    "SteinVI requested but numpyro.contrib.einstein is not available."
                ) from e
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

    # --- Lipschitz utilities -------------------------------------------------

    def _estimate_lipschitz_stats(self, X, params) -> Dict[str, Dict[str, float]]:
        """
        Estimate Lipschitz-related deterministic sites for the current params.

        Requirements:
          * Only used for SVI/SteinVI (guide-based).
          * The model must define deterministic sites with names in self.lip_sites.

        Returns:
          { site_name: {"mean": float, "std": float}, ... }
        """
        if not self.log_lipschitz or not self.lip_sites:
            return {}

        if self._backend_ not in ("svi", "steinvi"):
            # Only guide-based methods support simple Predictive-based logging
            return {}

        # Small batch just to trigger the model; Lipschitz depends only on weights.
        X = _to_device_array(X)
        if X.ndim >= 1 and X.shape[0] > self.lip_batch_size:
            X_lip = X[: self.lip_batch_size]
        else:
            X_lip = X

        # Predictive with current params and guide
        key = self._rng_pred_
        self._rng_pred_, key = _split_key(key)

        pred = Predictive(
            self.model,
            guide=self._inference_.guide,
            params=params,
            num_samples=self.lip_num_samples,
        )
        samples = pred(key, X_lip, y=None)

        stats: Dict[str, Dict[str, float]] = {}
        for site in self.lip_sites:
            if site not in samples:
                continue
            s = samples[site]
            # shape (S, ...) -> flatten spatial dims
            s_flat = jnp.reshape(s, (s.shape[0], -1))
            mean = jnp.mean(s_flat, axis=0)
            std = jnp.std(s_flat, axis=0)
            stats[site] = {
                "mean": float(jnp.mean(mean)),
                "std": float(jnp.mean(std)),
            }
        return stats

    def _record_lip_stats(
        self, step: int, loss_val: float, stats: Dict[str, Dict[str, float]]
    ) -> None:
        if self.lip_history_ is None:
            self.lip_history_ = {
                "step": [],
                "loss": [],
                "sites": {s: {"mean": [], "std": []} for s in self.lip_sites},
            }
        self.lip_history_["step"].append(int(step))
        self.lip_history_["loss"].append(float(loss_val))
        for s in self.lip_sites:
            if s not in self.lip_history_["sites"]:
                self.lip_history_["sites"][s] = {"mean": [], "std": []}
            if s in stats:
                self.lip_history_["sites"][s]["mean"].append(stats[s]["mean"])
                self.lip_history_["sites"][s]["std"].append(stats[s]["std"])
            else:
                self.lip_history_["sites"][s]["mean"].append(float("nan"))
                self.lip_history_["sites"][s]["std"].append(float("nan"))

    def get_lipschitz_history(self) -> Optional[Dict[str, object]]:
        """
        Return the Lipschitz logging history:

            {
              "step": [int, ...],
              "loss": [float, ...],
              "sites": {
                "Lip_network": {"mean": [...], "std": [...]},
                "Lip_spec1":   {"mean": [...], "std": [...]},
                ...
              }
            }

        or None if logging was disabled.
        """
        return self.lip_history_

    # public API --------------------------------------------------------------

    def fit(self, X, y=None):
        """Fit the estimator to (X, y)."""
        X = _to_device_array(X)
        y = None if y is None else _to_device_array(y)

        self._compile()
        self._init_rng()

        if self._backend_ == "mcmc":
            if self.log_lipschitz:
                warnings.warn(
                    "log_lipschitz=True but backend='mcmc'; per-step "
                    "Lipschitz logging is only supported for SVI/SteinVI.",
                    stacklevel=2,
                )
            self._inference_.run(self._rng_, X, y)
            self.posterior_samples_ = self._inference_.get_samples()
            self.params_ = None

        elif self._backend_ == "svi":
            svi_state = self._inference_.init(self._rng_, X, y)
            self.losses = []
            self.lip_history_ = None if not self.log_lipschitz else None

            for i in range(int(self.num_steps)):
                svi_state, loss = self._inference_.update(svi_state, X, y)
                loss_val = float(loss)
                self.losses.append(loss)

                if self.progress_bar:
                    print(f"\rSVI step {i+1}/{self.num_steps}", end="", flush=True)

                if self.log_lipschitz and ((i + 1) % self.lip_log_interval == 0):
                    params_i = self._inference_.get_params(svi_state)
                    stats = self._estimate_lipschitz_stats(X, params_i)
                    self._record_lip_stats(i + 1, loss_val, stats)

            if self.progress_bar:
                print()
            self.params_ = self._inference_.get_params(svi_state)
            self.posterior_samples_ = None

        elif self._backend_ == "steinvi":
            if self.log_lipschitz:
                warnings.warn(
                    "log_lipschitz=True with SteinVI: per-step logging is "
                    "not wired into SteinVI.run yet; only final posterior "
                    "Lipschitz can be sampled via sample_lipschitz().",
                    stacklevel=2,
                )
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
        elif self._backend_ in ("svi", "steinvi"):
            return Predictive(
                self.model,
                guide=self._inference_.guide,
                params=self.params_,
                num_samples=n,
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

    def sample_lipschitz(
        self,
        X,
        *,
        sites: Optional[Sequence[str]] = None,
        num_samples: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Convenience wrapper to sample Lipschitz-related deterministic sites.

        Requires that your model defines deterministic sites with those names
        (e.g. 'Lip_network', 'Lip_spec1', 'Lip_head').

        Returns:
            {site: samples} with shape (S, ...) for S=num_samples.
        """
        sites = tuple(sites) if sites is not None else tuple(self.lip_sites)
        return self.predict_posterior(
            X,
            sites=sites,
            num_samples=num_samples,
            random_state=random_state,
        )


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
        # Lipschitz logging
        log_lipschitz: bool = False,
        lip_sites: Optional[Sequence[str]] = None,
        lip_num_samples: int = 1,
        lip_log_interval: int = 10,
        lip_batch_size: int = 1,
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
            log_lipschitz=log_lipschitz,
            lip_sites=lip_sites,
            lip_num_samples=lip_num_samples,
            lip_log_interval=lip_log_interval,
            lip_batch_size=lip_batch_size,
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

    Expectation: your model defines a logits or probability site, which you
    point to via `logits_site` or `proba_site`. A common pattern is to
    define `numpyro.deterministic("out", logits)` and pass `logits_site="out"`.
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
        # Lipschitz logging
        log_lipschitz: bool = False,
        lip_sites: Optional[Sequence[str]] = None,
        lip_num_samples: int = 1,
        lip_log_interval: int = 10,
        lip_batch_size: int = 1,
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
            log_lipschitz=log_lipschitz,
            lip_sites=lip_sites,
            lip_num_samples=lip_num_samples,
            lip_log_interval=lip_log_interval,
            lip_batch_size=lip_batch_size,
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

    def _adv_loss_on_inputs(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        *,
        num_draws: int = 1,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> jnp.ndarray:
        """
        Expected cross-entropy loss E_q[CE(f_θ(X), y)] under the current posterior,
        approximated with `num_draws` posterior samples.

        This is the scalar loss we maximize w.r.t. X for FGSM/PGD.
        """
        self._ensure_fitted(attributes=["_inference_"])
        X = _to_device_array(X)
        y = jnp.asarray(y)

        pred = Predictive(
            self.model,
            guide=self._inference_.guide,
            params=self.params_,
            num_samples=int(num_draws),
        )

        key = (
            _as_prng_key(random_state) if random_state is not None else self._rng_pred_
        )

        samples = pred(key, X, y=None)

        # Prefer logits if available
        if self.proba_site is None:
            site = self.logits_site or "logits"
            if site not in samples:
                raise KeyError(
                    f"Adversarial loss: logits site '{site}' not found in samples."
                )
            draws = samples[site]  # (S, N) or (S, N, C/1)
            logits = jnp.mean(draws, axis=0)  # (N,) or (N,C) or (N,1)
            return _cross_entropy_from_logits(logits, y).mean()
        else:
            site = self.proba_site
            if site not in samples:
                raise KeyError(
                    f"Adversarial loss: proba site '{site}' not found in samples."
                )
            draws = samples[site]  # (S, N, ...) probabilities
            probs = jnp.mean(draws, axis=0)  # (N,) or (N,C) or (N,1)
            if probs.ndim == 1:
                probs = jnp.stack([1.0 - probs, probs], axis=-1)
            elif probs.shape[-1] == 1:
                p1 = probs[..., 0]
                probs = jnp.stack([1.0 - p1, p1], axis=-1)

            # y encoded as integers 0..C-1
            log_probs = jnp.log(jnp.clip(probs, 1e-8, 1.0))  # (N,C)
            ce = -log_probs[jnp.arange(y.shape[0]), y]
            return ce.mean()

    def fgsm_attack(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        *,
        epsilon: float,
        num_draws: int = 1,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> jnp.ndarray:
        """
        Untargeted L∞ FGSM attack on posterior predictive mean loss.

        Args:
            X: inputs (N, D...)
            y: labels (N,), integer-encoded (same as for fit()).
            epsilon: L∞ radius.
            num_draws: # posterior samples to approximate expectation in the loss.
        Returns:
            X_adv: adversarially perturbed inputs of same shape as X.
        """
        self._ensure_fitted()
        X = _to_device_array(X)
        y = jnp.asarray(y)

        def loss_fn(x):
            return self._adv_loss_on_inputs(
                x, y, num_draws=num_draws, random_state=random_state
            )

        grad_x = jax.grad(loss_fn)(X)
        x_adv = X + float(epsilon) * jnp.sign(grad_x)
        return x_adv

    def pgd_attack(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        *,
        epsilon: float,
        step_size: float,
        num_steps: int = 10,
        num_draws: int = 1,
        random_start: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> jnp.ndarray:
        """
        Untargeted L∞ PGD attack on posterior predictive mean loss.

        Args:
            X: inputs (N, D...).
            y: labels (N,).
            epsilon: L∞ radius.
            step_size: PGD step size α.
            num_steps: number of PGD iterations.
            num_draws: # posterior samples in loss.
            random_start: if True, start from a random point in the ε-ball.
            clip_min/clip_max: optional global clipping bounds (e.g. 0,1).
        Returns:
            X_adv: adversarially perturbed inputs.
        """
        self._ensure_fitted()
        X = _to_device_array(X)
        y = jnp.asarray(y)

        eps = float(epsilon)
        alpha = float(step_size)

        key = (
            _as_prng_key(random_state) if random_state is not None else self._rng_pred_
        )

        if random_start:
            key, sub = _split_key(key)
            noise = jax.random.uniform(sub, X.shape, minval=-eps, maxval=eps)
            x_adv = X + noise
        else:
            x_adv = X

        def projected(x):
            x = jnp.clip(x, X - eps, X + eps)
            if (clip_min is not None) or (clip_max is not None):
                lo = -jnp.inf if clip_min is None else float(clip_min)
                hi = jnp.inf if clip_max is None else float(clip_max)
                x = jnp.clip(x, lo, hi)
            return x

        x_adv = projected(x_adv)

        def step(x, _):
            def loss_fn(z):
                return self._adv_loss_on_inputs(
                    z, y, num_draws=num_draws, random_state=random_state
                )

            grad = jax.grad(loss_fn)(x)
            x_next = x + alpha * jnp.sign(grad)
            x_next = projected(x_next)
            return x_next, None

        x_adv, _ = jax.lax.scan(step, x_adv, None, length=int(num_steps))
        return x_adv

    def evaluate_adversarial_accuracy(
        self,
        X,
        y,
        *,
        attack: str = "fgsm",
        epsilon: float = 0.1,
        step_size: float = 0.02,
        num_steps: int = 10,
        num_draws: int = 1,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> float:
        """
        Convenience method: run an attack (FGSM/PGD) and compute accuracy on the
        adversarial examples.

        Args:
            attack: 'fgsm' or 'pgd'.
            epsilon, step_size, num_steps, num_draws: attack params.
        Returns:
            adversarial accuracy in [0,1].
        """
        X_jnp = _to_device_array(X)
        y_np = np.asarray(y)

        if attack.lower() == "fgsm":
            X_adv = self.fgsm_attack(
                X_jnp,
                jnp.asarray(y_np),
                epsilon=epsilon,
                num_draws=num_draws,
                random_state=random_state,
            )
        elif attack.lower() == "pgd":
            X_adv = self.pgd_attack(
                X_jnp,
                jnp.asarray(y_np),
                epsilon=epsilon,
                step_size=step_size,
                num_steps=num_steps,
                num_draws=num_draws,
                random_start=True,
                random_state=random_state,
                clip_min=clip_min,
                clip_max=clip_max,
            )
        else:
            raise ValueError(f"Unknown attack type: {attack}")

        # Use standard classifier predict() on adversarial inputs
        y_pred = self.predict(np.array(X_adv))
        acc = float((y_pred == y_np).mean())
        return acc


def _cross_entropy_from_logits(
    logits: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """
    CE(logits, y) per-example.

    - Binary: logits (N,) or (N,1).
    - Multiclass: logits (N,C) with y in {0..C-1}.
    """
    logits = jnp.asarray(logits)
    y = jnp.asarray(y)

    # Binary cases
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[-1] == 1):
        z = logits if logits.ndim == 1 else logits[..., 0]
        # log σ(z) and log(1-σ(z))
        logp1 = -jax.nn.softplus(-z)
        logp0 = -jax.nn.softplus(z)
        y_f = y.astype(z.dtype)
        return -(y_f * logp1 + (1.0 - y_f) * logp0)

    # Multiclass
    log_probs = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    return -log_probs[jnp.arange(y.shape[0]), y]
