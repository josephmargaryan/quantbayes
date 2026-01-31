import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["BNNEnsembleRegression", "BNNEnsembleBinary", "BNNEnsembleMulticlass"]


def softmax_logits(logits, axis=-1, eps=1e-20):
    """
    Numerically stable softmax for logits with small epsilon clipping.
    """
    exps = jnp.exp(logits - jnp.max(logits, axis=axis, keepdims=True))
    sums = jnp.sum(exps, axis=axis, keepdims=True)
    return jnp.clip(exps / sums, eps, 1.0 - eps)


def binary_log_likelihood(logits, y):
    """
    Compute log-likelihood for binary classification given logits.
    logits: shape (S, N)
    y: shape (N,) in {0,1}
    Returns: shape (S, N) array of log p(y | logits)
    """
    # p = sigmoid(logits)
    # log p(y=1|x) = log(sigmoid(logit)), log p(y=0|x) = log(1 - sigmoid(logit))
    # We'll vectorize:
    log_p_1 = -jax.nn.softplus(-logits)  # log(sigmoid(logits))
    log_p_0 = -jax.nn.softplus(logits)  # log(1 - sigmoid(logits))
    # Now pick the correct one via y
    return y * log_p_1 + (1 - y) * log_p_0


def multiclass_log_likelihood(logits, y):
    """
    Compute log-likelihood for multiclass given logits.
    logits: shape (S, N, C)
    y: shape (N,) in {0,1,...,C-1}
    Returns: shape (S, N)
    """
    # 1) Convert to probabilities via softmax
    probs = softmax_logits(logits, axis=-1)  # shape (S, N, C)
    # 2) Gather the probability corresponding to the correct class
    #    We do a fancy indexing
    gather_idxs = jnp.expand_dims(y, axis=0)  # shape (1, N)
    gather_idxs = jnp.expand_dims(gather_idxs, axis=-1)  # shape (1, N, 1)
    # Now we gather from probs along last axis
    correct_class_probs = jnp.take_along_axis(probs, gather_idxs, axis=-1).squeeze(
        -1
    )  # (S, N)
    # 3) log
    return jnp.log(correct_class_probs + 1e-20)


def regression_log_likelihood(predictions, y, sigma=1.0):
    """
    Compute log-likelihood for regression under a Gaussian with known sigma.
    predictions: shape (S, N)
    y: shape (N,)
    sigma: float, presumed noise std dev
    Returns: shape (S, N) for log p(y | predictions)
    """
    # For demonstration, assume y ~ Normal(prediction, sigma^2).
    # log p(y|mu, sigma) = -0.5 * log(2pi*sigma^2) - (y - mu)^2 / (2*sigma^2)
    # We'll do this elementwise:
    c = -0.5 * jnp.log(2.0 * jnp.pi * sigma**2)
    mse_term = -0.5 / (sigma**2) * (y - predictions) ** 2
    return c + mse_term


def compute_waic_and_weights(loglik_matrix):
    """
    Given loglik_matrix of shape (S, N), we compute an approximate WAIC
    and return (waic, weight).
    - WAIC = -2 * (lpd - p_waic), where
      lpd = sum over i of log(average of exp(loglik over S)),
      p_waic = sum over i of var(loglik over S).
    We'll do sums over data dimension i, i.e. dimension=1 in (S, N).

    Returns waic_value (float), and then you can transform it to weight with:
      weight = exp(-0.5 * waic_value)
    (Pseudo-BMA weighting).
    """
    # 1) pointwise log predictive density: logmeanexp over S
    # shape (N,)
    log_colmeans = jax.scipy.special.logsumexp(loglik_matrix, axis=0) - jnp.log(
        loglik_matrix.shape[0]
    )

    # 2) total log predictive density
    lpd = jnp.sum(log_colmeans)

    # 3) pointwise var of loglik -> shape (N,)
    #    then sum over N
    var_loglik = jnp.var(loglik_matrix, axis=0, ddof=1)
    p_waic = jnp.sum(var_loglik)

    waic = -2.0 * (lpd - p_waic)
    return waic


def normalize_weights(weights):
    """
    Given a list or array of weights, returns normalized version that sums to 1.
    """
    w = jnp.array(weights)
    w = jnp.clip(w, 1e-20, None)  # ensure non-negative
    return w / jnp.sum(w)


class _BNNEnsembleBase:
    """
    Extended base class for handling multiple BNN models and combining predictions.
    Adds advanced methods to compute WAIC-based weights or stacking weights.
    """

    def __init__(self, models_dict, ensemble_method="bma"):
        """
        :param models_dict: dict
            { 'modelA': myBNN_A, 'modelB': myBNN_B, ... }
            Each must inherit from your 'Module' base class.
        :param ensemble_method: str
            'bma', 'simple_average', 'stacking' (extended with advanced weighting).
        """
        self.models_dict = models_dict
        self.ensemble_method = ensemble_method
        # Store learned weights for BMA or stacking
        self.model_weights = None

    def compile_models(self, **compile_kwargs):
        for name, model in self.models_dict.items():
            print(f"[Ensemble] Compiling model: {name}")
            model.compile(**compile_kwargs)

    def fit_models(self, X_train, y_train, rng_keys, **fit_kwargs):
        models_list = list(self.models_dict.items())

        # If single rng_key is passed, split it
        if isinstance(rng_keys, jnp.ndarray):
            rng_keys = jax.random.split(rng_keys, len(models_list))

        for (name, model), key in zip(models_list, rng_keys):
            print(f"[Ensemble] Fitting model: {name}")
            model.fit(X_train, y_train, key, **fit_kwargs)

    def predict_models(
        self, X_test, rng_keys, posterior="likelihood", num_samples=None
    ):
        models_list = list(self.models_dict.items())

        if isinstance(rng_keys, jnp.ndarray):
            rng_keys = jax.random.split(rng_keys, len(models_list))

        predictions_dict = {}
        for (name, model), key in zip(models_list, rng_keys):
            print(f"[Ensemble] Predicting with model: {name}")
            preds = model.predict(
                X_test, key, posterior=posterior, num_samples=num_samples
            )
            predictions_dict[name] = preds
        return predictions_dict

    ##################################################################
    # 4) combine_predictions is specialized in sub-classes
    ##################################################################
    def combine_predictions(self, predictions_dict, ensemble_method=None):
        raise NotImplementedError

    ##################################################################
    # 5) High-level predict
    ##################################################################
    def predict(
        self,
        X_test,
        rng_key,
        posterior="likelihood",
        num_samples=None,
        ensemble_method=None,
    ):
        preds_dict = self.predict_models(X_test, rng_key, posterior, num_samples)
        return self.combine_predictions(preds_dict, ensemble_method=ensemble_method)

    def fit_ensemble_weights(self, X_val, y_val):
        """
        Optional method to fit ensemble weights using a hold-out set (X_val, y_val).
        Sub-classes typically override or extend this to:
          - compute WAIC for each model (pseudo-BMA),
          - or do stacking by optimizing negative log-likelihood wrt weights.
        """
        raise NotImplementedError


class BNNEnsembleRegression(_BNNEnsembleBase):
    """
    For regression tasks.
    predictions: shape (S, N) from each model.
    We'll assume a Normal likelihood with a fixed sigma in computing log-likelihood,
    but you can adapt as needed.
    """

    def fit_ensemble_weights(self, X_val, y_val, sigma=1.0, method="waic"):
        """
        For regression:
          - If method="waic", compute WAIC for each model, store pseudo-BMA weights.
          - If method="stacking", solve for weights that minimize negative log-likelihood
            of the ensemble distribution on the validation set.
        """
        # 1) get predictions from each model
        rng_tmp = jax.random.PRNGKey(9999)  # or pass in a separate rng_key
        preds_dict = self.predict_models(X_val, rng_tmp, posterior="likelihood")

        # shapes: M models each -> (S, N)
        model_names = list(preds_dict.keys())

        if method == "waic":
            # For each model, compute log-likelihood
            waic_values = []
            for m in model_names:
                # shape (S, N)
                samples = preds_dict[m]
                # loglik_matrix shape (S, N)
                loglik_matrix = regression_log_likelihood(samples, y_val, sigma=sigma)
                waic_m = compute_waic_and_weights(loglik_matrix)
                waic_values.append(waic_m)

            # Convert WAIC to weights
            # weight_m = exp(-0.5 * waic_m) / sum over all models
            weights_unnorm = jnp.exp(-0.5 * jnp.array(waic_values))
            self.model_weights = normalize_weights(weights_unnorm)
            print("[EnsembleRegression] WAIC-based weights:", self.model_weights)

        elif method == "stacking":
            # 1) For stacking, we want to directly optimize:
            #    max_w sum_i log( sum_m w_m * p_m(y_i|x_i) ).
            #    We'll do a simple gradient-based approach with jax.

            # Convert (S, N) samples from each model into a full distribution:
            # we assume predictions ~ Normal(mu, sigma^2).
            # But for a simpler approach, take each model's mean for each data point as an approximation:
            model_means = {}
            for m in model_names:
                model_means[m] = preds_dict[m].mean(axis=0)  # shape (N,)

            # We'll do a small function to compute negative log-likelihood of mixture of Gaussians
            # Weighted mixture: E[y|x] ~ sum_m w_m * Normal(model_means[m], sigma^2)
            # For regression stacking, a straightforward approach is to treat it as
            # an additive mixture of means (like a single normal with mean = sum_m w_m mu_m).
            # This is a simplification. More advanced approach might handle sample-by-sample.

            def nll(weights):
                # weights shape (M,) on the simplex
                w = jnn_softmax(
                    weights
                )  # we can do a softmax so the sum=1, or other param
                # combined mean -> shape (N,)
                combined_mean = jnp.zeros_like(y_val, dtype=jnp.float32)
                for i, m in enumerate(model_names):
                    combined_mean += w[i] * model_means[m]

                # log-likelihood under Normal(combined_mean, sigma^2)
                loglik_matrix = regression_log_likelihood(
                    combined_mean[None, :], y_val, sigma=sigma
                )
                # shape (1, N)
                return -jnp.sum(loglik_matrix)

            # We'll define an init for weights
            from jax import grad

            jnn_softmax = lambda x: jax.nn.softmax(x)  # map R^M -> simplex

            # We'll do simple gradient descent
            M = len(model_names)
            w_init = jnp.ones((M,)) / M
            # We'll param in unconstrained space, then softmax
            theta_init = jnp.zeros_like(w_init)

            def train_stacking(theta_init, lr=0.01, steps=1000):
                theta = theta_init
                for step in range(steps):
                    g = grad(nll)(theta)
                    theta = theta - lr * g
                return theta

            theta_opt = train_stacking(theta_init)
            w_opt = jnn_softmax(theta_opt)
            self.model_weights = w_opt
            print("[EnsembleRegression] Stacking weights:", self.model_weights)

        else:
            raise ValueError(f"Unknown method for ensemble weight fitting: {method}")

    def combine_predictions(self, predictions_dict, ensemble_method=None):
        """
        Combine regression predictions from each model: shape (S, N).
        If self.model_weights is not None, we do a weighted combination. Otherwise fallback to uniform.
        """
        if ensemble_method is None:
            ensemble_method = self.ensemble_method

        model_names = list(predictions_dict.keys())
        stacked_preds = jnp.stack(
            [predictions_dict[m] for m in model_names], axis=0
        )  # (M, S, N)

        # If advanced weights exist, let's use them in any method that needs weighting
        if self.model_weights is None:
            # fallback: uniform
            w = jnp.ones((len(model_names),)) / len(model_names)
        else:
            w = self.model_weights

        if ensemble_method == "bma":
            # Weighted average in probability space. For regression, we are typically just combining samples.
            # We'll do a simple weighted average across M of the sample means.
            # Or more fully, we can treat each sample as "from model m" with probability w[m].
            # For demonstration, do a mixture of means:
            # shape (M, S, N)
            combined_preds = jnp.tensordot(w, stacked_preds, axes=1)  # shape (S, N)
            return combined_preds

        elif ensemble_method == "simple_average":
            # same but ignoring any model_weights
            combined_preds = stacked_preds.mean(axis=0)  # shape (S, N)
            return combined_preds

        elif ensemble_method == "stacking":
            # If stacking weights exist, do the weighted average
            combined_preds = jnp.tensordot(w, stacked_preds, axes=1)  # shape (S, N)
            return combined_preds

        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")


class BNNEnsembleBinary(_BNNEnsembleBase):
    """
    For binary classification tasks, each model's predictions are logits (S, N).
    """

    def fit_ensemble_weights(self, X_val, y_val, method="waic"):
        """
        Fit weights based on hold-out set (X_val, y_val) for binary classification.
        method='waic': compute WAIC for each model, use pseudo-BMA weighting.
        method='stacking': optimize negative log-likelihood of ensemble mixture on validation data.
        """
        rng_tmp = jax.random.PRNGKey(9998)
        preds_dict = self.predict_models(X_val, rng_tmp, posterior="logits")
        model_names = list(preds_dict.keys())

        if method == "waic":
            waic_values = []
            for m in model_names:
                # shape (S, N)
                logits_samps = preds_dict[m]
                # loglik_matrix shape (S, N)
                loglik_matrix = binary_log_likelihood(logits_samps, y_val)
                waic_m = compute_waic_and_weights(loglik_matrix)
                waic_values.append(waic_m)

            weights_unnorm = jnp.exp(-0.5 * jnp.array(waic_values))
            self.model_weights = normalize_weights(weights_unnorm)
            print("[EnsembleBinary] WAIC-based weights:", self.model_weights)

        elif method == "stacking":
            # We'll do a direct optimization of negative log-likelihood:
            # negative sum_i log( sum_m w_m * p_m(y_i|x_i) ),
            # where p_m(y_i|x_i) is Bernoulli(prob= sigmoid(logits_m)).
            # We'll do it in a simplified manner with a small gradient routine.

            # First gather probability predictions from each model's mean over samples:
            # For better fidelity, we might keep the entire distribution of samples,
            # but let's do a single average probability to keep it simpler.
            probs_dict = {}
            for m in model_names:
                logits_samps = preds_dict[m]  # shape (S, N)
                # average across S -> shape (N,)
                mean_logits = logits_samps.mean(axis=0)
                probs_dict[m] = jax.nn.sigmoid(mean_logits)  # shape (N,)

            # Now define an objective in terms of mixture weights w
            def nll(theta):
                w = jax.nn.softmax(theta)  # force simplex
                # mixture probability = sum_m w_m * p_m
                mixture_prob = jnp.zeros_like(y_val, dtype=jnp.float32)
                for i, mn in enumerate(model_names):
                    mixture_prob += w[i] * probs_dict[mn]
                # log-likelihood for each data point i
                # log p(y_i=1) = log(mixture_prob[i]), log p(y_i=0) = log(1 - mixture_prob[i])
                # We'll clamp to avoid log(0).
                mixture_prob = jnp.clip(mixture_prob, 1e-20, 1.0 - 1e-20)
                log_p_1 = jnp.log(mixture_prob)
                log_p_0 = jnp.log(1 - mixture_prob)
                ll = y_val * log_p_1 + (1 - y_val) * log_p_0
                return -jnp.sum(ll)  # negative log-likelihood

            from jax import grad

            M = len(model_names)
            theta_init = jnp.zeros((M,))

            def train_stacking(theta_init, lr=0.01, steps=1000):
                theta = theta_init
                for step in range(steps):
                    g = grad(nll)(theta)
                    theta = theta - lr * g
                return theta

            theta_opt = train_stacking(theta_init)
            self.model_weights = jax.nn.softmax(theta_opt)
            print("[EnsembleBinary] Stacking weights:", self.model_weights)
        else:
            raise ValueError(
                f"Unknown method {method} for binary ensemble weight fitting."
            )

    def combine_predictions(self, predictions_dict, ensemble_method=None):
        """
        Combine binary classification predictions (logits).
        If self.model_weights is not None, we do a weighted mixture in probability space for BMA/stacking.
        """
        if ensemble_method is None:
            ensemble_method = self.ensemble_method

        model_names = list(predictions_dict.keys())
        stacked_preds = jnp.stack(
            [predictions_dict[m] for m in model_names], axis=0
        )  # (M, S, N)

        # If advanced weights exist
        if self.model_weights is None:
            # fallback: uniform
            w = jnp.ones((len(model_names),)) / len(model_names)
        else:
            w = self.model_weights

        if ensemble_method == "bma":
            # 1) Convert logits to probabilities
            prob_preds = jax.nn.sigmoid(stacked_preds)  # (M, S, N)
            # 2) Weighted average across M
            # shape (S, N)
            weighted_probs = jnp.tensordot(w, prob_preds, axes=1)
            # 3) Convert back to logits
            combined_logits = jnp.log(weighted_probs) - jnp.log1p(-weighted_probs)
            return combined_logits

        elif ensemble_method == "simple_average":
            # direct average of logits ignoring self.model_weights
            combined_logits = stacked_preds.mean(axis=0)  # (S, N)
            return combined_logits

        elif ensemble_method == "stacking":
            # Weighted mixture in probability space if model_weights is set
            prob_preds = jax.nn.sigmoid(stacked_preds)
            weighted_probs = jnp.tensordot(w, prob_preds, axes=1)  # (S, N)
            combined_logits = jnp.log(weighted_probs) - jnp.log1p(-weighted_probs)
            return combined_logits

        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")


######################################################################
# BNNEnsembleMulticlass
######################################################################


class BNNEnsembleMulticlass(_BNNEnsembleBase):
    """
    For multiclass classification tasks, each model's predictions are logits: (S, N, C).
    """

    def fit_ensemble_weights(self, X_val, y_val, method="waic"):
        """
        Fit weights based on hold-out set (X_val, y_val) for multiclass classification.
        method='waic': compute WAIC for each model, use pseudo-BMA weighting
        method='stacking': optimize negative log-likelihood of ensemble mixture
        """
        rng_tmp = jax.random.PRNGKey(9997)
        preds_dict = self.predict_models(X_val, rng_tmp, posterior="logits")
        model_names = list(preds_dict.keys())

        if method == "waic":
            waic_values = []
            for m in model_names:
                logits_samps = preds_dict[m]  # (S, N, C)
                loglik_matrix = multiclass_log_likelihood(logits_samps, y_val)  # (S, N)
                waic_m = compute_waic_and_weights(loglik_matrix)
                waic_values.append(waic_m)

            weights_unnorm = jnp.exp(-0.5 * jnp.array(waic_values))
            self.model_weights = normalize_weights(weights_unnorm)
            print("[EnsembleMulticlass] WAIC-based weights:", self.model_weights)

        elif method == "stacking":
            # We'll define a simpler approach, using the mean of each model's logits to produce a single (N, C).
            model_mean_probs = {}
            for m in model_names:
                # shape (S, N, C)
                logits_samps = preds_dict[m]
                mean_logits = logits_samps.mean(axis=0)  # shape (N, C)
                # Convert to probabilities
                mean_probs = softmax_logits(mean_logits, axis=-1)  # (N, C)
                model_mean_probs[m] = mean_probs

            def nll(theta):
                w = jax.nn.softmax(theta)  # shape (M,)
                # mixture of probabilities: sum_m w_m * p_m
                mixture_probs = jnp.zeros_like(model_mean_probs[model_names[0]])
                for i, mn in enumerate(model_names):
                    mixture_probs += w[i] * model_mean_probs[mn]
                # negative log-likelihood of mixture
                mixture_probs = jnp.clip(mixture_probs, 1e-20, 1.0 - 1e-20)
                # pick correct class
                gather_idxs = y_val[None, :].astype(int)  # shape (1, N)
                gather_idxs = jnp.expand_dims(gather_idxs, axis=-1)  # (1, N, 1)
                correct_probs = jnp.take_along_axis(
                    mixture_probs[None, ...], gather_idxs, axis=-1
                ).squeeze(-1)
                # shape (1, N)
                return -jnp.sum(jnp.log(correct_probs + 1e-20))

            from jax import grad

            M = len(model_names)
            theta_init = jnp.zeros((M,))

            def train_stacking(theta_init, lr=0.01, steps=1000):
                theta = theta_init
                for step in range(steps):
                    g = grad(nll)(theta)
                    theta = theta - lr * g
                return theta

            theta_opt = train_stacking(theta_init)
            self.model_weights = jax.nn.softmax(theta_opt)
            print("[EnsembleMulticlass] Stacking weights:", self.model_weights)

        else:
            raise ValueError(
                f"Unknown method {method} for multiclass ensemble weight fitting."
            )

    def combine_predictions(self, predictions_dict, ensemble_method=None):
        if ensemble_method is None:
            ensemble_method = self.ensemble_method

        model_names = list(predictions_dict.keys())
        stacked_preds = jnp.stack(
            [predictions_dict[m] for m in model_names], axis=0
        )  # (M, S, N, C)

        if self.model_weights is None:
            w = jnp.ones((len(model_names),)) / len(model_names)
        else:
            w = self.model_weights

        if ensemble_method == "bma":
            # Convert each model's logits to probabilities, then do a weighted average
            prob_preds = jax.nn.softmax(stacked_preds, axis=-1)  # (M, S, N, C)
            weighted_probs = jnp.tensordot(w, prob_preds, axes=1)  # (S, N, C)
            # Convert back to logits by log(prob)
            combined_logits = jnp.log(weighted_probs + 1e-20)
            return combined_logits

        elif ensemble_method == "simple_average":
            # Just average the logits
            return stacked_preds.mean(axis=0)  # (S, N, C)

        elif ensemble_method == "stacking":
            # Weighted mixture in probability space
            prob_preds = jax.nn.softmax(stacked_preds, axis=-1)
            weighted_probs = jnp.tensordot(w, prob_preds, axes=1)  # (S, N, C)
            combined_logits = jnp.log(weighted_probs + 1e-20)
            return combined_logits

        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
