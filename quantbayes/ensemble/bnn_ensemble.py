import numpy as np
import jax
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
from scipy.special import expit, softmax  # for sigmoid and softmax

__all__ = [
    "BNNEnsembleRegression",
    "BNNEnsembleBinary",
    "BNNEnsembleMulticlass",
    "BNNEnsembleForecast",
]

"""
Example usecase:
from sklearn.linear_model import LogisticRegression

# Instantiate your base BNN models for binary classification.
base_models = {
    "deep_spectral": DeepSpectralBNN(), 
    "mixture_experts": MixtureOfExpertsBNN(),
    "fully_dense": FullyDenseBNN()
}

meta_model = LogisticRegression()
ensemble_binary = BNNEnsembleBinary(base_models=base_models, meta_model=meta_model, n_folds=5, rng_key=jax.random.PRNGKey(0))

ensemble_binary.fit(X_train, y_train)
binary_probs = ensemble_binary.predict_proba(X_test)
binary_preds = ensemble_binary.predict(X_test)
"""


def compute_entropy_binary(probs):
    """Compute binary classification entropy."""
    return -probs * np.log(probs + 1e-10) - (1 - probs) * np.log(1 - probs + 1e-10)


class BNNEnsembleRegression(BaseEstimator, RegressorMixin):
    """
    Ensemble for regression using BNN base models.

    Base models should implement:
      - fit(X, y, rng_key)
      - predict(X, rng_key, posterior="logits") which returns posterior samples of predictions.
    The base model's predict() is assumed to output samples drawn from a Normal likelihood.

    Meta model should be a scikit-learn regressor.
    """

    def __init__(
        self,
        base_models,
        meta_model,
        n_folds=5,
        rng_key=jax.random.PRNGKey(0),
        **kwargs,
    ):
        self.base_models = base_models  # dict of base models
        self.meta_model = meta_model  # e.g. a GradientBoostingRegressor
        self.n_folds = n_folds
        self.rng_key = rng_key
        self.kwargs = kwargs

    def _generate_meta_features(self, X, y, is_train=True):
        num_warmup = self.kwargs.get("num_warmup", 500)
        num_samples = self.kwargs.get("num_samples", 1000)
        num_chains = self.kwargs.get("num_chains", 2)
        num_steps = self.kwargs.get("num_steps", 1000)
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        # For regression, we use two features per model: mean and std.
        meta_features = np.zeros((n_samples, n_models * 2))

        # Use regular KFold since y is continuous
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        meta_pred = {name: np.zeros(n_samples) for name in self.base_models}
        meta_std = {name: np.zeros(n_samples) for name in self.base_models}

        for name, model in self.base_models.items():
            oof_preds = np.zeros(n_samples)
            oof_stds = np.zeros(n_samples)
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                # Reinitialize model for each fold
                model_fold = type(model)()
                model_fold.compile(
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                )
                key, self.rng_key = jax.random.split(self.rng_key)
                model_fold.fit(X_train_fold, y_train_fold, key, num_steps=num_steps)
                # Get posterior samples on validation fold
                key, self.rng_key = jax.random.split(self.rng_key)
                samples = model_fold.predict(X_val_fold, key, posterior="logits")
                # For regression, no transformation is needed.
                mean_pred = np.mean(samples, axis=0)
                std_pred = np.std(samples, axis=0)
                oof_preds[val_idx] = mean_pred
                oof_stds[val_idx] = std_pred
            meta_pred[name] = oof_preds
            meta_std[name] = oof_stds

        feature_list = []
        for name in sorted(self.base_models.keys()):
            feature_list.append(meta_pred[name].reshape(-1, 1))
            feature_list.append(meta_std[name].reshape(-1, 1))
        meta_X = np.hstack(feature_list)
        return meta_X

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.meta_X_ = self._generate_meta_features(X, y, is_train=True)
        self.meta_model.fit(self.meta_X_, y)
        # Retrain each base model on full training data.
        self.fitted_base_models_ = {}
        for name, model in self.base_models.items():
            model_full = type(model)()
            model_full.compile(num_warmup=500, num_samples=1000, num_chains=2)
            key, self.rng_key = jax.random.split(self.rng_key)
            model_full.fit(X, y, key)
            self.fitted_base_models_[name] = model_full
        return self

    def _meta_features_for_test(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_models = len(self.fitted_base_models_)
        feature_list = []
        for name in sorted(self.fitted_base_models_.keys()):
            model = self.fitted_base_models_[name]
            key, self.rng_key = jax.random.split(self.rng_key)
            samples = model.predict(X, key, posterior="logits")
            mean_pred = np.mean(samples, axis=0)
            std_pred = np.std(samples, axis=0)
            feature_list.append(mean_pred.reshape(-1, 1))
            feature_list.append(std_pred.reshape(-1, 1))
        meta_X_test = np.hstack(feature_list)
        return meta_X_test

    def predict(self, X):
        meta_X = self._meta_features_for_test(X)
        return self.meta_model.predict(meta_X)

    def predict_proba(self, X):
        # For regression, predict_proba might not be relevant.
        raise NotImplementedError(
            "predict_proba is not implemented for regression tasks."
        )


class BNNEnsembleBinary(BaseEstimator, ClassifierMixin):
    """
    Ensemble for binary classification using BNN base models.

    Base models should implement:
      - fit(X, y, rng_key)
      - predict(X, rng_key, posterior="logits") which returns posterior samples of logits.
    For binary classification we convert logits to probabilities with a sigmoid.

    Meta model should be a scikit-learn classifier.
    """

    def __init__(
        self,
        base_models,
        meta_model,
        n_folds=5,
        rng_key=jax.random.PRNGKey(0),
        **kwargs,
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.rng_key = rng_key
        self.kwargs = kwargs

    def _generate_meta_features(self, X, y, is_train=True):
        num_warmup = self.kwargs.get("num_warmup", 500)
        num_samples = self.kwargs.get("num_samples", 1000)
        num_chains = self.kwargs.get("num_chains", 2)
        num_steps = self.kwargs.get("num_steps", 1000)
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models * 2))

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        meta_pred = {name: np.zeros(n_samples) for name in self.base_models}
        meta_std = {name: np.zeros(n_samples) for name in self.base_models}

        for name, model in self.base_models.items():
            oof_preds = np.zeros(n_samples)
            oof_stds = np.zeros(n_samples)
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                model_fold = type(model)()
                model_fold.compile(
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                )
                key, self.rng_key = jax.random.split(self.rng_key)
                model_fold.fit(X_train_fold, y_train_fold, key, num_steps=num_steps)
                key, self.rng_key = jax.random.split(self.rng_key)
                logits_samples = model_fold.predict(X_val_fold, key, posterior="logits")
                # Convert logits to probabilities using sigmoid.
                probs_samples = expit(logits_samples)
                mean_probs = np.mean(probs_samples, axis=0)
                std_probs = np.std(probs_samples, axis=0)
                oof_preds[val_idx] = mean_probs
                oof_stds[val_idx] = std_probs
            meta_pred[name] = oof_preds
            meta_std[name] = oof_stds

        feature_list = []
        for name in sorted(self.base_models.keys()):
            feature_list.append(meta_pred[name].reshape(-1, 1))
            feature_list.append(meta_std[name].reshape(-1, 1))
        meta_X = np.hstack(feature_list)
        return meta_X

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.meta_X_ = self._generate_meta_features(X, y, is_train=True)
        self.meta_model.fit(self.meta_X_, y)
        self.fitted_base_models_ = {}
        for name, model in self.base_models.items():
            model_full = type(model)()
            model_full.compile(num_warmup=500, num_samples=1000, num_chains=2)
            key, self.rng_key = jax.random.split(self.rng_key)
            model_full.fit(X, y, key)
            self.fitted_base_models_[name] = model_full
        return self

    def _meta_features_for_test(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        feature_list = []
        for name in sorted(self.fitted_base_models_.keys()):
            model = self.fitted_base_models_[name]
            key, self.rng_key = jax.random.split(self.rng_key)
            logits_samples = model.predict(X, key, posterior="logits")
            probs_samples = expit(logits_samples)
            mean_probs = np.mean(probs_samples, axis=0)
            std_probs = np.std(probs_samples, axis=0)
            feature_list.append(mean_probs.reshape(-1, 1))
            feature_list.append(std_probs.reshape(-1, 1))
        meta_X_test = np.hstack(feature_list)
        return meta_X_test

    def predict_proba(self, X):
        meta_X = self._meta_features_for_test(X)
        return self.meta_model.predict_proba(meta_X)

    def predict(self, X):
        meta_X = self._meta_features_for_test(X)
        return self.meta_model.predict(meta_X)


class BNNEnsembleMulticlass(BaseEstimator, ClassifierMixin):
    """
    Ensemble for multiclass classification using BNN base models.

    Base models should implement:
      - fit(X, y, rng_key)
      - predict(X, rng_key, posterior="logits") which returns posterior samples of logits.
    For multiclass, convert logits to probabilities using softmax.

    Meta model should be a scikit-learn classifier that supports multiclass tasks.
    """

    def __init__(
        self,
        base_models,
        meta_model,
        n_folds=5,
        rng_key=jax.random.PRNGKey(0),
        **kwargs,
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.rng_key = rng_key
        self.kwargs = kwargs

    def _generate_meta_features(self, X, y, is_train=True):
        num_warmup = self.kwargs.get("num_warmup", 500)
        num_samples = self.kwargs.get("num_samples", 1000)
        num_chains = self.kwargs.get("num_chains", 2)
        num_steps = self.kwargs.get("num_steps", 1000)
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        # Here each base model returns logits of shape (n_samples_post, n_samples, n_classes).
        # We will flatten mean and std across classes.
        meta_features = None  # to be built dynamically

        # For multiclass, we use KFold (stratified split can be used if needed)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        meta_means = {}
        meta_stds = {}
        for name in self.base_models:
            meta_means[name] = None
            meta_stds[name] = None

        for name, model in self.base_models.items():
            all_means = []
            all_stds = []
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                model_fold = type(model)()
                model_fold.compile(
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                )
                key, self.rng_key = jax.random.split(self.rng_key)
                model_fold.fit(X_train_fold, y_train_fold, key, num_steps=num_steps)
                key, self.rng_key = jax.random.split(self.rng_key)
                logits_samples = model_fold.predict(X_val_fold, key, posterior="logits")
                # Convert logits to probabilities using softmax over last axis.
                # logits_samples shape: (n_samples_post, n_val, n_classes)
                probs_samples = softmax(logits_samples, axis=-1)
                mean_probs = np.mean(probs_samples, axis=0)  # shape: (n_val, n_classes)
                std_probs = np.std(probs_samples, axis=0)  # shape: (n_val, n_classes)
                all_means.append((val_idx, mean_probs))
                all_stds.append((val_idx, std_probs))
            # Combine across folds into full arrays
            full_mean = np.zeros((n_samples, mean_probs.shape[-1]))
            full_std = np.zeros((n_samples, std_probs.shape[-1]))
            for idx, m in all_means:
                full_mean[idx, :] = m
            for idx, s in all_stds:
                full_std[idx, :] = s
            meta_means[name] = full_mean
            meta_stds[name] = full_std

        # Concatenate features from each model: for each base model, flatten mean and std.
        feature_list = []
        for name in sorted(self.base_models.keys()):
            feature_list.append(meta_means[name])
            feature_list.append(meta_stds[name])
        meta_X = np.hstack(feature_list)
        return meta_X

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.meta_X_ = self._generate_meta_features(X, y, is_train=True)
        self.meta_model.fit(self.meta_X_, y)
        self.fitted_base_models_ = {}
        for name, model in self.base_models.items():
            model_full = type(model)()
            model_full.compile(num_warmup=500, num_samples=1000, num_chains=2)
            key, self.rng_key = jax.random.split(self.rng_key)
            model_full.fit(X, y, key)
            self.fitted_base_models_[name] = model_full
        return self

    def _meta_features_for_test(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        feature_list = []
        for name in sorted(self.fitted_base_models_.keys()):
            model = self.fitted_base_models_[name]
            key, self.rng_key = jax.random.split(self.rng_key)
            logits_samples = model.predict(X, key, posterior="logits")
            probs_samples = softmax(logits_samples, axis=-1)
            mean_probs = np.mean(probs_samples, axis=0)  # shape: (n_samples, n_classes)
            std_probs = np.std(probs_samples, axis=0)  # shape: (n_samples, n_classes)
            feature_list.append(mean_probs)
            feature_list.append(std_probs)
        meta_X_test = np.hstack(feature_list)
        return meta_X_test

    def predict_proba(self, X):
        meta_X = self._meta_features_for_test(X)
        return self.meta_model.predict_proba(meta_X)

    def predict(self, X):
        meta_X = self._meta_features_for_test(X)
        return self.meta_model.predict(meta_X)


class BNNEnsembleForecast(BaseEstimator, RegressorMixin):
    """
    Ensemble forecasting for time series using Bayesian Neural Network (BNN) models.

    Base models must implement:
      - fit(X, y, rng_key)
      - predict(X, rng_key, posterior="logits") which returns posterior samples.

    The meta model is any scikit-learn regressor that is trained on meta features derived from the
    base model predictions (e.g., mean and standard deviation of the forecasts).
    """

    def __init__(
        self,
        base_models,
        meta_model,
        forecast_horizon=1,
        n_splits=5,
        rng_key=None,
        **kwargs,
    ):
        """
        Parameters:
            base_models (dict): Dictionary of BNN base models.
            meta_model: A scikit-learn regressor to be used as the meta learner.
            forecast_horizon (int): Number of time steps ahead to forecast.
            n_splits (int): Number of splits for time series cross validation.
            rng_key: JAX random key (if None, a default key is created).
            kwargs: Additional keyword arguments (e.g., for inference parameters).
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.forecast_horizon = forecast_horizon
        self.n_splits = n_splits
        self.kwargs = kwargs
        if rng_key is None:
            import jax

            self.rng_key = jax.random.PRNGKey(0)
        else:
            self.rng_key = rng_key

    def _generate_meta_features(self, X, y):
        """
        Generate meta features using TimeSeriesSplit.

        For each base model, we generate out-of-fold predictions on a time-series split.
        For forecasting, the meta features are the mean and standard deviation of the posterior
        predictions (assumed to be samples from a Normal likelihood) for a given forecast horizon.

        Parameters:
            X (np.ndarray): Feature matrix (e.g., lagged observations).
            y (np.ndarray): True target values.

        Returns:
            meta_X (np.ndarray): Meta feature matrix.
        """
        num_warmup = self.kwargs.get("num_warmup", 500)
        num_samples = self.kwargs.get("num_samples", 1000)
        num_chains = self.kwargs.get("num_chains", 2)
        num_steps = self.kwargs.get("num_steps", 1000)

        n_samples = X.shape[0]
        n_models = len(self.base_models)
        # For each base model, we extract two features: the forecast mean and std.
        meta_features = np.zeros((n_samples, n_models * 2))

        # Use TimeSeriesSplit to preserve temporal ordering.
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        # To store meta predictions for each base model
        meta_pred = {name: np.zeros(n_samples) for name in self.base_models}
        meta_std = {name: np.zeros(n_samples) for name in self.base_models}

        for name, model in self.base_models.items():
            oof_preds = np.zeros(n_samples)
            oof_stds = np.zeros(n_samples)

            # Loop over time series splits. In each fold, training always comes before validation.
            for train_idx, val_idx in tscv.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                # Instantiate a fresh instance of the base model for this fold.
                model_fold = type(model)()
                model_fold.compile(
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                )
                # Split the random key for reproducibility.
                import jax

                key, self.rng_key = jax.random.split(self.rng_key)
                model_fold.fit(X_train_fold, y_train_fold, key, num_steps=num_steps)

                # For forecasting, we assume the model's predict method is set up for forecast_horizon steps.
                key, self.rng_key = jax.random.split(self.rng_key)
                samples = model_fold.predict(X_val_fold, key, posterior="logits")
                # samples shape: (n_posterior_samples, n_val)
                mean_pred = np.mean(samples, axis=0)
                std_pred = np.std(samples, axis=0)

                oof_preds[val_idx] = mean_pred
                oof_stds[val_idx] = std_pred

            meta_pred[name] = oof_preds
            meta_std[name] = oof_stds

        # Concatenate features in a fixed order across base models.
        feature_list = []
        for name in sorted(self.base_models.keys()):
            feature_list.append(meta_pred[name].reshape(-1, 1))
            feature_list.append(meta_std[name].reshape(-1, 1))
        meta_X = np.hstack(feature_list)
        return meta_X

    def fit(self, X, y):
        """
        Fit the ensemble forecast model.

        Generates meta features using out-of-fold predictions from base models (via TimeSeriesSplit)
        and trains the meta model on these features. Also retrains each base model on the full data.

        Parameters:
            X (np.ndarray): Feature matrix (e.g., lagged observations).
            y (np.ndarray): True target values.

        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Generate meta features using time-series cross validation.
        self.meta_X_ = self._generate_meta_features(X, y)
        self.meta_model.fit(self.meta_X_, y)

        # Refit each base model on the full training data.
        self.fitted_base_models_ = {}
        for name, model in self.base_models.items():
            model_full = type(model)()
            model_full.compile(
                num_warmup=self.kwargs.get("num_warmup", 500),
                num_samples=self.kwargs.get("num_samples", 1000),
                num_chains=self.kwargs.get("num_chains", 2),
            )
            import jax

            key, self.rng_key = jax.random.split(self.rng_key)
            model_full.fit(X, y, key, num_steps=self.kwargs.get("num_steps", 1000))
            self.fitted_base_models_[name] = model_full
        return self

    def _meta_features_for_test(self, X):
        """
        Generate meta features for test/forecast data.

        For each fitted base model, we compute the mean and standard deviation of the posterior forecast.

        Parameters:
            X (np.ndarray): Feature matrix for which to generate forecasts.

        Returns:
            meta_X_test (np.ndarray): Meta feature matrix for test data.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        feature_list = []

        for name in sorted(self.fitted_base_models_.keys()):
            model = self.fitted_base_models_[name]
            import jax

            key, self.rng_key = jax.random.split(self.rng_key)
            samples = model.predict(X, key, posterior="logits")
            mean_pred = np.mean(samples, axis=0)
            std_pred = np.std(samples, axis=0)
            feature_list.append(mean_pred.reshape(-1, 1))
            feature_list.append(std_pred.reshape(-1, 1))
        meta_X_test = np.hstack(feature_list)
        return meta_X_test

    def predict(self, X):
        """
        Generate forecasts using the ensemble.

        The meta features are generated using the base models' predictions on X and then
        passed to the meta model.

        Parameters:
            X (np.ndarray): Feature matrix for forecasting.

        Returns:
            Forecasts (np.ndarray)
        """
        meta_X = self._meta_features_for_test(X)
        return self.meta_model.predict(meta_X)

    def forecast(self, X_future):
        """
        Alias for predict() to emphasize forecasting.

        Parameters:
            X_future (np.ndarray): Feature matrix for future time steps.

        Returns:
            Forecasted values.
        """
        return self.predict(X_future)

    def summary(self):
        """
        Print a summary of the ensemble forecasting model.
        """
        print("BNN Ensemble Forecasting Model Summary")
        print("--------------------------------------")
        print(f"Forecast Horizon: {self.forecast_horizon}")
        print(f"Number of Base Models: {len(self.base_models)}")
        print(f"Ensemble Meta Model: {self.meta_model.__class__.__name__}")
        print(f"Time Series CV Splits: {self.n_splits}")
        print("--------------------------------------")

    def plot_forecast(self, X, y_true):
        """
        Plot the forecasted values against the true values.

        Parameters:
            X (np.ndarray): Feature matrix for forecasting.
            y_true (np.ndarray): True target values.
        """
        y_pred = self.predict(X)
        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label="True Values", marker="o")
        plt.plot(y_pred, label="Forecasted Values", marker="x")
        plt.xlabel("Time")
        plt.ylabel("Forecast")
        plt.title("BNN Ensemble Forecast vs True Values")
        plt.legend()
        plt.grid(True)
        plt.show()
