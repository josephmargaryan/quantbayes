import numpy as np
import jax
import jax.numpy as jnp
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.special import expit, softmax  # for sigmoid and softmax

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
    def __init__(self, base_models, meta_model, n_folds=5, rng_key=jax.random.PRNGKey(0)):
        self.base_models = base_models  # dict of base models
        self.meta_model = meta_model    # e.g. a GradientBoostingRegressor
        self.n_folds = n_folds
        self.rng_key = rng_key

    def _generate_meta_features(self, X, y, is_train=True):
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
                model_fold.compile(num_warmup=500, num_samples=1000, num_chains=2)
                key, self.rng_key = jax.random.split(self.rng_key)
                model_fold.fit(X_train_fold, y_train_fold, key)
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
        raise NotImplementedError("predict_proba is not implemented for regression tasks.")

        
class BNNEnsembleBinary(BaseEstimator, ClassifierMixin):
    """
    Ensemble for binary classification using BNN base models.
    
    Base models should implement:
      - fit(X, y, rng_key)
      - predict(X, rng_key, posterior="logits") which returns posterior samples of logits.
    For binary classification we convert logits to probabilities with a sigmoid.
    
    Meta model should be a scikit-learn classifier.
    """
    def __init__(self, base_models, meta_model, n_folds=5, rng_key=jax.random.PRNGKey(0)):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.rng_key = rng_key

    def _generate_meta_features(self, X, y, is_train=True):
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
                model_fold.compile(num_warmup=500, num_samples=1000, num_chains=2)
                key, self.rng_key = jax.random.split(self.rng_key)
                model_fold.fit(X_train_fold, y_train_fold, key)
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
    def __init__(self, base_models, meta_model, n_folds=5, rng_key=jax.random.PRNGKey(0)):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.rng_key = rng_key

    def _generate_meta_features(self, X, y, is_train=True):
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
                model_fold.compile(num_warmup=500, num_samples=1000, num_chains=2)
                key, self.rng_key = jax.random.split(self.rng_key)
                model_fold.fit(X_train_fold, y_train_fold, key)
                key, self.rng_key = jax.random.split(self.rng_key)
                logits_samples = model_fold.predict(X_val_fold, key, posterior="logits")
                # Convert logits to probabilities using softmax over last axis.
                # logits_samples shape: (n_samples_post, n_val, n_classes)
                probs_samples = softmax(logits_samples, axis=-1)
                mean_probs = np.mean(probs_samples, axis=0)  # shape: (n_val, n_classes)
                std_probs = np.std(probs_samples, axis=0)      # shape: (n_val, n_classes)
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
            std_probs = np.std(probs_samples, axis=0)      # shape: (n_samples, n_classes)
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
