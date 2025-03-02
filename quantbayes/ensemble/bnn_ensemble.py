import numpy as np
import jax
import jax.numpy as jnp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from scipy.special import expit  # for sigmoid

"""
Example Usage 
from sklearn.linear_model import LogisticRegression

# Suppose you have defined your BNN models like DeepSpectral, MixtureOfExperts, and BayesNet
base_models = {
    "deep_spectral": DeepSpectralBNN(), 
    "mixture_experts": MixtureOfExpertsBNN(),
    "fully_dense": FullyDenseBNN()
}

# Define a meta learner (for example, logistic regression)
meta_model = LogisticRegression()

# Instantiate the meta ensemble
ensemble = BNNMetaEnsemble(base_models=base_models, meta_model=meta_model, n_folds=5, rng_key=jax.random.PRNGKey(0))

# Fit on training data
ensemble.fit(X_train, y_train)

# Predict on test data
probs = ensemble.predict_proba(X_test)
preds = ensemble.predict(X_test)
"""

def compute_entropy_binary(probs):
    """Compute entropy for binary classification."""
    return -probs * np.log(probs + 1e-10) - (1 - probs) * np.log(1 - probs + 1e-10)

class BNNMetaEnsemble(BaseEstimator, ClassifierMixin):
    """
    A meta-learner that stacks BNN posterior summaries as features.

    Parameters:
      base_models: dict
          A dictionary of base BNN models (keys are names, values are model instances).
          Each model must implement .fit(X, y, rng_key), .predict(X, rng_key, posterior="logits"), and .get_samples if needed.
      meta_model: object
          A scikit-learn–compatible classifier (e.g., LogisticRegression, GradientBoostingClassifier).
      n_folds: int, default=5
          Number of folds for out-of-fold meta-feature generation.
      rng_key: jax.random.PRNGKey
          A base RNG key for reproducibility.
    """
    def __init__(self, base_models, meta_model, n_folds=5, rng_key=jax.random.PRNGKey(0)):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.rng_key = rng_key

    def _generate_meta_features(self, X, y, is_train=True):
        """
        Generate meta-features from base models.
        
        For each base model, compute summary statistics (mean and std of posterior probabilities)
        on out-of-fold predictions if is_train=True, otherwise use full training.
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        # We'll use two features per base model: mean and standard deviation.
        meta_features = np.zeros((n_samples, n_models * 2))
        
        # Create a stratified KFold for out-of-fold predictions
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        # To store predictions for training set
        meta_pred = {name: np.zeros(n_samples) for name in self.base_models}
        meta_std = {name: np.zeros(n_samples) for name in self.base_models}
        
        for name, model in self.base_models.items():
            # Prepare an array to hold predictions for current model
            oof_preds = np.zeros((n_samples,))  # will store mean probability
            oof_stds = np.zeros((n_samples,))
            # For each fold, retrain model and generate predictions on validation fold
            for train_index, val_index in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_index], X[val_index]
                y_train_fold = y[train_index]
                # Clone or reinitialize the model for each fold.
                # (Assuming your BNN model can be re-instantiated easily)
                model_fold = type(model)()  # reinitialize with default constructor; adjust as needed.
                # Optionally, you might need to set any hyperparameters
                # Fit on the training fold; update RNG key for each fold
                key, self.rng_key = jax.random.split(self.rng_key)
                model_fold.compile(num_warmup=500, num_samples=1000, num_chains=2)
                model_fold.fit(X_train_fold, y_train_fold, key)
                # Get posterior samples over logits on the validation fold
                key, self.rng_key = jax.random.split(self.rng_key)
                logits_samples = model_fold.predict(X_val_fold, key, posterior="logits")
                # Transform logits to probabilities
                probs_samples = expit(logits_samples)  # shape: (n_posterior_samples, n_val_samples)
                mean_probs = np.mean(probs_samples, axis=0)
                std_probs = np.std(probs_samples, axis=0)
                # Store in corresponding indices
                oof_preds[val_index] = mean_probs
                oof_stds[val_index] = std_probs
            # Save out-of-fold features for current model
            meta_pred[name] = oof_preds
            meta_std[name] = oof_stds

        # Now, construct the feature matrix by concatenating features from each model.
        # The ordering will be: [model1_mean, model1_std, model2_mean, model2_std, ...]
        feature_list = []
        for name in sorted(self.base_models.keys()):
            feature_list.append(meta_pred[name].reshape(-1, 1))
            feature_list.append(meta_std[name].reshape(-1, 1))
        meta_X = np.hstack(feature_list)
        return meta_X

    def fit(self, X, y):
        """
        Fit the base models with out-of-fold predictions to generate meta features,
        and then train the meta learner.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        # Generate meta features from out-of-fold predictions on training set
        self.meta_X_ = self._generate_meta_features(X, y, is_train=True)
        # Fit the meta model on these features
        self.meta_model.fit(self.meta_X_, y)
        # After meta feature generation, retrain each base model on the full training data.
        self.fitted_base_models_ = {}
        for name, model in self.base_models.items():
            # Reinitialize and train on full training set
            model_full = type(model)()
            model_full.compile(num_warmup=500, num_samples=1000, num_chains=2)
            key, self.rng_key = jax.random.split(self.rng_key)
            model_full.fit(X, y, key)
            self.fitted_base_models_[name] = model_full
        return self

    def _meta_features_for_test(self, X):
        """
        Given new data X, use the full-trained base models to generate meta features.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_models = len(self.fitted_base_models_)
        meta_features = np.zeros((n_samples, n_models * 2))
        feature_list = []
        for name in sorted(self.fitted_base_models_.keys()):
            model = self.fitted_base_models_[name]
            # Generate posterior samples on X_test using the full model.
            key, self.rng_key = jax.random.split(self.rng_key)
            logits_samples = model.predict(X, key, posterior="logits")
            probs_samples = expit(logits_samples)  # shape: (n_samples_post, n_samples)
            mean_probs = np.mean(probs_samples, axis=0)
            std_probs = np.std(probs_samples, axis=0)
            feature_list.append(mean_probs.reshape(-1, 1))
            feature_list.append(std_probs.reshape(-1, 1))
        meta_X_test = np.hstack(feature_list)
        return meta_X_test

    def predict_proba(self, X):
        """
        Predict probabilities using the meta learner.
        """
        meta_X = self._meta_features_for_test(X)
        return self.meta_model.predict_proba(meta_X)

    def predict(self, X):
        """
        Predict class labels using the meta learner.
        """
        meta_X = self._meta_features_for_test(X)
        return self.meta_model.predict(meta_X)

