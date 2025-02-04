import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class EnsembleClassificationModel(BaseEstimator, ClassifierMixin):
    def __init__(self, models, n_splits=5, ensemble_method="weighted_average", weights=None, meta_learner=None):
        """
        Ensemble Classification Model for binary classification.

        Parameters:
            models (dict): Dictionary of base classifiers. Keys are model names and values are model instances.
            n_splits (int): Number of folds for cross-validation (used in stacking).
            ensemble_method (str): "weighted_average" or "stacking"
            weights (dict or None): Dictionary mapping model names to weights (only used if ensemble_method=="weighted_average").
                                    If None, equal weights are used.
            meta_learner: The meta-classifier used for stacking. If None and stacking is chosen, defaults to LogisticRegression.
        """
        self.models = models
        self.n_splits = n_splits
        if ensemble_method not in ["weighted_average", "stacking"]:
            raise ValueError("ensemble_method must be either 'weighted_average' or 'stacking'")
        self.ensemble_method = ensemble_method
        self.weights = weights
        if self.ensemble_method == "stacking":
            self.meta_learner = meta_learner if meta_learner is not None else LogisticRegression()
        else:
            self.meta_learner = None

        # Containers for fitted models
        self.fitted_models_ = {}  # base classifiers trained on full data
        self.meta_fitted_ = None  # meta classifier (for stacking)
        self.oof_predictions_ = None  # out-of-fold predicted probabilities for stacking
        self.train_predictions_proba_ = None  # training-set predicted probabilities from ensemble
        self.is_fitted_ = False

    def fit(self, X, y):
        """
        Fit the ensemble classifier on the training data.
        
        For stacking, out-of-fold predicted probabilities (for the positive class) are generated using KFold.
        For weighted average, each base model is simply fit on the full data.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        model_names = list(self.models.keys())
        
        if self.ensemble_method == "stacking":
            n_samples = X.shape[0]
            n_models = len(model_names)
            # We'll store the positive-class probabilities from each model
            oof_preds = np.zeros((n_samples, n_models))
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

            # For each model, generate out-of-fold predictions
            for idx, model_name in enumerate(model_names):
                model = self.models[model_name]
                oof_model_preds = np.zeros(n_samples)
                for train_idx, val_idx in kf.split(X):
                    model_clone = clone(model)
                    model_clone.fit(X[train_idx], y[train_idx])
                    # Assumes that model supports predict_proba; using probability for the positive class (index 1)
                    oof_model_preds[val_idx] = model_clone.predict_proba(X[val_idx])[:, 1]
                oof_preds[:, idx] = oof_model_preds
                # Refit each base model on the full training data
                fitted_model = clone(model)
                fitted_model.fit(X, y)
                self.fitted_models_[model_name] = fitted_model

            # Train meta classifier on the out-of-fold predictions
            self.meta_fitted_ = clone(self.meta_learner)
            self.meta_fitted_.fit(oof_preds, y)
            # Store training-set ensemble predicted probabilities using meta classifier
            self.train_predictions_proba_ = self.meta_fitted_.predict_proba(oof_preds)[:, 1]
            self.oof_predictions_ = oof_preds

        elif self.ensemble_method == "weighted_average":
            preds = []
            for model_name in model_names:
                model = clone(self.models[model_name])
                model.fit(X, y)
                self.fitted_models_[model_name] = model
                preds.append(model.predict_proba(X)[:, 1].reshape(-1, 1))
            preds = np.hstack(preds)
            # Use equal weights if not provided
            if self.weights is None:
                weights_arr = np.ones(len(model_names)) / len(model_names)
            else:
                weights_arr = np.array([self.weights.get(name, 0) for name in model_names], dtype=float)
                if np.sum(weights_arr) == 0:
                    raise ValueError("Sum of provided weights cannot be zero.")
                weights_arr = weights_arr / np.sum(weights_arr)
            # Compute weighted average predicted probability (for positive class)
            self.train_predictions_proba_ = np.dot(preds, weights_arr)
        else:
            raise ValueError("Unknown ensemble_method provided.")
        
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        Returns an array of shape (n_samples, 2) where column 0 is the probability for class 0,
        and column 1 is for class 1.
        """
        if not self.is_fitted_:
            raise RuntimeError("The ensemble model must be fitted before predicting.")

        X = np.asarray(X)
        model_names = list(self.models.keys())

        if self.ensemble_method == "stacking":
            # Get base model probabilities (for positive class)
            base_preds = []
            for model_name in model_names:
                model = self.fitted_models_[model_name]
                base_preds.append(model.predict_proba(X)[:, 1].reshape(-1, 1))
            base_preds = np.hstack(base_preds)
            # Meta learner gives final probability for positive class
            proba_pos = self.meta_fitted_.predict_proba(base_preds)[:, 1]
        elif self.ensemble_method == "weighted_average":
            preds = []
            for model_name in model_names:
                model = self.fitted_models_[model_name]
                preds.append(model.predict_proba(X)[:, 1].reshape(-1, 1))
            preds = np.hstack(preds)
            if self.weights is None:
                weights_arr = np.ones(len(model_names)) / len(model_names)
            else:
                weights_arr = np.array([self.weights.get(name, 0) for name in model_names], dtype=float)
                weights_arr = weights_arr / np.sum(weights_arr)
            proba_pos = np.dot(preds, weights_arr)
        else:
            raise ValueError("Unknown ensemble_method provided.")
        
        # Return two-column probability: [1 - proba, proba]
        proba = np.vstack([1 - proba_pos, proba_pos]).T
        return proba

    def predict(self, X):
        """
        Predict class labels for X based on a 0.5 probability threshold.
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def fit_predict(self, X, y):
        """
        Convenience method: fit the model and return predictions on X.
        """
        self.fit(X, y)
        return self.predict(X)

    def summary(self):
        """
        Print a summary of the ensemble classifier.
        """
        if not self.is_fitted_:
            raise RuntimeError("The model has not been fitted yet. Call fit or fit_predict first.")
        
        print("Ensemble Classification Model Summary")
        print("-------------------------------------")
        print(f"Ensemble Method: {self.ensemble_method}")
        print("Base Models:")
        for name, model in self.models.items():
            print(f" - {name}: {model.__class__.__name__}")
        if self.ensemble_method == "stacking":
            print(f"Meta Learner: {self.meta_learner.__class__.__name__}")
        if self.train_predictions_proba_ is not None:
            print("\nNote: Training-set ensemble predicted probabilities are stored in self.train_predictions_proba_.")
            # Optionally, one can compute training accuracy and log loss here.
        print("-------------------------------------")

    def plot_roc(self, X, y_true):
        """
        Plot the ROC curve for the provided data.
        """
        if not self.is_fitted_:
            raise RuntimeError("The model has not been fitted yet. Call fit or fit_predict first.")
        y_true = np.asarray(y_true)
        proba = self.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"Ensemble ROC curve (area = {roc_auc:.2f})", color="b")
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


# Test the ensemble classifier when the module is run as the main program.
if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    # Generate a synthetic binary classification dataset.
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
                               random_state=42, flip_y=0.03, class_sep=1.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define base classifiers.
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(max_depth=5),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42)
    }

    # Choose ensemble method: "weighted_average" or "stacking"
    ensemble_method = "stacking"  # Change to "weighted_average" if desired.
    weights = None  # Only used if ensemble_method=="weighted_average"

    # Create an ensemble instance.
    ensemble = EnsembleClassificationModel(models, n_splits=5,
                                             ensemble_method=ensemble_method,
                                             weights=weights,
                                             meta_learner=None)  # Defaults to LogisticRegression

    # Fit the ensemble and predict on training data.
    ensemble.fit(X_train, y_train)
    train_preds = ensemble.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    train_logloss = log_loss(y_train, ensemble.predict_proba(X_train))
    print("Training Accuracy: {:.4f}".format(train_acc))
    print("Training Log Loss: {:.4f}".format(train_logloss))
    ensemble.summary()

    # Evaluate on test set.
    test_preds = ensemble.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_logloss = log_loss(y_test, ensemble.predict_proba(X_test))
    print("\nTest Accuracy: {:.4f}".format(test_acc))
    print("Test Log Loss: {:.4f}".format(test_logloss))

    # Plot the ROC curve on test data.
    ensemble.plot_roc(X_test, y_test)
