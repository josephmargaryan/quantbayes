import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, log_loss, roc_curve
from sklearn.model_selection import KFold, train_test_split


class EnsembleBinary(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        models,
        n_splits=5,
        ensemble_method="weighted_average",
        weights=None,
        meta_learner=None,
        cv=None,  # new parameter
    ):
        """
        Ensemble Binary Classifier (weighted average or stacking).

        Parameters:
            models (dict): base classifiers
            n_splits (int): if cv=None
            ensemble_method (str): "weighted_average" or "stacking"
            weights (dict or None)
            meta_learner: for stacking (defaults to LogisticRegression)
            cv: cross‐validator instance (e.g. StratifiedKFold) or None
        """
        self.models = models
        self.n_splits = n_splits
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.cv = cv
        if ensemble_method not in ["weighted_average", "stacking"]:
            raise ValueError("ensemble_method must be 'weighted_average' or 'stacking'")
        self.meta_learner = (
            meta_learner or LogisticRegression()
            if ensemble_method == "stacking"
            else None
        )

        self.fitted_models_ = {}
        self.meta_fitted_ = None
        self.oof_predictions_ = None
        self.train_predictions_proba_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        names = list(self.models)

        if self.ensemble_method == "stacking":
            splitter = self.cv or KFold(
                n_splits=self.n_splits, shuffle=True, random_state=42
            )
            n, m = X.shape[0], len(names)
            oof = np.zeros((n, m))

            for idx, name in enumerate(names):
                model = self.models[name]
                temp = np.zeros(n)
                for tr, va in splitter.split(X, y):
                    mc = clone(model)
                    mc.fit(X[tr], y[tr])
                    temp[va] = mc.predict_proba(X[va])[:, 1]
                oof[:, idx] = temp
                fm = clone(model)
                fm.fit(X, y)
                self.fitted_models_[name] = fm

            self.meta_fitted_ = clone(self.meta_learner)
            self.meta_fitted_.fit(oof, y)
            self.oof_predictions_ = oof
            self.train_predictions_proba_ = self.meta_fitted_.predict_proba(oof)[:, 1]

        else:
            probs = []
            for name, model in self.models.items():
                m = clone(model)
                m.fit(X, y)
                self.fitted_models_[name] = m
                probs.append(m.predict_proba(X)[:, 1].reshape(-1, 1))
            P = np.hstack(probs)
            if self.weights is None:
                w = np.ones(P.shape[1]) / P.shape[1]
            else:
                w = np.array([self.weights.get(n, 0) for n in names], dtype=float)
                if w.sum() == 0:
                    raise ValueError("Sum of weights cannot be zero.")
                w /= w.sum()
            self.train_predictions_proba_ = P.dot(w)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Call fit first.")
        X = np.asarray(X)
        names = list(self.models)
        if self.ensemble_method == "stacking":
            P = np.hstack(
                [
                    self.fitted_models_[n].predict_proba(X)[:, 1].reshape(-1, 1)
                    for n in names
                ]
            )
            pos = self.meta_fitted_.predict_proba(P)[:, 1]
        else:
            P = np.hstack(
                [
                    m.predict_proba(X)[:, 1].reshape(-1, 1)
                    for m in self.fitted_models_.values()
                ]
            )
            w = (
                np.ones(P.shape[1]) / P.shape[1]
                if self.weights is None
                else np.array([self.weights.get(n, 0) for n in names])
            )
            w = w / w.sum()
            pos = P.dot(w)
        return np.vstack([1 - pos, pos]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

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
            raise RuntimeError(
                "The model has not been fitted yet. Call fit or fit_predict first."
            )

        print("Ensemble Classification Model Summary")
        print("-------------------------------------")
        print(f"Ensemble Method: {self.ensemble_method}")
        print("Base Models:")
        for name, model in self.models.items():
            print(f" - {name}: {model.__class__.__name__}")
        if self.ensemble_method == "stacking":
            print(f"Meta Learner: {self.meta_learner.__class__.__name__}")
        if self.train_predictions_proba_ is not None:
            print(
                "\nNote: Training-set ensemble predicted probabilities are stored in self.train_predictions_proba_."
            )
            # Optionally, one can compute training accuracy and log loss here.
        print("-------------------------------------")

    def plot_roc(self, X, y_true):
        """
        Plot the ROC curve for the provided data.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "The model has not been fitted yet. Call fit or fit_predict first."
            )
        y_true = np.asarray(y_true)
        proba = self.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, label=f"Ensemble ROC curve (area = {roc_auc:.2f})", color="b"
        )
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


# Test the ensemble classifier when the module is run as the main program.
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    # Generate a synthetic binary classification dataset.
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42,
        flip_y=0.03,
        class_sep=1.0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define base classifiers.
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(max_depth=5),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
    }

    # Choose ensemble method: "weighted_average" or "stacking"
    ensemble_method = "stacking"  # Change to "weighted_average" if desired.
    weights = None  # Only used if ensemble_method=="weighted_average"

    # Create an ensemble instance.
    ensemble = EnsembleBinary(
        models,
        n_splits=5,
        ensemble_method=ensemble_method,
        weights=weights,
        meta_learner=None,
    )  # Defaults to LogisticRegression

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
