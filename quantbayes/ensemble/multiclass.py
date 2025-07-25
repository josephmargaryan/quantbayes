import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, train_test_split


class EnsembleMulticlass(BaseEstimator, ClassifierMixin):
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
        Ensemble Multiclass Classifier.

        Parameters:
            models (dict): base classifiers
            n_splits (int): if cv=None
            ensemble_method (str): "weighted_average" or "stacking"
            weights (dict or None)
            meta_learner: for stacking (defaults to multinomial LogisticRegression)
            cv: cross‐validator instance (e.g. StratifiedKFold) or None
        """
        self.models = models
        self.n_splits = n_splits
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.cv = cv
        if ensemble_method not in ["weighted_average", "stacking"]:
            raise ValueError("ensemble_method must be 'weighted_average' or 'stacking'")
        if ensemble_method == "stacking":
            self.meta_learner = meta_learner or LogisticRegression(
                solver="lbfgs", max_iter=1000
            )
        else:
            self.meta_learner = None

        self.fitted_models_ = {}
        self.meta_fitted_ = None
        self.oof_predictions_ = None
        self.train_predictions_proba_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        names = list(self.models)
        classes = np.unique(y)
        C = len(classes)

        if self.ensemble_method == "stacking":
            splitter = self.cv or KFold(
                n_splits=self.n_splits, shuffle=True, random_state=42
            )
            n = X.shape[0]
            oof = np.zeros((n, len(names) * C))

            for idx, name in enumerate(names):
                model = self.models[name]
                tmp = np.zeros((n, C))
                for tr, va in splitter.split(X, y):
                    mc = clone(model)
                    mc.fit(X[tr], y[tr])
                    tmp[va] = mc.predict_proba(X[va])
                oof[:, idx * C : (idx + 1) * C] = tmp
                fm = clone(model)
                fm.fit(X, y)
                self.fitted_models_[name] = fm

            self.meta_fitted_ = clone(self.meta_learner)
            self.meta_fitted_.fit(oof, y)
            self.oof_predictions_ = oof
            self.train_predictions_proba_ = self.meta_fitted_.predict_proba(oof)

        else:
            prob_list = []
            for name, model in self.models.items():
                m = clone(model)
                m.fit(X, y)
                self.fitted_models_[name] = m
                prob_list.append(m.predict_proba(X))
            A = np.array(prob_list)  # (n_models, n_samples, n_classes)
            if self.weights is None:
                w = np.ones(len(names)) / len(names)
            else:
                w = np.array([self.weights.get(n, 0) for n in names], dtype=float)
                if w.sum() == 0:
                    raise ValueError("Sum of weights cannot be zero.")
                w /= w.sum()
            # weighted sum over models axis
            self.train_predictions_proba_ = np.tensordot(w, A, axes=1)

        self.is_fitted_ = True
        self.is_fitted_ = True
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Call fit first.")
        X = np.asarray(X)
        names = list(self.models)
        if self.ensemble_method == "stacking":
            blocks = [self.fitted_models_[n].predict_proba(X) for n in names]
            M = np.hstack(blocks)
            return self.meta_fitted_.predict_proba(M)
        else:
            A = np.array([m.predict_proba(X) for m in self.fitted_models_.values()])
            w = (
                np.ones(A.shape[0]) / A.shape[0]
                if self.weights is None
                else np.array([self.weights.get(n, 0) for n in names])
            )
            w = w / w.sum()
            return np.tensordot(w, A, axes=1)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit_predict(self, X, y):
        """
        Convenience method: fit the model and predict on X.
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

        print("Ensemble Multiclass Classification Model Summary")
        print("-------------------------------------------------")
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
        print("-------------------------------------------------")

    def plot_confusion(self, X, y_true, normalize=False, cmap="Blues"):
        """
        Plot the confusion matrix for the provided data.

        Parameters:
            X : array-like, feature matrix.
            y_true : array-like, true class labels.
            normalize (bool): If True, normalize the confusion matrix.
            cmap (str): Colormap.
        """
        from sklearn.metrics import ConfusionMatrixDisplay

        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

        fig, ax = plt.subplots(figsize=(8, 6))  # Create a single figure
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=cmap, ax=ax)  # Explicitly pass ax to prevent new figure creation

        plt.title("Confusion Matrix")
        plt.show()


# Test the ensemble multiclass classifier when running as the main program.
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    # Generate a synthetic multiclass classification dataset.
    # Here, we create 4 classes.
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=4,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define base classifiers.
    models = {
        "LogisticRegression": LogisticRegression(solver="lbfgs", max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(max_depth=7, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
        # Note: For SVC, probability=True is needed to call predict_proba.
        "SVC": SVC(probability=True, random_state=42),
    }

    # Choose ensemble method: "weighted_average" or "stacking"
    ensemble_method = (
        "stacking"  # You can change to "weighted_average" to test that method.
    )
    weights = None  # Only used for weighted_average if desired.

    # Create ensemble instance.
    ensemble = EnsembleMulticlass(
        models,
        n_splits=5,
        ensemble_method=ensemble_method,
        weights=weights,
        meta_learner=None,
    )  # Defaults to LogisticRegression for stacking.

    # Fit the ensemble on training data.
    ensemble.fit(X_train, y_train)
    train_preds = ensemble.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print("Training Accuracy: {:.4f}".format(train_acc))
    print("\nTraining Classification Report:")
    print(classification_report(y_train, train_preds))

    ensemble.summary()

    # Evaluate on test data.
    test_preds = ensemble.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_preds))

    # Plot confusion matrix on test data.
    ensemble.plot_confusion(X_test, y_test, normalize=True)
