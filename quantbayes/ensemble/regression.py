import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor


class EnsembleRegression(BaseEstimator, RegressorMixin):
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
        Ensemble Regression Model supporting weighted averaging or stacking.

        Parameters:
            models (dict): name→estimator
            n_splits (int): number of folds (if cv=None)
            ensemble_method (str): "weighted_average" or "stacking"
            weights (dict or None): weights for weighted average
            meta_learner: estimator for stacking (defaults to LinearRegression)
            cv: cross‐validator instance (e.g. KFold or custom); if None, uses KFold(n_splits)
        """
        self.models = models
        self.n_splits = n_splits
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.cv = cv
        if ensemble_method not in ["weighted_average", "stacking"]:
            raise ValueError("ensemble_method must be 'weighted_average' or 'stacking'")
        if self.ensemble_method == "stacking":
            self.meta_learner = meta_learner or LinearRegression()
        else:
            self.meta_learner = None

        self.fitted_models_ = {}
        self.meta_fitted_ = None
        self.oof_predictions_ = None
        self.train_predictions_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.ensemble_method == "stacking":
            n_samples, n_models = X.shape[0], len(self.models)
            oof_preds = np.zeros((n_samples, n_models))
            # choose cv splitter
            splitter = self.cv or KFold(
                n_splits=self.n_splits, shuffle=True, random_state=42
            )
            model_names = list(self.models)

            for idx, name in enumerate(model_names):
                model = self.models[name]
                oof = np.zeros(n_samples)
                for train_idx, val_idx in splitter.split(X, y):
                    m = clone(model)
                    m.fit(X[train_idx], y[train_idx])
                    oof[val_idx] = m.predict(X[val_idx])
                oof_preds[:, idx] = oof
                # refit on full data
                fm = clone(model)
                fm.fit(X, y)
                self.fitted_models_[name] = fm

            # train meta‐learner
            self.meta_fitted_ = clone(self.meta_learner)
            self.meta_fitted_.fit(oof_preds, y)
            self.oof_predictions_ = oof_preds
            self.train_predictions_ = self.meta_fitted_.predict(oof_preds)

        else:  # weighted_average
            preds = []
            for name, model in self.models.items():
                m = clone(model)
                m.fit(X, y)
                self.fitted_models_[name] = m
                preds.append(m.predict(X).reshape(-1, 1))
            P = np.hstack(preds)
            if self.weights is None:
                w = np.ones(len(self.models)) / len(self.models)
            else:
                w = np.array([self.weights.get(n, 0) for n in self.models], dtype=float)
                if w.sum() == 0:
                    raise ValueError("Sum of weights cannot be zero.")
                w /= w.sum()
            self.train_predictions_ = P.dot(w)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Call fit first.")
        X = np.asarray(X)
        if self.ensemble_method == "stacking":
            base_preds = [
                self.fitted_models_[n].predict(X).reshape(-1, 1) for n in self.models
            ]
            return self.meta_fitted_.predict(np.hstack(base_preds))
        else:
            P = np.hstack(
                [m.predict(X).reshape(-1, 1) for m in self.fitted_models_.values()]
            )
            w = (
                (np.ones(P.shape[1]) / P.shape[1])
                if self.weights is None
                else np.array([self.weights.get(n, 0) for n in self.models])
            )
            w = w / w.sum()
            return P.dot(w)

    def fit_predict(self, X, y):
        """
        Convenience method that fits the model and returns predictions on X.
        """
        self.fit(X, y)
        return self.predict(X)

    def summary(self):
        """
        Print a summary of the ensemble model performance on the training data.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "The model has not been fitted yet. Call fit or fit_predict first."
            )

        print("Ensemble Regression Model Summary")
        print("---------------------------------")
        print(f"Ensemble Method: {self.ensemble_method}")
        print("Base Models:")
        for name, model in self.models.items():
            print(f" - {name}: {model.__class__.__name__}")
        if self.ensemble_method == "stacking":
            print(f"Meta Learner: {self.meta_learner.__class__.__name__}")
        if self.train_predictions_ is not None:
            # Compute performance metrics on the training set
            mse = mean_squared_error(
                self.train_predictions_, self.train_predictions_
            )  # Dummy example; normally you would compare against y
            r2 = r2_score(
                self.train_predictions_, self.train_predictions_
            )  # This will be 1.0 by design.
            # Instead, if we have stored y, we could compute performance; for demonstration, we simply note predictions are available.
            print("\nNote: Training predictions are stored in self.train_predictions_.")
            print(
                "You can compare these predictions with your true y values externally."
            )
        print("---------------------------------")

    def plot_predictions(self, X=None, y_true=None):
        """
        Plot the predictions vs true values. If X and y_true are provided,
        predictions are computed on X; otherwise, training predictions are used.
        """
        if X is not None and y_true is not None:
            y_pred = self.predict(X)
            y_actual = np.asarray(y_true)
            title = "Predictions vs True Values (Test Data)"
        elif self.train_predictions_ is not None:
            y_pred = self.train_predictions_
            y_actual = y_pred  # When only training predictions exist, we have no true values here.
            title = "Training Predictions (No true y provided)"
            print("Warning: True values not provided. Plot will show predictions only.")
        else:
            raise ValueError(
                "Either provide X and y_true, or call fit_predict first to use training predictions."
            )

        plt.figure(figsize=(8, 6))
        plt.scatter(y_actual, y_pred, alpha=0.7, edgecolor="k")
        plt.plot(
            [y_actual.min(), y_actual.max()],
            [y_actual.min(), y_actual.max()],
            "r--",
            lw=2,
        )
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(title)
        plt.grid(True)
        plt.show()


# Test the ensemble model if this module is run as the main program.
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    # Generate a synthetic regression dataset.
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define base models.
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=5),
    }

    # Choose ensemble method: "weighted_average" or "stacking"
    ensemble_method = "stacking"
    weights = None  # Only used if ensemble_method=="weighted_average"

    # Create ensemble instance.
    ensemble = EnsembleRegression(
        models,
        n_splits=5,
        ensemble_method=ensemble_method,
        weights=weights,
        meta_learner=None,
    )

    # Fit the ensemble and predict on the training data.
    train_preds = ensemble.fit_predict(X_train, y_train)

    # Print summary (you might extend this to include evaluation metrics comparing train_preds and y_train).
    ensemble.summary()

    # Evaluate on test set.
    test_preds = ensemble.predict(X_test)
    test_r2 = r2_score(y_test, test_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    print(f"\nTest R^2: {test_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    # Plot predictions on test data.
    ensemble.plot_predictions(X_test, y_test)
