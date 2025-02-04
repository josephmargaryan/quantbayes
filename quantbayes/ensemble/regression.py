import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


class EnsembleRegressionModel(BaseEstimator, RegressorMixin):
    def __init__(self, models, n_splits=5, ensemble_method="weighted_average", weights=None, meta_learner=None):
        """
        Ensemble Regression Model that supports two ensemble methods: 
            - "weighted_average": predictions are the weighted average of base model predictions.
            - "stacking": a meta-learner is trained on the out-of-fold predictions from base models.
        
        Parameters:
            models (dict): Dictionary of base models. Keys are model names and values are model instances.
            n_splits (int): Number of folds for cross-validation (used in stacking).
            ensemble_method (str): "weighted_average" or "stacking"
            weights (dict or None): Dictionary mapping model names to weights (only used if ensemble_method=="weighted_average").
                                    If None, equal weights are used.
            meta_learner: The meta-model used for stacking. If None and stacking is chosen, defaults to LinearRegression.
        """
        self.models = models
        self.n_splits = n_splits
        if ensemble_method not in ["weighted_average", "stacking"]:
            raise ValueError("ensemble_method must be either 'weighted_average' or 'stacking'")
        self.ensemble_method = ensemble_method
        self.weights = weights
        # For stacking, set up a meta learner
        if self.ensemble_method == "stacking":
            self.meta_learner = meta_learner if meta_learner is not None else LinearRegression()
        else:
            self.meta_learner = None

        # Containers for fitted models (if refitting is needed)
        self.fitted_models_ = {}  # will store the base models trained on full data
        self.meta_fitted_ = None  # will store the fitted meta learner in stacking
        self.oof_predictions_ = None  # out-of-fold predictions (for stacking)
        self.train_predictions_ = None  # final predictions on training data
        self.is_fitted_ = False

    def fit(self, X, y):
        """
        Fit the ensemble model on training data.

        For stacking, use K-Fold to generate out-of-fold predictions for meta-learner training.
        For weighted average, simply fit each base model on the full data.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if self.ensemble_method == "stacking":
            # Initialize out-of-fold predictions array: shape = (n_samples, n_models)
            n_samples = X.shape[0]
            n_models = len(self.models)
            oof_preds = np.zeros((n_samples, n_models))
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            model_names = list(self.models.keys())
            
            # For each base model, perform K-Fold CV to generate out-of-fold predictions
            for idx, model_name in enumerate(model_names):
                model = self.models[model_name]
                oof_pred = np.zeros(n_samples)
                for train_idx, val_idx in kf.split(X):
                    model_clone = clone(model)
                    model_clone.fit(X[train_idx], y[train_idx])
                    oof_pred[val_idx] = model_clone.predict(X[val_idx])
                oof_preds[:, idx] = oof_pred
                # Refit the base model on the full data
                fitted_model = clone(model)
                fitted_model.fit(X, y)
                self.fitted_models_[model_name] = fitted_model

            # Train the meta learner on the out-of-fold predictions
            self.meta_fitted_ = clone(self.meta_learner)
            self.meta_fitted_.fit(oof_preds, y)
            # Store predictions on training data for summary/plotting
            meta_train_pred = self.meta_fitted_.predict(oof_preds)
            self.train_predictions_ = meta_train_pred
            self.oof_predictions_ = oof_preds

        elif self.ensemble_method == "weighted_average":
            # Fit each base model on the full data.
            self.fitted_models_ = {}
            model_names = list(self.models.keys())
            preds = []
            for model_name in model_names:
                model = clone(self.models[model_name])
                model.fit(X, y)
                self.fitted_models_[model_name] = model
                preds.append(model.predict(X).reshape(-1, 1))
            preds = np.hstack(preds)
            # Set equal weights if not provided
            if self.weights is None:
                weights_arr = np.ones(len(model_names)) / len(model_names)
            else:
                # Ensure the provided weights follow the order of model_names and sum to 1.
                weights_arr = np.array([self.weights.get(name, 0) for name in model_names], dtype=float)
                if np.sum(weights_arr) == 0:
                    raise ValueError("Sum of provided weights cannot be zero.")
                weights_arr = weights_arr / np.sum(weights_arr)
            # Compute weighted average predictions
            self.train_predictions_ = np.dot(preds, weights_arr)
        else:
            raise ValueError("Unknown ensemble_method provided.")
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict using the fitted ensemble model.
        """
        if not self.is_fitted_:
            raise RuntimeError("The ensemble model must be fitted before predicting.")

        X = np.asarray(X)
        model_names = list(self.models.keys())

        if self.ensemble_method == "stacking":
            # Get predictions from each base model
            base_preds = []
            for model_name in model_names:
                model = self.fitted_models_[model_name]
                base_preds.append(model.predict(X).reshape(-1, 1))
            base_preds = np.hstack(base_preds)
            # Use meta learner to make final predictions
            final_pred = self.meta_fitted_.predict(base_preds)
            return final_pred

        elif self.ensemble_method == "weighted_average":
            # Get predictions from each base model
            preds = []
            for model_name in model_names:
                model = self.fitted_models_[model_name]
                preds.append(model.predict(X).reshape(-1, 1))
            preds = np.hstack(preds)
            # Use provided or equal weights
            if self.weights is None:
                weights_arr = np.ones(len(model_names)) / len(model_names)
            else:
                weights_arr = np.array([self.weights.get(name, 0) for name in model_names], dtype=float)
                weights_arr = weights_arr / np.sum(weights_arr)
            final_pred = np.dot(preds, weights_arr)
            return final_pred

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
            raise RuntimeError("The model has not been fitted yet. Call fit or fit_predict first.")
        
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
            mse = mean_squared_error(self.train_predictions_, self.train_predictions_)  # Dummy example; normally you would compare against y
            r2 = r2_score(self.train_predictions_, self.train_predictions_)  # This will be 1.0 by design.
            # Instead, if we have stored y, we could compute performance; for demonstration, we simply note predictions are available.
            print("\nNote: Training predictions are stored in self.train_predictions_.")
            print("You can compare these predictions with your true y values externally.")
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
            raise ValueError("Either provide X and y_true, or call fit_predict first to use training predictions.")

        plt.figure(figsize=(8, 6))
        plt.scatter(y_actual, y_pred, alpha=0.7, edgecolor="k")
        plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define base models.
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=5),
    }

    # Choose ensemble method: "weighted_average" or "stacking"
    ensemble_method = "stacking"
    weights = None  # Only used if ensemble_method=="weighted_average"
    
    # Create ensemble instance.
    ensemble = EnsembleRegressionModel(models, n_splits=5,
                                         ensemble_method=ensemble_method,
                                         weights=weights,
                                         meta_learner=None)
    
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
