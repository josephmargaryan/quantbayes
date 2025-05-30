import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


class EnsembleForecast(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        models,
        forecast_horizon=1,
        n_splits=5,
        ensemble_method="weighted_average",
        weights=None,
        meta_learner=None,
    ):
        """
        Ensemble Forecasting Model for time series data.

        Parameters:
            models (dict): Dictionary of base forecasting models. Keys are model names and values are model instances.
            forecast_horizon (int): Number of future time steps to forecast.
            n_splits (int): Number of splits for time series cross-validation (used in stacking).
            ensemble_method (str): "weighted_average" or "stacking"
            weights (dict or None): Dictionary mapping model names to weights (only used if ensemble_method=="weighted_average").
                                    If None, equal weights are used.
            meta_learner: The meta model used for stacking. If None and stacking is chosen, defaults to LinearRegression.
        """
        self.models = models
        self.forecast_horizon = forecast_horizon
        self.n_splits = n_splits

        if ensemble_method not in ["weighted_average", "stacking"]:
            raise ValueError(
                "ensemble_method must be either 'weighted_average' or 'stacking'"
            )
        self.ensemble_method = ensemble_method
        self.weights = weights

        if self.ensemble_method == "stacking":
            self.meta_learner = (
                meta_learner if meta_learner is not None else LinearRegression()
            )
        else:
            self.meta_learner = None

        # Containers for fitted models
        self.fitted_models_ = {}  # Base models fitted on the full dataset
        self.meta_fitted_ = None  # Fitted meta learner (for stacking)
        self.oof_predictions_ = None  # Out-of-fold predictions from stacking
        self.train_predictions_ = None  # Ensemble predictions on training data
        self.is_fitted_ = False

    def fit(self, X, y):
        """
        Fit the ensemble forecast model on time series data.

        For stacking, we use TimeSeriesSplit to generate out-of-fold predictions (which respect temporal order)
        and train a meta learner on these predictions. For weighted average, each base model is simply fitted on the full data.

        Parameters:
            X (array-like): Feature matrix (e.g., lagged features).
            y (array-like): Target variable.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        model_names = list(self.models.keys())

        if self.ensemble_method == "stacking":
            n_samples = X.shape[0]
            n_models = len(model_names)
            # Initialize out-of-fold predictions: shape = (n_samples, n_models)
            oof_preds = np.zeros((n_samples, n_models))
            tscv = TimeSeriesSplit(n_splits=self.n_splits)

            # For each model, perform time-series cross-validation
            for idx, model_name in enumerate(model_names):
                model = self.models[model_name]
                oof_model_preds = np.full(n_samples, np.nan)
                for train_idx, val_idx in tscv.split(X):
                    model_clone = clone(model)
                    model_clone.fit(X[train_idx], y[train_idx])
                    # Predict forecast horizon steps ahead; here, we assume the model's predict method
                    # is set up to forecast one step ahead. To forecast multiple steps,
                    # you might need to implement a recursive strategy or a direct multi-step approach.
                    preds = model_clone.predict(X[val_idx])
                    oof_model_preds[val_idx] = preds
                oof_preds[:, idx] = oof_model_preds
                # Refit each base model on the full dataset
                fitted_model = clone(model)
                fitted_model.fit(X, y)
                self.fitted_models_[model_name] = fitted_model

            # Train meta learner on the out-of-fold predictions (rows with no NaNs)
            valid_idx = ~np.isnan(oof_preds).any(axis=1)
            self.meta_fitted_ = clone(self.meta_learner)
            self.meta_fitted_.fit(oof_preds[valid_idx], y[valid_idx])
            # Store ensemble predictions on training data using meta learner
            self.train_predictions_ = self.meta_fitted_.predict(oof_preds[valid_idx])
            self.oof_predictions_ = oof_preds

        elif self.ensemble_method == "weighted_average":
            preds_list = []
            for model_name in model_names:
                model = clone(self.models[model_name])
                model.fit(X, y)
                self.fitted_models_[model_name] = model
                preds_list.append(model.predict(X).reshape(-1, 1))
            preds_array = np.hstack(preds_list)
            # Set equal weights if not provided
            if self.weights is None:
                weights_arr = np.ones(len(model_names)) / len(model_names)
            else:
                weights_arr = np.array(
                    [self.weights.get(name, 0) for name in model_names], dtype=float
                )
                if np.sum(weights_arr) == 0:
                    raise ValueError("Sum of provided weights cannot be zero.")
                weights_arr = weights_arr / np.sum(weights_arr)
            # Compute weighted average predictions
            self.train_predictions_ = np.dot(preds_array, weights_arr)
        else:
            raise ValueError("Unknown ensemble_method provided.")

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict (forecast) future values using the fitted ensemble model.

        Parameters:
            X (array-like): Feature matrix for which to forecast.

        Returns:
            Array of predictions.
        """
        if not self.is_fitted_:
            raise RuntimeError("The ensemble model must be fitted before predicting.")

        X = np.asarray(X)
        model_names = list(self.models.keys())

        if self.ensemble_method == "stacking":
            base_preds = []
            for model_name in model_names:
                model = self.fitted_models_[model_name]
                base_preds.append(model.predict(X).reshape(-1, 1))
            base_preds = np.hstack(base_preds)
            # Meta learner gives final ensemble forecast
            final_pred = self.meta_fitted_.predict(base_preds)
            return final_pred

        elif self.ensemble_method == "weighted_average":
            preds_list = []
            for model_name in model_names:
                model = self.fitted_models_[model_name]
                preds_list.append(model.predict(X).reshape(-1, 1))
            preds_array = np.hstack(preds_list)
            if self.weights is None:
                weights_arr = np.ones(len(model_names)) / len(model_names)
            else:
                weights_arr = np.array(
                    [self.weights.get(name, 0) for name in model_names], dtype=float
                )
                weights_arr = weights_arr / np.sum(weights_arr)
            final_pred = np.dot(preds_array, weights_arr)
            return final_pred
        else:
            raise ValueError("Unknown ensemble_method provided.")

    def forecast(self, X_future):
        """
        Forecast future values for given feature matrix X_future.

        This is an alias for predict() and can be used to emphasize forecasting.

        Parameters:
            X_future (array-like): Future feature matrix for forecasting.

        Returns:
            Array of forecasts.
        """
        return self.predict(X_future)

    def summary(self):
        """
        Print a summary of the ensemble forecasting model.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "The model has not been fitted yet. Call fit or forecast first."
            )

        print("Ensemble Forecasting Model Summary")
        print("----------------------------------")
        print(f"Forecast Horizon: {self.forecast_horizon}")
        print(f"Ensemble Method: {self.ensemble_method}")
        print("Base Models:")
        for name, model in self.models.items():
            print(f" - {name}: {model.__class__.__name__}")
        if self.ensemble_method == "stacking":
            print(f"Meta Learner: {self.meta_learner.__class__.__name__}")
        if self.train_predictions_ is not None:
            mse = mean_squared_error(
                self.train_predictions_, self.train_predictions_
            )  # Placeholder
            r2 = r2_score(self.train_predictions_, self.train_predictions_)
            print(
                "\nNote: Ensemble training predictions stored in self.train_predictions_."
            )
        print("----------------------------------")

    def plot_forecast(self, X, y_true):
        """
        Plot forecasted values vs true values.

        Parameters:
            X (array-like): Feature matrix to forecast.
            y_true (array-like): True values corresponding to X.
        """
        y_pred = self.predict(X)
        y_true = np.asarray(y_true)

        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label="True Values", marker="o")
        plt.plot(y_pred, label="Forecasted Values", marker="x")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Ensemble Forecast vs True Values")
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage:
if __name__ == "__main__":
    # For demonstration, we create a synthetic time series dataset.
    # Assume the time series has been transformed into a supervised learning problem (e.g., using lag features).
    np.random.seed(42)
    n_samples = 200
    # Create a simple time series: a linear trend with noise.
    t = np.arange(n_samples)
    y = 0.5 * t + np.random.normal(scale=5.0, size=n_samples)
    # Create lag features (using a simple approach)
    lag = 3
    X = np.array([y[i : i + lag] for i in range(n_samples - lag)])
    y_supervised = y[lag:]  # target is the next value after the lag window

    # Define base forecasting models. For example purposes, we use LinearRegression and DecisionTreeRegressor.
    from sklearn.tree import DecisionTreeRegressor

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=5),
    }

    # Create an ensemble forecast instance.
    ensemble_forecast = EnsembleForecast(
        models,
        forecast_horizon=1,
        n_splits=5,
        ensemble_method="stacking",  # Try "weighted_average" if preferred.
        weights=None,
        meta_learner=None,  # Defaults to LinearRegression for stacking.
    )

    # Split data into train and test sets (keeping time order).
    split_index = int(0.8 * X.shape[0])
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y_supervised[:split_index], y_supervised[split_index:]

    # Fit the ensemble model and forecast.
    ensemble_forecast.fit(X_train, y_train)
    y_pred = ensemble_forecast.forecast(X_test)

    # Print a summary and plot forecast vs true values.
    ensemble_forecast.summary()
    print("Test MSE: {:.4f}".format(mean_squared_error(y_test, y_pred)))
    print("Test R^2: {:.4f}".format(r2_score(y_test, y_pred)))
    ensemble_forecast.plot_forecast(X_test, y_test)
