#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MissingValueImputer
-------------------
A production-level class for imputing missing values in large datasets. It supports:
  - Simple strategies (constant, mean, median, most_frequent, forward_fill, backward_fill)
  - A state-of-the-art iterative strategy using machine learning models
    with optional hyperparameter tuning (GridSearchCV).

Usage
-----
This imputer can handle both numeric and categorical columns:
  - If mean/median strategy is used for an entire dataset that has mixed dtypes,
    numeric columns get mean/median, while categorical columns use most_frequent by default.
  - Forward/backward fill uses pandas .fillna(method=...) and thus works on all columns.
  - Iterative imputation trains separate regressors/classifiers for numeric/categorical targets.

New in this version:
  - An additional parameter, `categorical_threshold`, is used to decide whether a numeric
    column (e.g. int dtype) with few unique values should be treated as categorical.
"""

import logging
import sys
import numpy as np
import pandas as pd

from copy import deepcopy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    An advanced imputer for missing values that can handle both numeric and categorical data.

    Parameters
    ----------
    strategy : str, default='mean'
        Imputation strategy. One of:
          ['constant', 'mean', 'median', 'most_frequent', 'forward_fill', 'backward_fill', 'iterative'].
        For 'mean' or 'median', only numeric columns use that strategy; categorical columns fallback to 'most_frequent'.

    fill_value : object, default=None
        Value used for missing values if strategy='constant'.

    numeric_iterative_estimator : estimator, default=RandomForestRegressor(random_state=42)
        Base estimator for numeric columns in iterative imputation.

    categorical_iterative_estimator : estimator, default=RandomForestClassifier(random_state=42)
        Base estimator for categorical columns in iterative imputation.

    numeric_grid_params : dict, default=None
        Hyperparameter grid for numeric estimator in iterative imputation, if tuning is enabled.

    categorical_grid_params : dict, default=None
        Hyperparameter grid for categorical estimator in iterative imputation, if tuning is enabled.

    iterative_max_iter : int, default=10
        Maximum number of imputation iterations (passes over each column) in iterative strategy.

    iterative_tune : bool, default=False
        Whether to perform hyperparameter tuning for each column in iterative strategy.

    random_state : int, default=42
        Random seed for reproducibility in ML models.

    categorical_threshold : int, default=10
        For numeric columns: if the number of unique (non-null) values is less than this threshold,
        the column will be treated as categorical.

    Raises
    ------
    ValueError
        If an unsupported strategy is passed.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'num_col': [1, 2, np.nan, 4],
    ...     'cat_col': ['A', 'B', 'A', np.nan]
    ... })
    >>> imputer = MissingValueImputer(strategy='mean', categorical_threshold=5)
    >>> df_transformed = imputer.fit_transform(df)
    """

    def __init__(
        self,
        strategy="mean",
        fill_value=None,
        numeric_iterative_estimator=None,
        categorical_iterative_estimator=None,
        numeric_grid_params=None,
        categorical_grid_params=None,
        iterative_max_iter=10,
        iterative_tune=False,
        random_state=42,
        categorical_threshold=10,
    ):
        self.strategy = strategy
        self.fill_value = fill_value

        # Default estimators
        self.numeric_iterative_estimator = (
            RandomForestRegressor(n_estimators=100, random_state=random_state)
            if numeric_iterative_estimator is None
            else numeric_iterative_estimator
        )

        self.categorical_iterative_estimator = (
            RandomForestClassifier(n_estimators=100, random_state=random_state)
            if categorical_iterative_estimator is None
            else categorical_iterative_estimator
        )

        self.numeric_grid_params = numeric_grid_params
        self.categorical_grid_params = categorical_grid_params
        self.iterative_max_iter = iterative_max_iter
        self.iterative_tune = iterative_tune
        self.random_state = random_state
        self.categorical_threshold = categorical_threshold

        # Internal attributes
        self.column_order_ = None
        self.col_is_categorical_ = dict()
        self.label_encoders_ = dict()  # {col: LabelEncoder}
        self.iterative_models_ = dict()
        self.X_iter_filled_ = None

        # For simple strategies on mixed data
        self._numeric_imputer = None
        self._categorical_imputer = None

    def fit(self, X, y=None):
        """
        Fit the MissingValueImputer on the provided DataFrame X.

        Parameters
        ----------
        X : pd.DataFrame
            Input data with potential missing values.
        y : None
            Not used. (Included for API consistency.)

        Returns
        -------
        self : MissingValueImputer
            Fitted imputer.
        """
        logger.info("Fitting MissingValueImputer with strategy='%s'", self.strategy)
        X = self._validate_input(X)
        self.column_order_ = X.columns.tolist()

        # Identify numeric vs. categorical columns robustly.
        # If a column is numeric but has fewer unique values than the threshold,
        # treat it as categorical.
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                unique_count = X[col].nunique(dropna=True)
                if unique_count < self.categorical_threshold:
                    self.col_is_categorical_[col] = True
                else:
                    self.col_is_categorical_[col] = False
            else:
                self.col_is_categorical_[col] = True

        numeric_cols = [col for col in X.columns if not self.col_is_categorical_[col]]
        categorical_cols = [col for col in X.columns if self.col_is_categorical_[col]]

        if self.strategy in ["mean", "median", "most_frequent", "constant"]:
            # Create separate imputers for numeric vs. categorical if needed
            self._build_simple_imputers(numeric_cols, categorical_cols)

            # Fit them
            if numeric_cols:
                numeric_data = X[numeric_cols]
                self._numeric_imputer.fit(numeric_data)

            if categorical_cols:
                categorical_data = X[categorical_cols]
                self._categorical_imputer.fit(categorical_data)

        elif self.strategy in ["forward_fill", "backward_fill"]:
            # No fitting needed for forward/backward fill
            pass

        elif self.strategy == "iterative":
            # Encode categoricals for model usage
            X_encoded = self._encode_categorical(X, fit=True)
            # Fit iterative models
            X_filled = self._iterative_impute_fit(X_encoded)
            self.X_iter_filled_ = X_filled.copy()

        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")

        return self

    def transform(self, X):
        """
        Impute missing values in X using the fitted imputation strategy.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            A copy of X with imputed values.
        """
        check_is_fitted(self, "column_order_")
        X = self._validate_input(X)

        # Align columns with training data
        X = X.reindex(columns=self.column_order_, copy=True)

        if self.strategy in ["mean", "median", "most_frequent", "constant"]:
            X_out = self._simple_imputers_transform(X)
            return X_out

        elif self.strategy in ["forward_fill", "backward_fill"]:
            if self.strategy == "forward_fill":
                return X.ffill()
            else:  # 'backward_fill'
                return X.bfill()

        elif self.strategy == "iterative":
            # Encode, then fill via iterative, then decode
            X_encoded = self._encode_categorical(X, fit=False)
            X_filled = self._iterative_impute_transform(X_encoded)
            X_out = self._decode_categorical(X_filled)
            return X_out

        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it. Equiv. to self.fit(X).transform(X).
        """
        return self.fit(X, y).transform(X)

    # -------------------------------------------------------------------------
    # INTERNAL METHODS
    # -------------------------------------------------------------------------
    def _validate_input(self, X):
        """Ensure X is a DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        return X.copy()

    def _build_simple_imputers(self, numeric_cols, categorical_cols):
        """
        Build self._numeric_imputer and self._categorical_imputer
        depending on the chosen strategy.
        """
        if self.strategy == "constant":
            logger.info(
                "Using SimpleImputer with constant fill_value='%s' for all columns",
                self.fill_value,
            )
            self._numeric_imputer = SimpleImputer(
                strategy="constant", fill_value=self.fill_value
            )
            self._categorical_imputer = SimpleImputer(
                strategy="constant", fill_value=self.fill_value
            )

        elif self.strategy == "most_frequent":
            logger.info("Using SimpleImputer(most_frequent) for all columns.")
            self._numeric_imputer = SimpleImputer(strategy="most_frequent")
            self._categorical_imputer = SimpleImputer(strategy="most_frequent")

        elif self.strategy in ["mean", "median"]:
            # For numeric columns: use mean/median
            # For categorical columns: fallback to most_frequent
            logger.info(
                "Using SimpleImputer('%s') for numeric columns and SimpleImputer('most_frequent') for categorical.",
                self.strategy,
            )
            self._numeric_imputer = SimpleImputer(strategy=self.strategy)
            self._categorical_imputer = SimpleImputer(strategy="most_frequent")

        else:
            raise ValueError(
                f"Invalid strategy '{self.strategy}' in _build_simple_imputers."
            )

    def _simple_imputers_transform(self, X):
        """
        Apply the fitted simple imputers to numeric and categorical columns.
        """
        numeric_cols = [col for col in X.columns if not self.col_is_categorical_[col]]
        categorical_cols = [col for col in X.columns if self.col_is_categorical_[col]]

        X_out = X.copy()
        if numeric_cols:
            X_out[numeric_cols] = self._numeric_imputer.transform(X[numeric_cols])
        if categorical_cols:
            X_out[categorical_cols] = self._categorical_imputer.transform(
                X[categorical_cols]
            )

        return X_out

    def _encode_categorical(self, X, fit=True):
        """
        Convert categorical columns to numeric via LabelEncoder.
        If fit=True, we fit new LabelEncoders. Otherwise, we use existing ones.
        """
        X_encoded = X.copy()
        for col, is_cat in self.col_is_categorical_.items():
            if is_cat:
                if fit:
                    le = LabelEncoder()
                    # Fill missing with placeholder string
                    X_encoded[col] = X_encoded[col].astype(str).fillna("NaNPlaceholder")
                    le.fit(X_encoded[col])
                    self.label_encoders_[col] = le
                else:
                    le = self.label_encoders_[col]

                    # Handle unseen categories by extending classes
                    existing_classes = set(le.classes_)
                    new_categories = (
                        set(X_encoded[col].dropna().astype(str)) - existing_classes
                    )
                    if new_categories:
                        extended = np.concatenate([le.classes_, list(new_categories)])
                        le.classes_ = extended

                X_encoded[col] = X_encoded[col].astype(str).fillna("NaNPlaceholder")
                X_encoded[col] = self.label_encoders_[col].transform(X_encoded[col])

        return X_encoded

    def _decode_categorical(self, X_encoded):
        """
        Inverse transform numeric codes back to original categorical labels.
        """
        X_decoded = X_encoded.copy()
        for col, is_cat in self.col_is_categorical_.items():
            if is_cat:
                le = self.label_encoders_[col]
                X_decoded[col] = le.inverse_transform(X_decoded[col].astype(int))
                # Replace 'NaNPlaceholder' with np.nan
                X_decoded[col] = X_decoded[col].replace("NaNPlaceholder", np.nan)
        return X_decoded

    def _iterative_impute_fit(self, X_encoded):
        """
        Perform iterative imputation on training data:
          For each column with missing values:
            - Treat column as target
            - Fit regressor/classifier on other columns
            - Predict missing entries
            - Fill them in, repeat
        """
        X_filled = X_encoded.copy()
        for _ in range(self.iterative_max_iter):
            any_updated = False
            for col in X_filled.columns:
                missing_mask = X_filled[col].isnull()
                if missing_mask.sum() == 0:
                    continue

                # Train data
                not_missing = ~missing_mask
                X_train = X_filled.loc[not_missing, X_filled.columns != col]
                y_train = X_filled.loc[not_missing, col]
                # Test data
                X_test = X_filled.loc[missing_mask, X_filled.columns != col]

                if self.col_is_categorical_[col]:
                    base_model = deepcopy(self.categorical_iterative_estimator)
                    param_grid = self.categorical_grid_params
                    scoring = "accuracy"
                else:
                    base_model = deepcopy(self.numeric_iterative_estimator)
                    param_grid = self.numeric_grid_params
                    scoring = "neg_mean_squared_error"

                if self.iterative_tune and param_grid:
                    gs = GridSearchCV(
                        estimator=base_model,
                        param_grid=param_grid,
                        scoring=scoring,
                        cv=3,
                        n_jobs=-1,
                    )
                    gs.fit(X_train, y_train)
                    model = gs.best_estimator_
                    logger.info(
                        "Column '%s': best model found via GridSearchCV: %s", col, model
                    )
                else:
                    base_model.fit(X_train, y_train)
                    model = base_model

                self.iterative_models_[col] = model
                preds = model.predict(X_test)
                X_filled.loc[missing_mask, col] = preds
                any_updated = True

            if not any_updated:
                break

        return X_filled

    def _iterative_impute_transform(self, X_encoded):
        """
        Apply trained iterative models to new data for imputation.
        """
        X_filled = X_encoded.copy()
        for _ in range(self.iterative_max_iter):
            any_updated = False
            for col in X_filled.columns:
                missing_mask = X_filled[col].isnull()
                if missing_mask.sum() == 0:
                    continue

                if col not in self.iterative_models_:
                    # Means no missing values at fit-time => no model was trained => skip
                    continue

                model = self.iterative_models_[col]
                X_test = X_filled.loc[missing_mask, X_filled.columns != col]
                preds = model.predict(X_test)
                X_filled.loc[missing_mask, col] = preds
                any_updated = True

            if not any_updated:
                break

        return X_filled


# -----------------------------------------------------------------------------
# Self-Testing / Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small dummy dataset
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "numeric1": [1, 2, np.nan, 4, 5, np.nan, 7],
            "numeric2": [10, np.nan, 12, 13, np.nan, 15, 16],
            # Although these are stored as numbers, they have few unique values so will be treated as categorical.
            "categorical_numeric": [0, 1, 0, np.nan, 1, 0, 1],
            "categorical1": ["A", "B", np.nan, "B", "A", "A", np.nan],
            "categorical2": ["X", "Y", "X", "Y", "X", np.nan, "Y"],
        }
    )

    logger.info("Original dataset with NaNs:\n%s", df)

    # Example 1: Mean strategy (for numeric), fallback to most_frequent for categorical.
    imputer_mean = MissingValueImputer(strategy="mean", categorical_threshold=3)
    df_mean_imputed = imputer_mean.fit_transform(df)
    logger.info(
        "Mean-imputed (numeric) + most_frequent-imputed (categorical):\n%s",
        df_mean_imputed,
    )

    # Example 2: Forward fill.
    imputer_ffill = MissingValueImputer(strategy="forward_fill")
    df_ffill_imputed = imputer_ffill.fit_transform(df)
    logger.info("Forward-fill-imputed dataset:\n%s", df_ffill_imputed)

    # Example 3: Iterative imputation (default RF, no hyperparam tuning)
    imputer_iter = MissingValueImputer(
        strategy="iterative", iterative_max_iter=5, iterative_tune=False
    )
    df_iter_imputed = imputer_iter.fit_transform(df)
    logger.info("Iterative-imputed dataset:\n%s", df_iter_imputed)

    # Example 4: Iterative + hyperparameter tuning (tiny grid)
    numeric_grid = {"n_estimators": [10, 20], "max_depth": [3, 5]}
    cat_grid = {"n_estimators": [10, 20], "max_depth": [3, 5]}
    imputer_iter_tuned = MissingValueImputer(
        strategy="iterative",
        numeric_grid_params=numeric_grid,
        categorical_grid_params=cat_grid,
        iterative_max_iter=3,
        iterative_tune=True,
    )
    df_iter_tuned_imputed = imputer_iter_tuned.fit_transform(df)
    logger.info("Iterative-imputed (with tuning) dataset:\n%s", df_iter_tuned_imputed)

    logger.info("Done with self-tests.")
