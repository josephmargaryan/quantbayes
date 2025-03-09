import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna

# Import model libraries
import xgboost as xgb
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

__all__ = [
    "XGBClassifierTuner",
    "XGBRegressorTuner",
    "LGBMClassifierTuner",
    "LGBMRegressorTuner",
    "CatBoostClassifierTuner",
    "CatBoostRegressorTuner",
    "HistGradientBoostingClassifierTuner",
    "HistGradientBoostingRegressorTuner",
]

"""
Example usage:
tuner = XGBRegressorTuner(X, y, n_trials=50, cv=5)
best_params, best_score = tuner.tune()

print("Regression Best Hyperparameters:", best_params)
print("Regression Best CV Score:", best_score)
"""


class BaseTuner:
    """
    Base class for hyperparameter tuning using Optuna.
    It automatically sets the CV splitter and scoring based on whether the problem is classification or regression.
    """

    def __init__(self, X, y, n_trials=50, cv=5, scoring=None, random_state=42):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        self.is_classifier = self._determine_if_classifier()

        # Default scoring: for classification (binary: roc_auc, multiclass: accuracy) and for regression (neg_mean_squared_error)
        if scoring is not None:
            self.scoring = scoring
        else:
            if self.is_classifier:
                # Use roc_auc if binary, accuracy if multiclass.
                unique_vals = np.unique(y)
                self.scoring = "roc_auc" if len(unique_vals) == 2 else "accuracy"
            else:
                self.scoring = "neg_mean_squared_error"

        # For classification, also store problem type for later (binary vs multiclass)
        if self.is_classifier:
            self.problem_type = "binary" if len(np.unique(y)) == 2 else "multiclass"
        else:
            self.problem_type = "regression"

    def _determine_if_classifier(self):
        # Simple heuristic: if target has <=20 unique values or is not float, assume classification.
        if np.issubdtype(self.y.dtype, np.floating) and len(np.unique(self.y)) > 20:
            return False
        else:
            return True

    def get_cv(self):
        # Use StratifiedKFold for classification, KFold for regression.
        if self.is_classifier:
            return StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
        else:
            return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

    def objective(self, trial):
        """
        Must be implemented by subclasses. This function should:
          - Suggest a set of hyperparameters,
          - Initialize the model (with any fixed parameters merged in),
          - Evaluate via cross-validation,
          - And return the mean CV score.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def tune(self):
        # Create an Optuna study (maximizing the score) and run optimization.
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(self.objective, n_trials=self.n_trials)
        self.best_params = study.best_trial.params
        self.best_score = study.best_value
        return self.best_params, self.best_score


#############################################
# XGBoost Tuners
#############################################
class XGBClassifierTuner(BaseTuner):
    def __init__(self, X, y, n_trials=50, cv=5, scoring=None, random_state=42):
        super().__init__(X, y, n_trials, cv, scoring, random_state)
        self.model_class = xgb.XGBClassifier
        self.fixed_params = {}
        # Automatically set the objective and (if needed) num_class for multiclass problems.
        if self.problem_type == "multiclass":
            self.fixed_params["objective"] = "multi:softprob"
            self.fixed_params["num_class"] = len(np.unique(y))
        else:
            self.fixed_params["objective"] = "binary:logistic"

    def objective(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.5, 1.0
            ),  # NEW
            "colsample_bynode": trial.suggest_float(
                "colsample_bynode", 0.5, 1.0
            ),  # NEW
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", 0.1, 10.0, log=True
            ),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),  # NEW
        }
        # Merge with fixed parameters
        params.update(self.fixed_params)
        model = self.model_class(
            **params, random_state=self.random_state, use_label_encoder=False
        )
        cv = self.get_cv()
        scores = cross_val_score(
            model, self.X, self.y, scoring=self.scoring, cv=cv, n_jobs=-1
        )
        return np.mean(scores)


class XGBRegressorTuner(BaseTuner):
    def __init__(self, X, y, n_trials=50, cv=5, scoring=None, random_state=42):
        super().__init__(X, y, n_trials, cv, scoring, random_state)
        self.model_class = xgb.XGBRegressor
        self.fixed_params = {}  # No fixed parameters here

    def objective(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.5, 1.0
            ),  # NEW
            "colsample_bynode": trial.suggest_float(
                "colsample_bynode", 0.5, 1.0
            ),  # NEW
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),  # NEW
        }
        params.update(self.fixed_params)
        model = self.model_class(**params, random_state=self.random_state)
        cv = self.get_cv()
        scores = cross_val_score(
            model, self.X, self.y, scoring=self.scoring, cv=cv, n_jobs=-1
        )
        return np.mean(scores)


#############################################
# LightGBM Tuners
#############################################
class LGBMClassifierTuner(BaseTuner):
    def __init__(self, X, y, n_trials=50, cv=5, scoring=None, random_state=42):
        super().__init__(X, y, n_trials, cv, scoring, random_state)
        self.model_class = lgb.LGBMClassifier
        # Set fixed parameter to silence verbose output.
        self.fixed_params = {"verbose": -1}
        if self.problem_type == "multiclass":
            self.fixed_params["objective"] = "multiclass"
            self.fixed_params["num_class"] = len(np.unique(y))
        else:
            self.fixed_params["objective"] = "binary"

    def objective(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),  # NEW
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.5, 1.0
            ),  # NEW
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),  # NEW
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.5, 1.0
            ),  # NEW
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),  # NEW
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),  # NEW
        }
        params.update(self.fixed_params)
        model = self.model_class(**params, random_state=self.random_state)
        cv = self.get_cv()
        scores = cross_val_score(
            model, self.X, self.y, scoring=self.scoring, cv=cv, n_jobs=-1
        )
        return np.mean(scores)


class LGBMRegressorTuner(BaseTuner):
    def __init__(self, X, y, n_trials=50, cv=5, scoring=None, random_state=42):
        super().__init__(X, y, n_trials, cv, scoring, random_state)
        self.model_class = lgb.LGBMRegressor
        self.fixed_params = {"verbose": -1}

    def objective(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),  # NEW
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.5, 1.0
            ),  # NEW
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),  # NEW
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.5, 1.0
            ),  # NEW
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),  # NEW
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),  # NEW
        }
        params.update(self.fixed_params)
        model = self.model_class(**params, random_state=self.random_state)
        cv = self.get_cv()
        scores = cross_val_score(
            model, self.X, self.y, scoring=self.scoring, cv=cv, n_jobs=-1
        )
        return np.mean(scores)


#############################################
# CatBoost Tuners
#############################################
class CatBoostClassifierTuner(BaseTuner):
    def __init__(self, X, y, n_trials=50, cv=5, scoring=None, random_state=42):
        super().__init__(X, y, n_trials, cv, scoring, random_state)
        self.model_class = cb.CatBoostClassifier
        # Fixed parameter to silence verbose output.
        self.fixed_params = {"silent": True}
        if self.problem_type == "multiclass":
            self.fixed_params["loss_function"] = "MultiClass"
        else:
            self.fixed_params["loss_function"] = "Logloss"

    def objective(self, trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 0, 10),  # NEW
            "rsm": trial.suggest_float("rsm", 0.5, 1.0),  # NEW
            "leaf_estimation_iterations": trial.suggest_int(
                "leaf_estimation_iterations", 1, 10
            ),  # NEW
            "od_type": trial.suggest_categorical(
                "od_type", ["IncToDec", "Iter"]
            ),  # NEW
            "od_wait": trial.suggest_int("od_wait", 10, 50),  # NEW
        }
        params.update(self.fixed_params)
        model = self.model_class(**params, random_state=self.random_state)
        cv = self.get_cv()
        scores = cross_val_score(
            model, self.X, self.y, scoring=self.scoring, cv=cv, n_jobs=-1
        )
        return np.mean(scores)


class CatBoostRegressorTuner(BaseTuner):
    def __init__(self, X, y, n_trials=50, cv=5, scoring=None, random_state=42):
        super().__init__(X, y, n_trials, cv, scoring, random_state)
        self.model_class = cb.CatBoostRegressor
        self.fixed_params = {"silent": True}

    def objective(self, trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 0, 10),  # NEW
            "rsm": trial.suggest_float("rsm", 0.5, 1.0),  # NEW
            "leaf_estimation_iterations": trial.suggest_int(
                "leaf_estimation_iterations", 1, 10
            ),  # NEW
            "od_type": trial.suggest_categorical(
                "od_type", ["IncToDec", "Iter"]
            ),  # NEW
            "od_wait": trial.suggest_int("od_wait", 10, 50),  # NEW
        }
        params.update(self.fixed_params)
        model = self.model_class(**params, random_state=self.random_state)
        cv = self.get_cv()
        scores = cross_val_score(
            model, self.X, self.y, scoring=self.scoring, cv=cv, n_jobs=-1
        )
        return np.mean(scores)


#############################################
# HistGradientBoosting Tuners (sklearn)
#############################################
class HistGradientBoostingClassifierTuner(BaseTuner):
    def __init__(self, X, y, n_trials=50, cv=5, scoring=None, random_state=42):
        super().__init__(X, y, n_trials, cv, scoring, random_state)
        self.model_class = HistGradientBoostingClassifier
        self.fixed_params = {}  # No fixed parameters

    def objective(self, trial):
        params = {
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 100),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-8, 1.0, log=True
            ),
            "early_stopping": trial.suggest_categorical(
                "early_stopping", [True, False]
            ),  # NEW
            "validation_fraction": trial.suggest_float(
                "validation_fraction", 0.1, 0.3
            ),  # NEW
        }
        params.update(self.fixed_params)
        model = self.model_class(**params, random_state=self.random_state)
        cv = self.get_cv()
        scores = cross_val_score(
            model, self.X, self.y, scoring=self.scoring, cv=cv, n_jobs=-1
        )
        return np.mean(scores)


class HistGradientBoostingRegressorTuner(BaseTuner):
    def __init__(self, X, y, n_trials=50, cv=5, scoring=None, random_state=42):
        super().__init__(X, y, n_trials, cv, scoring, random_state)
        self.model_class = HistGradientBoostingRegressor
        self.fixed_params = {}  # No fixed parameters

    def objective(self, trial):
        params = {
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 100),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-8, 1.0, log=True
            ),
            "early_stopping": trial.suggest_categorical(
                "early_stopping", [True, False]
            ),  # NEW
            "validation_fraction": trial.suggest_float(
                "validation_fraction", 0.1, 0.3
            ),  # NEW
            "n_iter_no_change": trial.suggest_int("n_iter_no_change", 5, 20),  # NEW
            "tol": trial.suggest_float("tol", 1e-4, 1e-2, log=True),  # NEW
        }
        params.update(self.fixed_params)
        model = self.model_class(**params, random_state=self.random_state)
        cv = self.get_cv()
        scores = cross_val_score(
            model, self.X, self.y, scoring=self.scoring, cv=cv, n_jobs=-1
        )
        return np.mean(scores)
