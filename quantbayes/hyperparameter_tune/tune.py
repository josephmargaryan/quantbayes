import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss, mean_squared_error, make_scorer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

# Import models
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# Base tuner class supporting different problem types
class BaseTuner:
    def __init__(self, X, y, cv_splits=5, problem_type: str = "binary", random_state=42):
        self.X = X
        self.y = y
        self.problem_type = problem_type.lower()
        self.random_state = random_state

        # Choose CV strategy based on problem type
        if self.problem_type in ["binary", "multiclass"]:
            self.cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        else:
            self.cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        self.scorer = self.get_scorer()
        self.best_params = None
        self.best_score = None
        self.study = None

    def get_scorer(self):
        if self.problem_type in ["binary", "multiclass"]:
            # Use negative log loss so that a higher score is better
            return make_scorer(log_loss, greater_is_better=False, needs_proba=True)
        elif self.problem_type == "regression":
            # Use negative MSE so that a higher score is better
            return make_scorer(lambda y_true, y_pred: -mean_squared_error(y_true, y_pred))
        else:
            raise ValueError("Invalid problem_type. Choose 'binary', 'multiclass', or 'regression'.")

    def objective(self, trial):
        raise NotImplementedError("Subclasses must implement this method.")

    def tune(self, n_trials=50, sampler_seed=42):
        sampler = TPESampler(seed=sampler_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(self.objective, n_trials=n_trials)
        self.best_params = study.best_trial.params
        self.best_score = study.best_trial.value
        self.study = study
        return study.best_trial

# Tuner for XGBoost
class XGBTuner(BaseTuner):
    def objective(self, trial):
        param = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
            "random_state": self.random_state,
        }

        if self.problem_type == "regression":
            model = XGBRegressor(**param)
        else:
            # For binary and multiclass
            model = XGBClassifier(**param, use_label_encoder=False, eval_metric="logloss")

        scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring=self.scorer, n_jobs=-1)
        return np.mean(scores)

# Tuner for CatBoost
class CatBoostTuner(BaseTuner):
    def objective(self, trial):
        param = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
            "depth": trial.suggest_int("depth", 3, 12),
            "iterations": trial.suggest_int("iterations", 100, 400),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "random_state": self.random_state,
            "silent": True,
        }

        if self.problem_type == "regression":
            model = CatBoostRegressor(**param)
        else:
            model = CatBoostClassifier(**param)

        scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring=self.scorer, n_jobs=-1)
        return np.mean(scores)

# Tuner for LightGBM
class LGBMTuner(BaseTuner):
    def objective(self, trial):
        param = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "num_leaves": trial.suggest_int("num_leaves", 20, 60),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": self.random_state,
            "verbose": -1
        }

        if self.problem_type == "regression":
            model = LGBMRegressor(**param)
        else:
            model = LGBMClassifier(**param)

        scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring=self.scorer, n_jobs=-1)
        return np.mean(scores)

# Tuner for HistGradientBoosting
class HistGBMTuner(BaseTuner):
    def objective(self, trial):
        param = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.05),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "max_iter": trial.suggest_int("max_iter", 100, 600),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "random_state": self.random_state,
        }

        if self.problem_type == "regression":
            model = HistGradientBoostingRegressor(**param)
        else:
            model = HistGradientBoostingClassifier(**param)

        scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring=self.scorer, n_jobs=-1)
        return np.mean(scores)

# Example usage:
if __name__ == "__main__":
    # For demonstration purposes, we use a synthetic dataset.
    from sklearn.datasets import make_classification, make_regression

    # Choose problem type: "binary", "multiclass", or "regression"
    problem_type = "binary"  # Change as needed

    if problem_type in ["binary", "multiclass"]:
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_classes=2 if problem_type == "binary" else 3,
            random_state=42,
        )
    else:
        X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

    print("Tuning XGB...")
    xgb_tuner = XGBTuner(X, y, problem_type=problem_type)
    best_xgb = xgb_tuner.tune(n_trials=100)
    print("Best trial for XGBoost:")
    print("  Score: {:.4f}".format(xgb_tuner.best_score))
    print("  Params:", xgb_tuner.best_params)

    print("\nTuning CatBoost...")
    catboost_tuner = CatBoostTuner(X, y, problem_type=problem_type)
    best_cat = catboost_tuner.tune(n_trials=100)
    print("Best trial for CatBoost:")
    print("  Score: {:.4f}".format(catboost_tuner.best_score))
    print("  Params:", catboost_tuner.best_params)

    print("\nTuning LightGBM...")
    lgbm_tuner = LGBMTuner(X, y, problem_type=problem_type)
    best_lgbm = lgbm_tuner.tune(n_trials=100)
    print("Best trial for LightGBM:")
    print("  Score: {:.4f}".format(lgbm_tuner.best_score))
    print("  Params:", lgbm_tuner.best_params)

    print("\nTuning HistGradientBoosting...")
    histgbm_tuner = HistGBMTuner(X, y, problem_type=problem_type)
    best_histgbm = histgbm_tuner.tune(n_trials=100)
    print("Best trial for HistGradientBoosting:")
    print("  Score: {:.4f}".format(histgbm_tuner.best_score))
    print("  Params:", histgbm_tuner.best_params)
