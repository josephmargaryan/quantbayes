import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from quantbayes.bnn.utils import (
    expected_calibration_error,
    plot_calibration_curve,
    plot_roc_curve,
)
from quantbayes.ensemble import EnsembleBinary
from quantbayes.preprocessing import Preprocessor

data = pd.read_csv("kaggle/train.csv")
data = data.drop("id", axis=1)
data["day"] = pd.to_datetime(data["day"], format="%j", errors="coerce").apply(
    lambda x: x.replace(year=2024)
)
data = data.drop("day", axis=1)


def preprocess_data(data: pd.DataFrame):
    preprocessor = Preprocessor(
        task_type="binary",
        target_col="rainfall",
        feature_scaler=StandardScaler(),
    )
    X, y = preprocessor.fit_transform(data=data)
    return X, y


X, y = preprocess_data(data=data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# LGBM parameters and best CV score
lgbm_params = {
    "n_estimators": 303,
    "num_leaves": 90,
    "learning_rate": 0.010785044095277689,
    "min_child_samples": 80,
    "subsample": 0.5455494890371395,
    "colsample_bytree": 0.6671700685284869,
    "reg_alpha": 1.8841873910753578e-08,
    "reg_lambda": 0.13294573193342574,
    "min_split_gain": 0.07185404918408647,
    "bagging_fraction": 0.6492233430156829,
    "bagging_freq": 7,
    "feature_fraction": 0.6403708392801084,
    "lambda_l1": 0.709494244246443,
    "lambda_l2": 0.5324520154597764,
    "verbose": -1,
}
lgbm_cv_score = 0.8960044893378228

# XGBoost parameters and best CV score
xgb_params = {
    "n_estimators": 187,
    "max_depth": 4,
    "learning_rate": 0.013165696385650986,
    "subsample": 0.5104634389290511,
    "colsample_bytree": 0.8968835584655324,
    "colsample_bylevel": 0.8794942226162635,
    "colsample_bynode": 0.9064733642035727,
    "gamma": 0.7664116062587596,
    "reg_alpha": 0.0001299880226842251,
    "reg_lambda": 6.210706785972902e-08,
    "min_child_weight": 9,
    "scale_pos_weight": 3.427146691071858,
    "max_delta_step": 0,
}
xgb_cv_score = 0.8961335578002245

# CatBoost parameters and best CV score
catboost_params = {
    "iterations": 327,
    "depth": 5,
    "learning_rate": 0.014343310642092282,
    "l2_leaf_reg": 2.729491081696202,
    "bagging_temperature": 0.677373954828049,
    "border_count": 149,
    "random_strength": 1.4937934739269683,
    "rsm": 0.9725122186497227,
    "leaf_estimation_iterations": 2,
    "od_type": "IncToDec",
    "od_wait": 40,
    "silent": True,
}
catboost_cv_score = 0.8952749719416386

# HistGradientBoosting parameters and best CV score
histboost_params = {
    "max_iter": 195,
    "learning_rate": 0.01414040168096487,
    "max_leaf_nodes": 10,
    "l2_regularization": 0.7090919680140552,
    "early_stopping": False,
    "validation_fraction": 0.19878674789560188,
}
histboost_cv_score = 0.892253086419753


models = {
    "lgbm": LGBMClassifier(**lgbm_params),
    "xgb": XGBClassifier(**xgb_params),
    "cat": CatBoostClassifier(**catboost_params),
    "hist": HistGradientBoostingClassifier(**histboost_params),
}
if __name__ == "__main__":
    ensemble = EnsembleBinary(
        models=models, ensemble_method="stacking", weights=None, meta_learner=None
    )
    ensemble.fit(X_train, y_train)
    ensemble.summary()
    preds = ensemble.predict_proba(X_test)[:, -1]
    loss = log_loss(y_test, preds)
    plot_calibration_curve(y_test, preds)
    plot_roc_curve(y_test.ravel(), preds)
    ece = expected_calibration_error(y_test, preds)

    print("Stacking")
    print(f"Ensemble loss: {loss:.3f}")
    print(f"Ensemble ECE: {ece:.3f}")

    ensemble2 = EnsembleBinary(
        models=models,
        ensemble_method="weighted_average",
        weights=None,
        meta_learner=None,
    )
    ensemble2.fit(X_train, y_train)
    ensemble2.summary()
    preds = ensemble2.predict_proba(X_test)[:, -1]
    loss = log_loss(y_test, preds)
    plot_calibration_curve(y_test, preds)
    plot_roc_curve(y_test, preds)
    ece = expected_calibration_error(y_test, preds)

    print("Weighted average")
    print(f"Ensemble loss: {loss:.3f}")
    print(f"Ensemble ECE: {ece:.3f}")
