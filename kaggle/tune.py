import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from quantbayes.hyperparameter_tune import XGBTuner
from quantbayes.preprocessing import Preprocessor

data = pd.read_csv("kaggle/train.csv")
data = data.drop("id", axis=1)
data["day"] = pd.to_datetime(data["day"], format="%j", errors="coerce").apply(lambda x: x.replace(year=2024))

def preprocess_data(data = pd.DataFrame):
    preprocessor = Preprocessor(
        task_type="binary",
        target_col="rainfall",
        feature_scaler=StandardScaler(),
    )
    data = data.drop("day", axis=1)
    X, y = preprocessor.fit_transform(
        data = data
    )
    return X, y

X, y = preprocess_data(data = data)

tuner = XGBTuner(X, y)
best_xgb = tuner.tune(n_trials=10)
print("Best trial for XGBoost:")
print("  Score: {:.4f}".format(tuner.best_score))
print("  Params:", tuner.best_params)