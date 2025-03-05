import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from quantbayes.hyperparameter_tune import LGBMClassifierTuner
from quantbayes.preprocessing import Preprocessor

data = pd.read_csv("kaggle/train.csv")
data = data.drop("id", axis=1)
data["day"] = pd.to_datetime(data["day"], format="%j", errors="coerce").apply(lambda x: x.replace(year=2024))
data = data.drop("day", axis=1)

def preprocess_data(data: pd.DataFrame):
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
tuner = LGBMClassifierTuner(X, y)
best_params, best_score = tuner.tune()

print("Regression Best Hyperparameters:", best_params)
print("Regression Best CV Score:", best_score)

