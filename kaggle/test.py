import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from quantbayes import bnn
from quantbayes.bnn.utils import (expected_calibration_error,
                                  plot_calibration_curve, plot_roc_curve)
from quantbayes.ensemble import EnsembleBinary
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "lgbm": LGBMClassifier(verbose=-1),
    "xgb": XGBClassifier(),
    "cat": CatBoostClassifier(silent=True)
}
ensemble = EnsembleBinary(
    models=models,
    ensemble_method="weighted_average",
    weights=None,
    meta_learner=None
)
ensemble.fit(X_train, y_train)
ensemble.summary()
preds = ensemble.predict_proba(X_test)[:, -1]
loss = log_loss(y_test.ravel(), preds)
plot_calibration_curve(y_test.ravel(), preds)
plot_roc_curve(y_test.ravel(), preds)
ece = expected_calibration_error(y_test.ravel(), preds)

print(f"Ensemble loss: {loss:.3f}")
print(f"Ensemble ECE: {ece:.3f}")