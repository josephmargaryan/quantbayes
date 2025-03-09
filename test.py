import pandas as pd
import gc
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import equinox as eqx
import jax 
import jax.random as jr
import jax.numpy as jnp

from quantbayes.stochax.tabular import BinaryModel
from quantbayes.ensemble import EnsembleBinary
from quantbayes.hyperparameter_tune import (
    XGBClassifierTuner,
    LGBMClassifierTuner,
    CatBoostClassifierTuner,
    HistGradientBoostingClassifierTuner
)