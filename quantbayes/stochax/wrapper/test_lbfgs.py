# test_eqx_ensembles_lbfgs.py

import numpy as np
import equinox as eqx
import jax
import jax.random as jr
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    log_loss,
    roc_auc_score,
)

from quantbayes.ensemble import EnsembleRegression, EnsembleBinary, EnsembleMulticlass
from quantbayes.stochax.wrapper.wrapper_lbfgs import (
    EQXRegressorLBFGS,
    EQXBinaryClassifierLBFGS,
    EQXMulticlassClassifierLBFGS,
)


class RegressorNN(eqx.Module):
    l1: eqx.nn.Linear
    out: eqx.nn.Linear

    def __init__(self, key, n_features: int):
        k1, k2 = jr.split(key, 2)
        self.l1 = eqx.nn.Linear(n_features, 10, key=k1)
        self.out = eqx.nn.Linear(10, 1, key=k2)

    def __call__(self, x, key, state):
        x = jax.nn.relu(self.l1(x))
        return self.out(x), state


class BinaryNN(eqx.Module):
    l1: eqx.nn.Linear
    out: eqx.nn.Linear

    def __init__(self, key, n_features: int):
        k1, k2 = jr.split(key, 2)
        self.l1 = eqx.nn.Linear(n_features, 10, key=k1)
        self.out = eqx.nn.Linear(10, 1, key=k2)

    def __call__(self, x, key, state):
        x = jax.nn.relu(self.l1(x))
        return self.out(x), state


class MulticlassNN(eqx.Module):
    l1: eqx.nn.Linear
    out: eqx.nn.Linear

    def __init__(self, key, n_features: int, n_classes: int):
        k1, k2 = jr.split(key, 2)
        self.l1 = eqx.nn.Linear(n_features, 10, key=k1)
        self.out = eqx.nn.Linear(10, n_classes, key=k2)

    def __call__(self, x, key, state):
        x = jax.nn.relu(self.l1(x))
        return self.out(x), state


def test_regression():
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.2, random_state=0)
    X_tr, X_te = X[:800], X[800:]
    y_tr, y_te = y[:800], y[800:]

    models = {
        "lin": LinearRegression(),
        "tree": DecisionTreeRegressor(max_depth=5),
        "eqx": EQXRegressorLBFGS(
            model_cls=RegressorNN,
            model_kwargs={"n_features": 10},
            num_epochs=10,
            patience=10,
        ),
    }

    ens = EnsembleRegression(models=models, ensemble_method="stacking", n_splits=3)
    ens.fit(X_tr, y_tr)
    preds = ens.predict(X_te)

    print("LBFGS REGRESSION")
    print("  MSE:", mean_squared_error(y_te, preds))
    print("  R2: ", r2_score(y_te, preds))


def test_binary():
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5, random_state=1
    )
    X_tr, X_te = X[:800], X[800:]
    y_tr, y_te = y[:800], y[800:]

    models = {
        "log": LogisticRegression(solver="lbfgs", max_iter=500),
        "tree": DecisionTreeClassifier(max_depth=5),
        "eqx": EQXBinaryClassifierLBFGS(
            model_cls=BinaryNN,
            model_kwargs={"n_features": 10},
            num_epochs=10,
            patience=10,
        ),
    }

    ens = EnsembleBinary(models=models, ensemble_method="weighted_average", n_splits=3)
    ens.fit(X_tr, y_tr)

    proba = ens.predict_proba(X_te)[:, 1]
    preds = ens.predict(X_te)

    print("\nLBFGS BINARY")
    print("  Accuracy:", accuracy_score(y_te, preds))
    print("  LogLoss: ", log_loss(y_te, proba))
    print("  ROC AUC: ", roc_auc_score(y_te, proba))


def test_multiclass():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=2,
    )
    X_tr, X_te = X[:800], X[800:]
    y_tr, y_te = y[:800], y[800:]

    models = {
        "log": LogisticRegression(solver="lbfgs", max_iter=500),
        "tree": DecisionTreeClassifier(max_depth=5),
        "eqx": EQXMulticlassClassifierLBFGS(
            model_cls=MulticlassNN,
            model_kwargs={"n_features": 10, "n_classes": 3},
            num_epochs=10,
            patience=10,
        ),
    }

    ens = EnsembleMulticlass(models=models, ensemble_method="stacking", n_splits=3)
    ens.fit(X_tr, y_tr)

    proba = ens.predict_proba(X_te)
    preds = ens.predict(X_te)

    print("\nLBFGS MULTICLASS")
    print("  Accuracy:", accuracy_score(y_te, preds))
    print("  LogLoss: ", log_loss(y_te, proba))


if __name__ == "__main__":
    # test_regression()
    test_binary()
    # test_multiclass()
