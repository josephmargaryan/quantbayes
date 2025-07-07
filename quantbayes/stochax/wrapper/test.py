# test_eqx_ensembles.py

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
from quantbayes.stochax import regression_loss, binary_loss, multiclass_loss
from quantbayes.stochax import (
    EQXRegressor,
    EQXBinaryClassifier,
    EQXMulticlassClassifier,
)


# ─── 0) Define nets with (key, ...) signature ────────────────────────────────────


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


# ─── 1) Regression smoke test ───────────────────────────────────────────────────


def test_regression():
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.2, random_state=0)
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    models = {
        "lin": LinearRegression(),
        "tree": DecisionTreeRegressor(max_depth=5),
        "eqx": EQXRegressor(
            model_cls=RegressorNN,
            model_kwargs={"n_features": 10},
            loss_fn=regression_loss,
            num_epochs=100,
            patience=10,
        ),
    }

    ens = EnsembleRegression(models=models, ensemble_method="stacking", n_splits=5)
    ens.fit(X_train, y_train)
    preds = ens.predict(X_test)

    print("REGRESSION")
    print("  MSE:", mean_squared_error(y_test, preds))
    print("  R2: ", r2_score(y_test, preds))


# ─── 2) Binary classification smoke test ────────────────────────────────────────


def test_binary():
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5, random_state=1
    )
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    models = {
        "log": LogisticRegression(solver="lbfgs", max_iter=500),
        "tree": DecisionTreeClassifier(max_depth=5),
        "eqx": EQXBinaryClassifier(
            model_cls=BinaryNN,
            model_kwargs={"n_features": 10},
            loss_fn=binary_loss,
            num_epochs=100,
            patience=10,
        ),
    }

    ens = EnsembleBinary(models=models, ensemble_method="weighted_average")
    ens.fit(X_train, y_train)

    proba = ens.predict_proba(X_test)[:, 1]
    preds = ens.predict(X_test)

    print("\nBINARY")
    print("  Accuracy:", accuracy_score(y_test, preds))
    print("  LogLoss: ", log_loss(y_test, proba))
    print("  ROC AUC: ", roc_auc_score(y_test, proba))


# ─── 3) Multiclass classification smoke test ────────────────────────────────────


def test_multiclass():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=2,
    )
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    models = {
        "log": LogisticRegression(solver="lbfgs", max_iter=500),
        "tree": DecisionTreeClassifier(max_depth=5),
        "eqx": EQXMulticlassClassifier(
            model_cls=MulticlassNN,
            model_kwargs={"n_features": 10, "n_classes": 3},
            loss_fn=multiclass_loss,
            num_epochs=100,
            patience=10,
        ),
    }

    ens = EnsembleMulticlass(models=models, ensemble_method="stacking", n_splits=5)
    ens.fit(X_train, y_train)

    proba = ens.predict_proba(X_test)
    preds = ens.predict(X_test)

    print("\nMULTICLASS")
    print("  Accuracy:", accuracy_score(y_test, preds))
    print("  LogLoss: ", log_loss(y_test, proba))


# ─── 4) Run all three ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_regression()
    test_binary()
    test_multiclass()
