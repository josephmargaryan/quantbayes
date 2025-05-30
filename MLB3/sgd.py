import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from typing import Literal, Dict


def run_sgd(
    model_type: Literal["binary", "multiclass", "regression"],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    alpha: float = 1.0,
    T: int = 200,
    batch_size: int = 32,
    learning_rate: Literal["constant", "optimal", "invscaling"] = "constant",
    eta0: float = 1e-3,
    power_t: float = 0.5,
    penalty: Literal["l2", "l1", "elasticnet"] = "l2",
    random_state: int = 0,
    record_every: int = 10,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Run SGD for binary/multiclass classification or regression.

    Returns:
        dict with keys:
            'iters': iteration indices,
            'train_loss': recorded training loss,
            'test_loss': recorded test loss,
            'train_metric': training metric (acc or MSE),
            'test_metric': test metric (acc or MSE)
    """
    # Choose estimator and metrics
    if model_type == "regression":
        Estimator = SGDRegressor
        train_metric_fn = mean_squared_error
        test_metric_fn = mean_squared_error
        train_loss_fn = mean_squared_error
        test_loss_fn = mean_squared_error
        estimator_kwargs = {"loss": "squared_error"}
    else:
        Estimator = SGDClassifier
        train_metric_fn = lambda y, p: accuracy_score(
            y, np.argmax(p, axis=1) if p.ndim > 1 else (p > 0.5).astype(int)
        )
        test_metric_fn = train_metric_fn
        train_loss_fn = log_loss
        test_loss_fn = log_loss
        estimator_kwargs = {"loss": "log_loss"}

    # Common parameters
    params = dict(
        penalty=penalty,
        alpha=alpha,
        fit_intercept=True,
        learning_rate=learning_rate,
        eta0=eta0,
        power_t=power_t,
        random_state=random_state,
        shuffle=True,
        **estimator_kwargs,
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize model
    model = Estimator(**params)
    rng = np.random.default_rng(random_state)

    # For classifier: classes
    classes = np.unique(y_train) if model_type != "regression" else None

    record_iters = np.arange(record_every, T + 1, record_every)
    train_losses = []
    test_losses = []
    train_metrics = []
    test_metrics = []

    for i in range(1, T + 1):  # each iteration = single gradient update
        # sample batch
        if batch_size >= X_train_scaled.shape[0]:
            Xb, yb = X_train_scaled, y_train
        else:
            idx = rng.choice(X_train_scaled.shape[0], batch_size, replace=False)
            Xb, yb = X_train_scaled[idx], y_train[idx]

        # partial fit
        if i == 1:
            if classes is not None:
                model.partial_fit(Xb, yb, classes=classes)
            else:
                model.partial_fit(Xb, yb)
        else:
            model.partial_fit(Xb, yb)

        # record
        if i % record_every == 0:
            # predictions
            if model_type == "regression":
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
            else:
                # for log_loss we need probabilities
                if hasattr(model, "predict_proba"):
                    train_pred = model.predict_proba(X_train_scaled)
                    test_pred = model.predict_proba(X_test_scaled)
                else:
                    # use decision_function then softmax or sigmoid
                    dp_train = model.decision_function(X_train_scaled)
                    dp_test = model.decision_function(X_test_scaled)
                    # binary case: sigmoid
                    if dp_train.ndim == 1:
                        train_pred = np.vstack(
                            [
                                1 - 1 / (1 + np.exp(-dp_train)),
                                1 / (1 + np.exp(-dp_train)),
                            ]
                        ).T
                        test_pred = np.vstack(
                            [1 - 1 / (1 + np.exp(-dp_test)), 1 / (1 + np.exp(-dp_test))]
                        ).T
                    else:
                        # multiclass: softmax
                        def softmax(z):
                            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                            return exp_z / np.sum(exp_z, axis=1, keepdims=True)

                        train_pred = softmax(dp_train)
                        test_pred = softmax(dp_test)

            # compute and store
            train_losses.append(train_loss_fn(y_train, train_pred))
            test_losses.append(test_loss_fn(y_test, test_pred))
            train_metrics.append(train_metric_fn(y_train, train_pred))
            test_metrics.append(test_metric_fn(y_test, test_pred))
            if verbose:
                print(
                    f"Iter {i}: train_loss={train_losses[-1]:.4f}, test_loss={test_losses[-1]:.4f}"
                )

    return {
        "iters": record_iters,
        "train_loss": np.array(train_losses),
        "test_loss": np.array(test_losses),
        "train_metric": np.array(train_metrics),
        "test_metric": np.array(test_metrics),
    }


if __name__ == "__main__":
    # --- Regression demo ---
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, log_loss

    # Data
    X, y = make_regression(n_samples=500, n_features=20, noise=0.1, random_state=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)

    # Closed-form Ridge (alpha matches)
    ridge = Ridge(alpha=0.01).fit(Xtr, ytr)
    preds_ridge = ridge.predict(Xte)
    mse_ridge = mean_squared_error(yte, preds_ridge)
    print(f"Ridge closed-form MSE: {mse_ridge:.4f}")

    # SGDRegressor (ridge)
    res_reg = run_sgd(
        "regression",
        Xtr,
        Xte,
        ytr,
        yte,
        alpha=0.01,
        T=2000,
        batch_size=len(Xtr),
        learning_rate="constant",
        eta0=0.1,
        record_every=500,
    )
    print(f"SGD (ridge) MSE: {res_reg['test_loss'][-1]:.4f}")

    # --- Binary Classification demo ---
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    # Data
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=5, n_classes=2, random_state=2
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=2)

    # Closed-form Logistic Regression
    logreg = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    prob_logreg = logreg.predict_proba(Xte)
    loss_logreg = log_loss(yte, prob_logreg)
    print(f"LogisticRegression log-loss: {loss_logreg:.4f}")

    # SGDClassifier (logistic)
    res_bin = run_sgd(
        "binary",
        Xtr,
        Xte,
        ytr,
        yte,
        alpha=0.01,
        T=1000,
        batch_size=32,
        learning_rate="optimal",
        eta0=1.0,
        record_every=100,
    )
    print(f"SGDClassifier log-loss: {res_bin['test_loss'][-1]:.4f}")
    print(f"SGDClassifier accuracy: {res_bin['test_metric'][-1]:.4f}")

    # --- Multiclass Classification demo ---
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=3)

    # Closed-form multinomial logistic
    from sklearn.linear_model import LogisticRegression

    mlog = LogisticRegression(solver="lbfgs", max_iter=1000).fit(Xtr, ytr)
    prob_mlog = mlog.predict_proba(Xte)
    loss_mlog = log_loss(yte, prob_mlog)
    acc_mlog = accuracy_score(yte, mlog.predict(Xte))
    print(f"Multinomial LR log-loss: {loss_mlog:.4f}, accuracy: {acc_mlog:.4f}")

    # SGDClassifier (multiclass)
    res_multi = run_sgd(
        "multiclass",
        Xtr,
        Xte,
        ytr,
        yte,
        alpha=0.01,
        T=1000,
        batch_size=16,
        learning_rate="optimal",
        eta0=1.0,
        record_every=100,
    )
    print(
        f"SGDClassifier log-loss: {res_multi['test_loss'][-1]:.4f}, accuracy: {res_multi['test_metric'][-1]:.4f}"
    )
