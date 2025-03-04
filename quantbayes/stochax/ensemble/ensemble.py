import copy
import numpy as np
from sklearn.model_selection import KFold

__all__ = [
    "EnsembleBinary",
    "EnsembleMulticlass",
    "EnsembleRegression",
    "EnsembleForecast",
]

"""
Demonstration:
key1, key2 = jr.split(jr.PRNGKey(1))
model1 = SimpleBinaryModel(X_train.shape[1], 32, key1)
model2 = SimpleBinaryModel(X_train.shape[1], 32, key2)

# Their state is None (since we're not using BN/Dropout) and we create BinaryModel wrappers.
wrapper1 = BinaryModel(lr=1e-2)
wrapper2 = BinaryModel(lr=1e-2)

# Build a dictionary of base models.
base_models = {
    "model1": {"wrapper": wrapper1, "model": model1, "state": None},
    "model2": {"wrapper": wrapper2, "model": model2, "state": None},
}

# 4. Create an EnsembleBinary instance.
# Here, we use stacking and let the ensemble create a default meta learner.
ensemble = EnsembleBinary(
    models=base_models,
    n_splits=3,  # Using 3 folds for a quicker demo.
    ensemble_method="stacking"
)

# 5. Fit the ensemble on the training data.
# We'll use the same hyperparameters as in the base model fit methods.
ensemble.fit(
    X_train=jnp.array(X_train),
    y_train=jnp.array(y_train),
    X_val=jnp.array(X_test),
    y_val=jnp.array(y_test),
    num_epochs=100,
    patience=5,
    key=jr.PRNGKey(999)
)

# 6. Make predictions and visualize performance.
preds = ensemble.predict(jnp.array(X_test), key=jr.PRNGKey(123))
print("Ensemble predictions (raw logits):", preds)

# Visualize using ROC and calibration plots.
ensemble.visualize(jnp.array(X_test), np.array(y_test))
"""


# ================= Ensemble for Binary Classification ====================
class EnsembleBinary:
    """
    Ensemble wrapper for binary classification.

    Expected parameters:
      - models (dict): keys are model names, values are dicts with keys:
            "wrapper": instance of BinaryModel wrapper,
            "model": Equinox model instance,
            "state": any state (or None) used in the wrapper.
      - n_splits (int): number of CV folds for stacking.
      - ensemble_method (str): "weighted_average" or "stacking".
      - weights (dict or None): for weighted averaging; if None, all models are equal.
      - meta_learner: a scikit-learn estimator (must support fit() and predict());
            if None and ensemble_method=="stacking", a default LogisticRegression is used.
    """

    def __init__(
        self,
        models,
        n_splits=5,
        ensemble_method="weighted_average",
        weights=None,
        meta_learner=None,
    ):
        self.models = models
        self.n_splits = n_splits
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.meta_learner = meta_learner
        self.fitted_models = {}  # will store fitted base models (wrapper, model, state)

    def fit(self, X_train, y_train, X_val, y_val, **fit_kwargs):
        if self.ensemble_method == "weighted_average":
            # Simply fit each base model on full training data.
            for name, base in self.models.items():
                wrapper = base["wrapper"]
                model = base["model"]
                state = base["state"]
                fitted_model, fitted_state = wrapper.fit(
                    model, state, X_train, y_train, X_val, y_val, **fit_kwargs
                )
                self.fitted_models[name] = {
                    "wrapper": wrapper,
                    "model": fitted_model,
                    "state": fitted_state,
                }
        elif self.ensemble_method == "stacking":
            # First, generate out-of-fold predictions for each base model.
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            oof_preds = {name: np.zeros(len(y_train)) for name in self.models.keys()}
            for name, base in self.models.items():
                wrapper = base["wrapper"]
                preds_model = np.zeros(len(y_train))
                for train_idx, val_idx in kf.split(X_train):
                    # Clone the base model/state for each fold.
                    model_clone = copy.deepcopy(base["model"])
                    state_clone = copy.deepcopy(base["state"])
                    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
                    X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
                    fitted_model, fitted_state = wrapper.fit(
                        model_clone,
                        state_clone,
                        X_tr,
                        y_tr,
                        X_val_fold,
                        y_val_fold,
                        **fit_kwargs,
                    )
                    preds = wrapper.predict(
                        fitted_model, fitted_state, X_train[val_idx]
                    )
                    # Assume output is logits; store (you can change to probabilities if needed)
                    preds_model[val_idx] = np.array(preds).flatten()
                oof_preds[name] = preds_model
                # Refit the base model on full data.
                fitted_model, fitted_state = wrapper.fit(
                    copy.deepcopy(base["model"]),
                    copy.deepcopy(base["state"]),
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    **fit_kwargs,
                )
                self.fitted_models[name] = {
                    "wrapper": wrapper,
                    "model": fitted_model,
                    "state": fitted_state,
                }
            # Stack the out-of-fold predictions (each column is one model)
            X_meta = np.column_stack(
                [oof_preds[name] for name in sorted(self.models.keys())]
            )
            if self.meta_learner is None:
                from sklearn.linear_model import LogisticRegression

                self.meta_learner = LogisticRegression()
            self.meta_learner.fit(X_meta, np.array(y_train))
        return self

    def predict(self, X, **predict_kwargs):
        if self.ensemble_method == "weighted_average":
            preds_list = []
            for name in sorted(self.fitted_models.keys()):
                base = self.fitted_models[name]
                wrapper = base["wrapper"]
                model = base["model"]
                state = base["state"]
                pred = wrapper.predict(model, state, X, **predict_kwargs)
                preds_list.append(np.array(pred))
            preds_arr = np.array(preds_list)  # shape: (n_models, n_samples)
            if self.weights is None:
                avg_pred = np.mean(preds_arr, axis=0)
            else:
                # Use weights from the provided dictionary (default to 1.0 if missing).
                w = np.array(
                    [
                        self.weights.get(name, 1.0)
                        for name in sorted(self.fitted_models.keys())
                    ]
                )
                w = w / np.sum(w)
                avg_pred = np.average(preds_arr, axis=0, weights=w)
            return avg_pred
        elif self.ensemble_method == "stacking":
            preds_list = []
            for name in sorted(self.fitted_models.keys()):
                base = self.fitted_models[name]
                wrapper = base["wrapper"]
                model = base["model"]
                state = base["state"]
                pred = wrapper.predict(model, state, X, **predict_kwargs)
                preds_list.append(np.array(pred).flatten())
            X_meta = np.column_stack(preds_list)
            return self.meta_learner.predict(X_meta)

    def visualize(self, X, y, **viz_kwargs):
        # For binary classification we show ROC and calibration plots.
        preds = self.predict(X, **viz_kwargs)
        # Convert raw logits to probabilities via sigmoid.
        probs = 1 / (1 + np.exp(-preds))
        from sklearn.metrics import roc_curve, auc, calibration_curve
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y, probs)
        roc_auc = auc(fpr, tpr)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})", color="darkred")
        axs[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axs[0].set_xlabel("False Positive Rate")
        axs[0].set_ylabel("True Positive Rate")
        axs[0].set_title("ROC Curve")
        axs[0].legend()
        prob_true, prob_pred = calibration_curve(y, probs, n_bins=10)
        axs[1].plot(prob_pred, prob_true, marker="o", linewidth=1, label="Calibration")
        axs[1].plot([0, 1], [0, 1], linestyle="--", label="Ideal", color="gray")
        axs[1].set_xlabel("Mean Predicted Probability")
        axs[1].set_ylabel("Fraction of Positives")
        axs[1].set_title("Calibration Plot")
        axs[1].legend()
        plt.suptitle("Ensemble Binary Performance")
        plt.tight_layout()
        plt.show()


# ================= Ensemble for Multiclass Classification ====================
class EnsembleMulticlass:
    """
    Ensemble wrapper for multiclass classification.

    Expects a dictionary of base models (each as a dict with keys:
      "wrapper", "model", "state").

    For stacking the out-of-fold predictions from each base model (each providing
    a vector of logits per sample) are concatenated into a feature matrix.
    A default LogisticRegression (multinomial) is used as meta learner if not provided.
    """

    def __init__(
        self,
        models,
        n_splits=5,
        ensemble_method="weighted_average",
        weights=None,
        meta_learner=None,
    ):
        self.models = models
        self.n_splits = n_splits
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.meta_learner = meta_learner
        self.fitted_models = {}

    def fit(self, X_train, y_train, X_val, y_val, **fit_kwargs):
        if self.ensemble_method == "weighted_average":
            for name, base in self.models.items():
                wrapper = base["wrapper"]
                model = base["model"]
                state = base["state"]
                fitted_model, fitted_state = wrapper.fit(
                    model, state, X_train, y_train, X_val, y_val, **fit_kwargs
                )
                self.fitted_models[name] = {
                    "wrapper": wrapper,
                    "model": fitted_model,
                    "state": fitted_state,
                }
        elif self.ensemble_method == "stacking":
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            # For each base model, we will collect out-of-fold predictions.
            # Here, each prediction is an array of shape (n_samples, n_classes).
            oof_preds = {
                name: np.zeros((len(y_train), None)).tolist()
                for name in self.models.keys()
            }
            # We will first determine n_classes from the first model's prediction.
            for name, base in self.models.items():
                wrapper = base["wrapper"]
                preds_list = [None] * len(y_train)
                for train_idx, val_idx in kf.split(X_train):
                    model_clone = copy.deepcopy(base["model"])
                    state_clone = copy.deepcopy(base["state"])
                    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
                    X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
                    fitted_model, fitted_state = wrapper.fit(
                        model_clone,
                        state_clone,
                        X_tr,
                        y_tr,
                        X_val_fold,
                        y_val_fold,
                        **fit_kwargs,
                    )
                    preds = wrapper.predict(
                        fitted_model, fitted_state, X_train[val_idx]
                    )
                    # Ensure preds is a 2D array.
                    preds = np.atleast_2d(np.array(preds))
                    for i, idx in enumerate(val_idx):
                        preds_list[idx] = preds[i]
                # Stack predictions for this model.
                oof_preds[name] = np.vstack(preds_list)
                # Refit the base model on full data.
                fitted_model, fitted_state = wrapper.fit(
                    copy.deepcopy(base["model"]),
                    copy.deepcopy(base["state"]),
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    **fit_kwargs,
                )
                self.fitted_models[name] = {
                    "wrapper": wrapper,
                    "model": fitted_model,
                    "state": fitted_state,
                }
            # Concatenate features: for each sample, concatenate predictions from all models.
            meta_features = np.hstack(
                [oof_preds[name] for name in sorted(self.models.keys())]
            )
            if self.meta_learner is None:
                from sklearn.linear_model import LogisticRegression

                self.meta_learner = LogisticRegression(
                    multi_class="multinomial", max_iter=1000
                )
            self.meta_learner.fit(meta_features, np.array(y_train))
        return self

    def predict(self, X, **predict_kwargs):
        if self.ensemble_method == "weighted_average":
            preds_list = []
            for name in sorted(self.fitted_models.keys()):
                base = self.fitted_models[name]
                wrapper = base["wrapper"]
                model = base["model"]
                state = base["state"]
                pred = wrapper.predict(model, state, X, **predict_kwargs)
                preds_list.append(
                    np.array(pred)
                )  # each of shape (n_samples, n_classes)
            preds_arr = np.array(preds_list)  # shape: (n_models, n_samples, n_classes)
            if self.weights is None:
                avg_pred = np.mean(preds_arr, axis=0)
            else:
                w = np.array(
                    [
                        self.weights.get(name, 1.0)
                        for name in sorted(self.fitted_models.keys())
                    ]
                )
                w = w / np.sum(w)
                # Weighted average along axis 0.
                avg_pred = np.tensordot(w, preds_arr, axes=([0], [0]))
            return avg_pred
        elif self.ensemble_method == "stacking":
            preds_list = []
            for name in sorted(self.fitted_models.keys()):
                base = self.fitted_models[name]
                wrapper = base["wrapper"]
                model = base["model"]
                state = base["state"]
                pred = wrapper.predict(model, state, X, **predict_kwargs)
                preds_list.append(np.atleast_2d(np.array(pred)))
            meta_features = np.hstack(preds_list)
            return self.meta_learner.predict(meta_features)

    def visualize(self, X, y, **viz_kwargs):
        # For multiclass, we can use a confusion matrix and a bar plot of average predicted probabilities.
        preds = self.predict(X, **viz_kwargs)
        # Get predicted class labels.
        pred_classes = np.argmax(preds, axis=-1)
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y, pred_classes)
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[0])
        axs[0].set_xlabel("Predicted")
        axs[0].set_ylabel("True")
        axs[0].set_title("Confusion Matrix")
        avg_probs = np.mean(preds, axis=0)
        axs[1].bar(
            range(len(avg_probs)), avg_probs, color="mediumseagreen", edgecolor="black"
        )
        axs[1].set_xlabel("Class")
        axs[1].set_ylabel("Average Predicted Probability")
        axs[1].set_title("Average Predicted Probabilities")
        plt.suptitle("Ensemble Multiclass Performance")
        plt.tight_layout()
        plt.show()


# ================= Ensemble for Regression ====================
class EnsembleRegression:
    """
    Ensemble wrapper for regression.

    Expects models (dict) similarly to the above wrappers.
    For stacking, out-of-fold predictions are concatenated and a meta learner
    (default is LinearRegression) is used.
    """

    def __init__(
        self,
        models,
        n_splits=5,
        ensemble_method="weighted_average",
        weights=None,
        meta_learner=None,
    ):
        self.models = models
        self.n_splits = n_splits
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.meta_learner = meta_learner
        self.fitted_models = {}

    def fit(self, X_train, y_train, X_val, y_val, **fit_kwargs):
        if self.ensemble_method == "weighted_average":
            for name, base in self.models.items():
                wrapper = base["wrapper"]
                model = base["model"]
                state = base["state"]
                fitted_model, fitted_state = wrapper.fit(
                    model, state, X_train, y_train, X_val, y_val, **fit_kwargs
                )
                self.fitted_models[name] = {
                    "wrapper": wrapper,
                    "model": fitted_model,
                    "state": fitted_state,
                }
        elif self.ensemble_method == "stacking":
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            oof_preds = {name: np.zeros(len(y_train)) for name in self.models.keys()}
            for name, base in self.models.items():
                wrapper = base["wrapper"]
                preds_model = np.zeros(len(y_train))
                for train_idx, val_idx in kf.split(X_train):
                    model_clone = copy.deepcopy(base["model"])
                    state_clone = copy.deepcopy(base["state"])
                    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
                    X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
                    fitted_model, fitted_state = wrapper.fit(
                        model_clone,
                        state_clone,
                        X_tr,
                        y_tr,
                        X_val_fold,
                        y_val_fold,
                        **fit_kwargs,
                    )
                    preds = wrapper.predict(
                        fitted_model, fitted_state, X_train[val_idx]
                    )
                    preds_model[val_idx] = np.array(preds).flatten()
                oof_preds[name] = preds_model
                fitted_model, fitted_state = wrapper.fit(
                    copy.deepcopy(base["model"]),
                    copy.deepcopy(base["state"]),
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    **fit_kwargs,
                )
                self.fitted_models[name] = {
                    "wrapper": wrapper,
                    "model": fitted_model,
                    "state": fitted_state,
                }
            meta_features = np.column_stack(
                [oof_preds[name] for name in sorted(self.models.keys())]
            )
            if self.meta_learner is None:
                from sklearn.linear_model import LinearRegression

                self.meta_learner = LinearRegression()
            self.meta_learner.fit(meta_features, np.array(y_train))
        return self

    def predict(self, X, **predict_kwargs):
        if self.ensemble_method == "weighted_average":
            preds_list = []
            for name in sorted(self.fitted_models.keys()):
                base = self.fitted_models[name]
                wrapper = base["wrapper"]
                model = base["model"]
                state = base["state"]
                pred = wrapper.predict(model, state, X, **predict_kwargs)
                preds_list.append(np.array(pred).flatten())
            preds_arr = np.array(preds_list)
            if self.weights is None:
                avg_pred = np.mean(preds_arr, axis=0)
            else:
                w = np.array(
                    [
                        self.weights.get(name, 1.0)
                        for name in sorted(self.fitted_models.keys())
                    ]
                )
                w = w / np.sum(w)
                avg_pred = np.average(preds_arr, axis=0, weights=w)
            return avg_pred
        elif self.ensemble_method == "stacking":
            preds_list = []
            for name in sorted(self.fitted_models.keys()):
                base = self.fitted_models[name]
                wrapper = base["wrapper"]
                model = base["model"]
                state = base["state"]
                pred = wrapper.predict(model, state, X, **predict_kwargs)
                preds_list.append(np.array(pred).flatten())
            meta_features = np.column_stack(preds_list)
            return self.meta_learner.predict(meta_features)

    def visualize(self, X, y, **viz_kwargs):
        preds = self.predict(X, **viz_kwargs)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.scatter(np.array(y), preds, alpha=0.7)
        plt.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], "k--", lw=2)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Ensemble Regression Predictions vs. True")
        plt.show()


# ================= Ensemble for Forecasting ====================
class EnsembleForecast:
    """
    Ensemble wrapper for forecasting (real-valued sequential predictions).

    Its design is similar to the regression ensemble. For stacking, out‐of‐fold predictions
    are generated and a meta learner (default is LinearRegression) is used.
    """

    def __init__(
        self,
        models,
        n_splits=5,
        ensemble_method="weighted_average",
        weights=None,
        meta_learner=None,
    ):
        self.models = models
        self.n_splits = n_splits
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.meta_learner = meta_learner
        self.fitted_models = {}

    def fit(self, X_train, Y_train, X_val, Y_val, **fit_kwargs):
        if self.ensemble_method == "weighted_average":
            for name, base in self.models.items():
                wrapper = base["wrapper"]
                model = base["model"]
                state = base["state"]
                fitted_model, fitted_state = wrapper.fit(
                    model, state, X_train, Y_train, X_val, Y_val, **fit_kwargs
                )
                self.fitted_models[name] = {
                    "wrapper": wrapper,
                    "model": fitted_model,
                    "state": fitted_state,
                }
        elif self.ensemble_method == "stacking":
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            oof_preds = {name: np.zeros(len(Y_train)) for name in self.models.keys()}
            for name, base in self.models.items():
                wrapper = base["wrapper"]
                preds_model = np.zeros(len(Y_train))
                for train_idx, val_idx in kf.split(X_train):
                    model_clone = copy.deepcopy(base["model"])
                    state_clone = copy.deepcopy(base["state"])
                    X_tr, Y_tr = X_train[train_idx], Y_train[train_idx]
                    X_val_fold, Y_val_fold = X_train[val_idx], Y_train[val_idx]
                    fitted_model, fitted_state = wrapper.fit(
                        model_clone,
                        state_clone,
                        X_tr,
                        Y_tr,
                        X_val_fold,
                        Y_val_fold,
                        **fit_kwargs,
                    )
                    preds = wrapper.predict(
                        fitted_model, fitted_state, X_train[val_idx]
                    )
                    preds_model[val_idx] = np.array(preds).flatten()
                oof_preds[name] = preds_model
                fitted_model, fitted_state = wrapper.fit(
                    copy.deepcopy(base["model"]),
                    copy.deepcopy(base["state"]),
                    X_train,
                    Y_train,
                    X_val,
                    Y_val,
                    **fit_kwargs,
                )
                self.fitted_models[name] = {
                    "wrapper": wrapper,
                    "model": fitted_model,
                    "state": fitted_state,
                }
            meta_features = np.column_stack(
                [oof_preds[name] for name in sorted(self.models.keys())]
            )
            if self.meta_learner is None:
                from sklearn.linear_model import LinearRegression

                self.meta_learner = LinearRegression()
            self.meta_learner.fit(meta_features, np.array(Y_train))
        return self

    def predict(self, X, **predict_kwargs):
        if self.ensemble_method == "weighted_average":
            preds_list = []
            for name in sorted(self.fitted_models.keys()):
                base = self.fitted_models[name]
                wrapper = base["wrapper"]
                model = base["model"]
                state = base["state"]
                pred = wrapper.predict(model, state, X, **predict_kwargs)
                preds_list.append(np.array(pred).flatten())
            preds_arr = np.array(preds_list)
            if self.weights is None:
                avg_pred = np.mean(preds_arr, axis=0)
            else:
                w = np.array(
                    [
                        self.weights.get(name, 1.0)
                        for name in sorted(self.fitted_models.keys())
                    ]
                )
                w = w / np.sum(w)
                avg_pred = np.average(preds_arr, axis=0, weights=w)
            return avg_pred
        elif self.ensemble_method == "stacking":
            preds_list = []
            for name in sorted(self.fitted_models.keys()):
                base = self.fitted_models[name]
                wrapper = base["wrapper"]
                model = base["model"]
                state = base["state"]
                pred = wrapper.predict(model, state, X, **predict_kwargs)
                preds_list.append(np.array(pred).flatten())
            meta_features = np.column_stack(preds_list)
            return self.meta_learner.predict(meta_features)

    def visualize(self, Y_true, Y_pred, title="Forecast vs. Ground Truth"):
        import matplotlib.pyplot as plt

        Y_true = np.array(Y_true).flatten()
        Y_pred = np.array(Y_pred).flatten()
        plt.figure(figsize=(10, 4))
        plt.plot(Y_true, marker="o", label="Ground Truth")
        plt.plot(Y_pred, marker="x", label="Predictions")
        plt.title(title)
        plt.xlabel("Sample Index")
        plt.legend()
        plt.show()
