import copy
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import optax
from sklearn.linear_model import LogisticRegression

from quantbayes.stochax.trainer.train import (
    data_loader,
    train,
    predict,
    binary_loss,
    multiclass_loss,
    regression_loss,
)


# ------------------------------------------------------------------------
# Base Class for Ensemble
# ------------------------------------------------------------------------


class _BaseEquinoxEnsemble:
    """
    A base class that trains multiple Equinox models and can combine predictions
    either via weighted average or stacking with a meta-learner.

    Child classes override how predictions are interpreted (e.g., sigmoid vs softmax vs raw).
    """

    def __init__(
        self,
        model_constructors,
        loss_fn,
        optimizer,
        ensemble_method="weighted_average",
        meta_learner=None,
        weights=None,
    ):
        """
        Parameters
        ----------
        model_constructors : list of callables
            Each callable takes a PRNG key and returns a freshly initialized Equinox model.
        loss_fn : callable
            The loss function used for training (binary_loss, multiclass_loss, or regression_loss).
        optimizer : optax.GradientTransformation
            The optimizer to use for training each model in the ensemble.
        ensemble_method : str
            "weighted_average" or "stacking".
        meta_learner : sklearn-like estimator, optional
            Used only if ensemble_method == "stacking".
            Defaults to LogisticRegression() if None.
        weights : list of floats or None
            Per-model weights if using weighted averaging. Must match length of model_constructors.
            If None, defaults to uniform weights.
        """

        self.model_constructors = model_constructors
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.ensemble_method = ensemble_method

        if meta_learner is None:
            self.meta_learner = LogisticRegression()
        else:
            self.meta_learner = meta_learner

        # Will hold tuples of (best_model, best_state) for each ensemble member
        self.ensemble_members = []

        # If weighting is not provided, use uniform
        if self.ensemble_method == "weighted_average":
            if weights is None:
                self.weights = [1.0] * len(model_constructors)
            else:
                assert len(weights) == len(
                    model_constructors
                ), "Length of weights must match number of model_constructors"
                self.weights = weights
        else:
            self.weights = None  # Not used in stacking

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size,
        num_epochs,
        patience,
        key,
    ):
        """
        Train each ensemble member, then (if stacking) train the meta-learner on the validation set.
        """
        num_models = len(self.model_constructors)
        keys = jr.split(key, num_models)

        self.ensemble_members = []
        for i, constructor in enumerate(self.model_constructors):
            print(f"\n=== Training Model {i+1}/{num_models} ===")
            model_key = keys[i]
            model = constructor(model_key)
            state = (
                None  # For models that track state (e.g., BatchNorm), adapt as needed
            )
            opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))

            best_model, best_state, _, _ = train(
                model=model,
                state=state,
                opt_state=opt_state,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                batch_size=batch_size,
                num_epochs=num_epochs,
                patience=patience,
                key=model_key,
            )
            self.ensemble_members.append((best_model, best_state))

        # If stacking, gather predictions for the meta-learner
        if self.ensemble_method == "stacking":
            print("\n=== Fitting Meta-Learner (Stacking) ===")
            # Use the logic specific to the child class to get meta-features from validation data
            meta_features = self._build_meta_features(
                X_val, key=jr.PRNGKey(9999)
            )  # separate key
            y_val_np = np.array(y_val).ravel()
            self.meta_learner.fit(meta_features, y_val_np)

    def predict(self, X, key):
        """
        Returns final predictions (discrete classes for classification or real values for regression).
        Child classes can override or call `_combine_predictions(...)`.
        """
        raise NotImplementedError("Override in child class.")

    def _combine_predictions(self, raw_predictions):
        """
        Combine the raw model outputs (logits or real values) using the chosen ensemble method.

        raw_predictions: list of JAX arrays, each shape (N, output_dim).
        Returns a single JAX array or NumPy array of shape (N, output_dim).
        """
        if self.ensemble_method == "weighted_average":
            # Weighted average of raw predictions
            weighted_sum = 0.0
            total_weight = sum(self.weights)
            for w, p in zip(self.weights, raw_predictions):
                weighted_sum += w * p
            avg_preds = weighted_sum / total_weight
            return avg_preds

        elif self.ensemble_method == "stacking":
            # For stacking, we do not directly combine here. Instead, we rely on
            # the meta-learner's predict or predict_proba using stacked features.
            # This base method won't be used in stacking logic. Child classes can override.
            raise ValueError(
                "Shouldn't directly combine predictions when using 'stacking' method."
            )

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def _build_meta_features(self, X, key):
        """
        Return model-level features for stacking, e.g. probabilities or raw outputs
        stacked horizontally. Child classes define how to convert raw outputs (logits, etc.)
        into features (e.g., apply sigmoid or softmax).
        """
        raise NotImplementedError("Override in child class.")


# ------------------------------------------------------------------------
# 1) Binary Classification Ensemble
# ------------------------------------------------------------------------


class EnsembleBinary(_BaseEquinoxEnsemble):
    """
    Ensemble specialized for binary classification:
      - Applies sigmoid to logits.
      - `predict_proba` returns probabilities in [0, 1].
      - `predict` returns discrete 0/1 (or possibly 0/1 classes from the meta-learner).
    """

    def predict(self, X, key):
        """
        Predict discrete class labels for binary classification.
        If stacking, uses meta-learner. Otherwise, uses threshold=0.5 on averaged probabilities.
        """
        num_models = len(self.ensemble_members)
        keys = jr.split(key, num_models)

        # Collect raw logits from each model
        model_logits = []
        for i, (model, state) in enumerate(self.ensemble_members):
            logits = predict(model, state, X, keys[i])  # shape (N, 1)
            model_logits.append(logits)

        if self.ensemble_method == "weighted_average":
            # Weighted average of logits -> apply sigmoid -> threshold
            logits_avg = self._combine_predictions(model_logits)
            probs = jax.nn.sigmoid(logits_avg)
            # Discrete 0/1
            return np.array(probs >= 0.5, dtype=int).ravel()

        elif self.ensemble_method == "stacking":
            # Build stacking features (sigmoid probabilities)
            meta_features = self._build_meta_features(X, key)
            # Use meta-learner's predict (classes)
            final_preds = self.meta_learner.predict(meta_features)
            return final_preds

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def predict_proba(self, X, key):
        """
        Return probabilities for binary classification in [0, 1].
        If stacking, uses meta-learner's predict_proba. Otherwise, does weighted avg of sigmoids.
        """
        num_models = len(self.ensemble_members)
        keys = jr.split(key, num_models)

        model_logits = []
        for i, (model, state) in enumerate(self.ensemble_members):
            logits = predict(model, state, X, keys[i])
            model_logits.append(logits)

        if self.ensemble_method == "weighted_average":
            # Weighted average of logits -> apply sigmoid
            logits_avg = self._combine_predictions(model_logits)
            probs = jax.nn.sigmoid(logits_avg)
            return np.array(probs).ravel()  # shape (N,)

        elif self.ensemble_method == "stacking":
            # Build stacking features (sigmoids from each model)
            meta_features = self._build_meta_features(X, key)
            if hasattr(self.meta_learner, "predict_proba"):
                # shape (N, 2) for binary
                final_probs = self.meta_learner.predict_proba(meta_features)
                return final_probs[:, 1]  # Probability of class "1"
            else:
                # fallback: if meta-learner lacks predict_proba, use predict and interpret as 0/1
                preds = self.meta_learner.predict(meta_features)
                return preds  # 0 or 1, you can adapt as needed

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def _build_meta_features(self, X, key):
        """
        For binary classification, each model's feature is sigmoid(logits).
        Stacked horizontally => shape (N, num_models).
        """
        num_models = len(self.ensemble_members)
        keys = jr.split(key, num_models)

        all_probs = []
        for i, (model, state) in enumerate(self.ensemble_members):
            logits = predict(model, state, X, keys[i])  # shape (N, 1)
            probs = jax.nn.sigmoid(logits)
            all_probs.append(np.array(probs).reshape(-1, 1))

        meta_features = np.hstack(all_probs)  # shape (N, num_models)
        return meta_features


# ------------------------------------------------------------------------
# 2) Multiclass Classification Ensemble
# ------------------------------------------------------------------------


class EnsembleMulticlass(_BaseEquinoxEnsemble):
    """
    Ensemble specialized for multi-class classification:
      - Applies softmax to logits for each model.
      - `predict_proba` returns a distribution over classes (N, C).
      - `predict` returns argmax class.
      - If stacking, you can feed per-class probabilities as meta-features to the meta-learner.
        Make sure your meta-learner is scikit-learn multi-class capable (e.g. `multi_class='multinomial'`).
    """

    def predict(self, X, key):
        """
        Predict argmax class for multiclass classification.
        If stacking, use meta-learner. Otherwise, average probabilities.
        """
        num_models = len(self.ensemble_members)
        keys = jr.split(key, num_models)

        # Collect raw logits from each model
        model_logits = []
        for i, (model, state) in enumerate(self.ensemble_members):
            logits = predict(model, state, X, keys[i])  # shape (N, num_classes)
            model_logits.append(logits)

        if self.ensemble_method == "weighted_average":
            # Weighted average of logits -> softmax -> argmax
            logits_avg = self._combine_predictions(model_logits)  # shape (N, C)
            probs = jax.nn.softmax(logits_avg, axis=-1)
            return np.array(jnp.argmax(probs, axis=-1))

        elif self.ensemble_method == "stacking":
            # Build meta-features from each model's softmax
            meta_features = self._build_meta_features(X, key)
            # meta_learner must handle multi-class
            final_preds = self.meta_learner.predict(meta_features)
            return final_preds

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def predict_proba(self, X, key):
        """
        Return class probabilities (N, C) for multiclass classification.
        If stacking, uses meta-learner's predict_proba. Otherwise, weighted average of softmaxes.
        """
        num_models = len(self.ensemble_members)
        keys = jr.split(key, num_models)

        model_logits = []
        for i, (model, state) in enumerate(self.ensemble_members):
            logits = predict(model, state, X, keys[i])
            model_logits.append(logits)

        if self.ensemble_method == "weighted_average":
            # Weighted average of logits -> softmax
            logits_avg = self._combine_predictions(model_logits)
            probs = jax.nn.softmax(logits_avg, axis=-1)
            return np.array(probs)

        elif self.ensemble_method == "stacking":
            # Build meta-features from each model's softmax
            meta_features = self._build_meta_features(X, key)
            if hasattr(self.meta_learner, "predict_proba"):
                # shape = (N, #classes)
                final_probs = self.meta_learner.predict_proba(meta_features)
                return final_probs
            else:
                # fallback: use predict => discrete classes => one-hot? (or adapt as needed)
                preds = self.meta_learner.predict(meta_features)
                # you might want to convert discrete predictions to one-hot
                # but normally you want a meta-learner that has predict_proba
                # For simplicity, return discrete classes
                return preds
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def _build_meta_features(self, X, key):
        """
        For multiclass classification, each model's feature is softmax(logits).
        We flatten or keep them separate?
        Suppose each model outputs (N, C) => we flatten row-wise => shape (N, C * num_models).
        """
        num_models = len(self.ensemble_members)
        keys = jr.split(key, num_models)

        all_probs = []
        for i, (model, state) in enumerate(self.ensemble_members):
            logits = predict(model, state, X, keys[i])  # shape (N, C)
            probs = jax.nn.softmax(logits, axis=-1)  # shape (N, C)
            # Flatten each row if you want them all in a single feature vector:
            all_probs.append(np.array(probs))

        # Concatenate along the feature dimension => (N, C * num_models)
        meta_features = np.concatenate(all_probs, axis=-1)
        return meta_features


# ------------------------------------------------------------------------
# 3) Regression Ensemble
# ------------------------------------------------------------------------


class EnsembleRegression(_BaseEquinoxEnsemble):
    """
    Ensemble specialized for regression tasks:
      - Models output raw real values (shape (N, 1) or (N,)).
      - Weighted average sums them up, or stacking can use a meta-regressor (e.g., LinearRegression).
      - No 'predict_proba' method is provided here.
    """

    def predict(self, X, key):
        """
        Returns final regression predictions.
        If stacking, uses meta-regressor's prediction on the stacked model outputs.
        """
        num_models = len(self.ensemble_members)
        keys = jr.split(key, num_models)

        # Collect raw predictions from each model
        model_outputs = []
        for i, (model, state) in enumerate(self.ensemble_members):
            preds = predict(model, state, X, keys[i])  # shape (N,) or (N, 1)
            model_outputs.append(preds)

        if self.ensemble_method == "weighted_average":
            avg_preds = self._combine_predictions(model_outputs)  # shape (N,) or (N,1)
            return np.array(avg_preds).ravel()

        elif self.ensemble_method == "stacking":
            meta_features = self._build_meta_features(X, key)
            final_preds = self.meta_learner.predict(meta_features)
            return final_preds

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def _build_meta_features(self, X, key):
        """
        For regression, each model's feature is just the raw output (N,1).
        Stacked horizontally => shape (N, num_models).
        """
        num_models = len(self.ensemble_members)
        keys = jr.split(key, num_models)

        all_preds = []
        for i, (model, state) in enumerate(self.ensemble_members):
            preds = predict(model, state, X, keys[i])  # shape (N,) or (N,1)
            preds_np = np.array(preds).reshape(-1, 1)  # enforce (N,1)
            all_preds.append(preds_np)

        meta_features = np.hstack(all_preds)  # shape (N, num_models)
        return meta_features

    # No predict_proba for regression
    def predict_proba(self, X, key):
        raise NotImplementedError(
            "`predict_proba` is not applicable for regression tasks."
        )


# ------------------------------------------------------------------------
# Example Usage (Replace placeholders with real data/model definitions)
# ------------------------------------------------------------------------
if __name__ == "__main__":
    import jax
    import equinox as eqx
    import optax
    import jax.numpy as jnp
    import numpy as np
    import jax.random as jr
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss

    from quantbayes.fake_data import generate_binary_classification_data
    from quantbayes.stochax.layers import CirculantProcess

    df = generate_binary_classification_data()
    X, y = df.drop("target", axis=1), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)

    class EQXNet1(eqx.Module):
        l1: eqx.nn.Linear

        def __init__(self, in_features, key):
            self.l1 = eqx.nn.Linear(in_features, 1, key=key)

        def __call__(self, x, key, state):
            x = self.l1(x)
            logits = jnp.squeeze(x, axis=-1)
            return logits, state

    class EQXNet2(eqx.Module):
        l1: eqx.Module
        l2: eqx.nn.Linear

        def __init__(self, in_features, key):
            k1, k2 = jr.split(key, 2)
            self.l1 = CirculantProcess(in_features, key=k1)
            self.l2 = eqx.nn.Linear(in_features, 1, key=k2)

        def __call__(self, x, key, state):
            x = self.l1(x)
            x = jax.nn.tanh(x)
            x = self.l2(x)
            logits = jnp.squeeze(x, axis=-1)
            return logits, state

    # Create ensemble for binary classification
    in_features = X_train.shape[-1]
    ensemble_binary = EnsembleBinary(
        model_constructors=[
            lambda key: EQXNet1(in_features, key),
            lambda key: EQXNet2(in_features, key),
        ],
        loss_fn=binary_loss,
        optimizer=optax.adam(learning_rate=1e-3),
        ensemble_method="stacking",  # or "weighted_average"
        meta_learner=LogisticRegression(),
        weights=None,  # Not used if stacking
    )

    # Fit ensemble
    key = jr.PRNGKey(42)
    ensemble_binary.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        batch_size=1000,
        num_epochs=10000,
        patience=100,
        key=key,
    )

    # Predict
    test_key = jr.PRNGKey(999)
    y_pred = ensemble_binary.predict(X_test, test_key)  # discrete classes for binary
    y_prob = ensemble_binary.predict_proba(X_test, test_key)  # probabilities for binary

    loss = log_loss(np.array(y_test), np.array(y_prob))
    print(f"Log loss: {loss:.3f}")

    # You can similarly instantiate EnsembleMulticlass or EnsembleRegression
    # for multiclass or regression tasks!
