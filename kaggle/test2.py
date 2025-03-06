import numpy as np
import jax
from copy import deepcopy
from sklearn.model_selection import KFold

def ensure_2d(arr):
    """
    Ensure the input array is 2D.
    If 1D, reshape to (n_samples, 1).
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

class BNNEnsembleBase:
    def __init__(self, base_models, meta_model=None, n_folds=1, rng_key=None, ensemble_method=None):
        """
        Base ensemble class.

        Parameters:
            base_models (dict): Dictionary of base models. Each model must have .fit() and .predict().
            meta_model: For stacking; a model with .fit() and .predict(). For classification tasks, it should also have .predict_proba().
            n_folds (int): Number of folds to use for out‐of‐fold meta feature generation (stacking).
            rng_key: jax.random.PRNGKey for randomness in training/prediction.
            ensemble_method (str): Either "weighted_average" or "stacking". If not provided, defaults to "stacking" if meta_model is provided,
                                   otherwise to "weighted_average".
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.rng_key = rng_key
        if ensemble_method is not None:
            self.ensemble_method = ensemble_method
        else:
            self.ensemble_method = "stacking" if meta_model is not None else "weighted_average"
        self.weights = None           # Used in weighted average
        self.meta_features_train = None  # Stores meta features generated in stacking

    def compile(self, **kwargs):
        """
        Optionally compile each base model and the meta model (if available).
        """
        for name, model in self.base_models.items():
            if hasattr(model, "compile"):
                model.compile(**kwargs)
        if self.meta_model is not None and hasattr(self.meta_model, "compile"):
            self.meta_model.compile(**kwargs)

    def fit(self, X_train, y_train, **kwargs):
        """
        Fit the ensemble on training data.

        For weighted averaging, each base model is trained on the full data and errors are used to compute weights.
        For stacking, out‐of‐fold predictions are generated to train the meta model. Then, base models are retrained on full data.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        # Ensure y_train is 2D for consistent error computation.
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        if self.ensemble_method == "weighted_average":
            errors = []
            for name, model in self.base_models.items():
                model.fit(X_train, y_train, self.rng_key, **kwargs)
                preds = model.predict(X_train, self.rng_key, **kwargs)
                # Use "logits" key if available.
                if isinstance(preds, dict):
                    features = preds.get("logits", preds)
                else:
                    features = preds
                features = ensure_2d(features)
                error = np.mean((features - y_train)**2)
                errors.append(error)
            errors = np.array(errors)
            inv_errors = 1 / (errors + 1e-8)  # add epsilon for stability
            self.weights = inv_errors / np.sum(inv_errors)
        elif self.ensemble_method == "stacking":
            # Generate out-of-fold predictions using KFold.
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=0)
            oof_predictions = {name: None for name in self.base_models.keys()}
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                for name, model in self.base_models.items():
                    # Create a fresh copy of the base model for this fold.
                    model_copy = deepcopy(model)
                    X_train_fold = X_train[train_idx]
                    y_train_fold = y_train[train_idx]
                    model_copy.fit(X_train_fold, y_train_fold, self.rng_key, **kwargs)
                    preds = model_copy.predict(X_train[val_idx], self.rng_key, **kwargs)
                    if isinstance(preds, dict):
                        features = preds.get("logits", preds)
                    else:
                        features = preds
                    features = ensure_2d(features)
                    # Initialize oof_predictions on the first fold.
                    if oof_predictions[name] is None:
                        oof_predictions[name] = np.zeros((len(X_train), features.shape[1]))
                    oof_predictions[name][val_idx, :] = features
            # Concatenate predictions from each model along the feature axis.
            meta_features = np.concatenate(
                [oof_predictions[name] for name in sorted(oof_predictions.keys())], axis=1
            )
            self.meta_features_train = meta_features
            self.meta_model.fit(meta_features, y_train)
            # Finally, retrain all base models on the full training set.
            for name, model in self.base_models.items():
                model.fit(X_train, y_train, self.rng_key, **kwargs)
        else:
            raise ValueError("Unknown ensemble method: choose either 'weighted_average' or 'stacking'.")

    def predict(self, X_test, **kwargs):
        """
        Predict using the ensemble.

        For weighted average, a weighted combination of base model predictions is returned.
        For stacking, meta features are generated and the meta model is used for the final prediction.
        """
        X_test = np.asarray(X_test)
        if self.ensemble_method == "weighted_average":
            preds_list = []
            for name, model in self.base_models.items():
                preds = model.predict(X_test, self.rng_key, **kwargs)
                if isinstance(preds, dict):
                    features = preds.get("logits", preds)
                else:
                    features = preds
                features = ensure_2d(features)
                preds_list.append(features)
            preds_array = np.array(preds_list)  # shape: (num_models, n_samples, feature_dim)
            combined = np.average(preds_array, axis=0, weights=self.weights)
            return combined
        elif self.ensemble_method == "stacking":
            meta_features_list = []
            for name, model in self.base_models.items():
                preds = model.predict(X_test, self.rng_key, **kwargs)
                if isinstance(preds, dict):
                    features = preds.get("logits", preds)
                else:
                    features = preds
                features = ensure_2d(features)
                meta_features_list.append(features)
            meta_features = np.concatenate(meta_features_list, axis=1)
            return self.meta_model.predict(meta_features)
        else:
            raise ValueError("Unknown ensemble method.")

    def predict_proba(self, X_test, **kwargs):
        """
        Predict probabilities (for binary/multiclass tasks) using the meta model.
        """
        if not hasattr(self.meta_model, "predict_proba"):
            raise ValueError("Meta model does not support predict_proba().")
        X_test = np.asarray(X_test)
        meta_features_list = []
        for name, model in self.base_models.items():
            preds = model.predict(X_test, self.rng_key, **kwargs)
            if isinstance(preds, dict):
                features = preds.get("logits", preds)
            else:
                features = preds
            features = ensure_2d(features)
            meta_features_list.append(features)
        meta_features = np.concatenate(meta_features_list, axis=1)
        return self.meta_model.predict_proba(meta_features)

    def summary(self):
        """
        Print a summary of the ensemble.
        """
        print("Ensemble Summary")
        print("----------------")
        print(f"Ensemble Method: {self.ensemble_method}")
        print("Base Models:")
        for name, model in self.base_models.items():
            print(f" - {name}: {model.__class__.__name__}")
        if self.ensemble_method == "weighted_average":
            print("Weights:")
            print(self.weights)
        elif self.ensemble_method == "stacking":
            print("Meta Model:")
            print(self.meta_model)
            if self.meta_features_train is not None:
                print("Meta features shape:", self.meta_features_train.shape)

class BNNEnsembleRegression(BNNEnsembleBase):
    def summary(self):
        print("BNNEnsembleRegression Summary:")
        super().summary()

class BNNEnsembleBinary(BNNEnsembleBase):
    def summary(self):
        print("BNNEnsembleBinary Summary:")
        super().summary()

class BNNEnsembleClassification(BNNEnsembleBase):
    def summary(self):
        print("BNNEnsembleClassification Summary:")
        super().summary()


# === Example Usage ===
if __name__ == "__main__":
    # For demonstration, we define dummy base models and a dummy meta model.
    import jax 
    import jax.numpy as jnp 
    import jax.random as jr
    import numpyro 
    import numpyro.distributions as dist 
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from quantbayes import bnn
    from quantbayes.fake_data import generate_binary_classification_data

    class Dense(bnn.Module):
        def __init__(self):
            super().__init__()
        def __call__(self, X, y=None):
            N, D = X.shape
            X = bnn.ParticleLinear(
                in_features=D,
                out_features=10,
                prior=lambda shape: dist.Cauchy(0, 1).expand(shape).to_event(len(shape))
            )(X)
            X = jax.nn.tanh(X)
            X = bnn.Linear(
                in_features=10, 
                out_features=5,
                weight_prior_fn=lambda shape: dist.Cauchy(0, 1).expand(shape).to_event(len(shape)),
                name="linea layer 1"
                )(X)
            X = jax.nn.tanh(X)
            X = bnn.Linear(
                in_features=5,
                out_features=1,
                weight_prior_fn=lambda shape: dist.Cauchy(0, 1).expand(shape).to_event(len(shape)),
                name="linea layer2"
            )(X)
            logits = X.squeeze()
            numpyro.deterministic("logits", logits)
            with numpyro.plate("data", N):
                numpyro.sample("likelihood", dist.Bernoulli(logits=logits), obs=y)

    class GP(bnn.Module):
        def __init__(self):
            super().__init__(method="nuts")
            self.gp_layer = bnn.GaussianProcessLayer(
                input_dim=1, kernel_type="spectralmixture", name="gp_layer"
            )

        def __call__(self, X, y=None):
            N, D = X.shape
            """        
            X = bnn.ParticleLinear(
                in_features=D,
                out_features=15,
                prior=lambda shape: dist.Cauchy(0, 1).expand(shape).to_event(len(shape))
            )(X)
            X = jax.nn.tanh(X)
            X = bnn.Linear(
                in_features=15,
                out_features=25,
                weight_prior_fn=lambda shape: dist.Cauchy(0, 1).expand(shape).to_event(len(shape))
            )(X)
            """
            # X = self.fft_layer(X)
            # X = bnn.LayerNorm(D)(X)
            kernel_matrix = self.gp_layer(X)
            f = numpyro.sample(
                "f", 
                dist.MultivariateNormal(
                    loc=jnp.zeros(N), 
                    covariance_matrix=kernel_matrix
                )
            )
            logits = f.squeeze()
            numpyro.deterministic("logits", logits)
            with numpyro.plate("data", N):
                numpyro.sample("likelihood", dist.Bernoulli(logits=logits), obs=y)
            self.kernel_matrix = kernel_matrix

    df = generate_binary_classification_data()
    X, y = df.drop("target", axis=1), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = jnp.array(X_train), jnp.array(X_test), jnp.array(y_train), jnp.array(y_test)


    # Create a dictionary of dummy base models.
    base_models = {
        "deep_spectral": GP(),
        "fully_dense": Dense()
    }
    meta_model = LogisticRegression()
    rng_key = jr.PRNGKey(0)

    # Instantiate a binary classification ensemble.
    # (Here we use stacking because a meta model is provided and n_folds=5 for OOF meta feature generation.)
    ensemble_binary = BNNEnsembleBinary(
        base_models=base_models,
        meta_model=meta_model,
        n_folds=5,
        rng_key=rng_key
    )

    # Compile, fit, and then print the summary.
    ensemble_binary.compile()
    ensemble_binary.fit(X_train, y_train)
    ensemble_binary.summary()

    preds = ensemble_binary.predict(X_test)
    print("Predictions:", preds)

    # For classification, predict probabilities if supported.
    probs = ensemble_binary.predict_proba(X_test)
    print("Predicted probabilities:", probs)
