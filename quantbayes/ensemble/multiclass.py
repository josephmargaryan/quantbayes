import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class EnsembleMulticlassClassificationModel(BaseEstimator, ClassifierMixin):
    def __init__(self, models, n_splits=5, ensemble_method="weighted_average", weights=None, meta_learner=None):
        """
        Ensemble Multiclass Classification Model.
        
        Parameters:
            models (dict): Dictionary of base classifiers. Keys are model names and values are model instances.
            n_splits (int): Number of folds for cross-validation (used in stacking).
            ensemble_method (str): "weighted_average" or "stacking"
            weights (dict or None): Dictionary mapping model names to weights (only used if ensemble_method=="weighted_average").
                                    If None, equal weights are used.
            meta_learner: The meta-classifier used for stacking. If None and stacking is chosen, defaults to LogisticRegression with multinomial settings.
        """
        self.models = models
        self.n_splits = n_splits
        
        if ensemble_method not in ["weighted_average", "stacking"]:
            raise ValueError("ensemble_method must be either 'weighted_average' or 'stacking'")
        self.ensemble_method = ensemble_method
        
        self.weights = weights
        
        if self.ensemble_method == "stacking":
            # Default meta learner for multiclass problems
            self.meta_learner = meta_learner if meta_learner is not None else LogisticRegression(solver='lbfgs', max_iter=1000)
        else:
            self.meta_learner = None
        
        # Containers for fitted models
        self.fitted_models_ = {}       # base classifiers fitted on full data
        self.meta_fitted_ = None         # meta classifier (for stacking)
        self.oof_predictions_ = None     # out-of-fold predictions for stacking (shape: [n_samples, n_models*n_classes])
        self.train_predictions_proba_ = None  # ensemble predicted probabilities on training set
        self.is_fitted_ = False

    def fit(self, X, y):
        """
        Fit the ensemble classifier on training data.
        
        For stacking, generate out-of-fold predicted probability vectors from each base model
        (for all classes) via KFold and train a meta learner on the concatenated predictions.
        For weighted average, simply fit each base classifier on the full data.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        model_names = list(self.models.keys())
        
        if self.ensemble_method == "stacking":
            n_samples = X.shape[0]
            # Determine number of classes from y by fitting first base model's predict_proba later,
            # or we can infer it from np.unique(y)
            classes = np.unique(y)
            n_classes = len(classes)
            n_models = len(model_names)
            # Will store out-of-fold predictions for each model; features for meta learner:
            # shape: (n_samples, n_models * n_classes)
            oof_preds = np.zeros((n_samples, n_models * n_classes))
            
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            
            for idx, model_name in enumerate(model_names):
                model = self.models[model_name]
                # Temporary container for current model's out-of-fold predicted probabilities
                oof_model_preds = np.zeros((n_samples, n_classes))
                
                for train_idx, val_idx in kf.split(X):
                    model_clone = clone(model)
                    model_clone.fit(X[train_idx], y[train_idx])
                    # Expect that the model supports predict_proba.
                    oof_model_preds[val_idx, :] = model_clone.predict_proba(X[val_idx])
                
                # Place the current model's predictions in the corresponding block of columns
                start_col = idx * n_classes
                end_col = start_col + n_classes
                oof_preds[:, start_col:end_col] = oof_model_preds
                
                # Refit the base model on the full dataset and store it.
                fitted_model = clone(model)
                fitted_model.fit(X, y)
                self.fitted_models_[model_name] = fitted_model
            
            # Train the meta learner on the concatenated out-of-fold predictions.
            self.meta_fitted_ = clone(self.meta_learner)
            self.meta_fitted_.fit(oof_preds, y)
            
            # Store training ensemble probabilities computed using meta learner.
            self.train_predictions_proba_ = self.meta_fitted_.predict_proba(oof_preds)
            self.oof_predictions_ = oof_preds
            
        elif self.ensemble_method == "weighted_average":
            preds_list = []
            for model_name in model_names:
                model = clone(self.models[model_name])
                model.fit(X, y)
                self.fitted_models_[model_name] = model
                # Get predicted probability matrix for each model (shape: [n_samples, n_classes])
                preds_list.append(model.predict_proba(X))
            # Stack predictions along new axis: shape becomes (n_models, n_samples, n_classes)
            preds_array = np.array(preds_list)  # shape: (n_models, n_samples, n_classes)
            
            # Use equal weights if not provided.
            if self.weights is None:
                weights_arr = np.ones(len(model_names)) / len(model_names)
            else:
                weights_arr = np.array([self.weights.get(name, 0) for name in model_names], dtype=float)
                if np.sum(weights_arr) == 0:
                    raise ValueError("Sum of provided weights cannot be zero.")
                weights_arr = weights_arr / np.sum(weights_arr)
            
            # Combine predictions using weighted average:
            # For each sample and each class, weighted sum over models.
            # Multiply each model's prediction matrix by its weight.
            weighted_preds = np.tensordot(weights_arr, preds_array, axes=1)
            # weighted_preds shape: (n_samples, n_classes)
            self.train_predictions_proba_ = weighted_preds
        else:
            raise ValueError("Unknown ensemble_method provided.")
        
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Returns an array of shape (n_samples, n_classes)
        """
        if not self.is_fitted_:
            raise RuntimeError("The ensemble model must be fitted before predicting.")
        
        X = np.asarray(X)
        model_names = list(self.models.keys())
        
        if self.ensemble_method == "stacking":
            # Determine number of classes from meta_fitted_'s predict_proba output.
            # Collect base models' predicted probabilities, then concatenate.
            preds_list = []
            for model_name in model_names:
                model = self.fitted_models_[model_name]
                preds_list.append(model.predict_proba(X))
            # Concatenate along columns.
            # If there are n_models and each gives shape (n_samples, n_classes),
            # the concatenated feature matrix will have shape (n_samples, n_models * n_classes).
            X_meta = np.hstack(preds_list)
            proba = self.meta_fitted_.predict_proba(X_meta)
        
        elif self.ensemble_method == "weighted_average":
            preds_list = []
            for model_name in model_names:
                model = self.fitted_models_[model_name]
                preds_list.append(model.predict_proba(X))
            preds_array = np.array(preds_list)  # shape: (n_models, n_samples, n_classes)
            
            if self.weights is None:
                weights_arr = np.ones(len(model_names)) / len(model_names)
            else:
                weights_arr = np.array([self.weights.get(name, 0) for name in model_names], dtype=float)
                weights_arr = weights_arr / np.sum(weights_arr)
            proba = np.tensordot(weights_arr, preds_array, axes=1)
        else:
            raise ValueError("Unknown ensemble_method provided.")
        
        return proba

    def predict(self, X):
        """
        Predict class labels for X.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def fit_predict(self, X, y):
        """
        Convenience method: fit the model and predict on X.
        """
        self.fit(X, y)
        return self.predict(X)

    def summary(self):
        """
        Print a summary of the ensemble classifier.
        """
        if not self.is_fitted_:
            raise RuntimeError("The model has not been fitted yet. Call fit or fit_predict first.")
        
        print("Ensemble Multiclass Classification Model Summary")
        print("-------------------------------------------------")
        print(f"Ensemble Method: {self.ensemble_method}")
        print("Base Models:")
        for name, model in self.models.items():
            print(f" - {name}: {model.__class__.__name__}")
        if self.ensemble_method == "stacking":
            print(f"Meta Learner: {self.meta_learner.__class__.__name__}")
        if self.train_predictions_proba_ is not None:
            print("\nNote: Training-set ensemble predicted probabilities are stored in self.train_predictions_proba_.")
        print("-------------------------------------------------")

    def plot_confusion(self, X, y_true, normalize=False, cmap="Blues"):
        """
        Plot the confusion matrix for the provided data.
        
        Parameters:
            X : array-like, feature matrix.
            y_true : array-like, true class labels.
            normalize (bool): If True, normalize the confusion matrix.
            cmap (str): Colormap.
        """
        from sklearn.metrics import ConfusionMatrixDisplay
        
        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

        fig, ax = plt.subplots(figsize=(8, 6))  # Create a single figure
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=cmap, ax=ax)  # Explicitly pass ax to prevent new figure creation
        
        plt.title("Confusion Matrix")
        plt.show()

# Test the ensemble multiclass classifier when running as the main program.
if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    # Generate a synthetic multiclass classification dataset.
    # Here, we create 4 classes.
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
                               n_classes=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define base classifiers.
    models = {
        "LogisticRegression": LogisticRegression(solver='lbfgs', max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(max_depth=7, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
        # Note: For SVC, probability=True is needed to call predict_proba.
        "SVC": SVC(probability=True, random_state=42)
    }

    # Choose ensemble method: "weighted_average" or "stacking"
    ensemble_method = "stacking"  # You can change to "weighted_average" to test that method.
    weights = None  # Only used for weighted_average if desired.

    # Create ensemble instance.
    ensemble = EnsembleMulticlassClassificationModel(models, n_splits=5,
                                                       ensemble_method=ensemble_method,
                                                       weights=weights,
                                                       meta_learner=None)  # Defaults to LogisticRegression for stacking.

    # Fit the ensemble on training data.
    ensemble.fit(X_train, y_train)
    train_preds = ensemble.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print("Training Accuracy: {:.4f}".format(train_acc))
    print("\nTraining Classification Report:")
    print(classification_report(y_train, train_preds))
    
    ensemble.summary()

    # Evaluate on test data.
    test_preds = ensemble.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_preds))

    # Plot confusion matrix on test data.
    ensemble.plot_confusion(X_test, y_test, normalize=True)
