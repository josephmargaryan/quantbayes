class SaveLoadMixin:
    """
    Uniform joblib-based persistence for sklearn-compatible estimators.

    Example Usage:
        ```python
        ens.save("ensemble.joblib")
        ens2 = EnsembleBinary.load("ensemble.joblib")
        ```

    """

    def save(self, path: str, compress: int = 3) -> None:
        from joblib import dump
        from sklearn.utils.validation import check_is_fitted

        try:
            check_is_fitted(self)
        except Exception:
            if not (
                getattr(self, "is_fitted_", False) or getattr(self, "is_fitted", False)
            ):
                raise RuntimeError("Estimator appears unfitted; call fit() first.")
        dump(self, path, compress=compress)

    @classmethod
    def load(cls, path: str):
        from joblib import load

        obj = load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
