from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np


class LabelMapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self._map = {-1: 0, +1: 1}
        self._inv = {v: k for k, v in self._map.items()}
        y2 = np.vectorize(self._map.__getitem__)(y)
        self.clf_ = clone(self.base_estimator).fit(X, y2)
        return self

    def predict(self, X):
        y2 = self.clf_.predict(X)
        return np.vectorize(self._inv.__getitem__)(y2)

    def predict_proba(self, X):
        p = self.clf_.predict_proba(X)
        return np.vstack([1 - p[:, 1], p[:, 1]]).T
