import numpy as np
import logging
from typing import Optional, Union, Callable, List
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score


class SGDLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    L2‑regularized logistic regression with configurable learning‑rate schedulers,
    batch‑size control, early stopping, and full input validation.
    Suitable for production‑level scientific research.

    Parameters
    ----------
    num_epochs : int
        Maximum number of training epochs.
    L2 : float
        L2 regularization strength (λ ≥ 0).
    eta : float
        Initial learning rate (η > 0).
    batch_size : Optional[int]
        None or > N    ⇒ full‑batch GD;
        1              ⇒ stochastic GD;
        otherwise      ⇒ mini‑batch of given size.
    scheduler : Optional[Union[
        Literal['inverse','exponential','step'],
        Callable[[int], float]
    ]]
        Learning‑rate schedule. If callable, must accept epoch index and return ηₜ.
    decay_rate : float
        γ for exponential/step schedulers (0 < γ < 1 recommended).
    step_size : int
        Interval (in epochs) for 'step' scheduler.
    tol : float
        Minimum loss improvement to continue if early_stopping=True.
    early_stopping : bool
        If True, stop when improvement < tol.
    seed : Optional[int]
        If given, sets `np.random.seed(seed)` for reproducibility in `fit`.
    verbose : bool
        If True, logs training‑loss at INFO level.
    """

    def __init__(
        self,
        num_epochs: int = 1000,
        L2: float = 1.0,
        eta: float = 1e-3,
        batch_size: Optional[int] = None,
        scheduler: Optional[Union[str, Callable[[int], float]]] = None,
        decay_rate: float = 0.9,
        step_size: int = 100,
        tol: float = 1e-4,
        early_stopping: bool = False,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        # only store hyperparameters here—no logic
        self.num_epochs = num_epochs
        self.L2 = L2
        self.eta = eta
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.decay_rate = decay_rate
        self.step_size = step_size
        self.tol = tol
        self.early_stopping = early_stopping
        self.seed = seed
        self.verbose = verbose

        # logger (not a hyperparameter)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        if not self._logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter(
                    "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
                )
            )
            self._logger.addHandler(h)

        # model parameters (initialized in fit)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.loss_history_: List[float] = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        pos = z >= 0
        out = np.empty_like(z, dtype=np.float64)
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        return out

    def _get_lr(self, epoch: int) -> float:
        sched = self.scheduler
        if callable(sched):
            lr = float(sched(epoch))
        elif sched == "inverse":
            lr = self.eta / (epoch + 1)
        elif sched == "exponential":
            lr = self.eta * (self.decay_rate**epoch)
        elif sched == "step":
            lr = self.eta * (self.decay_rate ** (epoch // self.step_size))
        else:
            lr = self.eta

        if lr <= 0:
            raise ValueError(f"Non‑positive learning rate {lr} at epoch {epoch}")
        return lr

    def fit(self, X, y) -> "SGDLogisticRegression":
        """
        Train the model on (X, y).

        X : array-like, shape (N, D), dtype float
        y : array-like, shape (N,), values in {0,1}
        """
        # validate inputs
        X, y = check_X_y(X, y, dtype=np.float64)
        N, D = X.shape
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y values must be 0 or 1")

        # parameter sanity checks
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.L2 < 0:
            raise ValueError("L2 must be ≥ 0")
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be None or positive")
        if isinstance(self.scheduler, str) and self.scheduler not in (
            "inverse",
            "exponential",
            "step",
        ):
            raise ValueError(f"Unknown scheduler '{self.scheduler}'")
        if not (0 < self.decay_rate < 1):
            raise ValueError("decay_rate must be in (0,1)")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.tol < 0:
            raise ValueError("tol must be ≥ 0")

        # reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)

        # initialize parameters
        bs = N if (self.batch_size is None or self.batch_size > N) else self.batch_size
        self.coef_ = np.random.randn(D).astype(np.float64) * 0.01
        self.intercept_ = 0.0
        self.loss_history_.clear()

        # training loop
        for epoch in range(self.num_epochs):
            lr = self._get_lr(epoch)
            perm = np.random.permutation(N)
            Xs, ys = X[perm], y[perm]

            for start in range(0, N, bs):
                xb = Xs[start : start + bs]
                yb = ys[start : start + bs]
                m = xb.shape[0]

                z = xb.dot(self.coef_) + self.intercept_
                yhat = self._sigmoid(z)
                error = yhat - yb

                dw = (xb.T.dot(error) / m) + (self.L2 * self.coef_)
                db = np.sum(error) / m

                self.coef_ -= lr * dw
                self.intercept_ -= lr * db

            # full‑batch loss
            z_full = X.dot(self.coef_) + self.intercept_
            y_full = self._sigmoid(z_full)
            bce = -(y * np.log(y_full + 1e-12) + (1 - y) * np.log(1 - y_full + 1e-12))
            reg = 0.5 * self.L2 * np.sum(self.coef_**2) / N
            loss = float(np.mean(bce) + reg)
            self.loss_history_.append(loss)

            self._logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} — loss: {loss:.6f} — lr: {lr:.6f}"
            )

            if (
                self.early_stopping
                and epoch > 0
                and abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tol
            ):
                self._logger.info(
                    f"Early stopping at epoch {epoch+1}; Δloss < {self.tol}"
                )
                break

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return the probability of class 1, shape (N, 2)."""
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(X, dtype=np.float64)
        z = X.dot(self.coef_) + self.intercept_
        p1 = self._sigmoid(z)
        # return 2‑col array as in sklearn
        return np.vstack([1 - p1, p1]).T

    def predict(self, X) -> np.ndarray:
        """Hard binary predictions {0,1}."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def score(self, X, y) -> float:
        """Mean accuracy."""
        y = check_array(y, ensure_2d=False, dtype=None)
        return accuracy_score(y, self.predict(X))

    @property
    def loss(self) -> List[float]:
        """Training‑loss history (one entry per epoch)."""
        return self.loss_history_.copy()


if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics import accuracy_score, log_loss
    import matplotlib.pyplot as plt

    # Example usage
    X_train = np.random.randn(100, 10)
    y_train = (np.random.rand(100) > 0.5).astype(int)

    model = SGDLogisticRegression(
        num_epochs=500,
        eta=0.01,
        batch_size=1,
        scheduler="inverse",  # use ηₜ = η₀ / (t+1)
        decay_rate=0.95,  # used for exponential or step schedulers
        step_size=50,  # used for step scheduler
        verbose=True,
    )
    model.fit(X_train, y_train)

    print("Final weights:", model.coef_)
    print("Final bias:", model.intercept_)

    predictions = model.predict(X_train)
    preds = model.predict(X_train)
    probs = model.predict_proba(X_train)
    print(f"probs shape: {probs.shape}")
    print(f"preds shape: {preds.shape}")

    acc = accuracy_score(y_train, preds)
    lss = log_loss(y_train, probs)

    print(f"Loss: {lss:.3f}")
    print(f"Accuracy: {acc:.3f}")

    loss = model.loss
    plt.plot(loss)
    plt.show()
