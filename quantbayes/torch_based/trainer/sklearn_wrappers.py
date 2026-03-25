import numpy as np
import torch.nn as nn
from typing import Optional, Callable, Dict, Any, Tuple
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.base import clone
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

from quantbayes.torch_based.trainer.trainer import (
    Trainer,
    TrainerConfig,
)  # adjust import as needed

__all__ = [
    "TorchRegressor",
    "TorchBinaryClassifier",
    "TorchMulticlassClassifier",
    "TorchImageSegmenter",
    "SegmentationEnsemble",
]


class AugmentedDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        augment_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ):
        self.X = X
        self.y = y
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.augment_fn:
            x, y = self.augment_fn(x, y)
        return x, y


class _TorchBase(BaseEstimator):
    def __init__(
        self,
        module_cls: Callable[..., nn.Module],
        module_kwargs: Optional[Dict[str, Any]] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 128,
        num_epochs: int = 50,
        patience: int = 10,
        val_frac: float = 0.1,
        augment_fn: Optional[Callable] = None,
        **unused,
    ):
        self.module_cls = module_cls
        self.module_kwargs = module_kwargs or {}
        self.trainer_kwargs = trainer_kwargs or {}
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.augment_fn = augment_fn

    def get_params(self, deep=True):
        return {
            "module_cls": self.module_cls,
            "module_kwargs": self.module_kwargs,
            "trainer_kwargs": self.trainer_kwargs,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "patience": self.patience,
            "val_frac": self.val_frac,
            "augment_fn": self.augment_fn,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class TorchRegressor(_TorchBase, RegressorMixin):
    _default_loss = nn.MSELoss()
    _is_classifier = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 1) tabular data → tensors
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

        # 2) split, loaders
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_t,
            y_t,
            test_size=self.val_frac,
            shuffle=True,
            random_state=self.trainer_kwargs.get("seed", 42),
        )
        tr_ds = AugmentedDataset(X_tr, y_tr, augment_fn=self.augment_fn)
        va_ds = TensorDataset(X_va, y_va)
        tr_ld = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
        va_ld = DataLoader(va_ds, batch_size=self.batch_size)

        # 3) model, optim, sched
        self.model = self.module_cls(**self.module_kwargs)
        loss_fn = self._default_loss
        optim = self.trainer_kwargs.get("optimizer") or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.trainer_kwargs.get("init_lr", 1e-3),
            weight_decay=self.trainer_kwargs.get("weight_decay", 1e-4),
        )
        sched = self.trainer_kwargs.get("scheduler", None)

        cfg = TrainerConfig(
            output_dir=self.trainer_kwargs.get("output_dir", "./torch_outputs"),
            early_stop_patience=self.patience,
            **self.trainer_kwargs,
        )
        self.trainer = Trainer(
            model=self.model,
            optimizer=optim,
            loss_fn=loss_fn,
            scheduler=sched,
            device=self.trainer_kwargs.get("device", "cpu"),
            config=cfg,
        )
        self.trainer.train(tr_ld, va_ld, num_epochs=self.num_epochs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.trainer.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(X_t)
        preds = out.squeeze(-1).cpu().numpy()
        return preds


class TorchBinaryClassifier(_TorchBase, ClassifierMixin):
    _default_loss = nn.BCEWithLogitsLoss()
    _is_classifier = True

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_t,
            y_t,
            test_size=self.val_frac,
            shuffle=True,
            random_state=self.trainer_kwargs.get("seed", 42),
        )
        tr_ds = AugmentedDataset(X_tr, y_tr, augment_fn=self.augment_fn)
        va_ds = TensorDataset(X_va, y_va)
        tr_ld = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
        va_ld = DataLoader(va_ds, batch_size=self.batch_size)

        self.model = self.module_cls(**self.module_kwargs)
        loss_fn = self._default_loss
        optim = self.trainer_kwargs.get("optimizer") or torch.optim.AdamW(
            self.model.parameters(), lr=self.trainer_kwargs.get("init_lr", 1e-3)
        )
        sched = self.trainer_kwargs.get("scheduler", None)
        cfg = TrainerConfig(
            output_dir=self.trainer_kwargs.get("output_dir", "./torch_outputs"),
            early_stop_patience=self.patience,
            **self.trainer_kwargs,
        )
        self.trainer = Trainer(
            self.model,
            optim,
            loss_fn,
            sched,
            self.trainer_kwargs.get("device", "cpu"),
            cfg,
        )
        self.trainer.train(tr_ld, va_ld, num_epochs=self.num_epochs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.trainer.device)
        self.model.eval()
        with torch.no_grad():
            logit = self.model(X_t).squeeze(-1).cpu().numpy()
        p = 1 / (1 + np.exp(-logit))
        return np.vstack([1 - p, p]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TorchMulticlassClassifier(_TorchBase, ClassifierMixin):
    """
    scikit-learn–compatible multiclass classifier wrapper around any nn.Module.
    Expects your module to output logits of shape (N, n_classes).
    """

    _default_loss = nn.CrossEntropyLoss()
    _is_classifier = True

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 1) numpy → torch
        X_t = torch.from_numpy(X.astype(np.float32))
        # CrossEntropyLoss wants targets of shape (N,) with dtype long
        y = np.asarray(y, dtype=np.int64).ravel()
        y_t = torch.from_numpy(y)

        # 2) train/val split
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_t,
            y_t,
            test_size=self.val_frac,
            shuffle=True,
            random_state=self.trainer_kwargs.get("seed", 42),
        )

        # 3) datasets + loaders
        tr_ds = AugmentedDataset(X_tr, y_tr, augment_fn=self.augment_fn)
        va_ds = TensorDataset(X_va, y_va)
        tr_ld = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
        va_ld = DataLoader(va_ds, batch_size=self.batch_size, shuffle=False)

        # 4) model, loss, optimizer, scheduler
        self.model = self.module_cls(**self.module_kwargs)
        loss_fn = self._default_loss
        optim = self.trainer_kwargs.get("optimizer") or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.trainer_kwargs.get("init_lr", 1e-3),
            weight_decay=self.trainer_kwargs.get("weight_decay", 1e-4),
        )
        sched = self.trainer_kwargs.get("scheduler", None)

        # 5) TrainerConfig + Trainer
        cfg = TrainerConfig(
            output_dir=self.trainer_kwargs.get("output_dir", "./torch_outputs"),
            early_stop_patience=self.patience,
            **self.trainer_kwargs,
        )
        self.trainer = Trainer(
            model=self.model,
            optimizer=optim,
            loss_fn=loss_fn,
            scheduler=sched,
            device=self.trainer_kwargs.get("device", "cpu"),
            config=cfg,
        )

        # 6) train
        self.trainer.train(tr_ld, va_ld, num_epochs=self.num_epochs)
        # record classes_
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.trainer.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)  # (N, n_classes)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


# ─────────────── Vision Subclasses ───────────────


class TorchVisionBase(_TorchBase):
    def _prepare_X(self, X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(X.astype(np.float32))


class TorchImageSegmenter(TorchVisionBase):
    _default_loss = nn.CrossEntropyLoss()
    _is_classifier = True

    def fit(self, X: np.ndarray, y: np.ndarray):
        # prepare data
        X_t = self._prepare_X(X)  # (N,C,H,W)
        y_np = np.asarray(y, dtype=np.int64)  # (N,H,W) or (N,1,H,W)

        # ensure y_np is (N,H,W)
        if y_np.ndim == 4 and y_np.shape[1] == 1:
            y_np = y_np[:, 0, :, :]

        y_t = torch.from_numpy(y_np)  # (N,H,W)

        # train/val split
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_t,
            y_t,
            test_size=self.val_frac,
            shuffle=True,
            random_state=self.trainer_kwargs.get("seed", 42),
        )

        # dataloaders
        tr_ds = AugmentedDataset(X_tr, y_tr, augment_fn=self.augment_fn)
        va_ds = TensorDataset(X_va, y_va)
        tr_ld = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
        va_ld = DataLoader(va_ds, batch_size=self.batch_size)

        # model & optimizer
        self.model = self.module_cls(**self.module_kwargs)
        loss_fn = self._default_loss
        optim = self.trainer_kwargs.get("optimizer") or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.trainer_kwargs.get("init_lr", 1e-3),
            weight_decay=self.trainer_kwargs.get("weight_decay", 1e-4),
        )
        sched = self.trainer_kwargs.get("scheduler", None)

        # **explicitly** construct TrainerConfig
        cfg = TrainerConfig(
            output_dir=self.trainer_kwargs.get("output_dir", "./torch_outputs"),
            save_best_k=self.trainer_kwargs.get("save_best_k", 3),
            grad_accum_steps=self.trainer_kwargs.get("grad_accum_steps", 1),
            max_grad_norm=self.trainer_kwargs.get("max_grad_norm", None),
            amp=self.trainer_kwargs.get("amp", False),
            early_stop_patience=self.patience,
            restore_best=self.trainer_kwargs.get("restore_best", True),
            log_interval=self.trainer_kwargs.get("log_interval", 50),
            deterministic=self.trainer_kwargs.get("deterministic", True),
            seed=self.trainer_kwargs.get("seed", 42),
            scheduler_step=self.trainer_kwargs.get("scheduler_step", "epoch"),
        )

        # train
        self.trainer = Trainer(
            model=self.model,
            optimizer=optim,
            loss_fn=loss_fn,
            scheduler=sched,
            device=self.trainer_kwargs.get("device", "cpu"),
            config=cfg,
        )
        self.trainer.train(tr_ld, va_ld, num_epochs=self.num_epochs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = self._prepare_X(X).to(self.trainer.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)  # (N,C,H,W)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()  # (N,C,H,W)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)  # (N,H,W)


# ─────────────── Segmentation Ensemble ───────────────


class SegmentationEnsemble(BaseEstimator):
    """
    Pixel-wise ensemble of multiple TorchImageSegmenter.
    Supports 'average' or 'stacking' (linear meta-learner).
    """

    def __init__(
        self,
        models: Dict[str, TorchImageSegmenter],
        ensemble_method: str = "average",
        weights: Optional[Dict[str, float]] = None,
        cv: Optional[KFold] = None,
        meta_learner: Optional[Any] = None,
    ):
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.cv = cv or KFold(n_splits=5, shuffle=True, random_state=42)
        self.meta_learner = meta_learner or LogisticRegression(max_iter=500)

    def fit(self, X: np.ndarray, y: np.ndarray):
        names = list(self.models)
        N, C, H, W = X.shape
        self.fitted_ = {}

        # train each base model
        for name in names:
            inst = clone(self.models[name])
            inst.fit(X, y)
            self.fitted_[name] = inst

        if self.ensemble_method == "stacking":
            # build out-of-fold pixel features
            oof_feats = np.zeros((N, len(names), H, W), dtype=float)
            for i, name in enumerate(names):
                m = self.models[name]
                for tr, va in self.cv.split(X, y.reshape(N, -1)[:, 0]):
                    inst = clone(m)
                    inst.fit(X[tr], y[tr])
                    probs = inst.predict_proba(X[va])[
                        :, 1, ...
                    ]  # single foreground channel
                    oof_feats[va, i] = probs
                # re-fit on full data
                self.fitted_[name].fit(X, y)

            # flatten pixels
            flat_X = oof_feats.transpose(0, 2, 3, 1).reshape(-1, len(names))
            flat_y = y.reshape(N, H * W)
            flat_y = flat_y.ravel()
            self.meta_ = clone(self.meta_learner)
            self.meta_.fit(flat_X, flat_y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        names = list(self.models)
        N, C, H, W = X.shape
        # collect per-model probs (take channel=1 if binary)
        probs = [
            self.fitted_[n].predict_proba(X) for n in names
        ]  # each (N,2,H,W) or (N,C,H,W)
        # if binary, shape is (N,2,H,W) -> take foreground channel
        if probs[0].shape[1] == 2:
            probs = [p[:, 1, ...] for p in probs]
        stacked = np.stack(probs, axis=-1)  # (N,H,W,M)
        if self.ensemble_method == "average":
            w = (
                np.array([self.weights.get(n, 1.0) for n in names])
                if self.weights
                else np.ones(len(names))
            )
            w /= w.sum()
            out = np.tensordot(stacked, w, axes=([3], [0]))  # (N,H,W)
            return np.stack([1 - out, out], axis=1)  # binary two-channel
        else:
            flat = stacked.reshape(-1, len(names))
            meta_p = self.meta_.predict_proba(flat)[:, 1]
            meta_p = meta_p.reshape(N, H, W)
            return np.stack([1 - meta_p, meta_p], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)  # (N,H,W)


if __name__ == "__main__":
    import numpy as np, torch, torch.nn as nn, random
    from sklearn.model_selection import train_test_split

    # toy model
    class ToySegModel(nn.Module):
        def __init__(self, in_channels=1, classes=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, classes, 1),
            )

        def forward(self, x):
            return self.net(x)

    def random_flip(x, y):
        if random.random() < 0.5:
            x = torch.flip(x, [-1])
            y = torch.flip(y, [-1])
        return x, y

    # tiny dataset
    N, C, H, W = 50, 1, 64, 64
    X = np.random.rand(N, C, H, W).astype(np.float32)
    y = np.zeros((N, H, W), dtype=np.int64)
    for i in range(N):
        cy, cx = np.random.randint(H), np.random.randint(W)
        r = np.random.randint(5, 15)
        yy, xx = np.ogrid[:H, :W]
        y[i] = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(np.int64)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    seg1 = TorchImageSegmenter(
        ToySegModel,
        {"in_channels": C, "classes": 2},
        {"device": "cpu"},
        batch_size=8,
        num_epochs=5,
        patience=2,
        augment_fn=random_flip,
    )
    seg2 = TorchImageSegmenter(
        ToySegModel,
        {"in_channels": C, "classes": 2},
        {"device": "cpu"},
        batch_size=8,
        num_epochs=5,
        patience=2,
        augment_fn=random_flip,
    )

    ensemble = SegmentationEnsemble(
        models={"s1": seg1, "s2": seg2}, ensemble_method="average"
    )

    print("Fitting ensemble...")
    ensemble.fit(X_tr, y_tr)

    print("Predicting...")
    proba = ensemble.predict_proba(X_te)
    preds = ensemble.predict(X_te)
    print("proba shape:", proba.shape, "preds shape:", preds.shape)
    print("Test complete.")
