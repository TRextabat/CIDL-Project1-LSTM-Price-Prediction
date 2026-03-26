"""Walk-forward safe RobustScaler wrapper."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import RobustScaler


class WalkForwardScaler:
    """RobustScaler wrapper ensuring no look-ahead bias in walk-forward CV.

    The scaler is fitted ONLY on training data for each fold,
    then applied to transform both training and test data.
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "WalkForwardScaler":
        """Fit scaler on training data only."""
        self.scaler.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")
        return self.scaler.inverse_transform(X)

    def save(self, path: str | Path) -> None:
        """Save scaler to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str | Path) -> "WalkForwardScaler":
        """Load scaler from disk."""
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance._is_fitted = True
        return instance

    @property
    def center_(self) -> np.ndarray | None:
        """Return the median used for centering."""
        if self._is_fitted:
            return self.scaler.center_
        return None

    @property
    def scale_(self) -> np.ndarray | None:
        """Return the IQR used for scaling."""
        if self._is_fitted:
            return self.scaler.scale_
        return None
