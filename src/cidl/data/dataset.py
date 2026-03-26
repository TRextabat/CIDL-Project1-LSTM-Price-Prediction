"""PyTorch Dataset for sliding-window time series data."""

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Sliding window dataset for LSTM time series prediction.

    Creates (X, y_regression, y_classification) tuples where:
    - X: feature window of shape (lookback, n_features)
    - y_regression: target log-return at horizon steps ahead
    - y_classification: direction label (1.0 if positive, 0.0 if negative)

    Args:
        features: Array of shape (n_samples, n_features) — scaled features.
        targets: Array of shape (n_samples,) — log returns.
        lookback: Number of past steps to include in each window.
        horizon: Number of steps ahead for prediction target.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        lookback: int,
        horizon: int,
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.features) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Feature window: [idx, idx + lookback)
        x = self.features[idx : idx + self.lookback]

        # Target: log-return at (idx + lookback + horizon - 1)
        target_idx = idx + self.lookback + self.horizon - 1
        y_regression = self.targets[target_idx]

        # Direction classification: positive = 1, negative/zero = 0
        y_classification = (y_regression > 0).float()

        return x, y_regression, y_classification
