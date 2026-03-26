"""Base class for all predictor models."""

from abc import ABC, abstractmethod

import torch.nn as nn


class BasePredictor(nn.Module, ABC):
    """Abstract base class for LSTM-based predictors.

    All models must:
    - Accept input of shape (batch, seq_len, input_size)
    - Return output of shape (batch, 1) for regression
    - Implement forward() method
    """

    @abstractmethod
    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Tensor of shape (batch, 1) — predicted log-return.
        """
        ...

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
