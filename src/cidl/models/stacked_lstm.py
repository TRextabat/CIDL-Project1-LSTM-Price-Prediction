"""Stacked (deep) LSTM model with 3 layers."""

import torch
import torch.nn as nn

from .base import BasePredictor


class StackedLSTM(BasePredictor):
    """3-layer stacked LSTM with dropout and two FC layers.

    Architecture:
        LSTM(input_size, hidden_size, 3 layers, dropout=0.2)
        -> last hidden state
        -> FC(hidden_size, hidden_size) -> ReLU -> Dropout
        -> FC(hidden_size, 1)

    Args:
        input_size: Number of input features.
        hidden_size: LSTM hidden dimension (default: 128).
        num_layers: Number of LSTM layers (default: 3).
        dropout: Dropout rate (default: 0.2).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, seq_len, input_size) -> (batch, 1)."""
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
