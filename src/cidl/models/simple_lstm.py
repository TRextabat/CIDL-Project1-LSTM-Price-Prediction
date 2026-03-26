"""Simple single-layer LSTM model."""

import torch
import torch.nn as nn

from .base import BasePredictor


class SimpleLSTM(BasePredictor):
    """Single-layer LSTM with fully connected output.

    Architecture:
        LSTM(input_size, hidden_size, 1 layer) -> last hidden state -> FC(hidden_size, 1)

    Args:
        input_size: Number of input features.
        hidden_size: LSTM hidden dimension (default: 128).
        num_layers: Number of LSTM layers (default: 1).
        dropout: Dropout rate (default: 0.0, unused for single layer).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
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
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, seq_len, input_size) -> (batch, 1)."""
        lstm_out, _ = self.lstm(x)
        # Use last time step hidden state
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out
