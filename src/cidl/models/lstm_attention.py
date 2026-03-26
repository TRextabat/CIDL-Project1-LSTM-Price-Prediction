"""LSTM with Bahdanau attention mechanism."""

import torch
import torch.nn as nn

from .base import BasePredictor


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention over LSTM hidden states.

    Computes attention weights for each time step and returns
    a weighted context vector.

    Args:
        hidden_size: Dimension of LSTM hidden states.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted context vector.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size) — weighted sum of hidden states.
            weights: (batch, seq_len) — attention weights per time step.
        """
        # Score each time step
        scores = self.V(torch.tanh(self.W(hidden_states)))  # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum
        context = (weights * hidden_states).sum(dim=1)  # (batch, hidden_size)

        return context, weights.squeeze(-1)  # weights: (batch, seq_len)


class LSTMAttention(BasePredictor):
    """2-layer LSTM with Bahdanau attention over all hidden states.

    Architecture:
        LSTM(input_size, hidden_size, 2 layers, dropout=0.2)
        -> Bahdanau attention over all seq_len hidden states
        -> context vector (hidden_size)
        -> FC(hidden_size, hidden_size // 2) -> ReLU
        -> FC(hidden_size // 2, 1)

    The attention weights are stored for visualization after forward pass.

    Args:
        input_size: Number of input features.
        hidden_size: LSTM hidden dimension (default: 128).
        num_layers: Number of LSTM layers (default: 2).
        dropout: Dropout rate (default: 0.2).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_weights = None  # Stored after forward pass

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = BahdanauAttention(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, seq_len, input_size) -> (batch, 1).

        Stores attention weights in self.attention_weights for visualization.
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Apply attention
        context, weights = self.attention(lstm_out)
        self.attention_weights = weights.detach()  # Store for visualization

        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
