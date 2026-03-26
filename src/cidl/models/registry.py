"""Model registry for creating models by name."""

from .base import BasePredictor
from .bilstm import BidirectionalLSTM
from .lstm_attention import LSTMAttention
from .simple_lstm import SimpleLSTM
from .stacked_lstm import StackedLSTM

MODEL_REGISTRY: dict[str, type[BasePredictor]] = {
    "simple_lstm": SimpleLSTM,
    "stacked_lstm": StackedLSTM,
    "bilstm": BidirectionalLSTM,
    "lstm_attention": LSTMAttention,
}


def create_model(name: str, input_size: int, **kwargs) -> BasePredictor:
    """Create a model instance by name.

    Args:
        name: Model name (must be in MODEL_REGISTRY).
        input_size: Number of input features.
        **kwargs: Additional arguments (hidden_size, num_layers, dropout).

    Returns:
        Instantiated model.

    Raises:
        KeyError: If model name is not registered.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](input_size=input_size, **kwargs)
