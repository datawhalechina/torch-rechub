"""Generative Recommendation Models."""

from .hllm import HLLMModel
from .hstu import HSTUModel
from .rqvae import RQVAEModel

__all__ = ['HSTUModel', 'HLLMModel', 'RQVAEModel', 'TIGERModel']


def __getattr__(name):
    if name == 'TIGERModel':
        try:
            from .tiger import TIGERModel
        except ModuleNotFoundError as exc:
            if exc.name == 'transformers':
                raise ImportError("TIGERModel requires the optional generative dependencies. Install with `pip install torch-rechub[generative]`.") from exc
            raise
        return TIGERModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
