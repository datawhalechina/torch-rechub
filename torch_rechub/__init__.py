"""Torch-RecHub: A PyTorch Toolbox for Recommendation Models."""

__version__ = "0.1.0"

# 导入主要模块
from . import basic, models, trainers, utils

__all__ = [
    "__version__",
    "basic",
    "models",
    "trainers",
    "utils",
]
