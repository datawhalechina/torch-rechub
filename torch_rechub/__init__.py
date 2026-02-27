"""Torch-RecHub: A PyTorch Toolbox for Recommendation Models."""

from importlib.metadata import metadata

_meta = metadata("torch-rechub")
__version__ = _meta["Version"]
__author__ = _meta.get("Author") or _meta.get("Author-email", "Unknown")
__license__ = _meta.get("License") or _meta.get("License-Expression", "MIT")
__url__ = _meta.get("Home-page") or _meta.get("Project-URL", "").split(", ")[-1]

# 导入主要模块
from . import basic, models, trainers, utils

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__url__",
    "basic",
    "models",
    "trainers",
    "utils",
]
