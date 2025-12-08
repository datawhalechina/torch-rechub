"""Reusable type aliases."""

import os
import typing as ty

# Type for path to a file.
FilePath = ty.Union[str, os.PathLike]

# Type for supported Python data types
SupportedPythonDType = ty.Union[bool, float, int]
