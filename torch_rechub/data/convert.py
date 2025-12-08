"""Utilities for converting array-like data structures into PyTorch tensors."""

import typing as ty

import numpy as np
import pyarrow as pa
import pyarrow.types as pt
import torch

from .type import SupportedPythonDType


def pa_array_to_tensor(
    arr: pa.Array,
    null_replace: ty.Optional[SupportedPythonDType] = None,
) -> torch.Tensor:
    """
    Convert a PyArrow array to a PyTorch tensor.

    Parameters
    ----------
    arr : pa.Array
        The given PyArrow array.
    null_replace : SupportedPythonDType
        A global value to replace nulls. If ``None``, nulls will not be replaced and it
        will raise when there are nulls because PyTorch tensor does not support nulls.

    Returns
    -------
    torch.Tensor: The result PyTorch tensor.

    Raises
    ------
    TypeError
        if the array type or the value type (when nested) is unsupported.
    ValueError
        if the nested array is ragged (unequal lengths of each row), or
        if nulls were found but ``null_replace`` is invalid.
    """
    if _is_supported_scalar(arr.type):
        arr = _fill_nulls_if_needed(arr, arr.type, null_replace)
        # Force the NumPy array to be writable. PyArrow's to_numpy() often returns a
        # read-only view for zero-copy, which PyTorch's from_numpy() does not support.
        return torch.from_numpy(arr.to_numpy(writable=True, zero_copy_only=False))

    if _is_supported_list(arr.type):
        if not _is_supported_scalar(val_type := arr.type.value_type):
            raise TypeError(f"Unsupported value type in the nested array: {val_type}")
        arr = _fill_nulls_if_needed(arr, val_type, null_replace)

        try:
            return torch.from_numpy(np.array(arr.to_pylist()))
        except ValueError as error:
            raise ValueError("Cannot convert the ragged nested array.") from error

    raise TypeError(f"Unsupported array type: {arr.type}")


# helper functions


def _fill_nulls_if_needed(
    arr: pa.Array,
    data_type: pa.DataType,
    null_replace: ty.Optional[SupportedPythonDType],
) -> pa.Array:
    """
    Fill nulls in the array if needed, validating the replacement value.

    Parameters
    ----------
    arr : pa.Array
        The input array (scalar or list-like).
    data_type : pa.DataType
        The logical scalar data type to validate against. For scalar arrays, this is
        ``arr.type``; for list arrays this is ``arr.type.value_type``.
    null_replace : SupportedPythonDType
        Replacement value for nulls.

    Returns
    -------
    pa.Array: Either the original array (if no nulls) or a new array with nulls filled.

    Raises
    ------
    ValueError
        if nulls are present but ``null_replace`` is invalid for ``value_type``.
    """
    if arr.null_count == 0:
        return arr

    if not _is_valid_null_replace(null_replace, data_type):
        raise ValueError("Found nulls and invalid null replacement.")

    return arr.fill_null(null_replace)


def _is_valid_null_replace(
    null_replace: ty.Optional[SupportedPythonDType],
    data_type: pa.DataType,
) -> bool:
    """
    Check if the null replace aligns with the PyArrow data type.

    Parameters
    ----------
    null_replace : SupportedPythonDType or None
        The replacement value.
    data_type : pa.DataType
        The Arrow scalar type.

    Returns
    -------
    bool: ``True`` if ``null_replace`` is valid otherwise ``False``.
    """
    if data_type == pa.bool8() or pt.is_boolean(data_type):
        return isinstance(null_replace, bool)

    if pt.is_floating(data_type):
        return isinstance(null_replace, float)

    if pt.is_integer(data_type):
        return isinstance(null_replace, int)

    return False


def _is_supported_list(t: pa.DataType) -> bool:
    """Check if the given PyArrow data type is a supported list."""
    return pt.is_fixed_size_list(t) or pt.is_large_list(t) or pt.is_list(t)


def _is_supported_scalar(t: pa.DataType) -> bool:
    """Check if the given PyArrow data type is a supported scalar."""
    return t == pa.bool8() or pt.is_boolean(t) or pt.is_floating(t) or pt.is_integer(t)
