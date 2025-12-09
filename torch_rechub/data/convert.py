"""Utilities for converting array-like data structures into PyTorch tensors."""

import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types as pt
import torch


def pa_array_to_tensor(arr: pa.Array) -> torch.Tensor:
    """
    Convert a PyArrow array to a PyTorch tensor.

    Parameters
    ----------
    arr : pa.Array
        The given PyArrow array.

    Returns
    -------
    torch.Tensor: The result PyTorch tensor.

    Raises
    ------
    TypeError
        if the array type or the value type (when nested) is unsupported.
    ValueError
        if the nested array is ragged (unequal lengths of each row).
    """
    if _is_supported_scalar(arr.type):
        arr = pc.cast(arr, pa.float32())
        return torch.from_numpy(_to_writable_numpy(arr))

    if not _is_supported_list(arr.type):
        raise TypeError(f"Unsupported array type: {arr.type}")

    if not _is_supported_scalar(val_type := arr.type.value_type):
        raise TypeError(f"Unsupported value type in the nested array: {val_type}")

    if len(pc.unique(pc.list_value_length(arr))) > 1:
        raise ValueError("Cannot convert the ragged nested array.")

    arr = pc.cast(arr, pa.list_(pa.float32()))
    np_arr = _to_writable_numpy(arr.values)  # type: ignore[attr-defined]

    # For empty list-of-lists, define output shape as (0, 0); otherwise infer width.
    return torch.from_numpy(np_arr.reshape(len(arr), -1 if len(arr) > 0 else 0))


# helper functions


def _is_supported_list(t: pa.DataType) -> bool:
    """Check if the given PyArrow data type is a supported list."""
    return pt.is_fixed_size_list(t) or pt.is_large_list(t) or pt.is_list(t)


def _is_supported_scalar(t: pa.DataType) -> bool:
    """Check if the given PyArrow data type is a supported scalar type."""
    return pt.is_boolean(t) or pt.is_floating(t) or pt.is_integer(t) or pt.is_null(t)


def _to_writable_numpy(arr: pa.Array) -> npt.NDArray:
    """Dump a PyArrow array into a writable NumPy array."""
    # Force the NumPy array to be writable. PyArrow's to_numpy() often returns a
    # read-only view for zero-copy, which PyTorch's from_numpy() does not support.
    return arr.to_numpy(writable=True, zero_copy_only=False)
