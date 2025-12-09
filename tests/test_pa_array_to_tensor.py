import numpy as np
import pyarrow as pa
import pytest
import torch

from torch_rechub.data.convert import pa_array_to_tensor

###############
# scalar arrays
###############


def test_scalar_empty_array() -> None:
    # Given
    array = pa.array([], type=pa.null())

    # When
    tensor = pa_array_to_tensor(array)

    # Then
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.tolist() == []


def test_scalar_bool_arrays() -> None:
    # Given
    array = pa.array([True, True, False], type=pa.bool_())

    # When
    tensor = pa_array_to_tensor(array)

    # Then
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.tolist(), [1.0, 1.0, 0.0])

    # Given
    array = pa.array([None, None, False], type=pa.bool_())

    # When
    tensor = pa_array_to_tensor(array)

    # Then
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.tolist(), [np.nan, np.nan, 0.0], equal_nan=True)


@pytest.mark.parametrize("dtype", [pa.float16(), pa.float32(), pa.float64()])
def test_scalar_float_arrays(dtype: pa.DataType) -> None:
    # Given
    array = pa.array([1.0, 2.0, 3.0], type=dtype)

    # When
    tensor = pa_array_to_tensor(array)

    # Then
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.tolist(), [1.0, 2.0, 3.0])

    # Given
    array = pa.array([None, None, 3.0], type=dtype)

    # When
    tensor = pa_array_to_tensor(array)

    # Then
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.tolist(), [np.nan, np.nan, 3.0], equal_nan=True)


@pytest.mark.parametrize("dtype", [pa.int16(), pa.int32(), pa.int64()])
def test_scalar_int_arrays(dtype: pa.DataType) -> None:
    # Given
    array = pa.array([1, 2, 3], type=dtype)

    # When
    tensor = pa_array_to_tensor(array)

    # Then
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.tolist(), [1.0, 2.0, 3.0])

    # Given
    array = pa.array([None, None, 3], type=dtype)

    # When
    tensor = pa_array_to_tensor(array)

    # Then
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.tolist(), [np.nan, np.nan, 3.0], equal_nan=True)


###############
# nested arrays
###############


def test_nested_empty_array() -> None:
    # Given
    array = pa.array([[]], type=pa.list_(pa.null()))

    # When
    tensor = pa_array_to_tensor(array)

    # Then
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.tolist() == [[]]


def test_nested_bool_arrays() -> None:
    # Given
    array = pa.array([[True], [True]], type=pa.list_(pa.bool_()))

    # When
    tensor = pa_array_to_tensor(array)

    # Then
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.tolist(), [[1.0], [1.0]])

    # Given
    array = pa.array([[None], [None]], type=pa.list_(pa.bool_()))

    # When
    tensor = pa_array_to_tensor(array)

    # Then
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.tolist(), [[np.nan], [np.nan]], equal_nan=True)


@pytest.mark.parametrize("dtype", [pa.float16(), pa.float32(), pa.float64()])
def test_nested_float_arrays(dtype: pa.DataType) -> None:
    # Given
    array = pa.array([[1.0], [2.0]], type=pa.list_(dtype))

    # When
    tensor = pa_array_to_tensor(array)

    # Then
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.tolist(), [[1.0], [2.0]])

    # Given
    array = pa.array([[None], [None]], type=pa.list_(dtype))

    # When
    tensor = pa_array_to_tensor(array)

    # Then
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.tolist(), [[np.nan], [np.nan]], equal_nan=True)


@pytest.mark.parametrize("dtype", [pa.int16(), pa.int32(), pa.int64()])
def test_nested_int_arrays(dtype: pa.DataType) -> None:
    # Given
    array = pa.array([[1], [2]], type=pa.list_(dtype))

    # When
    tensor = pa_array_to_tensor(array)

    # Then
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.tolist(), [[1.0], [2.0]])

    # Given
    array = pa.array([[None], [None]], type=pa.list_(dtype))

    # When
    tensor = pa_array_to_tensor(array)

    # Then
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.tolist(), [[np.nan], [np.nan]], equal_nan=True)


################
# invalid arrays
################


def test_unsupported_scalar_array() -> None:
    # Given
    array = pa.array(["a", "b", "c"], type=pa.string())

    # When/Then
    with pytest.raises(TypeError):
        pa_array_to_tensor(array)


def test_unsupported_nested_array() -> None:
    # Given
    array = pa.array([["a"], ["b"]], type=pa.list_(pa.string()))

    # When/Then
    with pytest.raises(TypeError):
        pa_array_to_tensor(array)


def test_ragged_nested_array() -> None:
    # Given
    array = pa.array([[True], [False, True]], type=pa.list_(pa.bool_()))

    # When/Then
    with pytest.raises(ValueError):
        pa_array_to_tensor(array)
