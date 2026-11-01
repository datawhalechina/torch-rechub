import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from torch_rechub.data.convert import pa_array_to_tensor
from torch_rechub.data.dataset import ParquetIterableDataset


@pytest.mark.parametrize("dtype", [pa.bool_(), pa.int64(), pa.float32()])
@pytest.mark.parametrize("nested", [False, True])
def test_array_to_tensor(dtype, nested):
    values = [True, None, False] if pa.types.is_boolean(dtype) else [1, None, 0]
    array = pa.array([values] if nested else values, type=pa.list_(dtype) if nested else dtype)
    tensor = pa_array_to_tensor(array)
    assert tensor.dtype == torch.float32
    expected = [[1, np.nan, 0]] if nested else [1, np.nan, 0]
    np.testing.assert_allclose(tensor.numpy(), expected, equal_nan=True)


def test_empty_and_unsupported_arrays():
    assert pa_array_to_tensor(pa.array([], type=pa.null())).numel() == 0
    assert pa_array_to_tensor(pa.array([[]], type=pa.list_(pa.null()))).shape == (1, 0)
    with pytest.raises(TypeError):
        pa_array_to_tensor(pa.array(["text"]))
    with pytest.raises(ValueError):
        pa_array_to_tensor(pa.array([[1], [2, 3]]))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_parquet_loading(tmp_path, num_workers):
    paths = []
    for part in range(3):
        path = tmp_path / f"part{part}.parquet"
        pq.write_table(pa.table({"id": range(part * 7, (part + 1) * 7)}), path)
        paths.append(str(path))
    dataset = ParquetIterableDataset(paths, columns=["id"], batch_size=4)
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    seen = []
    for batch in loader:
        assert isinstance(batch["id"], torch.Tensor)
        seen.extend(batch["id"].tolist())
    # Each input row must appear exactly once, including the last partial batch.
    assert sorted(seen) == list(range(21))
