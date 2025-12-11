import os.path as op
import tempfile
import typing as ty

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from torch_rechub.data.dataset import ParquetIterableDataset

_ParquetSpec = dict[str, ty.Iterable]
_ParquetCreator = ty.Callable[[list[_ParquetSpec]], list[str]]


@pytest.fixture(scope="function")
def parquet_files() -> ty.Generator[_ParquetCreator, None, None]:
    """Yield a helper that creates temporary Parquet files from specs.

    Parameters
    ----------
    None

    Yields
    ------
    Callable[[list[dict[str, ty.Iterable]]], list[str]]
        A function that takes a list of column specs and returns file paths.

    Examples
    --------
    >>> def test_something(parquet_files):
    ...     paths = parquet_files([
    ...         {"id": range(10)},        # file 1
    ...         {"id": range(10, 20)},    # file 2
    ...     ])
    ...     ...
    """
    with tempfile.TemporaryDirectory() as tmpdir:

        def _create(files_specs: list[_ParquetSpec]) -> list[str]:
            """Create temporary Parquet files given specs."""
            paths: list[str] = []
            for i, spec in enumerate(files_specs):
                table = pa.table({k: pa.array(v) for k, v in spec.items()})
                path = op.join(tmpdir, f"part{i}.parquet")
                pq.write_table(table, path)
                paths.append(path)
            return paths

        yield _create


def test_single_worker(parquet_files: _ParquetCreator) -> None:
    # Given
    specs = [{"id": range(0, 5)}, {"id": range(5, 10)}]
    paths = parquet_files(specs)

    ds = ParquetIterableDataset(paths, columns=["id"], batch_size=3)
    loader = DataLoader(ds, batch_size=None, num_workers=0)

    # When
    seen = []
    for batch in loader:
        assert isinstance(batch, dict)
        assert isinstance(batch["id"], torch.Tensor)
        seen.extend(batch["id"].tolist())

    # Then
    assert np.allclose(sorted(seen), np.arange(10))


def test_multiple_workers(parquet_files: _ParquetCreator) -> None:
    # Given
    specs = [{"id": range(0, 7)}, {"id": range(7, 14)}, {"id": range(14, 21)}]
    paths = parquet_files(specs)

    ds = ParquetIterableDataset(paths, columns=["id"], batch_size=4)
    loader = DataLoader(ds, batch_size=None, num_workers=3)

    # When
    seen = []
    for batch in loader:
        assert isinstance(batch, dict)
        assert isinstance(batch["id"], torch.Tensor)
        seen.extend(batch["id"].tolist())

    # Then
    assert np.allclose(sorted(seen), np.arange(21))
