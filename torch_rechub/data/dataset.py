"""Dataset implementations providing streaming, batch-wise data access for PyTorch."""

import typing as ty

import pyarrow.dataset as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

from torch_rechub.types import FilePath

from .convert import pa_array_to_tensor

# The default batch size when reading a Parquet dataset
_DEFAULT_BATCH_SIZE = 1024


class ParquetIterableDataset(IterableDataset):
    """Stream Parquet data as PyTorch tensors.

    Parameters
    ----------
    file_paths : list[FilePath]
        Paths to Parquet files.
    columns : list[str], optional
        Columns to select; if ``None``, read all columns.
    batch_size : int, default _DEFAULT_BATCH_SIZE
        Rows per streamed batch.

    Notes
    -----
    Reads lazily; no full Parquet load. Each worker gets a partition, builds its
    own PyArrow Dataset/Scanner, and yields dicts of column tensors batch by batch.

    Examples
    --------
    >>> ds = ParquetIterableDataset(
    ...     ["/data/train1.parquet", "/data/train2.parquet"],
    ...     columns=["x", "y", "label"],
    ...     batch_size=1024,
    ... )
    >>> loader = DataLoader(ds, batch_size=None)
    >>> for batch in loader:
    ...     x, y, label = batch["x"], batch["y"], batch["label"]
    ...     ...
    """

    def __init__(
        self,
        file_paths: ty.Sequence[FilePath],
        /,
        columns: ty.Optional[ty.Sequence[str]] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        """Initialize this instance."""
        self._file_paths = tuple(map(str, file_paths))
        self._columns = None if columns is None else tuple(columns)
        self._batch_size = batch_size

    def __iter__(self) -> ty.Iterator[dict[str, torch.Tensor]]:
        """Stream Parquet data as mapped PyTorch tensors.

        Builds a PyArrow Dataset from the current worker's file partition, then
        lazily scans selected columns. Each batch becomes a dict of Torch tensors.

        Returns
        -------
        Iterator[dict[str, torch.Tensor]]
            One converted batch at a time.
        """
        if not (partition := self._get_partition()):
            return

        # Build the dataset for the current worker.
        ds = pd.dataset(partition, format="parquet")

        # Create a scanner. This does not read data.
        columns = None if self._columns is None else list(self._columns)
        scanner = ds.scanner(columns=columns, batch_size=self._batch_size)

        for batch in scanner.to_batches():
            data_dict: dict[str, torch.Tensor] = {}
            for name, array in zip(batch.column_names, batch.columns):
                data_dict[name] = pa_array_to_tensor(array)
            yield data_dict

    # private interfaces

    def _get_partition(self) -> tuple[str, ...]:
        """Get file partition for the current worker.

        Splits file paths into contiguous partitions by number of workers and worker ID.
        In the main process (no worker info), returns all paths.

        Returns
        -------
        tuple[str, ...]
            Partition of file paths for this worker.
        """
        if (info := get_worker_info()) is None:
            return self._file_paths

        n = len(self._file_paths)
        per_worker = (n + info.num_workers - 1) // info.num_workers

        start = info.id * per_worker
        end = n if (end := start + per_worker) > n else end
        return self._file_paths[start:end]
