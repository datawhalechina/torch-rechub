"""Dataset implementations providing streaming, batch-wise data access for PyTorch."""

from os import PathLike
from typing import Iterator, Optional, Sequence, Union

from pyarrow.dataset import dataset
from torch import Tensor, from_numpy
from torch.utils.data import IterableDataset, get_worker_info

# The default batch size when reading a Parquet dataset
DEFAULT_BATCH_SIZE = 1024


class ParquetIterableDataset(IterableDataset):
    """
    IterableDataset that streams data from one or more Parquet files.

    Parameters
    ----------
    file_paths : list[Union[str, PathLike]]
        Paths to Parquet files.
    columns : list[str], optional
        Column names to select. If ``None``, all columns are read.
    batch_size : int, default DEFAULT_BATCH_SIZE
        Number of rows per streamed batch.

    Notes
    -----
    This dataset reads data lazily and never loads the entire Parquet dataset to memory.
    The current worker receives a partition of ``file_paths`` and builds its own PyArrow
    Dataset and Scanner. Iteration yields dictionaries mapping column names to PyTorch
    tensors created via NumPy, one batch at a time.

    Examples
    --------
    >>> ds = ParquetIterableDataset(
    ...     ["/data/train1.parquet", "/data/train2.parquet"],
    ...     columns=["x", "y", "label"],
    ...     batch_size=1024,
    ... )
    >>> loader = DataLoader(ds, batch_size=None)
    >>> # Now iterate over batches.
    >>> for batch in loader:
    ...     x, y, label = batch["x"], batch["y"], batch["label"]
    ...     # Do some work.
    ...     ...
    """

    def __init__(self, file_paths: Sequence[Union[str, PathLike]], /, columns: Optional[Sequence[str]] = None, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """Initialize this instance."""
        self._file_paths = tuple(map(str, file_paths))
        self._columns = None if columns is None else tuple(columns)
        self._batch_size = batch_size

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """
        Stream Parquet data as mapped PyTorch tensors.

        Build a PyArrow Dataset from the current worker's assigned file partition, then
        create a Scanner to lazily read batches of the selected columns. Each batch is
        converted to a dict mapping column names to PyTorch tensors (via NumPy).

        Returns
        -------
        Iterator[dict[str, Tensor]]
            An iterator that yields one converted batch at a time.
        """
        if not (partition := self._get_partition()):
            return

        # Build the dataset for the current worker.
        ds = dataset(partition, format="parquet")

        # Create a scanner. This does not read data.
        columns = None if self._columns is None else list(self._columns)
        scanner = ds.scanner(columns=columns, batch_size=self._batch_size)

        for batch in scanner.to_batches():
            yield {name: from_numpy(array.to_numpy(zero_copy_only=False)) for name, array in zip(batch.column_names, batch.columns)}

    # private interfaces

    def _get_partition(self) -> tuple[str, ...]:
        """
        Get the partition of file paths for the current worker.

        This method splits the full list of file paths into contiguous partitions with
        a nearly equal size by the total number of workers and the current worker ID.

        If running in the main process (i.e., no worker information is available), the
        entire list of file paths is returned.

        Returns
        -------
        tuple[str, ...]
            The partition of file paths for the current worker.
        """
        if (info := get_worker_info()) is None:
            return self._file_paths

        n = len(self._file_paths)
        per_worker = (n + info.num_workers - 1) // info.num_workers

        start = info.id * per_worker
        end = n if (end := start + per_worker) > n else end
        return self._file_paths[start:end]
