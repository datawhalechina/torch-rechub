"""FAISS-based vector index implementation for the retrieval stage."""

import contextlib
import typing as ty

import faiss
import torch

from torch_rechub.types import FilePath

from .base import BaseBuilder, BaseIndexer

# Type for indexing methods.
_FaissMethod = ty.Literal["flat", "hnsw", "ivf"]

# Type for indexing metrics.
_FaissMetric = ty.Union[faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2]

# Default indexing method.
_DEFAULT_FAISS_METHOD: _FaissMethod = "flat"

# Default indexing metric.
_DEFAULT_FAISS_METRIC: _FaissMetric = faiss.METRIC_L2

# Default number of clusters to build an IVF index.
_DEFAULT_N_LISTS = 100

# Default max number of neighbors to build an HNSW index.
_DEFAULT_M = 32


class FaissBuilder(BaseBuilder):
    """Implement ``BaseBuilder`` for FAISS vector index construction."""

    def __init__(
        self,
        method: _FaissMethod = _DEFAULT_FAISS_METHOD,
        metric: _FaissMetric = _DEFAULT_FAISS_METRIC,
        *,
        m: int = _DEFAULT_M,
        nlists: int = _DEFAULT_N_LISTS,
        efSearch: ty.Optional[int] = None,
        nprobe: ty.Optional[int] = None,
    ) -> None:
        """
        Initialize a FAISS builder.

        Parameters
        ----------
        method : _FaissMethod, optional
            The indexing method. Default to ``"flat"``.
        metric : _FaissMetric, optional
            The indexing metric. Default to ``faiss.METRIC_L2``.
        m : int, optional
            Max number of neighbors to build an HNSW index.
        nlists : int, optional
            Number of clusters to build an IVF index.
        efSearch : int or None, optional
            Number of candidate nodes during an HNSW search.
        nprobe : int or None, optional
            Number of clusters during an IVF search.
        """
        self._method = method
        self._metric = metric

        self._m = m
        self._nlists = nlists
        self._efSearch = efSearch
        self._nprobe = nprobe

    @contextlib.contextmanager
    def from_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> ty.Generator["FaissIndexer",
                      None,
                      None]:
        """Adhere to ``BaseBuilder.from_embeddings``."""
        index: faiss.Index = faiss.index_factory(
            embeddings.shape[1],
            _build_method_dsl(self._method,
                              m=self._m,
                              nlists=self._nlists),
            self._metric,
        )

        if isinstance(index, faiss.IndexHNSW):
            index.hnsw.efSearch = self._efSearch or index.hnsw.efSearch

        if isinstance(index, faiss.IndexIVF):
            index.nprobe = self._nprobe or index.nprobe

        index.train(embeddings)
        index.add(embeddings)

        try:
            yield FaissIndexer(index)
        finally:
            self.dispose()

    @contextlib.contextmanager
    def from_index_file(
        self,
        index_file: FilePath,
    ) -> ty.Generator["FaissIndexer",
                      None,
                      None]:
        """Adhere to ``BaseBuilder.from_index_file``."""
        index = faiss.read_index(str(index_file))

        try:
            yield FaissIndexer(index)
        finally:
            self.dispose()

    def dispose(self) -> None:
        """Adhere to ``BaseBuilder.dispose``."""


class FaissIndexer(BaseIndexer):
    """FAISS-based implementation of ``BaseIndexer``."""

    def __init__(self, index: faiss.Index) -> None:
        """Initialize a FAISS indexer."""
        self._index = index

    def query(
        self,
        embeddings: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor,
               torch.Tensor]:
        """Adhere to ``BaseIndexer.query``."""
        dists, ids = self._index.search(embeddings, top_k)

        nn_ids = torch.from_numpy(ids)
        nn_distances = torch.from_numpy(dists)

        return nn_ids, nn_distances

    def save(self, file_path: FilePath) -> None:
        """Adhere to ``BaseIndexer.save``."""
        faiss.write_index(self._index, str(file_path))


# helper functions


def _build_method_dsl(method: _FaissMethod, *, m: int, nlists: int) -> str:
    """Build the method DSL passed to ``faiss.index_factory``."""
    if method == "hnsw":
        return f"HNSW{m},Flat"

    if method == "ivf":
        return f"IVF{nlists},Flat"

    return "Flat"
