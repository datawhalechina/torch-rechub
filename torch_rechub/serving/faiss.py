"""FAISS-based vector index implementation for the retrieval stage."""

import contextlib
import typing as ty

import faiss
import torch

from torch_rechub.types import FilePath

from .base import BaseBuilder, BaseIndexer

# Type for indexing methods.
FaissIndexType = ty.Literal["Flat", "HNSW", "IVF"]

# Type for indexing metrics.
FaissMetric = ty.Union[faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2]

# Default indexing method.
_DEFAULT_FAISS_INDEX_TYPE: FaissIndexType = "Flat"

# Default indexing metric.
_DEFAULT_FAISS_METRIC: FaissMetric = faiss.METRIC_L2

# Default number of clusters to build an IVF index.
_DEFAULT_N_LISTS = 100

# Default max number of neighbors to build an HNSW index.
_DEFAULT_M = 32


class FaissBuilder(BaseBuilder):
    """Implement ``BaseBuilder`` for FAISS vector index construction."""

    def __init__(
        self,
        index_type: FaissIndexType = _DEFAULT_FAISS_INDEX_TYPE,
        metric: FaissMetric = _DEFAULT_FAISS_METRIC,
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
        index_type : FaissIndexType, optional
            The indexing index_type. Default to ``"Flat"``.
        metric : FaissMetric, optional
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
        self._index_type = index_type
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
            _build_index_type_dsl(
                self._index_type,
                m=self._m,
                nlists=self._nlists,
            ),
            self._metric,
        )

        if isinstance(index, faiss.IndexHNSW) and self._efSearch is not None:
            index.hnsw.efSearch = self._efSearch

        if isinstance(index, faiss.IndexIVF) and self._nprobe is not None:
            index.nprobe = self._nprobe

        index.train(embeddings)
        index.add(embeddings)

        try:
            yield FaissIndexer(index)
        finally:
            pass

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
            pass


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
        dists, ids = self._index.search(embeddings.numpy(), top_k)
        return torch.from_numpy(ids), torch.from_numpy(dists)

    def save(self, file_path: FilePath) -> None:
        """Adhere to ``BaseIndexer.save``."""
        faiss.write_index(self._index, str(file_path))


# helper functions


def _build_index_type_dsl(index_type: FaissIndexType, *, m: int, nlists: int) -> str:
    """Build the index_type DSL passed to ``faiss.index_factory``."""
    if index_type == "HNSW":
        return f"{index_type}{m},Flat"

    if index_type == "IVF":
        return f"{index_type}{nlists},Flat"

    return "Flat"
