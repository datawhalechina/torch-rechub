"""ANNOY-based vector index implementation for the retrieval stage."""

import contextlib
import typing as ty

import annoy
import torch

from torch_rechub.types import FilePath

from .base import BaseBuilder, BaseIndexer

# Type for distance metrics for the ANNOY index.
AnnoyMetric = ty.Literal["angular", "euclidean", "dot"]

# Default distance metric used by ANNOY.
_DEFAULT_METRIC: AnnoyMetric = "angular"

# Default number of trees to build in the ANNOY index.
_DEFAULT_N_TREES = 10

# Default number of worker threads for building the ANNOY index.
_DEFAULT_THREADS = -1

# Default number of nodes to inspect during an ANNOY search.
_DEFAULT_SEARCHK = -1


class AnnoyBuilder(BaseBuilder):
    """ANNOY-based implementation of ``BaseBuilder``."""

    def __init__(
        self,
        d: int,
        metric: AnnoyMetric = _DEFAULT_METRIC,
        *,
        n_trees: int = _DEFAULT_N_TREES,
        threads: int = _DEFAULT_THREADS,
        searchk: int = _DEFAULT_SEARCHK,
    ) -> None:
        """
        Initialize a ANNOY builder.

        Parameters
        ----------
        d : int
            The dimension of embeddings.
        metric : AnnoyMetric, optional
            The indexing metric. Default to ``"angular"``.
        n_trees : int, optional
            Number of trees to build an ANNOY index.
        threads : int, optional
            Number of worker threads to build an ANNOY index.
        searchk : int, optional
            Number of nodes to inspect during an ANNOY search.
        """
        self._d = d
        self._metric = metric

        self._n_trees = n_trees
        self._threads = threads
        self._searchk = searchk

    @contextlib.contextmanager
    def from_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> ty.Generator["AnnoyIndexer",
                      None,
                      None]:
        """Adhere to ``BaseBuilder.from_embeddings``."""
        index = annoy.AnnoyIndex(self._d, metric=self._metric)

        for idx, emb in enumerate(embeddings):
            index.add_item(idx, emb)

        index.build(self._n_trees, n_jobs=self._threads)

        try:
            yield AnnoyIndexer(index, self._searchk)
        finally:
            index.unload()

    @contextlib.contextmanager
    def from_index_file(
        self,
        index_file: FilePath,
    ) -> ty.Generator["AnnoyIndexer",
                      None,
                      None]:
        """Adhere to ``BaseBuilder.from_index_file``."""
        index = annoy.AnnoyIndex(self._d, metric=self._metric)
        index.load(str(index_file))

        try:
            yield AnnoyIndexer(index, searchk=self._searchk)
        finally:
            index.unload()


class AnnoyIndexer(BaseIndexer):
    """ANNOY-based implementation of ``BaseIndexer``."""

    def __init__(self, index: annoy.AnnoyIndex, searchk: int) -> None:
        """Initialize a ANNOY indexer."""
        self._index = index
        self._searchk = searchk

    def query(
        self,
        embeddings: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor,
               torch.Tensor]:
        """Adhere to ``BaseIndexer.query``."""
        n, _ = embeddings.shape

        nn_ids = torch.zeros(n, top_k, dtype=torch.int64)
        nn_distances = torch.zeros(n, top_k, dtype=torch.float32)

        for idx, emb in enumerate(embeddings):
            ids, distances = self._index.get_nns_by_vector(
                emb.numpy(),
                top_k,
                search_k=self._searchk,
                include_distances=True,
            )

            nn_ids[idx] = torch.tensor(ids, dtype=torch.int64)
            nn_distances[idx] = torch.tensor(distances, dtype=torch.float32)

        return nn_ids, nn_distances

    def save(self, file_path: FilePath) -> None:
        """Adhere to ``BaseIndexer.save``."""
        self._index.save(str(file_path))
