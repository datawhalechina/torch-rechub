"""Base abstraction for vector indexers used in the retrieval stage."""

import abc
import typing as ty

import torch

from torch_rechub.types import FilePath


class BaseBuilder(abc.ABC):
    """
    Abstract base class for vector index construction.

    A builder owns all build-time configuration and produces a ``BaseIndexer`` through a
    context-managed build operation.

    Examples
    --------
    >>> builder = BaseBuilder(...)
    >>> embeddings = torch.randn(1000, 128)
    >>> with builder.from_embeddings(embeddings) as indexer:
    ...     ids, scores = indexer.query(embeddings[:2], top_k=5)
    ...     indexer.save("index.bin")
    >>> with builder.from_index_file("index.bin") as indexer:
    ...     ids, scores = indexer.query(embeddings[:2], top_k=5)
    """

    @abc.abstractmethod
    def from_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> ty.ContextManager["BaseIndexer"]:
        """
        Build a vector index from the embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            A 2D tensor (n, d) containing embedding vectors to build a new index.

        Returns
        -------
        ContextManager[BaseIndexer]
            A context manager that yields a fully initialized ``BaseIndexer``.
        """

    @abc.abstractmethod
    def from_index_file(
        self,
        index_file: FilePath,
    ) -> ty.ContextManager["BaseIndexer"]:
        """
        Build a vector index from the index file.

        Parameters
        ----------
        index_file : FilePath
            Path to a serialized index on disk to be loaded.

        Returns
        -------
        ContextManager[BaseIndexer]
            A context manager that yields a fully initialized ``BaseIndexer``.
        """


class BaseIndexer(abc.ABC):
    """Abstract base class for vector indexers in the retrieval stage."""

    @abc.abstractmethod
    def query(
        self,
        embeddings: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor,
               torch.Tensor]:
        """
        Query the vector index.

        Parameters
        ----------
        embeddings : torch.Tensor
            A 2D tensor (n, d) containing embedding vectors to query the index.
        top_k : int
            The number of nearest items to retrieve for each vector.

        Returns
        -------
        torch.Tensor
            A 2D tensor of shape (n, top_k), containing the retrieved nearest neighbor
            IDs for each vector, ordered by descending relevance.
        torch.Tensor
            A 2D tensor of shape (n, top_k), containing the relevance distances of the
            nearest neighbors for each vector.
        """

    @abc.abstractmethod
    def save(self, file_path: FilePath) -> None:
        """
        Persist the index to local disk.

        Parameters
        ----------
        file_path : FilePath
            Destination path where the index will be saved.
        """
