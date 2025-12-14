import contextlib
import typing as ty

import torch

from torch_rechub.types import FilePath

from .annoy import AnnoyBuilder
from .base import BaseBuilder, BaseIndexer
from .faiss import FaissBuilder

# Type for supported retrieval models.
_RetrievalModel = ty.Literal["annoy", "faiss"]


@contextlib.contextmanager
def indexer_factory(
    model: _RetrievalModel,
    *,
    embeddings: ty.Optional[torch.Tensor] = None,
    index_file: ty.Optional[FilePath] = None,
    builder_config: ty.Optional[dict[str,
                                     ty.Any]] = None,
) -> ty.Generator[BaseIndexer,
                  None,
                  None]:
    """
    Context manager factory for creating a vector indexer.

    This function instantiates an index builder (ANNOY, FAISS, or Milvus) based on the
    specified retrieval model, and yields a ready-to-use ``BaseIndexer`` instance. The
    ndexer can be constructed either from in-memory embeddings or by loading a prebuilt
    index file.

    Exactly one of ``embeddings`` or ``index_file`` must be provided.

    Parameters
    ----------
    model : _RetrievalModel
        The retrieval backend to use. Determines which index builder to use.
    embeddings : torch.Tensor, optional
        A tensor of embeddings used to build a new index in memory.
        Must not be provided together with ``index_file``.
    index_file : FilePath, optional
        Path to a serialized index file to load.
        Must not be provided together with ``embeddings``.
    builder_config : dict[str, Any], optional
        Keyword arguments passed directly to the underlying index builder constructor.

    Yields
    ------
    BaseIndexer
        An initialized indexer instance, ready for similarity search.

    Raises
    ------
    NotImplementedError
        if the specified retrieval model is not supported.
    ValueError
        if neither or both of ``embeddings`` and ``index_file`` are provided.
    """
    builder_factory: ty.Optional[type[BaseBuilder]] = None

    if model == "annoy":
        builder_factory = AnnoyBuilder

    if model == "faiss":
        builder_factory = FaissBuilder

    if builder_factory is None:
        raise NotImplementedError(f"{model=} is not implemented yet!")

    if embeddings is None and index_file is None:
        raise ValueError("Either embeddings or index_file must be provided!")

    if embeddings is not None and index_file is not None:
        raise ValueError("Can only provide either embeddings or index_file!")

    builder_config = {} if builder_config is None else builder_config
    builder = builder_factory(**builder_config)

    if embeddings is not None:
        with builder.from_embeddings(embeddings) as indexer:
            yield indexer

    if index_file is not None:
        with builder.from_index_file(index_file) as indexer:
            yield indexer


__all__ = ["indexer_factory"]
