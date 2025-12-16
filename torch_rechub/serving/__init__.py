import typing as ty

from .annoy import AnnoyBuilder
from .base import BaseBuilder
from .faiss import FaissBuilder
from .milvus import MilvusBuilder

# Type for supported retrieval models.
_RetrievalModel = ty.Literal["annoy", "faiss", "milvus"]


def builder_factory(model: _RetrievalModel, **builder_config) -> BaseBuilder:
    """
    Factory function for creating a vector index builder.

    This function instantiates and returns a concrete implementation of ``BaseBuilder``
    based on the specified retrieval backend. The returned builder is responsible for
    constructing or loading the underlying ANN index via its own ``from_embeddings`` or
    ``from_index_file`` method.

    Parameters
    ----------
    model : "annoy", "faiss", or "milvus"
        The retrieval backend to use.
    **builder_config
        Keyword arguments passed directly to the selected builder constructor.

    Returns
    -------
    BaseBuilder
        A concrete builder instance corresponding to the specified retrieval backend.

    Raises
    ------
    NotImplementedError
        if the specified retrieval model is not supported.
    """
    if model == "annoy":
        return AnnoyBuilder(**builder_config)

    if model == "faiss":
        return FaissBuilder(**builder_config)

    if model == "milvus":
        return MilvusBuilder(**builder_config)

    raise NotImplementedError(f"{model=} is not implemented yet!")


__all__ = ["builder_factory"]
