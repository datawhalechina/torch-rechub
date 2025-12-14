import typing as ty

from .annoy import AnnoyBuilder
from .base import BaseBuilder
from .faiss import FaissBuilder

# Type for supported retrieval models.
_RetrievalModel = ty.Literal["annoy", "faiss"]


def builder_factory(model: _RetrievalModel, **builder_config) -> BaseBuilder:
    """
    Factory function for creating a vector index builder.

    This function instantiates and returns a concrete implementation of ``BaseBuilder``
    based on the specified retrieval backend. The returned builder is responsible for
    constructing or loading the underlying ANN index via its own ``from_embeddings`` or
    ``from_index_file`` method.

    Parameters
    ----------
    model : _RetrievalModel
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
    builder_factory: ty.Optional[type[BaseBuilder]] = None

    if model == "annoy":
        builder_factory = AnnoyBuilder

    if model == "faiss":
        builder_factory = FaissBuilder

    if builder_factory is None:
        raise NotImplementedError(f"{model=} is not implemented yet!")

    return builder_factory(**builder_config)


__all__ = ["builder_factory"]
