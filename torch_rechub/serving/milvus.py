"""Milvus-based vector index implementation for the retrieval stage."""

import contextlib
import typing as ty
import uuid

import numpy as np
import pymilvus as milvus
import torch

from torch_rechub.types import FilePath

from .base import BaseBuilder, BaseIndexer

# Type for indexing methods.
_MilvusIndexType = ty.Literal["FLAT", "HNSW", "IVF_FLAT"]

# Type for indexing metrics.
_MilvusMetric = ty.Literal["COSINE", "IP", "L2"]

# Default indexing method.
_DEFAULT_MILVUS_INDEX_TYPE: _MilvusIndexType = "FLAT"

# Default indexing metric.
_DEFAULT_MILVUS_METRIC: _MilvusMetric = "COSINE"

# Default number of clusters to build an IVF index.
_DEFAULT_N_LIST = 128

# Default max number of neighbors to build an HNSW index.
_DEFAULT_M = 30

# Default name of Milvus database connection.
_DEFAULT_NAME = "rechub"

# Default host of Milvus instance.
_DEFAULT_HOST = "localhost"

# Default port of Milvus instance.
_DEFAULT_PORT = 19530

# Name of the embedding column in the Milvus database table.
_EMBEDDING_COLUMN = "embedding"


class MilvusBuilder(BaseBuilder):
    """Implement ``BaseBuilder`` for Milvus vector index construction."""

    def __init__(
        self,
        d: int,
        index_type: _MilvusIndexType = _DEFAULT_MILVUS_INDEX_TYPE,
        metric: _MilvusMetric = _DEFAULT_MILVUS_METRIC,
        *,
        m: int = _DEFAULT_M,
        nlist: int = _DEFAULT_N_LIST,
        ef: ty.Optional[int] = None,
        nprobe: ty.Optional[int] = None,
        name: str = _DEFAULT_NAME,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
    ) -> None:
        """
        Initialize a Milvus builder.

        Parameters
        ----------
        d : int
            The dimension of embeddings.
        index_type : ``"FLAT"``, ``"HNSW"``, or ``"IVF_FLAT"``, optional
            The indexing index_type. Default to ``"FLAT"``.
        metric : ``"COSINE"``, ``"IP"``, or ``"L2"``, optional
            The indexing metric. Default to ``"COSINE"``.
        m : int, optional
            Max number of neighbors to build an HNSW index.
        nlist : int, optional
            Number of clusters to build an IVF index.
        ef : int or None, optional
            Number of candidate nodes during an HNSW search.
        nprobe : int or None, optional
            Number of clusters during an IVF search.
        name : str, optional
            The name of connection. Each name corresponds to one connection.
        host : str, optional
            The host of Milvus instance. Default at "localhost".
        port : int, optional
            The port of Milvus instance. Default at 19530
        """
        self._d = d

        # connection parameters
        self._name = name
        self._host = host
        self._port = port

        bparams: dict[str, ty.Any] = {}
        qparams: dict[str, ty.Any] = {}

        if index_type == "HNSW":
            bparams.update(M=m)
            if ef is not None:
                qparams.update(ef=ef)

        if index_type == "IVF_FLAT":
            bparams.update(nlist=nlist)
            if nprobe is not None:
                qparams.update(nprobe=nprobe)

        self._build_params = dict(
            index_type=index_type,
            metric_type=metric,
            params=bparams,
        )

        self._query_params = dict(
            metric_type=metric,
            params=qparams,
        )

    @contextlib.contextmanager
    def from_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> ty.Generator["MilvusIndexer",
                      None,
                      None]:
        """Adhere to ``BaseBuilder.from_embeddings``."""
        milvus.connections.connect(self._name, host=self._host, port=self._port)
        collection = self._build_collection(embeddings)

        try:
            yield MilvusIndexer(collection, self._query_params)
        finally:
            collection.drop()
            milvus.connections.disconnect(self._name)

    @contextlib.contextmanager
    def from_index_file(
        self,
        index_file: FilePath,
    ) -> ty.Generator["MilvusIndexer",
                      None,
                      None]:
        """Adhere to ``BaseBuilder.from_index_file``."""
        raise NotImplementedError("Milvus does not support index files!")

    def _build_collection(self, embeddings: torch.Tensor) -> milvus.Collection:
        """Build a Milvus collection with the current connection."""
        fields = [
            milvus.FieldSchema(
                name="id",
                dtype=milvus.DataType.INT64,
                is_primary=True,
            ),
            milvus.FieldSchema(
                name=_EMBEDDING_COLUMN,
                dtype=milvus.DataType.FLOAT_VECTOR,
                dim=self._d,
            ),
        ]

        collection = milvus.Collection(
            name=f"{self._name}_{uuid.uuid4().hex}",
            schema=milvus.CollectionSchema(fields=fields),
            using=self._name,
        )

        n, _ = embeddings.shape
        collection.insert([np.arange(n, dtype=np.int64), embeddings.cpu().numpy()])
        collection.create_index(_EMBEDDING_COLUMN, index_params=self._build_params)
        collection.load()

        return collection


class MilvusIndexer(BaseIndexer):
    """Milvus-based implementation of ``BaseIndexer``."""

    def __init__(
        self,
        collection: milvus.Collection,
        query_params: dict[str,
                           ty.Any],
    ) -> None:
        """Initialize a Milvus indexer."""
        self._collection = collection
        self._query_params = query_params

    def query(
        self,
        embeddings: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor,
               torch.Tensor]:
        """Adhere to ``BaseIndexer.query``."""
        results = self._collection.search(
            data=embeddings.cpu().numpy(),
            anns_field=_EMBEDDING_COLUMN,
            param=self._query_params,
            limit=top_k,
        )

        n, _ = embeddings.shape
        nn_ids = np.zeros((n, top_k), dtype=np.int64)
        nn_distances = np.zeros((n, top_k), dtype=np.float32)

        for i, result in enumerate(results):
            nn_ids[i] = result.ids
            nn_distances[i] = result.distances

        return torch.from_numpy(nn_ids), torch.from_numpy(nn_distances)

    def save(self, file_path: FilePath) -> None:
        """Adhere to ``BaseIndexer.save``."""
        raise NotImplementedError("Milvus does not support index files!")
