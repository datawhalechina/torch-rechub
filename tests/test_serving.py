import os
import pathlib
import typing as ty

import pytest
import torch

from torch_rechub.serving import builder_factory

# Skip the Milvus tests by the envvar boolean flag.
SKIP_MILVUS_TESTS = os.getenv("SKIP_MILVUS_TESTS") == "1"


@pytest.mark.parametrize("metric", ["angular", "euclidean", "dot"])
@pytest.mark.parametrize("n_trees", [1, 10])
@pytest.mark.parametrize("threads", [-1, 2])
@pytest.mark.parametrize("searchk", [-1, 100])
def test_annoy_indexing(
    metric: str,
    n_trees: int,
    threads: int,
    searchk: int,
    tmp_path: pathlib.Path,
) -> None:
    # Given
    n = 100
    d = 5
    top_k = 5
    index_file = tmp_path / "annoy.index"

    item_embeddings = torch.randn(n, d, dtype=torch.float32)
    user_embeddings = torch.randn(n, d, dtype=torch.float32)

    # When
    builder = builder_factory(
        "annoy",
        d=d,
        metric=metric,
        n_trees=n_trees,
        threads=threads,
        searchk=searchk,
    )

    with builder.from_embeddings(item_embeddings) as indexer:
        ids, distances = indexer.query(user_embeddings, top_k)
        indexer.save(tmp_path / index_file)

    # Then
    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (n, top_k)
    assert ids.dtype == torch.int64

    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (n, top_k)
    assert distances.dtype == torch.float32

    # When
    with builder.from_index_file(index_file) as indexer:
        ids, distances = indexer.query(user_embeddings, top_k)

    # Then
    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (n, top_k)
    assert ids.dtype == torch.int64

    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (n, top_k)
    assert distances.dtype == torch.float32


@pytest.mark.parametrize("metric", ["IP", "L2"])
def test_faiss_flat_indexing(metric: str, tmp_path: pathlib.Path) -> None:
    # Given
    n = 100
    d = 5
    top_k = 5
    index_file = tmp_path / "faiss.index"

    item_embeddings = torch.randn(n, d, dtype=torch.float32)
    user_embeddings = torch.randn(n, d, dtype=torch.float32)

    # When
    builder = builder_factory("faiss", index_type="Flat", metric=metric)

    with builder.from_embeddings(item_embeddings) as indexer:
        ids, distances = indexer.query(user_embeddings, top_k)
        indexer.save(tmp_path / index_file)

    # Then
    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (n, top_k)
    assert ids.dtype == torch.int64

    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (n, top_k)
    assert distances.dtype == torch.float32

    # When
    with builder.from_index_file(index_file) as indexer:
        ids, distances = indexer.query(user_embeddings, top_k)

    # Then
    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (n, top_k)
    assert ids.dtype == torch.int64

    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (n, top_k)
    assert distances.dtype == torch.float32


@pytest.mark.parametrize("metric", ["IP", "L2"])
@pytest.mark.parametrize("m", [16, 32])
@pytest.mark.parametrize("efSearch", [None, 50])
def test_faiss_hnsw_indexing(
    metric: str,
    m: int,
    efSearch: ty.Optional[int],
    tmp_path: pathlib.Path,
) -> None:
    # Given
    n = 100
    d = 5
    top_k = 5
    index_file = tmp_path / "faiss.index"

    item_embeddings = torch.randn(n, d, dtype=torch.float32)
    user_embeddings = torch.randn(n, d, dtype=torch.float32)

    # When
    builder = builder_factory(
        "faiss",
        index_type="HNSW",
        metric=metric,
        m=m,
        efSearch=efSearch,
    )

    with builder.from_embeddings(item_embeddings) as indexer:
        ids, distances = indexer.query(user_embeddings, top_k)
        indexer.save(tmp_path / index_file)

    # Then
    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (n, top_k)
    assert ids.dtype == torch.int64

    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (n, top_k)
    assert distances.dtype == torch.float32

    # When
    with builder.from_index_file(index_file) as indexer:
        ids, distances = indexer.query(user_embeddings, top_k)

    # Then
    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (n, top_k)
    assert ids.dtype == torch.int64

    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (n, top_k)
    assert distances.dtype == torch.float32


@pytest.mark.parametrize("metric", ["IP", "L2"])
@pytest.mark.parametrize("nlists", [1, 2])
@pytest.mark.parametrize("nprobe", [None, 5])
def test_faiss_ivf_indexing(
    metric: str,
    nlists: int,
    nprobe: ty.Optional[int],
    tmp_path: pathlib.Path,
) -> None:
    # Given
    n = 100
    d = 5
    top_k = 5
    index_file = tmp_path / "faiss.index"

    item_embeddings = torch.randn(n, d, dtype=torch.float32)
    user_embeddings = torch.randn(n, d, dtype=torch.float32)

    # When
    builder = builder_factory(
        "faiss",
        index_type="IVF",
        metric=metric,
        nlists=nlists,
        nprobe=nprobe,
    )

    with builder.from_embeddings(item_embeddings) as indexer:
        ids, distances = indexer.query(user_embeddings, top_k)
        indexer.save(tmp_path / index_file)

    # Then
    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (n, top_k)
    assert ids.dtype == torch.int64

    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (n, top_k)
    assert distances.dtype == torch.float32

    # When
    with builder.from_index_file(index_file) as indexer:
        ids, distances = indexer.query(user_embeddings, top_k)

    # Then
    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (n, top_k)
    assert ids.dtype == torch.int64

    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (n, top_k)
    assert distances.dtype == torch.float32


@pytest.mark.skipif(SKIP_MILVUS_TESTS, reason="Facilitate CI builds.")
@pytest.mark.parametrize("metric", ["COSINE", "IP", "L2"])
def test_milvus_flat_indexing(metric: str) -> None:
    # Given
    n = 100
    d = 5
    top_k = 5

    item_embeddings = torch.randn(n, d, dtype=torch.float32)
    user_embeddings = torch.randn(n, d, dtype=torch.float32)

    # When
    builder = builder_factory("milvus", d=d, index_type="FLAT", metric=metric)

    with builder.from_embeddings(item_embeddings) as indexer:
        ids, distances = indexer.query(user_embeddings, top_k)

    # Then
    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (n, top_k)
    assert ids.dtype == torch.int64

    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (n, top_k)
    assert distances.dtype == torch.float32


@pytest.mark.skipif(SKIP_MILVUS_TESTS, reason="Facilitate CI builds.")
@pytest.mark.parametrize("metric", ["COSINE", "IP", "L2"])
@pytest.mark.parametrize("m", [16, 32])
@pytest.mark.parametrize("ef", [None, 50])
def test_milvus_hnsw_indexing(
    metric: str,
    m: int,
    ef: ty.Optional[int],
) -> None:
    # Given
    n = 100
    d = 5
    top_k = 5

    item_embeddings = torch.randn(n, d, dtype=torch.float32)
    user_embeddings = torch.randn(n, d, dtype=torch.float32)

    # When
    builder = builder_factory(
        "milvus",
        d=d,
        index_type="HNSW",
        metric=metric,
        m=m,
        ef=ef,
    )

    with builder.from_embeddings(item_embeddings) as indexer:
        ids, distances = indexer.query(user_embeddings, top_k)

    # Then
    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (n, top_k)
    assert ids.dtype == torch.int64

    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (n, top_k)
    assert distances.dtype == torch.float32


@pytest.mark.skipif(SKIP_MILVUS_TESTS, reason="Facilitate CI builds.")
@pytest.mark.parametrize("metric", ["COSINE", "IP", "L2"])
@pytest.mark.parametrize("nlist", [1, 2])
@pytest.mark.parametrize("nprobe", [None, 5])
def test_milvus_ivf_indexing(
    metric: str,
    nlist: int,
    nprobe: ty.Optional[int],
) -> None:
    # Given
    n = 100
    d = 5
    top_k = 5

    item_embeddings = torch.randn(n, d, dtype=torch.float32)
    user_embeddings = torch.randn(n, d, dtype=torch.float32)

    # When
    builder = builder_factory(
        "milvus",
        d=d,
        index_type="IVF_FLAT",
        metric=metric,
        nlist=nlist,
        nprobe=nprobe,
    )

    with builder.from_embeddings(item_embeddings) as indexer:
        ids, distances = indexer.query(user_embeddings, top_k)

    # Then
    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (n, top_k)
    assert ids.dtype == torch.int64

    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (n, top_k)
    assert distances.dtype == torch.float32
