import os

import pytest
import torch

from torch_rechub.serving import builder_factory


@pytest.mark.parametrize(
    "backend, options",
    [
        ("annoy",
         {
             "d": 8,
             "metric": "angular",
             "n_trees": 2
         }),
        ("faiss",
         {
             "index_type": "Flat",
             "metric": "IP"
         }),
        ("faiss",
         {
             "index_type": "HNSW",
             "metric": "L2",
             "m": 16
         }),
        ("faiss",
         {
             "index_type": "IVF",
             "metric": "IP",
             "nlists": 2
         }),
    ]
)
def test_local_index(tmp_path, backend, options):
    pytest.importorskip(backend)
    torch.manual_seed(2026)
    items = torch.randn(100, 8)
    users = torch.randn(3, 8)
    path = tmp_path / "items.index"
    builder = builder_factory(backend, **options)
    with builder.from_embeddings(items) as index:
        ids, distances = index.query(users, 5)
        assert ids.shape == distances.shape == (3, 5)
        assert ids.dtype == torch.int64
        assert torch.isfinite(distances).all()
        assert ((ids >= 0) & (ids < len(items))).all()
        index.save(path)
    with builder.from_index_file(path) as index:
        restored_ids, restored_distances = index.query(users, 5)
    torch.testing.assert_close(restored_ids, ids)
    torch.testing.assert_close(restored_distances, distances)


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_MILVUS_TESTS") != "1", reason="Start Milvus and set RUN_MILVUS_TESTS=1 to run")
@pytest.mark.parametrize("index_type, options", [("FLAT", {}), ("HNSW", {"m": 16}), ("IVF_FLAT", {"nlist": 2})])
def test_milvus_index(index_type, options):
    pytest.importorskip("pymilvus")
    items = torch.randn(100, 8)
    builder = builder_factory("milvus", d=8, index_type=index_type, metric="IP", **options)
    with builder.from_embeddings(items) as index:
        ids, distances = index.query(items[:3], 5)
        assert ids.shape == distances.shape == (3, 5)
        assert torch.isfinite(distances).all()
        assert ((ids >= 0) & (ids < len(items))).all()
