"""Full Criteo CSV benchmark with training-only feature statistics and bounded-memory batches."""

import gc
from pathlib import Path
from tempfile import TemporaryFile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.parquet as pq
import torch

from benchmarks.datasets import RankingDatasetBundle
from benchmarks.full_runs._evaluation import FullCTRTrainer
from benchmarks.models import build_ranking_model
from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DeepFM

DENSE = [f"I{i}" for i in range(1, 14)]
CATEGORICAL = [f"C{i}" for i in range(1, 27)]
SPARSE = CATEGORICAL + [name + "_cat" for name in DENSE]
SEED = 2022
BATCH_SIZE = 16384


def _prepare(data, parquet, dense_cache, sparse_cache):
    # Parquet lets us fit one vocabulary at a time, including on a 46M-row CSV.
    types = {"label": pa.int8(), **dict.fromkeys(DENSE, pa.float64()), **dict.fromkeys(CATEGORICAL, pa.string())}
    options = csv.ConvertOptions(column_types=types, include_columns=list(types), strings_can_be_null=True)
    with csv.open_csv(data, convert_options=options, read_options=csv.ReadOptions(block_size=16 << 20)) as reader:
        with pq.ParquetWriter(parquet, reader.schema) as writer:
            for batch in reader:
                writer.write_batch(batch)
    parquet.seek(0)
    with pq.ParquetFile(parquet) as source:
        n = source.metadata.num_rows
        train_end = int(0.7 * n)
        valid_end = train_end + int(0.1 * n)
        permutation = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).numpy()
        splits = [np.sort(rows) for rows in np.split(permutation, [train_end, valid_end])]
        del permutation
        train = np.zeros(n, dtype=bool)
        train[splits[0]] = True
        dense = np.memmap(dense_cache, mode="w+", dtype="float32", shape=(len(DENSE), n))
        sparse = np.memmap(sparse_cache, mode="w+", dtype="int32", shape=(len(SPARSE), n))
        vocab_sizes = {}

        def read(name):
            return source.read(columns=[name]).column(0).combine_chunks()

        def encode(name, column):
            vocabulary = pc.drop_null(pc.unique(pc.filter(column, train)))
            # Index zero represents missing and unseen values in validation and testing.
            indices = pc.index_in(column, value_set=vocabulary)
            sparse[SPARSE.index(name)] = pc.fill_null(pc.add(indices, 1), 0).to_numpy()
            vocab_sizes[name] = len(vocabulary) + 1

        for i, name in enumerate(DENSE):
            values = pc.fill_null(read(name), 0).to_numpy()
            training = values[train]
            low, high = training.min(), training.max()
            dense[i] = (values - low) / (high - low if high > low else 1)
            whole = values.astype(np.int64)
            bins = whole - 2
            mask = whole > 2
            bins[mask] = np.square(np.log(whole[mask])).astype(np.int64)
            encode(name + "_cat", pa.array(bins))
            del values, training, whole, bins, mask
        for name in CATEGORICAL:
            encode(name, read(name))
            print(f"{name}: {vocab_sizes[name]:,} embedding rows", flush=True)
        labels = read("label").to_numpy()
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Criteo expects binary click labels.")
        dense.flush()
        sparse.flush()
        return labels, splits, vocab_sizes


def _category_maps(sparse, train_rows, sizes):
    # Count training impressions only. Rare and unseen categories share index zero.
    maps = []
    compact_sizes = sizes.copy()
    for i, name in enumerate(CATEGORICAL):
        counts = np.bincount(sparse[i, train_rows], minlength=sizes[name])
        keep = counts >= 10
        keep[0] = False
        mapping = np.zeros(len(counts), dtype=np.int32)
        n_kept = int(keep.sum())
        mapping[keep] = np.arange(1, n_kept + 1, dtype=np.int32)
        maps.append(mapping)
        compact_sizes[name] = n_kept + 1
    return maps, compact_sizes


class _Batches:
    """Shuffle blocks and rows within each training block to keep disk reads local."""

    def __init__(self, dense, sparse, labels, rows, shuffle, device, category_maps=None):
        self.category_maps = category_maps
        self.dense, self.sparse, self.labels = dense, sparse, labels
        self.rows, self.shuffle, self.device = rows, shuffle, device
        self.generator = np.random.default_rng(SEED)

    def __len__(self):
        return (len(self.rows) + BATCH_SIZE - 1) // BATCH_SIZE

    def __iter__(self):
        block_size = 16 * BATCH_SIZE
        blocks = np.arange(0, len(self.rows), block_size)
        if self.shuffle:
            self.generator.shuffle(blocks)
        for offset in blocks:
            rows = self.rows[offset:offset + block_size]
            dense = torch.from_numpy(self.dense[:, rows].T.copy())
            sparse_block = self.sparse[:, rows].T.copy()
            for i, mapping in enumerate(self.category_maps or []):
                sparse_block[:, i] = mapping[sparse_block[:, i]]
            sparse = torch.from_numpy(sparse_block)
            labels = torch.from_numpy(self.labels[rows])
            order = self.generator.permutation(len(rows)) if self.shuffle else np.arange(len(rows))
            for start in range(0, len(rows), BATCH_SIZE):
                idx = order[start:start + BATCH_SIZE]
                dense_batch = dense[idx].to(self.device)
                sparse_batch = sparse[idx].to(self.device)
                x = dict(zip(DENSE, dense_batch.unbind(dim=1)))
                x.update(zip(SPARSE, sparse_batch.unbind(dim=1)))
                yield x, labels[idx]


def _build_model(name, sizes):
    dense = [DenseFeature(f) for f in DENSE]
    sparse = [SparseFeature(f, sizes[f], embed_dim=16, padding_idx=0) for f in SPARSE]
    if name == "DeepFM":
        # Share raw category embeddings across FM and DNN. Numeric buckets stay in FM.
        categories = [feature for feature in sparse if feature.name in CATEGORICAL]
        return DeepFM(deep_features=dense + categories, fm_features=sparse, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
    features = RankingDatasetBundle(x=None, y=None, dense_features=dense, sparse_features=sparse)
    return build_ranking_model({"name": name}, features)


def run(data: Path, output: Path, models: list[str], device: str, training_seed: int = SEED, epochs=None) -> dict[str, dict[str, float]]:
    """Train on the fixed split. training_seed controls initialization and dropout."""
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    # Windows can release these files even if a failed training step still holds a mapped array.
    with TemporaryFile(dir=output) as dense_cache, TemporaryFile(dir=output) as sparse_cache:
        with TemporaryFile(dir=output) as parquet:
            labels, splits, sizes = _prepare(data, parquet, dense_cache, sparse_cache)
        # Release the preprocessing buffers before mapping the encoded features for training.
        gc.collect()
        pa.default_memory_pool().release_unused()
        dense = np.memmap(dense_cache, mode="r", dtype="float32", shape=(len(DENSE), len(labels)))
        sparse = np.memmap(sparse_cache, mode="r", dtype="int32", shape=(len(SPARSE), len(labels)))
        print("train/validation/test rows:", [len(rows) for rows in splits], flush=True)
        print("positive labels:", [int(labels[rows].sum()) for rows in splits], flush=True)
        for name in models:
            torch.manual_seed(training_seed)
            category_maps, model_sizes = _category_maps(sparse, splits[0], sizes) if name == "DeepFM" else (None, sizes)
            model = _build_model(name, model_sizes)
            loaders = [_Batches(dense, sparse, labels, rows, i == 0, device, category_maps) for i, rows in enumerate(splits)]
            directory = output / name
            directory.mkdir(exist_ok=True)
            weight_decay = 1e-6 if name == "DeepFM" else 0.001
            optimizer = {"lr": 0.001, "weight_decay": weight_decay, "fused": torch.device(device).type == "cuda"}
            trainer = FullCTRTrainer(model, n_epoch=epochs or 2, earlystop_patience=epochs or 10, device=device, model_path=str(directory), optimizer_params=optimizer)
            trainer.fit(loaders[0], loaders[1], log_interval=100)
            results[name] = trainer.test_metrics(trainer.model, loaders[2])
            print(name, "test", results[name], flush=True)
            del trainer, model, loaders
        del dense, sparse
    return results
