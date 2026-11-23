"""AliExpress-US full-exposure AUC, with validation search IDs held out from train.csv."""

import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import torch

from benchmarks.full_runs._evaluation import FullMTLTrainer
from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.multi_task import AITM, SharedBottom

_CATEGORICAL = [f"categorical_{i}" for i in range(1, 17)]
_NUMERICAL = [f"numerical_{i}" for i in range(1, 64)]
_TARGETS = ["conversion", "click"]
_COLUMNS = (_CATEGORICAL, _NUMERICAL, _TARGETS)
_DTYPES = ("int32", "float32", "uint8")
BATCH_SIZE = 10240


def _validation_mask(search_ids):
    # Keep all impressions for a search in the same split, independent of input order.
    hashed = search_ids.astype("uint64") + np.uint64(2022)
    hashed = (hashed ^ (hashed >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
    hashed = (hashed ^ (hashed >> 27)) * np.uint64(0x94d049bb133111eb)
    return (hashed ^ (hashed >> 31)) % 10 == 0


def _prepare(archive):
    groups = {"train": [], "val": [], "test": []}
    maximum = np.zeros(len(_CATEGORICAL), dtype=np.int64)
    types = {name: pa.from_numpy_dtype(dtype) for fields, dtype in zip(_COLUMNS, _DTYPES) for name in fields}
    types["search_id"] = pa.int64()
    options = pcsv.ConvertOptions(include_columns=list(types), column_types=types)
    # Use the same block size as the full-data run because training shuffles blocks.
    read_options = pcsv.ReadOptions(block_size=24 * 1024**2, use_threads=False)
    for split in ("test", "train"):
        member = f"AliExpress_US/{split}.csv"
        if member not in archive.namelist():
            member = f"{split}.csv"
        with archive.open(member) as stream:
            reader = pcsv.open_csv(stream, read_options=read_options, convert_options=options)
            for block in reader:
                arrays = [np.stack([block.column(col).to_numpy() for col in fields]) for fields in _COLUMNS]
                np.maximum(maximum, arrays[0].max(axis=1), out=maximum)
                if split == "test":
                    parts = [("test", slice(None))]
                else:
                    valid = _validation_mask(block.column("search_id").to_numpy())
                    parts = [("train", ~valid), ("val", valid)]
                for name, mask in parts:
                    chunk = [np.ascontiguousarray(a[:, mask]) for a in arrays]
                    rows = chunk[0].shape[1]
                    if rows:
                        groups[name].append((rows, [pa.compress(a, codec="zstd") for a in chunk]))
    if not all(groups.values()):
        raise ValueError("Need train, validation and test examples; check the number of distinct search IDs")
    return groups, maximum + 1


def _decompress(group):
    rows, buffers = group
    arrays = []
    for fields, dtype, buf in zip(_COLUMNS, _DTYPES, buffers):
        nbytes = len(fields) * rows * np.dtype(dtype).itemsize
        data = pa.decompress(buf, nbytes, codec="zstd")
        arrays.append(np.frombuffer(data, dtype=dtype).reshape(len(fields), rows))
    return arrays


class _Batches:

    def __init__(self, groups, train):
        self.groups = groups
        self.rows = sum(rows for rows, _ in groups)
        self.train = train
        self.epoch = 0

    def __iter__(self):
        order = np.arange(len(self.groups))
        rng = np.random.default_rng(2022 + self.epoch)
        self.epoch += 1
        # Shuffle both the compressed block order and the impressions in each group of 12.
        if self.train:
            rng.shuffle(order)
        group_size = 12 if self.train else 1
        for start in range(0, len(order), group_size):
            blocks = [_decompress(self.groups[i]) for i in order[start:start + group_size]]
            columns = [np.concatenate(parts, axis=1) for parts in zip(*blocks)] if len(blocks) > 1 else blocks[0]
            del blocks
            rows = columns[0].shape[1]
            indices = rng.permutation(rows) if self.train else None
            for begin in range(0, rows, BATCH_SIZE):
                batch = slice(begin, begin + BATCH_SIZE)
                take = indices[batch] if self.train else batch
                categories, numbers, labels = [torch.from_numpy(np.array(a[:, take], copy=True, order="C")) for a in columns]
                features = dict(zip(_CATEGORICAL, categories))
                features.update(zip(_NUMERICAL, numbers))
                yield features, labels.T


def run(data, output, models, device, epochs=None):
    with zipfile.ZipFile(data) as archive:
        groups, sizes = _prepare(archive)
    results = {}
    for name in models:
        torch.manual_seed(2022)
        features = [SparseFeature(field, int(size), embed_dim=5) for field, size in zip(_CATEGORICAL, sizes)]
        features.extend(DenseFeature(field) for field in _NUMERICAL)
        types = ["classification", "classification"]
        towers = [{"dims": [8]}, {"dims": [8]}]
        if name == "SharedBottom":
            model = SharedBottom(features, types, bottom_params={"dims": [192, 96, 48]}, tower_params_list=towers)
        elif name == "AITM":
            model = AITM(features, 2, bottom_params={"dims": [128, 64, 32]}, tower_params_list=towers)
        else:
            raise ValueError(f"Unknown AliExpress-US model: {name}")

        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        trainer = FullMTLTrainer(
            model,
            task_types=types,
            optimizer_params={
                "lr": 0.001,
                "weight_decay": 0.00001
            },
            n_epoch=epochs or 2,
            earlystop_patience=epochs or 1,
            device=device,
            model_path=str(directory),
        )
        train, val, test = [_Batches(groups[s], train=(s == "train")) for s in ("train", "val", "test")]
        trainer.fit(train, val, seed=2022, log_interval=100)
        scores = trainer.evaluate(trainer.model, test)
        results[name] = {f"AUC[{target}]": score for target, score in zip(_TARGETS, scores)}
    return results
