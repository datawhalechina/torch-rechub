"""Full Avazu_x1 benchmark using the splits and feature definitions in run_avazu.py."""

import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from benchmarks.full_runs._evaluation import FullCTRTrainer
from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DCN, DeepFM, WideDeep

_FIELDS = [f"feat_{i}" for i in range(1, 23)]
_SPLITS = ("train", "valid", "test")
BATCH_SIZE = 8192


def _chunks(archive, split):
    with archive.open(split + ".csv") as stream:
        yield from pd.read_csv(stream, usecols=["label", *_FIELDS], dtype="int32", chunksize=150_000)


def _encode(archive, directory, device):
    # The original recipe fits vocabularies and numeric ranges across the supplied splits.
    seen = [set() for _ in _FIELDS]
    sizes = dict.fromkeys(_SPLITS, 0)
    for split in _SPLITS:
        for frame in _chunks(archive, split):
            for col, values in zip(_FIELDS, seen):
                values.update(frame[col].unique().tolist())
            sizes[split] += len(frame)
    values = [np.asarray(sorted(field), dtype="int32") for field in seen]

    # On smaller devices, read batches from a temporary disk array instead.
    gpu_bytes = sum(sizes.values()) * (len(_FIELDS) + 1) * 4
    on_gpu = device.type == "cuda" and torch.cuda.mem_get_info(device)[0] > gpu_bytes + 3_000_000_000
    matrices = {}
    for split in _SPLITS:
        shape = (sizes[split], len(_FIELDS) + 1)
        if on_gpu:
            matrix = torch.empty(shape, dtype=torch.int32, device=device)
        else:
            matrix = torch.from_file(str(directory / (split + ".bin")), shared=True, size=shape[0] * shape[1], dtype=torch.int32).view(shape)
        offset = 0
        for frame in _chunks(archive, split):
            block = np.empty((len(frame), shape[1]), dtype="int32")
            block[:, 0] = frame["label"]
            for index, (col, vocabulary) in enumerate(zip(_FIELDS, values), start=1):
                block[:, index] = np.searchsorted(vocabulary, frame[col])
            matrix[offset:offset + len(block)].copy_(torch.from_numpy(block))
            offset += len(block)
        matrices[split] = matrix
    return matrices, values


class _Batches:

    def __init__(self, matrix, dense_lookup, train):
        self.matrix = matrix
        self.rows = len(matrix)
        self.dense_lookup = dense_lookup
        self.train = train
        self.rng = torch.Generator().manual_seed(2022)

    def __len__(self):
        return (self.rows + BATCH_SIZE - 1) // BATCH_SIZE

    def __iter__(self):
        order = torch.randperm(self.rows, generator=self.rng).to(self.matrix.device) if self.train else None
        for begin in range(0, self.rows, BATCH_SIZE):
            batch = slice(begin, begin + BATCH_SIZE)
            fields = self.matrix[order[batch] if self.train else batch].T
            features = dict(zip(_FIELDS, fields[1:]))
            for name, lookup in zip(_FIELDS, self.dense_lookup):
                features[name] = lookup[features[name].long()]
            yield features, fields[0]


def run(data, output, models, device, epochs=None):
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        with zipfile.ZipFile(data) as archive:
            matrices, values = _encode(archive, Path(tmp), torch.device(device))
        try:
            return _train(matrices, values, output, models, device, epochs)
        finally:
            # Release file-backed tensors before Windows removes the temporary directory.
            matrices.clear()


def _train(matrices, values, output, models, device, epochs=None):
    storage_device = matrices["train"].device
    lookup = [torch.as_tensor(MinMaxScaler().fit_transform(v.reshape(-1, 1)).ravel(), device=storage_device) for v in values[:3]]
    results = {}
    for name in models:
        torch.manual_seed(2022)
        dense = [DenseFeature(col) for col in _FIELDS[:3]]
        sparse = [SparseFeature(col, len(v), embed_dim=16) for col, v in zip(_FIELDS, values)]
        # Match the example: the first three fields use their scaled values even for the sparse lookup.
        mlp = {"dims": [256, 128], "dropout": 0.2, "activation": "relu"}
        if name == "WideDeep":
            model = WideDeep(wide_features=dense, deep_features=sparse, mlp_params=mlp)
        elif name == "DeepFM":
            model = DeepFM(deep_features=dense, fm_features=sparse, mlp_params=mlp)
        elif name == "DCN":
            model = DCN(features=dense + sparse, n_cross_layers=3, mlp_params={"dims": [256, 128]})
        else:
            raise ValueError(f"Unknown Avazu model: {name}")
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        trainer = FullCTRTrainer(
            model,
            optimizer_params={
                "lr": 0.001,
                "weight_decay": 0.00001
            },
            earlystop_patience=epochs or 10,
            n_epoch=epochs or 2,
            device=device,
            model_path=str(directory),
        )
        train, val, test = [_Batches(matrices[s], lookup, train=(s == "train")) for s in _SPLITS]
        trainer.fit(train, val, log_interval=500)
        results[name] = trainer.test_metrics(trainer.model, test)
    return results
