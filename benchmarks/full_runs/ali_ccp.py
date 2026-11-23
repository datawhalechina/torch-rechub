"""Full-file Ali-CCP benchmark using the published train, validation and test splits."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmarks.full_runs._evaluation import FullCTRTrainer, FullMTLTrainer
from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.multi_task import MMOE
from torch_rechub.models.ranking import DeepFM

DENSE = ["D109_14", "D110_14", "D127_14", "D150_14", "D508", "D509", "D702", "D853"]
LABELS = ["purchase", "click"]
BATCH_SIZE = 8192
SEED = 2022


def _prepare(data, cache):
    columns = pd.read_csv(data / "ali_ccp_train.csv", nrows=0).columns
    sparse = [name for name in columns if name not in DENSE + LABELS]
    sizes = pd.Series(0, index=sparse, dtype="int64")
    splits = {}
    dtypes = {**dict.fromkeys(sparse, "int32"), **dict.fromkeys(DENSE, "float32"), **dict.fromkeys(LABELS, "uint8")}
    for split in ("train", "val", "test"):
        parts = []
        positives = np.zeros(2, dtype="int64")
        for index, frame in enumerate(pd.read_csv(data / f"ali_ccp_{split}.csv", dtype=dtypes, chunksize=16 * BATCH_SIZE)):
            # The supplied CSVs already contain encoded IDs and scaled numeric features.
            ids = frame[sparse]
            sizes = np.maximum(sizes, ids.max() + 1)
            labels = frame[LABELS].to_numpy()
            positives += labels.sum(axis=0, dtype="int64")
            path = cache / f"{split}_{index}.npz"
            np.savez(path, ids=ids.to_numpy(), dense=frame[DENSE].to_numpy(), labels=labels)
            parts.append((path, len(frame)))
        print(f"{split}: {sum(n for _, n in parts):,} rows, purchase/click positives {positives.tolist()}", flush=True)
        splits[split] = parts
    return sizes, splits


class _Batches:

    def __init__(self, parts, sparse, device, multitask, shuffle):
        self.parts = parts
        self.sparse = sparse
        self.device = device
        self.multitask = multitask
        self.shuffle = shuffle
        self.rng = np.random.default_rng(SEED)

    def __len__(self):
        return sum((n + BATCH_SIZE - 1) // BATCH_SIZE for _, n in self.parts)

    def __iter__(self):
        # One chunk at a time. Shuffle training chunks and their rows each epoch.
        indices = self.rng.permutation(len(self.parts)) if self.shuffle else range(len(self.parts))
        for index in indices:
            path, n = self.parts[index]
            with np.load(path) as block:
                ids = torch.as_tensor(block["ids"], device=self.device)
                dense = torch.as_tensor(block["dense"], device=self.device)
                labels = torch.as_tensor(block["labels"], device=self.device)
            order = torch.as_tensor(self.rng.permutation(n), device=self.device) if self.shuffle else None
            for start in range(0, n, BATCH_SIZE):
                rows = slice(start, start + BATCH_SIZE) if order is None else order[start:start + BATCH_SIZE]
                x = dict(zip(self.sparse, ids[rows].T.contiguous()))
                x.update(zip(DENSE, dense[rows].T.contiguous()))
                yield x, labels[rows] if self.multitask else labels[rows, 1]


def run(data: Path, output: Path, models: list[str], device: str, epochs=None):
    """Run DeepFM (click AUC) and/or MMOE (purchase and click AUC)."""
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(dir=output, prefix="ali-ccp-batches-") as cache:
        sizes, splits = _prepare(data, Path(cache))
        for name in models:
            torch.manual_seed(SEED)
            directory = output / name
            directory.mkdir(exist_ok=True)
            dense = [DenseFeature(col) for col in DENSE]
            multitask = name == "MMOE"
            sparse = [SparseFeature(col, int(size), embed_dim=4 if multitask else 16) for col, size in sizes.items()]
            if name == "DeepFM":
                model = DeepFM(deep_features=dense, fm_features=sparse, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
                trainer = FullCTRTrainer(model, n_epoch=epochs or 2, earlystop_patience=epochs or 10, optimizer_params={"lr": 1e-3, "weight_decay": 1e-3}, device=device, model_path=str(directory))
            elif name == "MMOE":
                tasks = ["classification", "classification"]
                model = MMOE(sparse + dense, tasks, 8, expert_params={"dims": [16]}, tower_params_list=[{"dims": [8]}, {"dims": [8]}])
                trainer = FullMTLTrainer(model, task_types=tasks, n_epoch=epochs or 1, earlystop_patience=epochs or 10, optimizer_params={"lr": 1e-3, "weight_decay": 1e-4}, device=device, model_path=str(directory))
            else:
                raise ValueError(f"Unknown Ali-CCP model {name!r}. Choose DeepFM or MMOE.")
            train, val, test = [_Batches(splits[split], sizes.index, device, multitask, split == "train") for split in ("train", "val", "test")]
            trainer.fit(train, val, log_interval=500)
            auc = trainer.evaluate(trainer.model, test)
            results[name] = {"AUC[purchase]": auc[0], "AUC[click]": auc[1]} if multitask else {"AUC[click]": auc}
            print(f"{name} test: {results[name]}", flush=True)
            del model, trainer
    return results
