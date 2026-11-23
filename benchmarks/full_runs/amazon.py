"""DIN with one uniform unseen item per positive and the last two interactions held out."""

import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models.ranking import DIN
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import neg_sample

HISTORY_LENGTH = 50


def _events(frame):
    for col in frame:
        frame[col] = (LabelEncoder().fit_transform(frame[col]) + 1).astype("int32")
    item_category = np.zeros(frame.item_id.max() + 1, dtype="int32")
    item_category[frame.item_id] = frame.cate_id

    frame = frame.sort_values("time").sort_values("user_id", kind="stable")
    user = frame.user_id.to_numpy()
    item = frame.item_id.to_numpy()
    first, counts = np.unique(user, return_index=True, return_counts=True)[1:]
    start = np.repeat(first, counts)
    last = start + np.repeat(counts, counts) - 1
    rows = np.arange(len(frame))
    negatives = np.empty_like(item)
    n_items = len(item_category) - 1
    for begin, count in zip(first, counts):
        history = item[begin:begin + count].tolist()
        negatives[begin:begin + count] = [neg_sample(history, n_items) for _ in history]

    pairs = []
    for mask in (rows < last - 1, rows == last - 1, rows == last):
        # Each target row supplies a positive (even index) and a negative (odd index).
        indices = (2 * rows[mask & (rows > start), None] + [0, 1]).ravel()
        random.shuffle(indices)
        pairs.append(indices)
    return (user, item, frame.cate_id.to_numpy(), item_category, start, negatives), pairs


class _HistoryBatches(Dataset):
    """Build just the current batch of histories from the underlying event arrays."""

    def __init__(self, events, indices):
        self.events = events
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, keys):
        users, items, categories, item_category, start, negatives = self.events
        paired = self.indices[keys]
        target = paired // 2
        positive = paired % 2 == 0
        item = np.where(positive, items[target], negatives[target])
        positions = np.maximum(start[target], target - HISTORY_LENGTH)[:, None] + np.arange(HISTORY_LENGTH)
        valid = positions < target[:, None]
        positions = np.minimum(positions, target[:, None])
        item_history = np.where(valid, items[positions], 0)
        category_history = np.where(valid, categories[positions], 0)
        features = {
            "target_item_id": item,
            "target_cate_id": item_category[item],
            "user_id": users[target],
            "hist_item_id": item_history,
            "hist_cate_id": category_history,
        }
        return {key: torch.as_tensor(value, dtype=torch.long) for key, value in features.items()}, torch.from_numpy(positive).float()


def run(data, output, models, device, epochs=None):
    if models != ["DIN"]:
        raise ValueError("The Amazon Electronics recipe supports DIN")
    random.seed(2022)
    torch.manual_seed(2022)
    frame = pd.read_csv(data, usecols=["user_id", "item_id", "cate_id", "time"])
    events, pairs = _events(frame)
    users, items, categories = events[:3]
    sizes = (int(items.max()) + 1, int(categories.max()) + 1)
    targets = [SparseFeature(name, size, embed_dim=8, padding_idx=0) for name, size in zip(("target_item_id", "target_cate_id"), sizes)]
    targets.append(SparseFeature("user_id", int(users.max()) + 1, embed_dim=8))
    history = [SequenceFeature(name, size, embed_dim=8, pooling="concat", shared_with=owner.name, padding_idx=0) for name, size, owner in zip(("hist_item_id", "hist_cate_id"), sizes, targets)]
    model = DIN(features=targets, target_features=targets, history_features=history, mlp_params={"dims": [256, 128]}, attention_mlp_params={"dims": [256, 128]})
    loaders = []
    for sampler, indices in zip((RandomSampler, SequentialSampler, SequentialSampler), pairs):
        dataset = _HistoryBatches(events, indices)
        batches = BatchSampler(sampler(dataset), batch_size=4096, drop_last=False)
        loaders.append(DataLoader(dataset, batch_size=None, sampler=batches))
    directory = output / "DIN"
    directory.mkdir(parents=True, exist_ok=True)
    trainer = CTRTrainer(model, optimizer_params={"lr": 0.001, "weight_decay": 0.001}, n_epoch=epochs or 2, earlystop_patience=epochs or 4, device=device, model_path=str(directory))
    trainer.fit(loaders[0], loaders[1])
    return {"DIN": {"AUC": float(trainer.evaluate(trainer.model, loaders[2]))}}
