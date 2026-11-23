"""MovieLens next-item retrieval with chronological holdouts and exact catalog ranking."""

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models.matching import DSSM, MIND, NARM, SINE, STAMP, ComirecDR, ComirecSA, GRU4Rec, SASRec, YoutubeDNN
from torch_rechub.trainers import MatchTrainer

SEED = 2022
HISTORY = 20
BATCH_SIZE = 512
EMBED_DIM = 16


class _Interactions:
    """Keep one copy of each event and gather its most recent history on demand."""

    def __init__(self, path, device):
        data = pd.read_csv(path, usecols=["user_id", "movie_id", "timestamp"])
        data = data[data.groupby("user_id")["movie_id"].transform("size") >= 4]
        if data.empty:
            raise ValueError("Need users with at least four events for training, validation and testing.")
        data = data.sort_values(["user_id", "timestamp"], kind="stable")
        user, users = pd.factorize(data.user_id, sort=True)
        item, items = pd.factorize(data.movie_id, sort=True)
        self.n_users, self.n_items = len(users), len(items)
        sizes = np.bincount(user)
        end = sizes.cumsum()
        start = end - sizes
        rows = np.arange(len(data))
        self.train = torch.as_tensor(rows[(rows > start[user]) & (rows < end[user] - 2)], device=device)
        self.valid = end - 2
        self.test = end - 1
        self.user = torch.as_tensor(user, device=device)
        self.item = torch.as_tensor(item + 1, device=device)
        self.start = torch.as_tensor(start, device=device)
        self.offsets = torch.arange(-HISTORY, 0, device=device)

        # Evaluation excludes the entire observed prefix, including events outside the encoder window.
        def seen_before(holdout):
            prefix = rows < holdout[user]
            return csr_matrix((np.ones(prefix.sum(), dtype=bool), (user[prefix], item[prefix])), shape=(self.n_users, self.n_items))

        self.valid_seen = seen_before(self.valid)
        self.test_seen = seen_before(self.test)

    def features(self, rows, right_pad=False):
        users = self.user[rows]
        if right_pad:
            # Packed GRUs and STAMP index the last event as length - 1.
            first = torch.maximum(rows - HISTORY, self.start[users])
            positions = first[:, None] + (self.offsets + HISTORY)
            history = self.item[positions.minimum(rows[:, None])].masked_fill(positions >= rows[:, None], 0)
        else:
            positions = rows[:, None] + self.offsets
            history = self.item[positions.clamp_min(0)].masked_fill(positions < self.start[users, None], 0)
        return {"user_id": users + 1, "hist_movie_id": history, "movie_id": self.item[rows]}

    def batches(self, generator, right_pad=False):
        order = torch.randperm(len(self.train), generator=generator).to(self.train.device)
        for rows in self.train[order].split(BATCH_SIZE):
            features = self.features(rows, right_pad)
            yield features, features["movie_id"]


def _model(name, dataset):
    users = [SparseFeature("user_id", dataset.n_users + 1, EMBED_DIM)]
    items = [SparseFeature("movie_id", dataset.n_items + 1, EMBED_DIM, padding_idx=0)]
    history = SequenceFeature("hist_movie_id", dataset.n_items + 1, EMBED_DIM, pooling="mean" if name in ("YoutubeDNN", "DSSM") else "concat", shared_with=None if name == "SASRec" else "movie_id", padding_idx=0)
    if name == "SASRec":
        return SASRec([history], max_len=HISTORY, item_feature=items[0])
    if name == "NARM":
        return NARM(history, hidden_dim=32, emb_dropout_p=0.25, session_rep_dropout_p=0.5, item_feature=items[0])
    if name == "STAMP":
        # STAMP multiplies two projected vectors; tiny initial weights suppress both gradients.
        return STAMP(history, weight_std=0.05, emb_std=0.1, item_feature=items[0])
    if name == "SINE":
        return SINE([history.name], [items[0].name], [], dataset.n_items + 1, EMBED_DIM, hidden_dim=64, num_concept=10, num_intention=2, seq_max_len=HISTORY, temperature=0.1)
    if name == "DSSM":
        return DSSM(users + [history], items, user_params={"dims": [128, 64, EMBED_DIM]}, item_params={"dims": [64, EMBED_DIM]}, temperature=0.1)
    if name == "GRU4Rec":
        return GRU4Rec(users, [history], items, [], user_params={"dims": [128, 64, EMBED_DIM]}, temperature=0.1)
    if name == "YoutubeDNN":
        return YoutubeDNN(users + [history], items, [], user_params={"dims": [128, 64, EMBED_DIM]}, temperature=0.1)
    kwargs = dict(user_features=users, history_features=[history], item_features=items, neg_item_feature=[], temperature=0.1, interest_num=4)
    if name == "ComirecSA":
        return ComirecSA(**kwargs)
    return {"MIND": MIND, "ComirecDR": ComirecDR}[name](max_length=HISTORY, **kwargs)


@torch.no_grad()
def _evaluate(model, data, rows, seen, right_pad=False):
    device = data.item.device
    model.eval()
    movie_ids = torch.arange(1, data.n_items + 1, device=device)
    ranks, aucs = [], []
    # Fix MIND's routing noise during evaluation without advancing the training RNG.
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(SEED)
        try:
            model.mode = "item"
            items = model({"movie_id": movie_ids})
            model.mode = "user"
            for begin in range(0, len(rows), BATCH_SIZE):
                end = begin + BATCH_SIZE
                features = data.features(torch.as_tensor(rows[begin:end], device=device), right_pad)
                target_ids = features.pop("movie_id")
                users = model(features).reshape(len(target_ids), -1, EMBED_DIM)
                scores = (users @ items.T).amax(dim=1)
                if not torch.isfinite(scores).all():
                    raise ValueError("Found a non-finite catalog score before ranking.")
                target = target_ids[:, None] - 1
                positive = scores.gather(1, target)
                scores.masked_fill_(torch.as_tensor(seen[begin:end].toarray(), device=device), -torch.inf)
                scores.scatter_(1, target, positive)  # Retain the target even if it was seen before.
                ahead = (scores > positive).sum(dim=1)
                tied = scores == positive
                # Break top-k ties by ascending movie ID; AUC assigns tied pairs half credit.
                ranks.append(1 + ahead + (tied & (movie_ids < target_ids[:, None])).sum(dim=1))
                negatives = torch.isfinite(scores).sum(dim=1) - 1
                aucs.append(1 - (ahead + 0.5 * (tied.sum(dim=1) - 1)) / negatives)
        finally:
            model.mode = None
    rank = torch.cat(ranks)
    auc = torch.cat(aucs)
    result = {"GAUC": auc.nanmean().item()}
    for k in (5, 10):
        hit = rank <= k
        result[f"Recall@{k}"] = result[f"Hit@{k}"] = hit.float().mean().item()
        result[f"NDCG@{k}"] = (hit / torch.log2(rank + 1)).mean().item()
    return result


def run(data, output, models, device, epochs=None):
    dataset = _Interactions(data, device)
    print(f"MovieLens users={dataset.n_users:,}, movies={dataset.n_items:,}, training targets={len(dataset.train):,}")
    results = {}
    for name in models:
        torch.manual_seed(SEED)
        generator = torch.Generator().manual_seed(SEED)
        model = _model(name, dataset)
        right_pad = name in ("NARM", "STAMP")
        trainer = MatchTrainer(model, mode=2, in_batch_neg=True, device=device, optimizer_params={"lr": 0.001, "weight_decay": 1e-6})
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "model.pth"
        best = -float("inf")
        for epoch in range(epochs or 10):
            loss = trainer.train_one_epoch(dataset.batches(generator, right_pad), log_interval=100)
            score = _evaluate(model, dataset, dataset.valid, dataset.valid_seen, right_pad)["NDCG@10"]
            print(f"{name} epoch {epoch + 1}: loss={loss:.6f}, validation NDCG@10={score:.6f}", flush=True)
            if score > best:
                best = score
                torch.save(model.state_dict(), checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        results[name] = _evaluate(model, dataset, dataset.test, dataset.test_seen, right_pad)
        print(f"{name} test: {results[name]}", flush=True)
    return results
