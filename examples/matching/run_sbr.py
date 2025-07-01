import argparse
import sys
from collections import defaultdict
from itertools import accumulate

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
from data.session_based.preprocess_session_based import INTERNAL_ITEM_ID_FIELD, TEST_DATA_PREFIX, TRAIN_DATA_PREFIX
from torch.utils.data import DataLoader

from torch_rechub.basic.features import SequenceFeature
from torch_rechub.basic.metric import topk_metrics
from torch_rechub.models.matching import NARM, STAMP
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import TorchDataset

sys.path.append("../..")

HISTORY_ITEM_FILED = "hist_item_id"


def generate_model_input(df, max_seq_len=19):
    df["x"] = df[INTERNAL_ITEM_ID_FIELD].apply(lambda a: list(accumulate(a[1:-1], lambda t, e: t + [e], initial=[a[0]])))
    df["y"] = df[INTERNAL_ITEM_ID_FIELD].apply(lambda a: a[1:])
    df_input = df[["x", "y"]].explode(["x", "y"])
    df_input["x"] = df_input["x"].apply(lambda x: x[:max_seq_len])
    return {HISTORY_ITEM_FILED: df_input["x"].values.tolist()}, df_input["y"].values.tolist()


def get_sbr_data(config):
    df_train_sessions = pd.read_pickle(config.data_path + f"/{TRAIN_DATA_PREFIX}.pkl")
    df_test_sessions = pd.read_pickle(config.data_path + f"/{TEST_DATA_PREFIX}.pkl")
    x_train, y_train = generate_model_input(df_train_sessions, config.max_seq_len)
    x_test, y_test = generate_model_input(df_test_sessions, config.max_seq_len)

    item_history_feature = (SequenceFeature(HISTORY_ITEM_FILED, vocab_size=pd.concat([df_train_sessions, df_test_sessions])[INTERNAL_ITEM_ID_FIELD].explode().max() + 1, embed_dim=config.item_emb_dim, pooling="concat"))
    return item_history_feature, x_train, y_train, x_test, y_test


def collate_fn(batch):
    x = rnn_utils.pad_sequence([torch.LongTensor(b[0][HISTORY_ITEM_FILED]) for b in batch], batch_first=True)
    y = torch.LongTensor([b[1] for b in batch])
    return {HISTORY_ITEM_FILED: x}, y


def evaluate(model, test_dl, device, top_k=20):
    res = defaultdict(list)
    model.eval()
    for input_dict, y_true in test_dl:
        input_dict = {k: v.to(device) for k, v in input_dict.items()}
        _, y_pred = torch.topk(model(input_dict).detach(), top_k)

        y_true = dict(zip(range(len(y_true)), y_true.cpu().numpy()[:, None]))
        y_pred = dict(zip(range(len(y_pred)), y_pred.cpu().numpy()))

        metrics = topk_metrics(y_true, y_pred, topKs=[top_k])
        for _, m in metrics.items():
            k, v = m[0].split(":")
            res[k.strip()].append(float(v.strip()))
    res = {k: np.mean(v) for k, v in res.items()}
    return res


def main(config):
    torch.manual_seed(config.seed)
    item_history_feature, x_train, y_train, x_test, y_test = get_sbr_data(config)

    train_dl = DataLoader(TorchDataset(x_train, y_train), collate_fn=collate_fn, batch_size=config.batch_size)
    test_dl = DataLoader(TorchDataset(x_test, y_test), collate_fn=collate_fn, batch_size=config.batch_size)

    if config.model_name == "narm":
        model = NARM(item_history_feature, config.hidden_dim, config.emb_dropout, config.session_rep_dropout)
    elif config.model_name == "stamp":
        model = STAMP(item_history_feature, config.weight_std, config.emb_std)
    else:
        raise NotImplementedError(f"Unknown model {config.model_name}")

    trainer = MatchTrainer(model, mode=2, optimizer_params={"lr": config.learning_rate, "weight_decay": config.weight_decay}, n_epoch=config.epoch, device=config.device, model_path=config.save_dir)

    trainer.fit(train_dl)

    metrics = evaluate(model, test_dl, config.device, top_k=config.top_k)
    print(f"test metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/session_based/diginetica")
    parser.add_argument("--model_name", default="narm")
    parser.add_argument("--max_seq_len", default=19, type=int)
    parser.add_argument("--item_emb_dim", default=100, type=int)
    parser.add_argument("--hidden_dim", default=50, type=int)
    parser.add_argument("--emb_dropout", default=0.25, type=float)
    parser.add_argument("--session_rep_dropout", default=0.5, type=float)
    parser.add_argument("--weight_std", default=0.05, type=float)
    parser.add_argument("--emb_std", default=0.002, type=float)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_dir", default="./")

    args = parser.parse_args()
    main(args)
"""
python run_sbr.py --model_name narm --data_path ./data/session_based/diginetica --top_k 3
python run_sbr.py --model_name narm --data_path ./data/session_based/yoochoose --top_k 3

python run_sbr.py --model_name stamp --data_path ./data/session_based/diginetica --top_k 3
python run_sbr.py --model_name stamp --data_path ./data/session_based/yoochoose --top_k 3
"""
