import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models.ranking import DIEN
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature

sys.path.append("../..")


def build_item2cate(data, item_col, cate_col):
    return data[[item_col, cate_col]].drop_duplicates().set_index(item_col)[cate_col].to_dict()


def build_neg_history(split, hist_item_col, item2cate, n_items):
    """Per-timestep negative sampling: sample a neg item, then look up its cate."""
    seqs = split[hist_item_col]  # [N, T]
    neg_items = np.zeros_like(seqs)
    neg_cates = np.zeros_like(seqs)
    for i, row in enumerate(seqs):
        for t, item in enumerate(row):
            if item == 0:  # padding
                continue
            neg = item
            while neg == item:
                neg = random.randint(1, n_items)
            neg_items[i, t] = neg
            neg_cates[i, t] = item2cate.get(neg, 1)
    return neg_items, neg_cates


def get_data(dataset_path):
    raw = pd.read_csv(dataset_path)
    # replicate label encoding done inside generate_seq_feature to build item2cate
    enc_data = raw.copy()
    for feat in enc_data:
        le = LabelEncoder()
        enc_data[feat] = le.fit_transform(enc_data[feat]) + 1
    enc_data = enc_data.astype('int32')
    item2cate = build_item2cate(enc_data, 'item_id', 'cate_id')
    n_items = enc_data['item_id'].max()
    n_users = enc_data['user_id'].max()
    n_cates = enc_data['cate_id'].max()

    train, val, test = generate_seq_feature(data=raw, user_col="user_id", item_col="item_id", time_col="time", item_attribute_cols=["cate_id"])

    features = [SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)]
    target_features = [
        SparseFeature("target_item_id",
                      vocab_size=n_items + 1,
                      embed_dim=8,
                      padding_idx=0),
        SparseFeature("target_cate_id",
                      vocab_size=n_cates + 1,
                      embed_dim=8,
                      padding_idx=0),
    ]
    history_features = [
        SequenceFeature("hist_item_id",
                        vocab_size=n_items + 1,
                        embed_dim=8,
                        pooling="concat",
                        shared_with="target_item_id",
                        padding_idx=0),
        SequenceFeature("hist_cate_id",
                        vocab_size=n_cates + 1,
                        embed_dim=8,
                        pooling="concat",
                        shared_with="target_cate_id",
                        padding_idx=0),
    ]
    neg_history_features = [
        SequenceFeature("neg_hist_item_id",
                        vocab_size=n_items + 1,
                        embed_dim=8,
                        pooling="concat",
                        shared_with="target_item_id",
                        padding_idx=0),
        SequenceFeature("neg_hist_cate_id",
                        vocab_size=n_cates + 1,
                        embed_dim=8,
                        pooling="concat",
                        shared_with="target_cate_id",
                        padding_idx=0),
    ]

    train, val, test = df_to_dict(train), df_to_dict(val), df_to_dict(test)
    train_y, val_y, test_y = train.pop("label"), val.pop("label"), test.pop("label")

    for split in [train, val, test]:
        neg_items, neg_cates = build_neg_history(split, "hist_item_id", item2cate, n_items)
        split["neg_hist_item_id"] = neg_items
        split["neg_hist_cate_id"] = neg_cates

    return (features, target_features, history_features, neg_history_features, (train, train_y), (val, val_y), (test, test_y))


def main(dataset_path, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    torch.manual_seed(seed)
    features, target_features, history_features, neg_history_features, \
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_data(dataset_path)

    dg = DataGenerator(train_x, train_y)
    train_dl, val_dl, test_dl = dg.generate_dataloader(x_val=val_x, y_val=val_y, x_test=test_x, y_test=test_y, batch_size=batch_size)

    model = DIEN(
        features=features,
        history_features=history_features,
        neg_history_features=neg_history_features,
        target_features=target_features,
        mlp_params={"dims": [256,
                             128]},
        alpha=0.2,
    )

    trainer = CTRTrainer(
        model,
        optimizer_params={
            "lr": learning_rate,
            "weight_decay": weight_decay
        },
        n_epoch=epoch,
        earlystop_patience=4,
        device=device,
        model_path=save_dir,
        loss_mode=False,
    )
    trainer.fit(train_dl, val_dl)
    auc = trainer.evaluate(trainer.model, test_dl)
    print(f"test auc: {auc:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="./data/amazon-electronics/amazon_electronics_sample.csv")
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default="./")
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    main(args.dataset_path, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed)
"""
python run_dien.py
"""
