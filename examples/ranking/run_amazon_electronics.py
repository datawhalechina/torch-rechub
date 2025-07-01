import sys

import numpy as np
import pandas as pd
import torch

from torch_rechub.basic.features import DenseFeature, SequenceFeature, SparseFeature
from torch_rechub.models.ranking import DIN
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature, pad_sequences

sys.path.append("../..")


def get_amazon_data_dict(dataset_path):
    data = pd.read_csv(dataset_path)
    print('========== Start Amazon ==========')
    train, val, test = generate_seq_feature(data=data, user_col="user_id", item_col="item_id", time_col='time', item_attribute_cols=["cate_id"])
    print('INFO: Now, the dataframe named: ', train.columns)
    n_users, n_items, n_cates = data["user_id"].max(), data["item_id"].max(), data["cate_id"].max()
    print(train)

    features = [SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8), SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8), SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)]
    target_features = features
    history_features = [
        SequenceFeature("hist_item_id",
                        vocab_size=n_items + 1,
                        embed_dim=8,
                        pooling="concat",
                        shared_with="target_item_id"),
        SequenceFeature("hist_cate_id",
                        vocab_size=n_cates + 1,
                        embed_dim=8,
                        pooling="concat",
                        shared_with="target_cate_id")
    ]

    print('========== Generate input dict ==========')
    train = df_to_dict(train)
    val = df_to_dict(val)
    test = df_to_dict(test)
    train_y, val_y, test_y = train["label"], val["label"], test["label"]

    del train["label"]
    del val["label"]
    del test["label"]
    train_x, val_x, test_x = train, val, test
    return features, target_features, history_features, (train_x, train_y), (val_x, val_y), (test_x, test_y)


def main(dataset_path, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    torch.manual_seed(seed)
    features, target_features, history_features, (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_amazon_data_dict(dataset_path)
    dg = DataGenerator(train_x, train_y)

    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=val_x, y_val=val_y, x_test=test_x, y_test=test_y, batch_size=batch_size)
    model = DIN(features=features, history_features=history_features, target_features=target_features, mlp_params={"dims": [256, 128]}, attention_mlp_params={"dims": [256, 128]})

    ctr_trainer = CTRTrainer(model, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=4, device=device, model_path=save_dir)
    ctr_trainer.fit(train_dataloader, val_dataloader)
    auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data/amazon-electronics/amazon_electronics_sample.csv")
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')  # cuda:0
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    main(args.dataset_path, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed)
"""
python run_amazon_electronics.py
"""
