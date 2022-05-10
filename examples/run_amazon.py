import sys

sys.path.append("../")

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_rechub.models.ranking import DIN
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.features import DenseFeature, SparseFeature, SequenceFeature
from torch_rechub.basic.utils import DataGenerator, create_seq_features, df_to_input_dict


def get_amazon_data_dict(dataset_path):
    data = pd.read_csv(dataset_path)
    print('========== Start Amazon ==========')
    for feat in data:
        le = LabelEncoder()
        data[feat] = le.fit_transform(data[feat])
    n_users, n_items, n_cates = data["user_id"].max(), data["item_id"].max(), data["cate_id"].max()

    features = [SparseFeature("target_item", vocab_size=n_items + 2, embed_dim=8), SparseFeature("target_cate", vocab_size=n_cates + 2, embed_dim=8)]
    target_features = features
    history_features = [SequenceFeature("history_item", vocab_size=n_items + 2, embed_dim=8, pooling="concat", shared_with="target_item"), SequenceFeature("history_cate", vocab_size=n_cates + 2, embed_dim=8, pooling="concat", shared_with="target_cate")]

    print('========== create sequence features ==========')
    train, val, test = create_seq_features(data)

    print('========== generate input dict ==========')
    train = df_to_input_dict(train)
    val = df_to_input_dict(val)
    test = df_to_input_dict(test)

    train_y, val_y, test_y = train["label"], val["label"], test["label"]

    del train["label"]
    del val["label"]
    del test["label"]
    train_x, val_x, test_x = train, val, test
    return features, target_features, history_features, (train_x, train_y), (val_x, val_y), (test_x, test_y)


def main(dataset_path, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    torch.manual_seed(seed)
    features, target_features, history_features, (train_x, train_y), (val_x, val_y), (test_x, test_y) = \
        get_amazon_data_dict(dataset_path)
    dg = DataGenerator(train_x, train_y)

    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=val_x, y_val=val_y, x_test=test_x, y_test=test_y, batch_size=batch_size)
    model = DIN(features=features, history_features=history_features, target_features=target_features, mlp_params={"dims": [256, 128]})

    ctr_trainer = CTRTrainer(model, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=4, device=device, model_path=save_dir)
    # scheduler_fn=torch.optim.lr_scheduler.StepLR,scheduler_params={"step_size": 2,"gamma": 0.8},
    ctr_trainer.fit(train_dataloader, val_dataloader)
    auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data/amazon/amazon_sample.csv")
    parser.add_argument('--epoch', type=int, default=2)  #100
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')  #cuda:0
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    main(args.dataset_path, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed)
"""
python run_amazon.py
"""
