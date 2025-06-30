import sys

import pandas as pd
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.multi_task import AITM, MMOE, PLE, SharedBottom
from torch_rechub.trainers import MTLTrainer
from torch_rechub.utils.data import DataGenerator

sys.path.append("../..")


def get_aliexpress_data_dict(data_path='./data/aliexpress'):
    df_train = pd.read_csv(data_path + '/aliexpress_train_sample.csv')
    df_test = pd.read_csv(data_path + '/aliexpress_test_sample.csv')
    print("train : test = %d %d" % (len(df_train), len(df_test)))
    train_idx = df_train.shape[0]
    data = pd.concat([df_train, df_test], axis=0)
    col_names = data.columns.values.tolist()
    sparse_cols = [name for name in col_names if name.startswith("categorical")]  # categorical
    dense_cols = [name for name in col_names if name.startswith("numerical")]  # numerical
    print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))
    label_cols = ["conversion", "click"]

    used_cols = sparse_cols + dense_cols
    features = [SparseFeature(col, data[col].max() + 1, embed_dim=5)for col in sparse_cols] \
        + [DenseFeature(col) for col in dense_cols]
    x_train, y_train = {name: data[name].values[:train_idx] for name in used_cols}, data[label_cols].values[:train_idx]
    x_test, y_test = {name: data[name].values[train_idx:] for name in used_cols}, data[label_cols].values[train_idx:]
    return features, x_train, y_train, x_test, y_test


def main(model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    torch.manual_seed(seed)
    features, x_train, y_train, x_test, y_test = get_aliexpress_data_dict()
    task_types = ["classification", "classification"]
    if model_name == "SharedBottom":
        model = SharedBottom(features, task_types, bottom_params={"dims": [512, 256]}, tower_params_list=[{"dims": [128, 64]}, {"dims": [128, 64]}])
    elif model_name == "MMOE":
        model = MMOE(features, task_types, n_expert=8, expert_params={"dims": [512, 256]}, tower_params_list=[{"dims": [128, 64]}, {"dims": [128, 64]}])
    elif model_name == "PLE":
        model = PLE(features,
                    task_types,
                    n_level=1,
                    n_expert_specific=4,
                    n_expert_shared=4,
                    expert_params={
                        "dims": [512,
                                 256],
                    },
                    tower_params_list=[{
                        "dims": [128,
                                 64]
                    },
                                       {
                                           "dims": [128,
                                                    64]
                                       }])
    elif model_name == "AITM":
        model = AITM(features, n_task=2, bottom_params={"dims": [512, 256]}, tower_params_list=[{"dims": [128, 64]}, {"dims": [128, 64]}])

    dg = DataGenerator(x_train, y_train)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=x_test, y_val=y_test, x_test=x_test, y_test=y_test, batch_size=batch_size)

    # adaptive weight loss:
    # mtl_trainer = MTLTrainer(model, task_types=task_types, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, adaptive_params={"method": "uwl"}, n_epoch=epoch, earlystop_patience=10, device=device, model_path=save_dir)
    # metabalance
    mtl_trainer = MTLTrainer(model, task_types=task_types, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=3, device=device, adaptive_params={'method': 'gradnorm'}, model_path=save_dir)
    mtl_trainer.fit(train_dataloader, val_dataloader)
    auc = mtl_trainer.evaluate(mtl_trainer.model, test_dataloader)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='SharedBottom')
    parser.add_argument('--epoch', type=int, default=100)  # 100
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=10240)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cpu')  # cuda:0
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    main(args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed)
"""
python run_gradnorm.py --model_name SharedBottom
python run_gradnorm.py --model_name MMOE
python run_gradnorm.py --model_name PLE
python run_gradnorm.py --model_name AITM
"""
