import sys

import pandas as pd
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.multi_task import AITM, ESMM, MMOE, PLE, SharedBottom
from torch_rechub.trainers import MTLTrainer
from torch_rechub.utils.data import DataGenerator

sys.path.append("../..")


def get_ali_ccp_data_dict(model_name, data_path='./data/ali-ccp'):
    df_train = pd.read_csv(data_path + '/ali_ccp_train_sample.csv')
    df_val = pd.read_csv(data_path + '/ali_ccp_val_sample.csv')
    df_test = pd.read_csv(data_path + '/ali_ccp_test_sample.csv')
    print("train : val : test = %d %d %d" % (len(df_train), len(df_val), len(df_test)))
    train_idx, val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)
    # task 1 (as cvr): main task, purchase prediction
    # task 2(as ctr): auxiliary task, click prediction
    data.rename(columns={'purchase': 'cvr_label', 'click': 'ctr_label'}, inplace=True)
    data["ctcvr_label"] = data['cvr_label'] * data['ctr_label']

    col_names = data.columns.values.tolist()
    dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    sparse_cols = [col for col in col_names if col not in dense_cols and col not in ['cvr_label', 'ctr_label', 'ctcvr_label']]
    print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))
    # define dense and sparse features
    if model_name == "ESMM":
        # the order of 3 labels must fixed as this
        label_cols = ['cvr_label', 'ctr_label', "ctcvr_label"]
        # ESMM only for sparse features in origin paper
        # assumption features split for user and item
        item_cols = ['129', '205', '206', '207', '210', '216']
        user_cols = [col for col in sparse_cols if col not in item_cols]
        user_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in user_cols]
        item_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in item_cols]
        x_train, y_train = {name: data[name].values[:train_idx] for name in sparse_cols}, data[label_cols].values[:train_idx]
        x_val, y_val = {name: data[name].values[train_idx:val_idx] for name in sparse_cols}, data[label_cols].values[train_idx:val_idx]
        x_test, y_test = {name: data[name].values[val_idx:] for name in sparse_cols}, data[label_cols].values[val_idx:]
        return user_features, item_features, x_train, y_train, x_val, y_val, x_test, y_test
    else:
        # the order of labels can be any
        label_cols = ['cvr_label', 'ctr_label']
        used_cols = sparse_cols + dense_cols
        features = [SparseFeature(col, data[col].max() + 1, embed_dim=4)for col in sparse_cols] \
            + [DenseFeature(col) for col in dense_cols]
        x_train, y_train = {name: data[name].values[:train_idx] for name in used_cols}, data[label_cols].values[:train_idx]
        x_val, y_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}, data[label_cols].values[train_idx:val_idx]
        x_test, y_test = {name: data[name].values[val_idx:] for name in used_cols}, data[label_cols].values[val_idx:]
        return features, x_train, y_train, x_val, y_val, x_test, y_test


def main(model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    torch.manual_seed(seed)
    if model_name == "SharedBottom":
        features, x_train, y_train, x_val, y_val, x_test, y_test = get_ali_ccp_data_dict(model_name)
        task_types = ["classification", "classification"]
        model = SharedBottom(features, task_types, bottom_params={"dims": [117]}, tower_params_list=[{"dims": [8]}, {"dims": [8]}])
    elif model_name == "ESMM":
        user_features, item_features, x_train, y_train, x_val, y_val, x_test, y_test = get_ali_ccp_data_dict(model_name)
        task_types = ["classification", "classification", "classification"]  # cvr,ctr,ctcvr
        model = ESMM(user_features, item_features, cvr_params={"dims": [16, 8]}, ctr_params={"dims": [16, 8]})
    elif model_name == "MMOE":
        features, x_train, y_train, x_val, y_val, x_test, y_test = get_ali_ccp_data_dict(model_name)
        task_types = ["classification", "classification"]
        model = MMOE(features, task_types, 8, expert_params={"dims": [16]}, tower_params_list=[{"dims": [8]}, {"dims": [8]}])
    elif model_name == "PLE":
        features, x_train, y_train, x_val, y_val, x_test, y_test = get_ali_ccp_data_dict(model_name)
        task_types = ["classification", "classification"]
        model = PLE(features, task_types, n_level=1, n_expert_specific=2, n_expert_shared=1, expert_params={"dims": [16]}, tower_params_list=[{"dims": [8]}, {"dims": [8]}])
    elif model_name == "AITM":
        task_types = ["classification", "classification"]
        features, x_train, y_train, x_val, y_val, x_test, y_test = get_ali_ccp_data_dict(model_name)
        model = AITM(features, 2, bottom_params={"dims": [32, 16]}, tower_params_list=[{"dims": [8]}, {"dims": [8]}])

    dg = DataGenerator(x_train, y_train)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, batch_size=batch_size)

    # adaptive weight loss:
    # mtl_trainer = MTLTrainer(model, task_types=task_types, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, adaptive_params={"method": "uwl"}, n_epoch=epoch, earlystop_patience=10, device=device, model_path=save_dir)

    mtl_trainer = MTLTrainer(model, task_types=task_types, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=30, device=device, model_path=save_dir)
    mtl_trainer.fit(train_dataloader, val_dataloader)
    auc = mtl_trainer.evaluate(mtl_trainer.model, test_dataloader)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='SharedBottom')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu')  # cuda:0
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    main(args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed)
"""
python run_ali_ccp_multi_task.py --model_name SharedBottom
python run_ali_ccp_multi_task.py --model_name ESMM
python run_ali_ccp_multi_task.py --model_name MMOE
python run_ali_ccp_multi_task.py --model_name PLE
python run_ali_ccp_multi_task.py --model_name AITM
"""
