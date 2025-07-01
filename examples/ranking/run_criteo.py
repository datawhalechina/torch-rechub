import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DCN, EDCN, DCNv2, DeepFFM, DeepFM, FatDeepFFM, FiBiNet, WideDeep
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

sys.path.append("../..")


def convert_numeric_feature(val):
    v = int(val)
    if v > 2:
        return int(np.log(v)**2)
    else:
        return v - 2


def get_criteo_data_dict(data_path):
    if data_path.endswith(".gz"):  # if the raw_data is gz file:
        data = pd.read_csv(data_path, compression="gzip")
    else:
        data = pd.read_csv(data_path)
    print("data load finished")
    dense_features = [f for f in data.columns.tolist() if f[0] == "I"]
    sparse_features = [f for f in data.columns.tolist() if f[0] == "C"]

    data[sparse_features] = data[sparse_features].fillna('0')
    data[dense_features] = data[dense_features].fillna(0)

    for feat in tqdm(dense_features):  # discretize dense feature and as new sparse feature
        sparse_features.append(feat + "_cat")
        data[feat + "_cat"] = data[feat].apply(lambda x: convert_numeric_feature(x))

    sca = MinMaxScaler()  # scaler dense feature
    data[dense_features] = sca.fit_transform(data[dense_features])

    for feat in tqdm(sparse_features):  # encode sparse feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name in sparse_features]
    ffm_linear_feas = [SparseFeature(feature.name, vocab_size=feature.vocab_size, embed_dim=1) for feature in sparse_feas]
    ffm_cross_feas = [SparseFeature(feature.name, vocab_size=feature.vocab_size * len(sparse_feas), embed_dim=10) for feature in sparse_feas]
    y = data["label"]
    del data["label"]
    x = data
    return dense_feas, sparse_feas, ffm_linear_feas, ffm_cross_feas, x, y


def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    torch.manual_seed(seed)
    dense_feas, sparse_feas, ffm_linear_feas, ffm_cross_feas, x, y = get_criteo_data_dict(dataset_path)
    dg = DataGenerator(x, y)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=batch_size)
    if model_name == "widedeep":
        model = WideDeep(wide_features=dense_feas, deep_features=sparse_feas, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
    elif model_name == "deepfm":
        model = DeepFM(deep_features=dense_feas, fm_features=sparse_feas, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
    elif model_name == "dcn":
        model = DCN(features=dense_feas + sparse_feas, n_cross_layers=3, mlp_params={"dims": [256, 128]})
    elif model_name == "dcn_v2":
        model = DCNv2(features=dense_feas + sparse_feas, n_cross_layers=3, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
    elif model_name == "fibinet":
        model = FiBiNet(features=dense_feas + sparse_feas, reduction_ratio=3, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
    elif model_name == "edcn":
        model = EDCN(features=dense_feas + sparse_feas, n_cross_layers=3, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
    elif model_name == "deepffm":
        model = DeepFFM(linear_features=ffm_linear_feas, cross_features=ffm_cross_feas, embed_dim=10, mlp_params={"dims": [1600, 1600], "dropout": 0.5, "activation": "relu"})
    elif model_name == "fat_deepffm":
        model = FatDeepFFM(linear_features=ffm_linear_feas, cross_features=ffm_cross_feas, embed_dim=10, reduction_ratio=1, mlp_params={"dims": [1600, 1600], "dropout": 0.5, "activation": "relu"})
    ctr_trainer = CTRTrainer(model, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=10, device=device, model_path=save_dir)
    # scheduler_fn=torch.optim.lr_scheduler.StepLR,scheduler_params={"step_size": 2,"gamma": 0.8},
    ctr_trainer.fit(train_dataloader, val_dataloader)
    auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data/criteo/criteo_sample.csv")
    parser.add_argument('--model_name', default='fibinet')
    parser.add_argument('--epoch', type=int, default=2)  # 100
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2048)  # 4096
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')  # cuda:0
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed)
"""
python run_criteo.py --model_name widedeep
python run_criteo.py --model_name deepfm
python run_criteo.py --model_name dcn
python run_criteo.py --model_name dcn_v2
python run_criteo.py --model_name edcn
python run_criteo.py --model_name deepffm
python run_criteo.py --model_name fat_deepffm
"""
