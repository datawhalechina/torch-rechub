import sys

sys.path.append("../")

import numpy as np
import pickle
import torch
from torch_ctr.models import WideDeep, DeepFM, DIN
from torch_ctr.basic.trainer import CTRTrainer
from torch_ctr.basic.features import DenseFeature, SparseFeature, SequenceFeature
from torch_ctr.basic.utils import DataGenerator


def get_amazon_data_dict(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)  #five key: history_item, history_cate,target_item,target_cate,label
    n_item = len(np.unique(data["target_item"]))
    n_cate = len(np.unique(data["target_cate"]))
    features = [SparseFeature("target_item", vocab_size=n_item, embed_dim=8), SparseFeature("target_cate", vocab_size=n_cate, embed_dim=8)]
    target_features = features
    history_features = [
        SequenceFeature("history_item", vocab_size=n_item, embed_dim=8, pooling="concat", shared_with="target_item"),
        SequenceFeature("history_cate", vocab_size=n_cate, embed_dim=8, pooling="concat", shared_with="target_cate")
    ]
    y = data["label"]
    del data["label"]
    x = data
    return features, target_features, history_features, x, y


def main(dataset_name, dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    torch.manual_seed(seed)
    features, target_features, history_features, x, y = get_amazon_data_dict(dataset_path)
    dg = DataGenerator(x, y)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.8, 0.1], batch_size=batch_size)
    model = DIN(features=features, history_features=history_features, target_features=target_features, mlp_params={"dims": [256, 128]})

    ctr_trainer = CTRTrainer(model,
                             optimizer_params={
                                 "lr": learning_rate,
                                 "weight_decay": weight_decay
                             },
                             n_epoch=epoch,
                             earlystop_patience=4,
                             device=device,
                             model_path=save_dir)
    #scheduler_fn=torch.optim.lr_scheduler.StepLR,scheduler_params={"step_size": 2,"gamma": 0.8},
    ctr_trainer.train(train_dataloader, val_dataloader)
    auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='amazon')
    parser.add_argument('--dataset_path', default="./amazon/data/amazon_dict.pkl")
    parser.add_argument('--model_name', default='din')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./amazon/saved_model')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    main(args.dataset_name, args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device,
         args.save_dir, args.seed)
"""
调用参考：
python run_amazon.py
"""