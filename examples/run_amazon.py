import sys
import random

sys.path.append("../")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_ctr.models import WideDeep, DeepFM, DIN
from torch_ctr.basic.trainer import CTRTrainer
from torch_ctr.basic.features import DenseFeature, SparseFeature, SequenceFeature
from torch_ctr.basic.utils import DataGenerator


def get_amazon_data_dict(reviews_file_path, meta_file_path):
    data = generate_seq_data(reviews_file_path, meta_file_path)
    n_item = len(np.unique(data["target_item"]))
    n_cate = len(np.unique(data["target_cate"]))
    features = [SparseFeature("target_item", vocab_size=n_item, embed_dim=8),
                SparseFeature("target_cate", vocab_size=n_cate, embed_dim=8)]
    target_features = features
    history_features = [
        SequenceFeature("history_item", vocab_size=n_item, embed_dim=8, pooling="concat", shared_with="target_item"),
        SequenceFeature("history_cate", vocab_size=n_cate, embed_dim=8, pooling="concat", shared_with="target_cate")
    ]
    y = data["label"]
    del data["label"]
    x = data
    return features, target_features, history_features, x, y


def json_to_df(file_path):
    with open(file_path, 'r') as f:
        df = {}
        i = 0
        for line in f:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


def generate_seq_data(reviews_file_path, meta_file_path):
    print('========== Start reading data ==========')
    reviews_df = json_to_df(reviews_file_path)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

    meta_df = json_to_df(meta_file_path)
    meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
    meta_df = meta_df.reset_index(drop=True)
    meta_df = meta_df[['asin', 'categories']]
    meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1]) # Category features keep only one
    print('========== DataFrame file successfully read ==========')

    asin_map, asin_key = build_map(meta_df, 'asin')
    cate_map, cate_key = build_map(meta_df, 'categories')
    revi_map, revi_key = build_map(reviews_df, 'reviewerID')

    user_count, item_count, cate_count, example_count = \
        len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]

    meta_df = meta_df.sort_values('asin')
    meta_df = meta_df.reset_index(drop=True)

    reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
    reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
    reviews_df = reviews_df.reset_index(drop=True)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

    cate_list = np.array(meta_df['categories'], dtype='int32')

    reviews_df = reviews_df
    reviews_df.columns = ['user_id', 'item_id', 'time']

    # Generate sequence features
    data = []
    max_len = 50
    print('========== Start generating sequence features ==========')
    for user_id, hist in tqdm(reviews_df.groupby('user_id')):
        pos_list = hist['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]
        hist = []
        cate = []
        for i in range(1, len(pos_list)):
            hist.append(pos_list[i - 1])
            cate.append(cate_list[pos_list[i - 1]])
            hist_pad = hist[: max_len]
            cate_pad = cate[: max_len]
            if len(hist) < max_len:
                hist_pad = hist + [0] * (max_len - len(hist))
                cate_pad = cate + [0] * (max_len - len(cate))
            data.append([np.array(hist_pad), np.array(cate_pad), pos_list[i], cate_list[pos_list[i]], 1]) # generate positive samples
            data.append([np.array(hist_pad), np.array(cate_pad), neg_list[i], cate_list[neg_list[i]], 0]) # generate negative samples

    random.shuffle(data)

    # generate datasets
    data_dict = {'history_item': [],
                 'history_cate': [],
                 'target_item': [],
                 'target_cate': [],
                 'label': []}
    for item in data:
        data_dict['history_item'].append(item[0])
        data_dict['history_cate'].append(item[1])
        data_dict['target_item'].append(item[2])
        data_dict['target_cate'].append(item[3])
        data_dict['label'].append(item[4])

    for key in data_dict.keys(): # Convert to ndarray
        data_dict[key] = np.array(data_dict[key])

    return data_dict


def main(dataset_name,
         reviews_file_path,
         meta_file_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         seed):

    torch.manual_seed(seed)
    features, target_features, history_features, x, y = get_amazon_data_dict(reviews_file_path, meta_file_path)
    dg = DataGenerator(x, y)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.8, 0.1],
                                                                               batch_size=batch_size)
    model = DIN(features=features, history_features=history_features, target_features=target_features,
                mlp_params={"dims": [256, 128]})

    ctr_trainer = CTRTrainer(model,
                             optimizer_params={
                                 "lr": learning_rate,
                                 "weight_decay": weight_decay
                             },
                             n_epoch=epoch,
                             earlystop_patience=4,
                             device=device,
                             model_path=save_dir)
    # scheduler_fn=torch.optim.lr_scheduler.StepLR,scheduler_params={"step_size": 2,"gamma": 0.8},
    ctr_trainer.train(train_dataloader, val_dataloader)
    auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='amazon')
    parser.add_argument('--reviews_file_path', default="./amazon/data/reviews_Electronics_5.json")
    parser.add_argument('--meta_file_path', default="./amazon/data/meta_Electronics.json")
    parser.add_argument('--model_name', default='din')
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./amazon/saved_model')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    main(args.dataset_name,
         args.reviews_file_path,
         args.meta_file_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.seed)
"""
调用参考：
python run_amazon.py
"""
