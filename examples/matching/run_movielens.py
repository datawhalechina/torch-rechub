import sys

sys.path.append("../..")

import os
import numpy as np
import pandas as pd
import collections
import torch

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.basic.features import DenseFeature, SparseFeature, SequenceFeature
from torch_rechub.utils.match import Annoy, generate_seq_feature_match, gen_model_input
from torch_rechub.utils.data import df_to_dict, MatchDataGenerator
from torch_rechub.basic.metric import topk_metrics


def get_movielens_data(data_path, load_cache=False):
    data = pd.read_csv(data_path)
    data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', "cate_id"]
    user_col, item_col, label_col = "user_id", "movie_id", "label"

    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        if feature == user_col:
            user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)
                       }  #encode user id: raw user id
        if feature == item_col:
            item_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)
                       }  #encode item id: raw item id
    np.save("./data/ml-1m/saved/raw_id_maps.npy", (user_map, item_map))

    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
    item_profile = data[["movie_id", "cate_id"]].drop_duplicates('movie_id')

    if load_cache:  #if you have run this script before and saved the preprocessed data
        x_train, y_train, x_test, y_test = np.load("./data/ml-1m/saved/data_preprocess.npy", allow_pickle=True)
    else:
        df_train, df_test = generate_seq_feature_match(data,
                                                       user_col,
                                                       item_col,
                                                       time_col="timestamp",
                                                       item_attribute_cols=[],
                                                       sample_method=0,
                                                       mode=0,
                                                       neg_ratio=3,
                                                       min_item=0)
        x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50)
        y_train = df_train["label"].values
        x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50)
        y_test = df_test["label"].values
        np.save("./data/ml-1m/saved/data_preprocess.npy", (x_train, y_train, x_test, y_test))

    user_cols = ['user_id', 'gender', 'age', 'occupation', 'zip']
    item_cols = ['movie_id', "cate_id"]

    user_features = [
        SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name], embed_dim=16)
        for feature_name in user_cols
    ]
    user_features += [
        SequenceFeature("hist_movie_id",
                        vocab_size=feature_max_idx["movie_id"],
                        embed_dim=16,
                        pooling="mean",
                        shared_with="movie_id")
    ]

    item_features = [
        SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name], embed_dim=16)
        for feature_name in item_cols
    ]

    all_item = df_to_dict(item_profile)
    test_user = x_test
    return user_features, item_features, x_train, y_train, all_item, test_user


def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.manual_seed(seed)
    user_features, item_features, x_train, y_train, all_item, test_user = get_movielens_data(dataset_path)
    dg = MatchDataGenerator(x=x_train, y=y_train)

    if model_name == "dssm":
        model = DSSM(user_features,
                     item_features,
                     sim_func="cosine",
                     temperature=0.05,
                     user_params={
                         "dims": [256, 128, 64, 32],
                     },
                     item_params={
                         "dims": [256, 128, 64, 32],
                     })

    trainer = MatchTrainer(model,
                           optimizer_params={
                               "lr": learning_rate,
                               "weight_decay": weight_decay
                           },
                           n_epoch=epoch,
                           device=device,
                           model_path=save_dir,
                           gpus=[0, 1])

    train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=batch_size)
    trainer.fit(train_dl)

    print("inference embedding")
    user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path=save_dir)
    item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=save_dir)
    torch.save(user_embedding.data.cpu(), save_dir + "user_embedding.pth")
    torch.save(item_embedding.data.cpu(), save_dir + "item_embedding.pth")

    print("evaluate embedding matching on test data")
    annoy = Annoy(n_trees=10)
    annoy.fit(item_embedding)

    #for each user of test dataset, get ann search topk result
    print("matching for topk")
    user_map, item_map = np.load("./data/ml-1m/saved/raw_id_maps.npy", allow_pickle=True)
    match_res = collections.defaultdict(dict)
    topk = 100
    for user_id, user_emb in zip(test_user["user_id"], user_embedding):
        items_idx, items_scores = annoy.query(v=user_emb, n=topk)  #the index of topk match items
        match_res[user_map[user_id]] = all_item["movie_id"][items_idx]

    #get ground truth
    print("generate ground truth")
    user_col = "user_id"
    item_col = "movie_id"

    data = pd.DataFrame({"user_id": all_item["user_id"], "movie_id": all_item["movie_id"]})
    data[user_col] = data[user_col].map(user_map)
    data[item_col] = data[item_col].map(item_map)
    user_pos_item = data.groupby(user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))

    print("compute topk metrics")
    out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=[topk])
    print(out)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data/ml-1m/ml-1m.csv")
    parser.add_argument('--model_name', default='dssm')
    parser.add_argument('--epoch', type=int, default=5)  #100x
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=2048)  #4096
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')  #cuda:0
    parser.add_argument('--save_dir', default='./data/ml-1m/saved/')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay,
         args.device, args.save_dir, args.seed)
"""
python run_movielens.py --model_name dssm
"""