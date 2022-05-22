import sys

sys.path.append("../")

import os
import random
import numpy as np
import pandas as pd
import collections
import torch
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from torch_rechub.models.ranking import DeepFM
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.basic.features import DenseFeature, SparseFeature, SequenceFeature
from torch_rechub.basic.match_utils import Annoy, generate_seq_feature_match
from torch_rechub.basic.utils import PredictDataset, pad_sequences, TorchDataset, df_to_input_dict
from torch_rechub.basic.metric import  topk_metrics
    
def gen_model_input(df, user_profile, user_col, item_profile, item_col, seq_max_len):
    df = pd.merge(df, user_profile, on=user_col)
    df = pd.merge(df, item_profile, on=item_col)
    for col in df.columns.to_list():
        if col.startswith("hist_"):
            df[col] = pad_sequences(df[col], maxlen=seq_max_len, value=0).tolist()
    input_dict = df_to_input_dict(df)
    return input_dict


def get_movielens_data(data_path, load_cache=False):
    data = pd.read_csv(data_path)[:100000]
    data["cate_id"] = data["genres"].apply(lambda x:x.split("|")[0])
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip',"cate_id"]
    user_col, item_col, label_col = "user_id", "movie_id", "label"

    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        if feature == user_col:
            user_map = {encode_id+1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)} #encode user id: raw user id
        if feature == item_col:
            item_map = {encode_id+1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)} #encode item id: raw item id
    np.save("./data/ml-1m/saved/raw_id_maps.npy", (user_map, item_map))
    
    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
    item_profile = data[["movie_id", "cate_id"]].drop_duplicates('movie_id')
    
    
    if load_cache: #if you have run this script before and saved the preprocessed data
        x_train, y_train, x_test, y_test = np.load("./data/ml-1m/saved/data_preprocess.npy",allow_pickle=True)
    else:
        df_train, df_test = generate_seq_feature_match(data, user_col, item_col, time_col="timestamp", item_attribute_cols=[], sample_method=0, mode=0, neg_ratio=3, min_item=0)
        x_train =  gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50)
        y_train = df_train["label"].values
        x_test =  gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50)
        y_test = df_test["label"].values
        np.save("./data/ml-1m/saved/data_preprocess.npy", (x_train, y_train, x_test, y_test)) 
    
    user_cols = ['user_id', 'gender', 'age', 'occupation', 'zip']
    item_cols = ['movie_id',"cate_id"]
    
    user_features = [SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name], embed_dim=8) for feature_name in user_cols]
    user_features += [SequenceFeature("hist_movie_id", vocab_size=feature_max_idx["movie_id"], embed_dim=8, pooling="mean", shared_with="movie_id")]
    
    item_features = [SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name], embed_dim=8) for feature_name in item_cols]

    all_item_model_input = df_to_input_dict(item_profile)
    test_user_model_input = x_test
    return user_features, item_features, x_train, y_train, all_item_model_input, test_user_model_input



def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.manual_seed(seed)
    user_features, item_features, x_train, y_train, all_item_model_input, test_user_model_input = get_movielens_data(dataset_path)
    dataset = TorchDataset(x_train, y_train)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    if model_name == "dssm":
        model = DSSM(user_features, item_features, sim_func="cosine", temperature=0.01, user_params={"dims":[256, 128,64,16],"output_layer":False}, item_params={"dims":[256, 128,64,16],"output_layer":False})

    trainer = MatchTrainer(model, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, 
                             earlystop_patience=5, device=device, model_path=save_dir,
                             scheduler_fn=torch.optim.lr_scheduler.StepLR,
                             scheduler_params={"step_size": 2,"gamma": 0.8})

    trainer.fit(train_dataloader)
    
    print("inference embedding")
    test_dl = DataLoader(PredictDataset(test_user_model_input), batch_size=batch_size, shuffle=True, num_workers=8)
    user_embedding = trainer.inference_embedding(model=model, mode="user_tower", data_loader=test_dl, model_path=save_dir)
    
    all_item_dl = DataLoader(PredictDataset(all_item_model_input), batch_size=batch_size, shuffle=True, num_workers=8)
    item_embedding = trainer.inference_embedding(model=model, mode="item_tower", data_loader=all_item_dl, model_path=save_dir)

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
    for user_id, user_emb in zip(test_user_model_input["user_id"], user_embedding):
        items_idx, items_scores = annoy.query(v=user_emb, n=topk) #the index of topk match items
        match_res[user_map[user_id]] = all_item_model_input["movie_id"][items_idx]
    
    #get ground truth
    print("generate ground truth")
    user_col = "user_id"
    item_col = "movie_id"

    data = pd.DataFrame({"user_id":test_user_model_input["user_id"],"movie_id":test_user_model_input["movie_id"]})
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
    parser.add_argument('--dataset_path', default="./data/ml-1m/movielens_sample.csv")
    parser.add_argument('--model_name', default='dssm')
    parser.add_argument('--epoch', type=int, default=1)  #100
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=2048)  #4096
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cpu')  #cuda:0
    parser.add_argument('--save_dir', default='./data/ml-1m/saved/')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed)
"""
python run_movielens.py --model_name dssm
"""