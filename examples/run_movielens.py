import sys

sys.path.append("../")
import random
import numpy as np
import pandas as pd
import collections
import torch

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.features import DenseFeature, SparseFeature, SequenceFeature
from torch.utils.data import DataLoader
from torch_rechub.basic.match_utils import full_predict, Annoy
from torch_rechub.basic.utils import pad_sequences, TorchDataset, df_to_input_dict
from torch_rechub.basic.metric import  topk_metrics

def preprocess_data(data, neg_ratio=0, min_item=3):
    print("preprocess data")
    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()
    
    movie_cate_map = dict(zip(data["movie_id"], data["cate_id"]))

    train_set = []
    test_set = []
    n_cold_user = 0
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        rating_list = hist['rating'].tolist()
        if len(pos_list)<min_item: #drop this user when his pos items < min_item
            n_cold_user += 1
            continue

        if neg_ratio > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set,size=len(pos_list)*neg_ratio,replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, movie_cate_map[pos_list[i]],len(hist[::-1]), rating_list[i]))
                for negi in range(neg_ratio):
                    neg_item = neg_list[i*neg_ratio+negi]
                    train_set.append((reviewerID, hist[::-1], neg_item, 0, movie_cate_map[neg_item],len(hist[::-1])))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i],1, movie_cate_map[pos_list[i]], len(hist[::-1]), rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print("n_train: %d, n_test: %d" %(len(train_set),len(test_set)))
    print("%d cold start user droped "%(n_cold_user))
    return train_set,test_set

def gen_model_input(train_set,user_profile,seq_max_len):
    
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_cid = np.array([line[4] for line in train_set])
    train_hist_len = np.array([line[5] for line in train_set])
    

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len, "cate_id":train_cid}

    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label


def get_movielens_data(data_path, load_cache=False):
    data = pd.read_csv(data_path)
    print(data.shape)
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
    user_profile.set_index("user_id", inplace=True)
    
    item_profile = data[["movie_id", "cate_id"]].drop_duplicates('movie_id')
    #user_item_list = data.groupby("user_id")['movie_id'].apply(list)
    if load_cache:
        x_train, y_train, x_test, y_test = np.load("./data/ml-1m/saved/data_preprocess.npy",allow_pickle=True)
    else:
        train_set, test_set = preprocess_data(data, neg_ratio=5, min_item=0)
        x_train, y_train = gen_model_input(train_set, user_profile, seq_max_len=50)
        x_test, y_test = gen_model_input(test_set, user_profile, seq_max_len=50)
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
    torch.manual_seed(seed)
    user_features, item_features, x_train, y_train, all_item_model_input, test_user_model_input = get_movielens_data(dataset_path)
    dataset = TorchDataset(x_train, y_train)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    if model_name == "dssm":
        model = DSSM(user_features, item_features, sim_func="cosine", temperature=0.01, user_params={"dims":[128,64,32],"output_layer":False}, item_params={"dims":[128,64,32],"output_layer":False})

    ctr_trainer = CTRTrainer(model, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, 
                             earlystop_patience=5, device=device, model_path=save_dir,
                             scheduler_fn=torch.optim.lr_scheduler.StepLR,
                             scheduler_params={"step_size": 2,"gamma": 0.8})

    ctr_trainer.fit(train_dataloader)
    
    print("inference embedding")
    model.mode = "user_tower"
    user_embedding = full_predict(ctr_trainer.model, test_user_model_input,  device)
    
    model.mode = "item_tower"
    item_embedding = full_predict(ctr_trainer.model, all_item_model_input,  device)
    
    torch.save(user_embedding.data.cpu(), save_dir + "user_embedding.pth")
    torch.save(item_embedding.data.cpu(), save_dir + "item_embedding.pth")
    
    print("evaluate embedding matching on test data")
    annoy = Annoy(n_trees=10)
    annoy.fit(item_embedding)

    #for each user of test dataset, get ann search topk result
    print("matching for topk")
    user_map, item_map = np.load("./data/ml-1m/saved/raw_id_maps.npy", allow_pickle=True)
    match_res = collections.defaultdict(dict)
    topk = 10
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