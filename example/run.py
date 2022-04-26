import sys
sys.path.append("../")

import torch
from torch.utils.data import DataLoader

from taobao import TaobaoDataset
from criteo import CriteoDataset
from amazon import AmazonDataset

from torch_ctr.models import WideAndDeep, DeepFM, DIN
from torch_ctr.learner import CTRLearner

def get_model_data(model_name, dataset_path, dataset_name, emb_dim):
    if dataset_name == "criteo":
        dataset = CriteoDataset(dataset_path)
    elif dataset_name == "taobao":
        dataset = TaobaoDataset(dataset_path)
    elif dataset_name == "amazon":
        dataset = AmazonDataset(dataset_path)
    
    if model_name == "widedeep":
        model = WideAndDeep(dataset.dense_field_nums, dataset.sparse_field_dims, embed_dim=emb_dim, mlp_dims=(256, 128), dropout=0.2)
    elif model_name == "deepfm":
        model = DeepFM(dataset.dense_field_nums, dataset.sparse_field_dims, embed_dim=emb_dim, mlp_dims=(256, 128), dropout=0.2)
    elif model_name == "din":
        model = DIN(dataset.sequence_field_dims, dataset.sparse_field_dims, embed_dim=emb_dim, mlp_dims=[200, 80], activation="dice",dropout=0.2)
    return model, dataset

def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         emb_dim,
         weight_decay,
         device,
         save_dir, 
         seed):
    torch.manual_seed(seed)
    device = torch.device(device)

    model, dataset = get_model_data(model_name, dataset_path, dataset_name, emb_dim)

    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    
    ctr_learner = CTRLearner(model, learning_rate=learning_rate, weight_decay=weight_decay, n_epoch=epoch,
                            earlystop_patience=10, device=device, model_path=save_dir)
    ctr_learner.train(train_data_loader, valid_data_loader)
    auc = ctr_learner.evaluate(model, test_data_loader)
    print(f'test auc: {auc}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', default='./criteo/data/criteo_sample_50w.csv')
    parser.add_argument('--model_name', default='din')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--emb_dim', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./model.pth')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.emb_dim,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.seed)

"""
调用参考：
python run.py --dataset_name criteo --model_name deepfm --dataset_path ./criteo/data/criteo_sample_50w.csv --emb_dim 16 --save_dir ./criteo/saved_model/model.pth --learning_rate 0.001
ython run.py --dataset_name amazon --model_name  --dataset_path ./amazon/data/amazon_dict.pkl --emb_dim 16 --save_dir ./amazon/saved_model/model.pth --learning_rate 0.001   

"""