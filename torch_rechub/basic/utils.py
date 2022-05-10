import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, mean_squared_error


class TorchDataset(Dataset):

    def __init__(self, x, y):
        super(TorchDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}, self.y[index]

    def __len__(self):
        return len(self.y)


class DataGenerator(object):

    def __init__(self, x, y):
        super(DataGenerator, self).__init__()
        self.dataset = TorchDataset(x, y)
        self.length = len(self.dataset)

    def generate_dataloader(self, x_val=None, y_val=None, x_test=None, y_test=None, split_ratio=None, batch_size=16, num_workers=8):
        if split_ratio != None:
            train_length = int(self.length * split_ratio[0])
            val_length = int(self.length * split_ratio[1])
            test_length = self.length - train_length - val_length
            print("the samples of train : val : test are  %d : %d : %d" % (train_length, val_length, test_length))
            train_dataset, val_dataset, test_dataset = random_split(self.dataset, (train_length, val_length, test_length))
        else:
            train_dataset = self.dataset
            val_dataset = TorchDataset(x_val, y_val)
            test_dataset = TorchDataset(x_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
        return train_dataloader, val_dataloader, test_dataloader


class PredictDataset(Dataset):

    def __init__(self, x):
        super(TorchDataset, self).__init__()
        self.x = x

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}

    def __len__(self):
        return len(self.x[list(self.x.keys())[0]])


def get_auto_embedding_dim(num_classes):
    """ Calculate the dim of embedding vector according to number of classes in the category
    emb_dim = [6 * (num_classes)^(1/4)]
    reference: Deep & Cross Network for Ad Click Predictions.(ADKDD'17)
    
    Args:
        num_classes: number of classes in the category
    
    Returns:
        the dim of embedding vector
    """
    return np.floor(6 * np.pow(num_classes, 0.26))


def get_loss_func(task_type="classification"):
    if task_type == "classification":
        return torch.nn.BCELoss()
    elif task_type == "regression":
        return torch.nn.MSELoss()
    else:
        raise ValueError("task_type must be classification or regression")


def get_metric_func(task_type="classification"):
    if task_type == "classification":
        return roc_auc_score
    elif task_type == "regression":
        return mean_squared_error
    else:
        raise ValueError("task_type must be classification or regression")


def create_seq_features(data, max_len=50, drop_short=3, shuffle=True):
    """Build a sequence of user's history by time.
    
    Args:
        data (pd.DataFrame): must contain keys: `user_id, item_id, cate_id, time`.
        max_len (int): the max length of a user history sequence.
        drop_short (int): remove some inactive user who's sequence length < drop_short.
        shuffle (bool): shuffle data if true.
    
    Returns: 
        train (pd.DataFrame): target item will be each item before last two items.
        val (pd.DataFrame): target item is the second to last item of user's history sequence.
        test (pd.DataFrame): target item is the last item of user's history sequence.
    """
    n_users, n_items, n_cates = data["user_id"].max(), data["item_id"].max(), data["cate_id"].max()
    # 0 to be used as the symbol for padding
    data = data.astype('int32')
    data['item_id'] = data['item_id'].apply(lambda x: x + 1)
    data['cate_id'] = data['cate_id'].apply(lambda x: x + 1)

    item_cate_map = data[['item_id', 'cate_id']]
    item2cate_dict = item_cate_map.set_index(['item_id'])['cate_id'].to_dict()

    data = data.sort_values(['user_id', 'time']).groupby('user_id').agg(click_hist_list=('item_id', list), cate_hist_hist=('cate_id', list)).reset_index()

    # Sliding window to construct negative samples
    train_data, val_data, test_data = [], [], []
    for item in data.itertuples():
        if len(item[2]) < drop_short:
            continue
        click_hist_list = item[2][:max_len]
        cate_hist_list = item[3][:max_len]

        def neg_sample():
            neg = click_hist_list[0]
            while neg in click_hist_list:
                neg = random.randint(1, n_items)
            return neg

        neg_list = [neg_sample() for _ in range(len(click_hist_list))]
        hist_list = []
        cate_list = []
        for i in range(1, len(click_hist_list)):
            hist_list.append(click_hist_list[i - 1])
            cate_list.append(cate_hist_list[i - 1])
            hist_list_pad = hist_list + [0] * (max_len - len(hist_list))
            cate_list_pad = cate_list + [0] * (max_len - len(cate_list))
            if i == len(click_hist_list) - 1:
                test_data.append([hist_list_pad, cate_list_pad, click_hist_list[i], cate_hist_list[i], 1])
                test_data.append([hist_list_pad, cate_list_pad, neg_list[i], item2cate_dict[neg_list[i]], 0])
            if i == len(click_hist_list) - 2:
                val_data.append([hist_list_pad, cate_list_pad, click_hist_list[i], cate_hist_list[i], 1])
                val_data.append([hist_list_pad, cate_list_pad, neg_list[i], item2cate_dict[neg_list[i]], 0])
            else:
                train_data.append([hist_list_pad, cate_list_pad, click_hist_list[i], cate_hist_list[i], 1])
                train_data.append([hist_list_pad, cate_list_pad, neg_list[i], item2cate_dict[neg_list[i]], 0])

    # shuffle
    if shuffle:
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

    col_name = ['history_item', 'history_cate', 'target_item', 'target_cate', 'label']
    train = pd.DataFrame(train_data, columns=col_name)
    val = pd.DataFrame(val_data, columns=col_name)
    test = pd.DataFrame(test_data, columns=col_name)

    return train, val, test


def df_to_input_dict(data):
    data_dict = data.to_dict('list')
    for key in data.keys():
        data_dict[key] = np.array(data_dict[key])
    return data_dict