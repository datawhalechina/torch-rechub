import torch
import numpy as np
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

    :param num_classes: number of classes in the category
    :return: the dim of embedding vector
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