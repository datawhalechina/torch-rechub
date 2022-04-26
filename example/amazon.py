
import numpy as np
from torch.utils.data import Dataset
import torch
import pickle


class AmazonDataset(Dataset):
    """
        Amazon 
    """
    def __init__(self, data_path="./amazon/data/amazon_dict.pkl"):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        sequence_field_dims = np.zeros(2, dtype=np.int32)
        for i in range(2):
            sequence_field_dims[i] = np.unique(data["candicate"][:,i]).size

        self.max_len = 50
        self.sparse_field_dims = sequence_field_dims
        self.sequence_field_dims = sequence_field_dims
        
        self.x_sparse = data["candicate"] #统一接口 先多train两个特征
        self.x_sequence = data["history"]   #batch_size,n_seq_fields,seq_len
        self.x_candidate = data["candicate"] #batch_size,n_seq_fields

        self.length = self.x_candidate.shape[0]
        self.y = data["label"]

    def __getitem__(self, index):
        x_dict = {
                  "x_sparse":torch.LongTensor(self.x_candidate[index]),
                  "x_sequence":torch.LongTensor(self.x_sequence[index]),
                  "x_candidate":torch.LongTensor(self.x_candidate[index])
                  }
        y = self.y[index]
        return x_dict, y

    def __len__(self):
        return self.length

