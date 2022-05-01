import numpy as np
from torch.utils.data import Dataset
import pickle

from torch_ctr.basic.features import DenseFeature, SparseFeature, SequenceFeature


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


class AmazonDataset(Dataset):
    """
        Amazon 
    """

    def __init__(self, data_path="./amazon/data/amazon_dict.pkl"):
        with open(data_path, "rb") as f:
            data = pickle.load(f)  #five key: history_item, history_cate,target_item,target_cate,label
        n_item = len(np.unique(data["target_item"]))
        n_cate = len(np.unique(data["target_cate"]))

        self.features = [SparseFeature("target_item", vocab_size=n_item, embed_dim=8), SparseFeature("target_cate", vocab_size=n_cate, embed_dim=8)]
        self.target_features = self.features
        self.history_features = [
            SequenceFeature("history_item", vocab_size=n_item, embed_dim=8, pooling="concat", shared_with="target_item"),
            SequenceFeature("history_cate", vocab_size=n_cate, embed_dim=8, pooling="concat", shared_with="target_cate")
        ]

        self.length = len(data["target_item"])
        self.y = data["label"]
        del data["label"]
        self.data = data

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data.items()}, self.y[index]

    def __len__(self):
        return self.length
