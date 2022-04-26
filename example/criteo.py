
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import torch

class CriteoDataset(Dataset):
    """
    Criteo Display Advertising Challenge Dataset

    :param dataset_path: criteo train.csv path.
    Reference:
        https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
        https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf
    """

    def __init__(self, dataset_path="./criteo_sample_50w.csv"):
        # if data_mode=="full":
        #     sparse_features = ['C' + str(i) for i in range(1, 27)]
        #     dense_features = ['I' + str(i) for i in range(1, 14)]
        #     target_columns = ['label']
        #     columns = target_columns + dense_features + sparse_features
        #     data = pd.read_csv(dataset_path, sep='\t', names = columns)
        # elif data_mode=="mini":
        data = pd.read_csv(dataset_path)
        dense_features = [f for f in data.columns.tolist() if f[0] == "I"]
        sparse_features = [f for f in data.columns.tolist() if f[0] == "C"]
        
        data[sparse_features] = data[sparse_features].fillna('-10086', ) #一定要padding负数，Embedding层默认使用0作为padding值
        data[dense_features] = data[dense_features].fillna(0, )


        # ## 数值特征离散化
        for feat in tqdm(dense_features):
            data[feat] = data[feat].apply(lambda x:self.convert_numeric_feature(x))
        
        ## 数值特征标准化
        # for feat in tqdm(dense_features):
        #     mean = data[feat].mean()
        #     std = data[feat].std()
        #     data[feat] = (data[feat] - mean) / (std + 1e-12)   # 防止除零

        ## 类别特征labelencoder
        sparse_features = sparse_features + dense_features
        for feat in tqdm(sparse_features):
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat]) #会先按从小到大排序，再编码

        #features = dense_features + sparse_features
        field_dims = np.zeros(len(sparse_features), dtype=np.int32)
        for i, fea in enumerate(sparse_features):
           field_dims[i] = data[fea].nunique()
        
        self.sparse_field_dims = field_dims #每个sparse特征的unique值个数 list
        self.dense_field_nums = len(dense_features)
        self.length = data.shape[0]
        
        self.x_dense = data[dense_features].values 
        self.x_sparse = data[sparse_features].values 
        #self.x_sequence = data[features].values 

        self.y = data["label"].values

    def __getitem__(self, index):
        x_dict = {
                  "x_dense":torch.FloatTensor(self.x_dense[index]),
                  "x_sparse":torch.LongTensor(self.x_sparse[index]),
                  }
        return x_dict, self.y[index]

    def __len__(self):
        return self.length

    def convert_numeric_feature(self, val):
        v = int(val)
        if v > 2:
            return int(np.log(v) ** 2)
        else:
            return v - 2
