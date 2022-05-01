import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from torch_ctr.basic.features import DenseFeature, SparseFeature, SequenceFeature


class CriteoDataset(Dataset):
    """
    Criteo Display Advertising Challenge Dataset

    :param dataset_path: criteo train.csv path.
    Reference:
        https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
        https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf
    """

    def __init__(self, dataset_path="./criteo_sample_50w.csv"):
        data = pd.read_csv(dataset_path)
        dense_features = [f for f in data.columns.tolist() if f[0] == "I"]
        sparse_features = [f for f in data.columns.tolist() if f[0] == "C"]

        data[sparse_features] = data[sparse_features].fillna('-10086',)  #一定要padding负数，Embedding层默认使用0作为padding值
        data[dense_features] = data[dense_features].fillna(0,)

        #数值特征离散化
        for fea in tqdm(dense_features):
            sparse_features.append(fea + "_cat")
            data[fea + "_cat"] = data[fea].apply(lambda x: self.convert_numeric_feature(x))

        # 数值特征标准化
        for fea in tqdm(dense_features):
            sca = MinMaxScaler()
            data[fea] = sca.fit_transform(data[fea])

        ## 类别特征labelencoder
        #sparse_features = dense_features + sparse_features  #离散化后，dense->sparse

        for feat in tqdm(sparse_features):
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])  #会先按从小到大排序，再编码

        #dense_features = []

        self.dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
        self.sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name in sparse_features]

        self.features = self.dense_feas + self.sparse_feas
        self.data = data
        self.length = data.shape[0]
        self.y = data["label"].values
        print(len(self.features))
        print(data.columns)

    def __getitem__(self, index):
        return self.data.loc[index].to_dict(), self.y[index]

    def __len__(self):
        return self.length

    def convert_numeric_feature(self, val):
        v = int(val)
        if v > 2:
            return int(np.log(v)**2)
        else:
            return v - 2
