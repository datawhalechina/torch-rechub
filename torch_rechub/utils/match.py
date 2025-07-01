import copy
import random
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import tqdm

from .data import df_to_dict, pad_sequences

# Optional imports with fallbacks
try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False

try:
    import torch
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def gen_model_input(df, user_profile, user_col, item_profile, item_col, seq_max_len, padding='pre', truncating='pre'):
    """Merge user_profile and item_profile to df, pad and truncate history sequence feature.

    Args:
        df (pd.DataFrame): data with history sequence feature
        user_profile (pd.DataFrame): user data
        user_col (str): user column name
        item_profile (pd.DataFrame): item data
        item_col (str): item column name
        seq_max_len (int): sequence length of every data
        padding (str, optional): padding style, {'pre', 'post'}. Defaults to 'pre'.
        truncating (str, optional): truncate style, {'pre', 'post'}. Defaults to 'pre'.

    Returns:
        dict: The converted dict, which can be used directly into the input network
    """
    df = pd.merge(df, user_profile, on=user_col, how='left')  # how=left to keep samples order same as the input
    df = pd.merge(df, item_profile, on=item_col, how='left')
    for col in df.columns.to_list():
        if col.startswith("hist_"):
            df[col] = pad_sequences(df[col], maxlen=seq_max_len, value=0, padding=padding, truncating=truncating).tolist()
    for col in df.columns.to_list():
        if col.startswith("tag_"):
            df[col] = pad_sequences(df[col], maxlen=seq_max_len, value=0, padding=padding, truncating=truncating).tolist()

    input_dict = df_to_dict(df)
    return input_dict


def negative_sample(items_cnt_order, ratio, method_id=0):
    """Negative Sample method for matching model.

    Reference: https://github.com/wangzhegeek/DSSM-Lookalike/blob/master/utils.py
    Updated with more methods and redesigned this function.

    Args:
        items_cnt_order (dict): the item count dict, the keys(item) sorted by value(count) in reverse order.
        ratio (int): negative sample ratio, >= 1
        method_id (int, optional):
        `{
            0: "random sampling",
            1: "popularity sampling method used in word2vec",
            2: "popularity sampling method by `log(count+1)+1e-6`",
            3: "tencent RALM sampling"}`.
            Defaults to 0.

    Returns:
        list: sampled negative item list
    """
    items_set = [item for item, count in items_cnt_order.items()]
    if method_id == 0:
        neg_items = np.random.choice(items_set, size=ratio, replace=True)
    elif method_id == 1:
        # items_cnt_freq = {item: count/len(items_cnt) for item, count in items_cnt_order.items()}
        # p_sel = {item: np.sqrt(1e-5/items_cnt_freq[item]) for item in items_cnt_order}
        # The most popular paramter is item_cnt**0.75:
        p_sel = {item: count**0.75 for item, count in items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=True, p=p_value)
    elif method_id == 2:
        p_sel = {item: np.log(count + 1) + 1e-6 for item, count in items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=True, p=p_value)
    elif method_id == 3:
        p_sel = {item: (np.log(k + 2) - np.log(k + 1)) / np.log(len(items_cnt_order) + 1) for item, k in items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=False, p=p_value)
    else:
        raise ValueError("method id should in (0,1,2,3)")
    return neg_items


def generate_seq_feature_match(data, user_col, item_col, time_col, item_attribute_cols=None, sample_method=0, mode=0, neg_ratio=0, min_item=0):
    """Generate sequence feature and negative sample for match.

    Args:
        data (pd.DataFrame): the raw data.
        user_col (str): the col name of user_id
        item_col (str): the col name of item_id
        time_col (str): the col name of timestamp
        item_attribute_cols (list[str], optional): the other attribute cols of item which you want to generate sequence feature. Defaults to `[]`.
        sample_method (int, optional): the negative sample method `{
            0: "random sampling",
            1: "popularity sampling method used in word2vec",
            2: "popularity sampling method by `log(count+1)+1e-6`",
            3: "tencent RALM sampling"}`.
            Defaults to 0.
        mode (int, optional): the training mode, `{0:point-wise, 1:pair-wise, 2:list-wise}`. Defaults to 0.
        neg_ratio (int, optional): negative sample ratio, >= 1. Defaults to 0.
        min_item (int, optional): the min item each user must have. Defaults to 0.

    Returns:
        pd.DataFrame: split train and test data with sequence features.
    """
    if item_attribute_cols is None:
        item_attribute_cols = []
    if mode == 2:  # list wise learning
        assert neg_ratio > 0, 'neg_ratio must be greater than 0 when list-wise learning'
    elif mode == 1:  # pair wise learning
        neg_ratio = 1
    print("preprocess data")
    data.sort_values(time_col, inplace=True)  # sort by time from old to new
    train_set, test_set = [], []
    n_cold_user = 0

    items_cnt = Counter(data[item_col].tolist())
    items_cnt_order = OrderedDict(sorted((items_cnt.items()), key=lambda x: x[1], reverse=True))  # item_id:item count
    neg_list = negative_sample(items_cnt_order, ratio=data.shape[0] * neg_ratio, method_id=sample_method)
    neg_idx = 0
    for uid, hist in tqdm.tqdm(data.groupby(user_col), desc='generate sequence features'):
        pos_list = hist[item_col].tolist()
        if len(pos_list) < min_item:  # drop this user when his pos items < min_item
            n_cold_user += 1
            continue

        for i in range(1, len(pos_list)):
            hist_item = pos_list[:i]
            sample = [uid, pos_list[i], hist_item, len(hist_item)]
            if len(item_attribute_cols) > 0:
                for attr_col in item_attribute_cols:  # the history of item attribute features
                    sample.append(hist[attr_col].tolist()[:i])
            if i != len(pos_list) - 1:
                if mode == 0:  # point-wise, the last col is label_col, include label 0 and 1
                    last_col = "label"
                    train_set.append(sample + [1])
                    for _ in range(neg_ratio):
                        sample[1] = neg_list[neg_idx]
                        neg_idx += 1
                        train_set.append(sample + [0])
                elif mode == 1:  # pair-wise, the last col is neg_col, include one negative item
                    last_col = "neg_items"
                    for _ in range(neg_ratio):
                        sample_copy = copy.deepcopy(sample)
                        sample_copy.append(neg_list[neg_idx])
                        neg_idx += 1
                        train_set.append(sample_copy)
                elif mode == 2:  # list-wise, the last col is neg_col, include neg_ratio negative items
                    last_col = "neg_items"
                    sample.append(neg_list[neg_idx:neg_idx + neg_ratio])
                    neg_idx += neg_ratio
                    train_set.append(sample)
                else:
                    raise ValueError("mode should in (0,1,2)")
            else:
                # Note: if mode=1 or 2, the label col is useless.
                test_set.append(sample + [1])

    random.shuffle(train_set)
    random.shuffle(test_set)

    print("n_train: %d, n_test: %d" % (len(train_set), len(test_set)))
    print("%d cold start user dropped " % n_cold_user)

    attr_hist_col = ["hist_" + col for col in item_attribute_cols]
    df_train = pd.DataFrame(train_set, columns=[user_col, item_col, "hist_" + item_col, "histlen_" + item_col] + attr_hist_col + [last_col])
    df_test = pd.DataFrame(test_set, columns=[user_col, item_col, "hist_" + item_col, "histlen_" + item_col] + attr_hist_col + [last_col])

    return df_train, df_test


class Annoy(object):
    """A vector matching engine using Annoy library"""

    def __init__(self, metric='angular', n_trees=10, search_k=-1):
        if not ANNOY_AVAILABLE:
            raise ImportError("Annoy is not available. To use Annoy engine, please install it first:\n"
                              "pip install annoy\n"
                              "Or use other available engines like Faiss or Milvus")
        self._n_trees = n_trees
        self._search_k = search_k
        self._metric = metric

    def fit(self, X):
        """Build the Annoy index from input vectors.

        Args:
            X (np.ndarray): input vectors with shape (n_samples, n_features)
        """
        self._annoy = AnnoyIndex(X.shape[1], metric=self._metric)
        for i, x in enumerate(X):
            self._annoy.add_item(i, x.tolist())
        self._annoy.build(self._n_trees)

    def set_query_arguments(self, search_k):
        """Set query parameters for searching.

        Args:
            search_k (int): number of nodes to inspect during searching
        """
        self._search_k = search_k

    def query(self, v, n):
        """Find the n nearest neighbors to vector v.

        Args:
            v (np.ndarray): query vector
            n (int): number of nearest neighbors to return

        Returns:
            tuple: (indices, distances) - lists of nearest neighbor indices and their distances
        """
        return self._annoy.get_nns_by_vector(v.tolist(), n, self._search_k, include_distances=True)

    def __str__(self):
        return 'Annoy(n_trees=%d, search_k=%d)' % (self._n_trees, self._search_k)


class Milvus(object):
    """A vector matching engine using Milvus database"""

    def __init__(self, dim=64, host="localhost", port="19530"):
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus is not available. To use Milvus engine, please install it first:\n"
                              "pip install pymilvus\n"
                              "Or use other available engines like Annoy or Faiss")
        self.dim = dim
        has = utility.has_collection("rechub")
        if has:
            utility.drop_collection("rechub")


# Create collection with schema definition
        fields = [
            FieldSchema(name="id",
                        dtype=DataType.INT64,
                        is_primary=True),
            FieldSchema(name="embeddings",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=dim),
        ]
        schema = CollectionSchema(fields=fields)
        self.milvus = Collection("rechub", schema=schema)

    def fit(self, X):
        """Insert vectors into Milvus collection and build index.

        Args:
            X (np.ndarray or torch.Tensor): input vectors with shape (n_samples, n_features)
        """
        if hasattr(X, 'cpu'):  # Handle PyTorch tensor
            X = X.cpu().numpy()
        self.milvus.release()
        entities = [[i for i in range(len(X))], X]
        self.milvus.insert(entities)
        print(f"Number of entities in Milvus: {self.milvus.num_entities}")

        # Create IVF_FLAT index for efficient search
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {
                "nlist": 128
            },
        }
        self.milvus.create_index("embeddings", index)

    @staticmethod
    def process_result(results):
        """Process Milvus search results into standard format.

        Args:
            results: raw search results from Milvus

        Returns:
            tuple: (indices_list, distances_list) - processed results
        """
        idx_list = []
        score_list = []
        for r in results:
            temp_idx_list = []
            temp_score_list = []
            for i in range(len(r)):
                temp_idx_list.append(r[i].id)
                temp_score_list.append(r[i].distance)
            idx_list.append(temp_idx_list)
            score_list.append(temp_score_list)
        return idx_list, score_list

    def query(self, v, n):
        """Query Milvus for the n nearest neighbors to vector v.

        Args:
            v (np.ndarray or torch.Tensor): query vector
            n (int): number of nearest neighbors to return

        Returns:
            tuple: (indices, distances) - lists of nearest neighbor indices and their distances
        """
        if torch.is_tensor(v):
            v = v.cpu().numpy()
        self.milvus.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
        results = self.milvus.search(v, "embeddings", search_params, n)
        return self.process_result(results)


class Faiss(object):
    """A vector matching engine using Faiss library"""

    def __init__(self, dim, index_type='flat', nlist=100, m=32, metric='l2'):
        self.dim = dim
        self.index_type = index_type.lower()
        self.nlist = nlist
        self.m = m
        self.metric = metric.lower()
        self.index = None
        self.is_trained = False

        # Create index based on different index types and metrics
        if self.metric == 'l2':
            if self.index_type == 'flat':
                self.index = faiss.IndexFlatL2(dim)
            elif self.index_type == 'ivf':
                quantizer = faiss.IndexFlatL2(dim)
                self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            elif self.index_type == 'hnsw':
                self.index = faiss.IndexHNSWFlat(dim, m)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
        elif self.metric == 'ip':
            if self.index_type == 'flat':
                self.index = faiss.IndexFlatIP(dim)
            elif self.index_type == 'ivf':
                quantizer = faiss.IndexFlatIP(dim)
                self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            elif self.index_type == 'hnsw':
                self.index = faiss.IndexHNSWFlat(dim, m)
                # HNSW defaults to L2, need to change to inner product
                self.index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def fit(self, X):
        """Train and build the index from input vectors.

        Args:
            X (np.ndarray): input vectors with shape (n_samples, dim)
        """

        # For index types that require training (like IVF), train first
        if self.index_type == 'ivf' and not self.is_trained:
            print(f"Training {self.index_type.upper()} index with {X.shape[0]} vectors...")
            self.index.train(X)
            self.is_trained = True

# Add vectors to the index
        print(f"Adding {X.shape[0]} vectors to index...")
        self.index.add(X)
        print(f"Index built successfully. Total vectors: {self.index.ntotal}")

    def query(self, v, n):
        """Query the nearest neighbors for given vector.

        Args:
            v (np.ndarray or torch.Tensor): query vector
            n (int): number of nearest neighbors to return

        Returns:
            tuple: (indices, distances) - lists of nearest neighbor indices and distances
        """
        if hasattr(v, 'cpu'):  # Handle PyTorch tensor
            v = v.cpu().numpy()

# Ensure query vector has correct shape
        if v.ndim == 1:
            v = v.reshape(1, -1)

        v = v.astype(np.float32)

        # Set search parameters for IVF index
        if self.index_type == 'ivf':
            # Set number of clusters to search
            nprobe = min(self.nlist, max(1, self.nlist // 4))
            self.index.nprobe = nprobe


# Execute search
        distances, indices = self.index.search(v, n)

        return indices.tolist(), distances.tolist()

    def set_query_arguments(self, nprobe=None, efSearch=None):
        """Set query parameters for search.

        Args:
            nprobe (int): number of clusters to search for IVF index
            efSearch (int): search parameter for HNSW index
        """
        if self.index_type == 'ivf' and nprobe is not None:
            self.index.nprobe = min(nprobe, self.nlist)
        elif self.index_type == 'hnsw' and efSearch is not None:
            self.index.hnsw.efSearch = efSearch

    def save_index(self, filepath):
        """Save index to file for later use."""
        faiss.write_index(self.index, filepath)

    def load_index(self, filepath):
        """Load index from file."""
        self.index = faiss.read_index(filepath)
        self.is_trained = True

    def __str__(self):
        return f'Faiss(index_type={self.index_type}, dim={self.dim}, metric={self.metric}, ntotal={self.index.ntotal if self.index else 0})'

if __name__ == '__main__':
    # Generate random item embeddings (100 items, each with 64 dimensions)
    item_embeddings = np.random.rand(100, 64).astype(np.float32)

    # Generate random user embedding (1 user, 64 dimensions)
    user_embedding = np.random.rand(1, 64).astype(np.float32)

    # Create FAISS index
    faiss_index = Faiss(dim=64, index_type='ivf', nlist=100, metric='l2')

    # Train and build the index
    faiss_index.fit(item_embeddings)

    # Query nearest neighbors
    indices, distances = faiss_index.query(user_embedding, n=10)

    print("Top 10 nearest neighbors:")
    print(indices)  # Output indices of nearest neighbors
    print(distances)  # Output distances of nearest neighbors
