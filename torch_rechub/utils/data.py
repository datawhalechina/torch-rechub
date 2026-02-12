import os
import json
import random

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, random_split


class TorchDataset(Dataset):

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}, self.y[index]

    def __len__(self):
        return len(self.y)


class PredictDataset(Dataset):

    def __init__(self, x):
        super().__init__()
        self.x = x

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}

    def __len__(self):
        return len(self.x[list(self.x.keys())[0]])


class MatchDataGenerator(object):

    def __init__(self, x, y=[]):
        super().__init__()
        if len(y) != 0:
            self.dataset = TorchDataset(x, y)
        else:  # For pair-wise model, trained without given label
            self.dataset = PredictDataset(x)

    def generate_dataloader(self, x_test_user, x_all_item, batch_size, num_workers=8):
        train_dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = PredictDataset(x_test_user)

        # shuffle = False to keep same order as ground truth
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        item_dataset = PredictDataset(x_all_item)
        item_dataloader = DataLoader(item_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dataloader, test_dataloader, item_dataloader


class DataGenerator(object):

    def __init__(self, x, y):
        super().__init__()
        self.dataset = TorchDataset(x, y)
        self.length = len(self.dataset)

    def generate_dataloader(self, x_val=None, y_val=None, x_test=None, y_test=None, split_ratio=None, batch_size=16, num_workers=0):
        if split_ratio is not None:
            train_length = int(self.length * split_ratio[0])
            val_length = int(self.length * split_ratio[1])
            test_length = self.length - train_length - val_length
            print("the samples of train : val : test are  %d : %d : %d" % (train_length, val_length, test_length))
            train_dataset, val_dataset, test_dataset = random_split(self.dataset, (train_length, val_length, test_length))
        else:
            train_dataset = self.dataset
            val_dataset = TorchDataset(x_val, y_val)
            test_dataset = TorchDataset(x_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dataloader, val_dataloader, test_dataloader


def get_auto_embedding_dim(num_classes):
    """Calculate embedding dim by category size.

    Uses ``emb_dim = floor(6 * num_classes**0.25)`` from DCN (ADKDD'17).

    Parameters
    ----------
    num_classes : int
        Number of categorical classes.

    Returns
    -------
    int
        Recommended embedding dimension.
    """
    return int(np.floor(6 * np.pow(num_classes, 0.25)))


def get_loss_func(task_type="classification"):
    """Return default loss by task type."""
    if task_type == "classification":
        return torch.nn.BCELoss()
    if task_type == "regression":
        return torch.nn.MSELoss()
    raise ValueError("task_type must be classification or regression")


def get_metric_func(task_type="classification"):
    """Return default metric by task type."""
    if task_type == "classification":
        return roc_auc_score
    if task_type == "regression":
        return mean_squared_error
    raise ValueError("task_type must be classification or regression")


def generate_seq_feature(data, user_col, item_col, time_col, item_attribute_cols=[], min_item=0, shuffle=True, max_len=50):
    """Generate sequence features and negatives for ranking.

    Parameters
    ----------
    data : pd.DataFrame
        Raw interaction data.
    user_col : str
        User id column name.
    item_col : str
        Item id column name.
    time_col : str
        Timestamp column name.
    item_attribute_cols : list[str], optional
        Additional item attribute columns to include in sequences.
    min_item : int, default=0
        Minimum items per user; users below are dropped.
    shuffle : bool, default=True
        Shuffle train/val/test.
    max_len : int, default=50
        Max history length.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test data with sequence features.
    """
    for feat in data:
        le = LabelEncoder()
        data[feat] = le.fit_transform(data[feat])
        # 0 to be used as the symbol for padding
        data[feat] = data[feat].apply(lambda x: x + 1)
    data = data.astype('int32')

    # generate item to attribute mapping
    n_items = data[item_col].max()
    item2attr = {}
    if len(item_attribute_cols) > 0:
        for col in item_attribute_cols:
            map = data[[item_col, col]]
            item2attr[col] = map.set_index([item_col])[col].to_dict()

    train_data, val_data, test_data = [], [], []
    data.sort_values(time_col, inplace=True)
    # Sliding window to construct negative samples
    for uid, hist in tqdm.tqdm(data.groupby(user_col), desc='generate sequence features'):
        pos_list = hist[item_col].tolist()
        len_pos_list = len(pos_list)
        if len_pos_list < min_item:  # drop this user when his pos items < min_item
            continue

        neg_list = [neg_sample(pos_list, n_items) for _ in range(len_pos_list)]
        for i in range(1, min(len_pos_list, max_len)):
            hist_item = pos_list[:i]
            hist_item = hist_item + [0] * (max_len - len(hist_item))
            pos_item = pos_list[i]
            neg_item = neg_list[i]
            pos_seq = [1, pos_item, uid, hist_item]
            neg_seq = [0, neg_item, uid, hist_item]
            if len(item_attribute_cols) > 0:
                for attr_col in item_attribute_cols:  # the history of item attribute features
                    hist_attr = hist[attr_col].tolist()[:i]
                    hist_attr = hist_attr + [0] * (max_len - len(hist_attr))
                    pos2attr = [hist_attr, item2attr[attr_col][pos_item]]
                    neg2attr = [hist_attr, item2attr[attr_col][neg_item]]
                    pos_seq += pos2attr
                    neg_seq += neg2attr
            if i == len_pos_list - 1:
                test_data.append(pos_seq)
                test_data.append(neg_seq)
            elif i == len_pos_list - 2:
                val_data.append(pos_seq)
                val_data.append(neg_seq)
            else:
                train_data.append(pos_seq)
                train_data.append(neg_seq)

    col_name = ['label', 'target_item_id', user_col, 'hist_item_id']
    if len(item_attribute_cols) > 0:
        for attr_col in item_attribute_cols:  # the history of item attribute features
            name = ['hist_' + attr_col, 'target_' + attr_col]
            col_name += name


# shuffle
    if shuffle:
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

    train = pd.DataFrame(train_data, columns=col_name)
    val = pd.DataFrame(val_data, columns=col_name)
    test = pd.DataFrame(test_data, columns=col_name)

    return train, val, test


def df_to_dict(data):
    """Convert DataFrame to dict inputs accepted by models.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.

    Returns
    -------
    dict
        Mapping of column name to numpy array.
    """
    data_dict = data.to_dict('list')
    for key in data.keys():
        data_dict[key] = np.array(data_dict[key])
    return data_dict


def neg_sample(click_hist, item_size):
    neg = random.randint(1, item_size)
    while neg in click_hist:
        neg = random.randint(1, item_size)
    return neg


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """Pad list-of-lists sequences to equal length.

    Equivalent to ``tf.keras.preprocessing.sequence.pad_sequences``.

    Parameters
    ----------
    sequences : Sequence[Sequence]
        Input sequences.
    maxlen : int, optional
        Maximum length; computed if None.
    dtype : str, default='int32'
    padding : {'pre', 'post'}, default='pre'
        Padding direction.
    truncating : {'pre', 'post'}, default='pre'
        Truncation direction.
    value : float, default=0.0
        Padding value.

    Returns
    -------
    np.ndarray
        Padded array of shape (n_samples, maxlen).
    """

    assert padding in ["pre", "post"], "Invalid padding={}.".format(padding)
    assert truncating in ["pre", "post"], "Invalid truncating={}.".format(truncating)

    if maxlen is None:
        maxlen = max(len(x) for x in sequences)
    arr = np.full((len(sequences), maxlen), value, dtype=dtype)
    for idx, x in enumerate(sequences):
        if len(x) == 0:
            continue  # empty list
        if truncating == 'pre':
            trunc = x[-maxlen:]
        else:
            trunc = x[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'pre':
            arr[idx, -len(trunc):] = trunc
        else:
            arr[idx, :len(trunc)] = trunc
    return arr


def array_replace_with_dict(array, dic):
    """Replace values in numpy array using a mapping dict.

    Parameters
    ----------
    array : np.ndarray
        Input array.
    dic : dict
        Mapping from old to new values.

    Returns
    -------
    np.ndarray
        Array with values replaced.
    """
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    idx = k.argsort()
    return v[idx[np.searchsorted(k, array, sorter=idx)]]


# Temporarily reserved for testing purposes(1985312383@qq.com)
def create_seq_features(data, seq_feature_col=['item_id', 'cate_id'], max_len=50, drop_short=3, shuffle=True):
    """Build user history sequences by time.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain ``user_id, item_id, cate_id, time``.
    seq_feature_col : list, default ['item_id', 'cate_id']
        Columns to generate sequence features.
    max_len : int, default=50
        Max history length.
    drop_short : int, default=3
        Drop users with sequence length < drop_short.
    shuffle : bool, default=True
        Shuffle outputs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train/val/test splits with sequence features.
    """
    for feat in data:
        le = LabelEncoder()
        data[feat] = le.fit_transform(data[feat])
        # 0 to be used as the symbol for padding
        data[feat] = data[feat].apply(lambda x: x + 1)
    data = data.astype('int32')

    n_items = data["item_id"].max()

    item_cate_map = data[['item_id', 'cate_id']]
    item2cate_dict = item_cate_map.set_index(['item_id'])['cate_id'].to_dict()

    data = data.sort_values(['user_id', 'time']).groupby('user_id').agg(click_hist_list=('item_id', list), cate_hist_hist=('cate_id', list)).reset_index()

    # Sliding window to construct negative samples
    train_data, val_data, test_data = [], [], []
    for item in data.itertuples():
        if len(item[2]) < drop_short:
            continue
        user_id = item[1]
        click_hist_list = item[2][:max_len]
        cate_hist_list = item[3][:max_len]

        neg_list = [neg_sample(click_hist_list, n_items) for _ in range(len(click_hist_list))]
        hist_list = []
        cate_list = []
        for i in range(1, len(click_hist_list)):
            hist_list.append(click_hist_list[i - 1])
            cate_list.append(cate_hist_list[i - 1])
            hist_list_pad = hist_list + [0] * (max_len - len(hist_list))
            cate_list_pad = cate_list + [0] * (max_len - len(cate_list))
            if i == len(click_hist_list) - 1:
                test_data.append([user_id, hist_list_pad, cate_list_pad, click_hist_list[i], cate_hist_list[i], 1])
                test_data.append([user_id, hist_list_pad, cate_list_pad, neg_list[i], item2cate_dict[neg_list[i]], 0])
            if i == len(click_hist_list) - 2:
                val_data.append([user_id, hist_list_pad, cate_list_pad, click_hist_list[i], cate_hist_list[i], 1])
                val_data.append([user_id, hist_list_pad, cate_list_pad, neg_list[i], item2cate_dict[neg_list[i]], 0])
            else:
                train_data.append([user_id, hist_list_pad, cate_list_pad, click_hist_list[i], cate_hist_list[i], 1])
                train_data.append([user_id, hist_list_pad, cate_list_pad, neg_list[i], item2cate_dict[neg_list[i]], 0])

# shuffle
    if shuffle:
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

    col_name = ['user_id', 'history_item', 'history_cate', 'target_item', 'target_cate', 'label']
    train = pd.DataFrame(train_data, columns=col_name)
    val = pd.DataFrame(val_data, columns=col_name)
    test = pd.DataFrame(test_data, columns=col_name)

    return train, val, test


# ============ Sequence Data Classes (新增) ============


class SeqDataset(Dataset):
    """Sequence dataset for HSTU-style next-item prediction.

    Parameters
    ----------
    seq_tokens : np.ndarray
        Token ids, shape ``(num_samples, seq_len)``.
    seq_positions : np.ndarray
        Position indices, shape ``(num_samples, seq_len)``.
    targets : np.ndarray
        Target token ids, shape ``(num_samples,)``.
    seq_time_diffs : np.ndarray
        Time-difference features, shape ``(num_samples, seq_len)``.

    Shape
    -----
    Output tuple: ``(seq_tokens, seq_positions, seq_time_diffs, target)``

    Examples
    --------
    >>> seq_tokens = np.random.randint(0, 1000, (100, 256))
    >>> seq_positions = np.arange(256)[np.newaxis, :].repeat(100, axis=0)
    >>> seq_time_diffs = np.random.randint(0, 86400, (100, 256))
    >>> targets = np.random.randint(0, 1000, (100,))
    >>> dataset = SeqDataset(seq_tokens, seq_positions, targets, seq_time_diffs)
    >>> len(dataset)
    100
    """

    def __init__(self, seq_tokens, seq_positions, targets, seq_time_diffs):
        super().__init__()
        self.seq_tokens = seq_tokens
        self.seq_positions = seq_positions
        self.targets = targets
        self.seq_time_diffs = seq_time_diffs

        # Validate basic shape consistency
        assert len(seq_tokens) == len(targets), "seq_tokens and targets must have same length"
        assert len(seq_tokens) == len(seq_positions), "seq_tokens and seq_positions must have same length"
        assert len(seq_tokens) == len(seq_time_diffs), "seq_tokens and seq_time_diffs must have same length"
        assert seq_tokens.shape[1] == seq_positions.shape[1], "seq_tokens and seq_positions must have same seq_len"
        assert seq_tokens.shape[1] == seq_time_diffs.shape[1], "seq_tokens and seq_time_diffs must have same seq_len"

    def __getitem__(self, index):
        """Return a single sample.

        Args:
            index (int): Sample index.

        Returns:
            tuple: ``(seq_tokens, seq_positions, seq_time_diffs, target)``.
        """
        return (torch.LongTensor(self.seq_tokens[index]), torch.LongTensor(self.seq_positions[index]), torch.LongTensor(self.seq_time_diffs[index]), torch.LongTensor([self.targets[index]]))

    def __len__(self):
        """Return the dataset size."""
        return len(self.targets)


class SequenceDataGenerator(object):
    """Sequence data generator for HSTU-style models.

    Wraps :class:`SeqDataset` and builds train/val/test loaders.

    Parameters
    ----------
    seq_tokens : np.ndarray
        Token ids, shape ``(num_samples, seq_len)``.
    seq_positions : np.ndarray
        Position indices, shape ``(num_samples, seq_len)``.
    targets : np.ndarray
        Target token ids, shape ``(num_samples,)``.
    seq_time_diffs : np.ndarray
        Time-difference features, shape ``(num_samples, seq_len)``.

    Examples
    --------
    >>> gen = SequenceDataGenerator(seq_tokens, seq_positions, targets, seq_time_diffs)
    >>> train_loader, val_loader, test_loader = gen.generate_dataloader(batch_size=32)
    """

    def __init__(self, seq_tokens, seq_positions, targets, seq_time_diffs):
        super().__init__()
        self.seq_tokens = seq_tokens
        self.seq_positions = seq_positions
        self.targets = targets
        self.seq_time_diffs = seq_time_diffs

        # Underlying dataset
        self.dataset = SeqDataset(seq_tokens, seq_positions, targets, seq_time_diffs)

    def generate_dataloader(self, batch_size=32, num_workers=0, split_ratio=None):
        """Generate train/val/test dataloaders.

        Parameters
        ----------
        batch_size : int, default=32
        num_workers : int, default=0
        split_ratio : tuple, default (0.7, 0.1, 0.2)
            Train/val/test split.

        Returns
        -------
        tuple
            (train_loader, val_loader, test_loader)
        """
        if split_ratio is None:
            split_ratio = (0.7, 0.1, 0.2)

        # 验证分割比例
        assert abs(sum(split_ratio) - 1.0) < 1e-6, "split_ratio must sum to 1.0"

        # 计算分割大小
        total_size = len(self.dataset)
        train_size = int(total_size * split_ratio[0])
        val_size = int(total_size * split_ratio[1])
        test_size = total_size - train_size - val_size

        # 分割数据集
        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader

class TigerSeqDataset(Dataset):

    def __init__(self, inters_json, indices_json, max_his_len, mode="train",sample_num =0):

        super().__init__()

        self.max_his_len = max_his_len
        self.sample_num = sample_num
        self.mode = mode

        # cache
        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None

        # load data    
        self.inters = inters_json
        self.indices = indices_json
        self._remap_items()

        # process
        self.inter_data = self._process_data()


    # =========================================================
    # Load Data
    # =========================================================

    def _remap_items(self):
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    # =========================================================
    # Process Data
    # =========================================================

    def _process_data(self):
        if self.mode == 'train':
            return self._process_train_data()
        elif self.mode == 'valid':
            return self._process_valid_data()
        elif self.mode == 'test':
            return self._process_test_data()
        else:
            raise NotImplementedError

    def _truncate_history(self, history):
        if self.max_his_len > 0:
            history = history[-self.max_his_len:]
        return history

    def _process_train_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                history = items[:i]
                history = self._truncate_history(history)

                inter_data.append({
                    "item": items[i],
                    "inters": "".join(history)
                })

        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]

            history = items[:-2]
            history = self._truncate_history(history)

            inter_data.append({
                "item": items[-2],
                "inters": "".join(history)
            })

        return inter_data

    def _process_test_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]

            history = items[:-1]
            history = self._truncate_history(history)

            inter_data.append({
                "item": items[-1],
                "inters": "".join(history)
            })

        if self.sample_num > 0:
            all_idx = np.arange(len(inter_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_test_data_ids(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]

            history = items[:-1]
            history = self._truncate_history(history)

            inter_data.append({
                "item": items[-1],
                "inters": history
            })

        if self.sample_num > 0:
            all_idx = np.arange(len(inter_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    # =========================================================
    # Token Utilities
    # =========================================================

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        tokens = set()
        for index in self.indices.values():
            for token in index:
                tokens.add(token)

        self.new_tokens = sorted(list(tokens))
        return self.new_tokens

    def get_all_items(self, as_list=False):
        if self.all_items is not None:
            return self.all_items

        items = ["".join(index) for index in self.indices.values()]
        self.all_items = items if as_list else set(items)

        return self.all_items

    def get_prefix_allowed_tokens_fn(self, tokenizer):

        if self.allowed_tokens is None:
            self.allowed_tokens = {}

            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][0]
                    if i not in self.allowed_tokens:
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)

            self.allowed_tokens[len(self.allowed_tokens)] = {
                tokenizer.eos_token_id
            }

        sep = [0]

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]

            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    # =========================================================
    # Dataset API
    # =========================================================

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        index = index % len(self.inter_data)
        d = self.inter_data[index]
        return {
            "input_ids": d["inters"],
            "labels": d["item"]
        }
    
    # =========================================================
    # Collate Function
    # =========================================================

    def get_collate_fn(self, tokenizer):

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0

        def collate_fn(batch):

            input_texts = [d["input_ids"] for d in batch]
            label_texts = [d["labels"] for d in batch]

            inputs = tokenizer(
                input_texts,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True
            )

            labels = tokenizer(
                label_texts,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True
            )

            inputs["labels"] = labels["input_ids"]
            inputs["labels"][inputs["labels"] == tokenizer.pad_token_id] = -100

            return inputs

        return collate_fn

class Trie(object):
    def __init__(self, sequences=[]):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None
        
    def def_prefix_allowed_tokens_fn(self, candidate_trie):
        def prefix_allowed_tokens(batch_id, sentence):
            sentence = sentence.tolist()
            trie_out = candidate_trie.get(sentence)
            return trie_out

        return prefix_allowed_tokens


    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence, trie_dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence,
        trie_dict,
        append_trie=None,
        bos_token_id= None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)