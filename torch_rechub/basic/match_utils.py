import tqdm
import pandas as pd
import numpy as np
import copy
import random
from collections import OrderedDict, Counter
from annoy import AnnoyIndex

def negative_sample(items_cnt_order, items_set, ratio, method_id=0):
    """Negative Sample method for matching model
    reference: https://github.com/wangzhegeek/DSSM-Lookalike/blob/master/utils.py
    update more method and redesign this function.

    Args:
        items (list): item list of raw data
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
    if not isinstance(ratio, int) or ratio < 1:
        raise ValueError("ratio means neg/pos, it should be greater than or equal to 1")
    if method_id == 0:
        neg_items = np.random.choice(items_set, size=ratio, replace=True)
    elif method_id == 1:
        #items_cnt_freq = {item: count/len(items_cnt) for item, count in items_cnt_order.items()}
        #p_sel = {item: np.sqrt(1e-5/items_cnt_freq[item]) for item in items_cnt_order}
        #The most popular paramter is item_cnt**0.75:
        p_sel = {item: count**0.75 for item, count in items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=True, p=p_value)
    elif method_id == 2:
        p_sel = {item: np.log(count+1)+1e-6 for item, count in items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=True, p=p_value)
    elif method_id == 3:
        p_sel = {item: (np.log(k + 2) - np.log(k + 1) / np.log(len(items_cnt_order) + 1)) for item, k in
                 items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=False, p=p_value)
    else:
        raise ValueError("method id should in (0,1,2,3)")
    return neg_items


def generate_seq_feature_match(data, user_col, item_col, time_col, item_attribute_cols=[], sample_method=0, mode=0, neg_ratio=0, min_item=0):
    """generate sequence feature and negative sample for match.

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
    print("preprocess data")
    data.sort_values(time_col, inplace=True) #sort by time from old to new
    train_set, test_set = [], []
    n_cold_user = 0
    for uid, hist in tqdm.tqdm(data.groupby(user_col)):
        pos_list = hist[item_col].tolist()
        if len(pos_list)<min_item: #drop this user when his pos items < min_item
            n_cold_user += 1
            continue

        if neg_ratio>0:
            items_cnt = Counter(data[item_col].tolist())
            items_set = list(items_cnt.keys()) #unique items
            items_cnt_order = OrderedDict(sorted((items_cnt.items()), key=lambda x: x[1], reverse=True)) #item_id:item count
            neg_list = negative_sample(items_cnt_order, items_set, ratio=len(pos_list)*neg_ratio, method_id=sample_method)
        
        for i in range(1, len(pos_list)):
            hist_item = pos_list[:i]
            sample = [uid, pos_list[i], hist_item, len(hist_item)]
            if len(item_attribute_cols)>0:
                for attr_col in item_attribute_cols: #the history of item attribute features
                    sample.append(hist[attr_col].tolist()[:i])
            if i != len(pos_list) - 1:
                    if mode == 0: #point-wise, the last col is label_col, include label 0 and 1
                        last_col = "label"
                        train_set.append(sample+[1])
                        for negi in range(neg_ratio):
                            sample[1] = neg_list[i*neg_ratio+negi]
                            train_set.append(sample+[0])
                    elif mode==1: #pair-wise, the last col is neg_col, include one negative item
                        last_col = "neg_sample"
                        for negi in range(neg_ratio):
                            sample_copy = copy.deepcopy(sample)
                            sample_copy.append(neg_list[i*neg_ratio+negi])
                            train_set.append(sample_copy)
                    elif mode==2: #list-wise, the last col is neg_col, include neg_ratio negative items
                        last_col = "neg_sample"
                        sample.append(neg_list[i*neg_ratio:(i+1)*neg_ratio])
                        train_set.append(sample)
                    else:
                        raise ValueError("mode should in (0,1,2)")
            else:
                test_set.append(sample+[1]) #Note: if mode=1/2, the label col is useless.

    random.shuffle(train_set)
    random.shuffle(test_set)

    print("n_train: %d, n_test: %d" %(len(train_set),len(test_set)))
    print("%d cold start user droped "%(n_cold_user))
    
    attr_hist_col = ["hist_" + col for col in item_attribute_cols]
    df_train = pd.DataFrame(train_set, columns=[user_col,item_col,"hist_"+item_col,"histlen_"+item_col]+attr_hist_col+[last_col])
    df_test = pd.DataFrame(test_set, columns=[user_col,item_col,"hist_"+item_col,"histlen_"+item_col]+attr_hist_col+[last_col])

    return df_train, df_test



class Annoy(object):
    """Vector matching by Annoy

    Args:
        metric (str): distance metric
        n_trees (int): n_trees
        search_k (int): search_k
    """
    def __init__(self, metric='angular', n_trees=10, search_k=-1):
        self._n_trees = n_trees
        self._search_k = search_k
        self._metric = metric

    def fit(self, X):
        self._annoy = AnnoyIndex(X.shape[1], metric=self._metric)
        for i, x in enumerate(X):
            self._annoy.add_item(i, x.tolist())
        self._annoy.build(self._n_trees)

    def set_query_arguments(self, search_k):
        self._search_k = search_k

    def query(self, v, n):
        return self._annoy.get_nns_by_vector(v.tolist(), n, self._search_k, include_distances=True)#

    def __str__(self):
        return 'Annoy(n_trees=%d, search_k=%d)' % (self._n_trees,
                                                   self._search_k)
        
#annoy = Annoy(n_trees=10)
#annoy.fit(item_embs)