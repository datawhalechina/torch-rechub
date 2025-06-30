"""The metric module, it is used to provide some metrics for recommenders.
Available function:
- auc_score: compute AUC
- gauc_score: compute GAUC
- log_loss: compute LogLoss
- topk_metrics: compute topk metrics contains 'ndcg', 'mrr', 'recall', 'hit'
Authors: Qida Dong, dongjidan@126.com
"""
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_score(y_true, y_pred):

    return roc_auc_score(y_true, y_pred)


def get_user_pred(y_true, y_pred, users):
    """divide the result into different group by user id

        Args:
                y_true (array): all true labels of the data
                y_pred (array): the predicted score
                users (array): user id

        Return:
                user_pred (dict): {userid: values}, key is user id and value is the labels and scores of each user
        """
    user_pred = {}
    for i, u in enumerate(users):
        if u not in user_pred:
            user_pred[u] = {'y_true': [y_true[i]], 'y_pred': [y_pred[i]]}
        else:
            user_pred[u]['y_true'].append(y_true[i])
            user_pred[u]['y_pred'].append(y_pred[i])

    return user_pred


def gauc_score(y_true, y_pred, users, weights=None):
    """compute GAUC

        Args:
                y_true (array): dim(N, ), all true labels of the data
                y_pred (array): dim(N, ), the predicted score
                users (array): dim(N, ), user id
                weight (dict): {userid: weight_value}, it contains weights for each group.
                                if it is None, the weight is equal to the number
                                of times the user is recommended
        Return:
                score: float, GAUC
        """
    assert len(y_true) == len(y_pred) and len(y_true) == len(users)

    user_pred = get_user_pred(y_true, y_pred, users)
    score = 0
    num = 0
    for u in user_pred.keys():
        auc = auc_score(user_pred[u]['y_true'], user_pred[u]['y_pred'])
        if weights is None:
            user_weight = len(user_pred[u]['y_true'])
        else:
            user_weight = weights[u]
        auc *= user_weight
        num += user_weight
        score += auc
    return score / num


def ndcg_score(y_true, y_pred, topKs=None):
    if topKs is None:
        topKs = [5]
    result = topk_metrics(y_true, y_pred, topKs)
    return result['NDCG']


def hit_score(y_true, y_pred, topKs=None):
    if topKs is None:
        topKs = [5]
    result = topk_metrics(y_true, y_pred, topKs)
    return result['Hit']


def mrr_score(y_true, y_pred, topKs=None):
    if topKs is None:
        topKs = [5]
    result = topk_metrics(y_true, y_pred, topKs)
    return result['MRR']


def recall_score(y_true, y_pred, topKs=None):
    if topKs is None:
        topKs = [5]
    result = topk_metrics(y_true, y_pred, topKs)
    return result['Recall']


def precision_score(y_true, y_pred, topKs=None):
    if topKs is None:
        topKs = [5]
    result = topk_metrics(y_true, y_pred, topKs)
    return result['Precision']


def topk_metrics(y_true, y_pred, topKs=None):
    """choice topk metrics and compute it
        the metrics contains 'ndcg', 'mrr', 'recall', 'precision' and 'hit'

        Args:
                y_true (dict): {userid, item_ids}, the key is user id and the value is the list that contains the items the user interacted
                y_pred (dict): {userid, item_ids}, the key is user id and the value is the list that contains the items recommended
                topKs (list or tuple): if you want to get top5 and top10, topKs=(5, 10)

        Return:
                results (dict): {metric_name: metric_values}, it contains five metrics, 'ndcg', 'recall', 'mrr', 'hit', 'precision'

        """
    if topKs is None:
        topKs = [5]
    assert len(y_true) == len(y_pred)

    if not isinstance(topKs, (tuple, list)):
        raise ValueError('topKs wrong, it should be tuple or list')

    pred_array = []
    true_array = []
    for u in y_true.keys():
        pred_array.append(y_pred[u])
        true_array.append(y_true[u])
    ndcg_result = []
    mrr_result = []
    hit_result = []
    precision_result = []
    recall_result = []
    for idx in range(len(topKs)):
        ndcgs = 0
        mrrs = 0
        hits = 0
        precisions = 0
        recalls = 0
        gts = 0
        for i in range(len(true_array)):
            if len(true_array[i]) != 0:
                mrr_tmp = 0
                mrr_flag = True
                hit_tmp = 0
                dcg_tmp = 0
                idcg_tmp = 0
                for j in range(topKs[idx]):
                    if pred_array[i][j] in true_array[i]:
                        hit_tmp += 1.
                        if mrr_flag:
                            mrr_flag = False
                            mrr_tmp = 1. / (1 + j)
                        dcg_tmp += 1. / (np.log2(j + 2))
                    if j < len(true_array[i]):
                        idcg_tmp += 1. / (np.log2(j + 2))
                gts += len(true_array[i])
                hits += hit_tmp
                mrrs += mrr_tmp
                recalls += hit_tmp / len(true_array[i])
                precisions += hit_tmp / topKs[idx]
                if idcg_tmp != 0:
                    ndcgs += dcg_tmp / idcg_tmp
        hit_result.append(round(hits / gts, 4))
        mrr_result.append(round(mrrs / len(pred_array), 4))
        recall_result.append(round(recalls / len(pred_array), 4))
        precision_result.append(round(precisions / len(pred_array), 4))
        ndcg_result.append(round(ndcgs / len(pred_array), 4))

    results = defaultdict(list)
    for idx in range(len(topKs)):

        output = f'NDCG@{topKs[idx]}: {ndcg_result[idx]}'
        results['NDCG'].append(output)

        output = f'MRR@{topKs[idx]}: {mrr_result[idx]}'
        results['MRR'].append(output)

        output = f'Recall@{topKs[idx]}: {recall_result[idx]}'
        results['Recall'].append(output)

        output = f'Hit@{topKs[idx]}: {hit_result[idx]}'
        results['Hit'].append(output)

        output = f'Precision@{topKs[idx]}: {precision_result[idx]}'
        results['Precision'].append(output)
    return results


def log_loss(y_true, y_pred):
    score = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    return -score.sum() / len(y_true)


def Coverage(y_pred, all_items, topKs=None):
    """compute the coverage
        This method measures the diversity of the recommended items
        and the ability to explore the long-tailed items
        Arg:
                y_pred (dict): {userid, item_ids}, the key is user id and the value is the list that contains the items recommended
                all_items (set): all unique items
        Return:
                result (list[float]): the list of coverage scores
        """
    if topKs is None:
        topKs = [5]
    result = []
    for k in topKs:
        rec_items = set([])
        for u in y_pred.keys():
            tmp_items = set(y_pred[u][:k])
            rec_items = rec_items | tmp_items
        score = len(rec_items) * 1. / len(all_items)
        score = round(score, 4)
        result.append(f'Coverage@{k}: {score}')
    return result


# print(Coverage({'0':[0,1,2],'1':[1,3,4]}, {0,1,2,3,4,5}, [2,3]))

# pred = np.array([  0.3, 0.2, 0.5, 0.9, 0.7, 0.31, 0.8, 0.1, 0.4, 0.6])
# label = np.array([   1,   0,   0,   1,   0,   0,    1,   0,   0,   1])
# users_id = np.array([ 2,   1,   0,   2,   1,   0,    0,   2,   1,   1])

# print('auc: ', auc_score(label, pred))
# print('gauc: ', gauc_score(label, pred, users_id))
# print('log_loss: ', log_loss(label, pred))

# for mt in ['ndcg', 'mrr', 'recall', 'hit','s']:
# 	tm = topk_metrics(y_true, y_pred, users_id, 3, metric_type=mt)
# 	print(f'{mt}: {tm}')
# y_pred = {'0': [0, 1], '1': [0, 1], '2': [2, 3]}
# y_true = {'0': [1, 2], '1': [0, 1, 2], '2': [2, 3]}
# out = topk_metrics(y_true, y_pred, topKs=(1,2))
# ndcgs = ndcg_score(y_true,y_pred, topKs=(1,2))
# print(out)
# print(ndcgs)

# ground_truth, match_res = np.load("C:\\Users\\dongj\\Desktop/res.npy", allow_pickle=True)
# print(len(ground_truth),len(match_res))
# out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=[50])
# print(out)

if __name__ == "__main__":
    y_pred = {'0': [0, 1], '1': [0, 1], '2': [2, 3]}
    y_true = {'0': [1, 2], '1': [0, 1, 2], '2': [2, 3]}
    out = topk_metrics(y_true, y_pred, topKs=(1, 2))
    print(out)
