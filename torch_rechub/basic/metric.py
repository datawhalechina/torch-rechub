"""Evaluation metrics for recommender systems.

Accuracy metrics:
- auc_score / gauc_score / log_loss
- topk_metrics (ndcg, mrr, recall, hit, precision)

Beyond-accuracy metrics:
- diversity_score: Intra-List Diversity (ILD)
- coverage_score: Catalog Coverage
- novelty_score: Mean Self-Information

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


def diversity_score(y_pred, item_embeddings, topKs=None):
    """Intra-List Diversity (ILD): average pairwise cosine distance within each user's recommendation list.

    A higher score means the recommended items are more different from each other,
    indicating the model is not just recommending similar items repeatedly.

    Args:
        y_pred (dict): {userid: [item_ids]}, recommended items per user
        item_embeddings (dict or np.ndarray): item vectors. If dict: {item_id: np.array};
            if 2D array: indexed by item_id (row = item_id)
        topKs (list or tuple): e.g. [5, 10]

    Return:
        results (dict): {'Diversity': ['Diversity@5: 0.xxxx', ...]}
    """
    if topKs is None:
        topKs = [5]
    results = defaultdict(list)
    for k in topKs:
        user_diversities = []
        for u in y_pred:
            items = y_pred[u][:k]
            if len(items) < 2:
                continue
            # collect embeddings
            embs = []
            for item in items:
                if isinstance(item_embeddings, dict):
                    if item in item_embeddings:
                        embs.append(np.array(item_embeddings[item], dtype=np.float64))
                else:
                    if item < len(item_embeddings):
                        embs.append(np.array(item_embeddings[item], dtype=np.float64))
            if len(embs) < 2:
                continue
            embs = np.stack(embs)
            # cosine similarity matrix
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            normed = embs / norms
            sim_matrix = normed @ normed.T
            # average pairwise distance (upper triangle, excluding diagonal)
            n = len(embs)
            pair_count = n * (n - 1) / 2
            dist_sum = (1 - sim_matrix)[np.triu_indices(n, k=1)].sum()
            user_diversities.append(dist_sum / pair_count)
        score = round(np.mean(user_diversities), 4) if user_diversities else 0.0
        results['Diversity'].append(f'Diversity@{k}: {score}')
    return results


def coverage_score(y_pred, all_items, topKs=None):
    """Catalog Coverage: fraction of all items that appear in at least one user's recommendation list.

    A higher score means the model recommends a wider variety of items across all users,
    rather than always recommending the same popular items.

    Args:
        y_pred (dict): {userid: [item_ids]}, recommended items per user
        all_items (set or list): all unique item ids in the catalog
        topKs (list or tuple): e.g. [5, 10]

    Return:
        results (dict): {'Coverage': ['Coverage@5: 0.xxxx', ...]}
    """
    if topKs is None:
        topKs = [5]
    results = defaultdict(list)
    for k in topKs:
        rec_items = set()
        for u in y_pred:
            rec_items.update(y_pred[u][:k])
        score = round(len(rec_items) / len(all_items), 4)
        results['Coverage'].append(f'Coverage@{k}: {score}')
    return results


def novelty_score(y_pred, item_popularity, topKs=None):
    """Mean Self-Information: measures how "surprising" or niche the recommendations are.

    For each recommended item, self-information = -log2(popularity).
    Popular items have low self-information; long-tail items have high self-information.
    A higher novelty score means the model recommends more niche items.

    Args:
        y_pred (dict): {userid: [item_ids]}, recommended items per user
        item_popularity (dict): {item_id: float}, interaction probability of each item
            (e.g. item_count / total_interactions). Values should be in (0, 1].
        topKs (list or tuple): e.g. [5, 10]

    Return:
        results (dict): {'Novelty': ['Novelty@5: x.xxxx', ...]}
    """
    if topKs is None:
        topKs = [5]
    results = defaultdict(list)
    for k in topKs:
        user_novelties = []
        for u in y_pred:
            items = y_pred[u][:k]
            if not items:
                continue
            self_info = []
            for item in items:
                pop = item_popularity.get(item, 1e-10)
                pop = max(pop, 1e-10)  # avoid log(0)
                self_info.append(-np.log2(pop))
            user_novelties.append(np.mean(self_info))
        score = round(np.mean(user_novelties), 4) if user_novelties else 0.0
        results['Novelty'].append(f'Novelty@{k}: {score}')
    return results


if __name__ == "__main__":
    # Test data: 3 users, each with 2 recommended items
    y_pred = {'0': [0, 1], '1': [0, 1], '2': [2, 3]}
    y_true = {'0': [1, 2], '1': [0, 1, 2], '2': [2, 3]}

    # --- Verify existing topk_metrics (hand-calculated) ---
    out = topk_metrics(y_true, y_pred, topKs=(1, 2))
    assert out['NDCG'][1] == 'NDCG@2: 0.7956', f"NDCG@2 wrong: {out['NDCG'][1]}"
    assert out['MRR'][1] == 'MRR@2: 0.8333', f"MRR@2 wrong: {out['MRR'][1]}"
    assert out['Recall'][1] == 'Recall@2: 0.7222', f"Recall@2 wrong: {out['Recall'][1]}"
    assert out['Hit'][1] == 'Hit@2: 0.7143', f"Hit@2 wrong: {out['Hit'][1]}"
    assert out['Precision'][1] == 'Precision@2: 0.8333', f"Precision@2 wrong: {out['Precision'][1]}"
    print("[PASS] topk_metrics")

    # --- Verify coverage_score ---
    # @1: rec_items={0,2} -> 2/6=0.3333; @2: rec_items={0,1,2,3} -> 4/6=0.6667
    all_items = {0, 1, 2, 3, 4, 5}
    cov = coverage_score(y_pred, all_items, topKs=[1, 2])
    assert cov['Coverage'][0] == 'Coverage@1: 0.3333', f"Coverage@1 wrong: {cov['Coverage'][0]}"
    assert cov['Coverage'][1] == 'Coverage@2: 0.6667', f"Coverage@2 wrong: {cov['Coverage'][1]}"
    print("[PASS] coverage_score")

    # --- Verify diversity_score ---
    # embs: 0=[1,0,0], 1=[0.9,0.1,0], 2=[0,1,0], 3=[0,0,1]
    # User0/1: cos_dist(0,1)=0.0061; User2: cos_dist(2,3)=1.0
    # avg = (0.0061 + 0.0061 + 1.0) / 3 = 0.3374
    item_embs = {
        0: [1.0,
            0.0,
            0.0],
        1: [0.9,
            0.1,
            0.0],
        2: [0.0,
            1.0,
            0.0],
        3: [0.0,
            0.0,
            1.0],
    }
    div = diversity_score(y_pred, item_embs, topKs=[2])
    assert div['Diversity'][0] == 'Diversity@2: 0.3374', f"Diversity@2 wrong: {div['Diversity'][0]}"
    # edge case: topK=1 means only 1 item per user, no pairs -> should be 0.0
    div1 = diversity_score(y_pred, item_embs, topKs=[1])
    assert div1['Diversity'][0] == 'Diversity@1: 0.0', f"Diversity@1 wrong: {div1['Diversity'][0]}"
    print("[PASS] diversity_score")

    # --- Verify novelty_score ---
    # SI(0)=-log2(0.5)=1.0, SI(1)=-log2(0.3)=1.737, SI(2)=-log2(0.05)=4.3219, SI(3)=-log2(0.01)=6.6439
    # User0/1: mean(1.0,1.737)=1.3685; User2: mean(4.3219,6.6439)=5.4829
    # avg@2 = (1.3685+1.3685+5.4829)/3 = 2.74
    item_pop = {0: 0.5, 1: 0.3, 2: 0.05, 3: 0.01}
    nov = novelty_score(y_pred, item_pop, topKs=[1, 2])
    assert nov['Novelty'][1] == 'Novelty@2: 2.74', f"Novelty@2 wrong: {nov['Novelty'][1]}"
    # @1: User0=SI(0)=1.0, User1=SI(0)=1.0, User2=SI(2)=4.3219 -> avg=2.1073
    assert nov['Novelty'][0] == 'Novelty@1: 2.1073', f"Novelty@1 wrong: {nov['Novelty'][0]}"
    print("[PASS] novelty_score")

    print("\nAll tests passed.")
