import tqdm
import torch
from torch.utils.data import DataLoader
from .utils import PredictDataset
from annoy import AnnoyIndex


def full_predict(model, input_data, device):
    dataset = PredictDataset(input_data)
    data_loader = DataLoader(dataset, batch_size=1024)

    model = model.to(device)
    model.eval()
    predicts = []
    with torch.no_grad():
        tk0 = tqdm.tqdm(data_loader, desc="full_predict ", smoothing=0, mininterval=1.0)
        for i, x_dict in enumerate(tk0):
            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            y_pred = model(x_dict)
            predicts.append(y_pred.data)
    return torch.cat(predicts, axis=0)


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