import copy


class EarlyStopper(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience):
        """
        Args:
            patience (int): How long to wait after last time validation auc improved.
            TODO：delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.trial_counter = 0
        self.best_auc = 0
        self.best_weights = None

    def stop_training(self, val_auc, weights):
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.trial_counter = 0
            self.best_weights = copy.deepcopy(weights)
            return False
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1
            return False
        else:
            return True