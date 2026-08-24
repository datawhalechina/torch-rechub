import copy


class EarlyStopper(object):
    """Early stops training if the validation metric does not improve.

    Args:
        patience (int): How long to wait after the validation metric last improved.
        mode (str): ``"max"`` for metrics where larger is better and ``"min"``
            for metrics where smaller is better (default = ``"max"``).
    """

    def __init__(self, patience, mode="max"):
        if mode not in {"max", "min"}:
            raise ValueError("mode must be either 'max' or 'min'")

        self.patience = patience
        self.mode = mode
        self.trial_counter = 0
        # Keep the previous initial value in the default mode so existing
        # callers retain the same behavior.
        self.best_auc = 0 if mode == "max" else float("inf")
        self.best_weights = None

    def stop_training(self, val_auc, weights):
        """whether to stop training.

        Args:
            val_auc (float): metric score on validation data. The name is kept
                for backward compatibility.
            weights (tensor): the weights of model
        """
        improved = val_auc > self.best_auc if self.mode == "max" else val_auc < self.best_auc
        if improved:
            self.best_auc = val_auc
            self.trial_counter = 0
            self.best_weights = copy.deepcopy(weights)
            return False
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1
            return False
        else:
            return True
