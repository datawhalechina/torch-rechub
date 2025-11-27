import os

import torch
import tqdm
from sklearn.metrics import roc_auc_score

from ..basic.callback import EarlyStopper
from ..basic.loss_func import BPRLoss, RegularizationLoss
from ..utils.match import gather_inbatch_logits, inbatch_negative_sampling


class MatchTrainer(object):
    """A general trainer for Matching/Retrieval

    Args:
        model (nn.Module): any matching model.
        mode (int, optional): the training mode, `{0:point-wise, 1:pair-wise, 2:list-wise}`. Defaults to 0.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
        in_batch_neg (bool): whether to use in-batch negative sampling instead of global negatives.
        in_batch_neg_ratio (int): number of negatives to draw from the batch per positive sample when in_batch_neg is True.
        hard_negative (bool): whether to choose hardest negatives within batch (top-k by score) instead of uniform random.
        sampler_seed (int): optional random seed for in-batch sampler to ease reproducibility/testing.
    """

    def __init__(
        self,
        model,
        mode=0,
        in_batch_neg=False,
        in_batch_neg_ratio=None,
        hard_negative=False,
        sampler_seed=None,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        regularization_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
        model_path="./",
    ):
        self.model = model  # for uniform weights save method in one gpu or multi gpu
        if gpus is None:
            gpus = []
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.in_batch_neg = in_batch_neg
        self.in_batch_neg_ratio = in_batch_neg_ratio
        self.hard_negative = hard_negative
        self._sampler_generator = None
        if sampler_seed is not None:
            self._sampler_generator = torch.Generator(device=self.device)
            self._sampler_generator.manual_seed(sampler_seed)
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        if regularization_params is None:
            regularization_params = {"embedding_l1": 0.0, "embedding_l2": 0.0, "dense_l1": 0.0, "dense_l2": 0.0}
        self.mode = mode
        if mode == 0:  # point-wise loss, binary cross_entropy
            # With in-batch negatives we treat it as list-wise classification over sampled negatives
            self.criterion = torch.nn.CrossEntropyLoss() if in_batch_neg else torch.nn.BCELoss()
        elif mode == 1:  # pair-wise loss
            self.criterion = BPRLoss()
        elif mode == 2:  # list-wise loss, softmax
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("mode only contain value in %s, but got %s" % ([0, 1, 2], mode))
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.evaluate_fn = roc_auc_score  # default evaluate function
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path
        # Initialize regularization loss
        self.reg_loss_fn = RegularizationLoss(**regularization_params)

    def train_one_epoch(self, data_loader, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (x_dict, y) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  # tensor to GPU
            y = y.to(self.device)
            if self.mode == 0:
                y = y.float()  # torch._C._nn.binary_cross_entropy expected Float
            else:
                y = y.long()  #
            if self.in_batch_neg:
                base_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                user_embedding = base_model.user_tower(x_dict)
                item_embedding = base_model.item_tower(x_dict)
                if user_embedding is None or item_embedding is None:
                    raise ValueError("Model must return user/item embeddings when in_batch_neg is True.")
                if user_embedding.dim() > 2 and user_embedding.size(1) == 1:
                    user_embedding = user_embedding.squeeze(1)
                if item_embedding.dim() > 2 and item_embedding.size(1) == 1:
                    item_embedding = item_embedding.squeeze(1)
                if user_embedding.dim() != 2 or item_embedding.dim() != 2:
                    raise ValueError(f"In-batch negative sampling requires 2D embeddings, got shapes {user_embedding.shape} and {item_embedding.shape}")

                scores = torch.matmul(user_embedding, item_embedding.t())  # bs x bs
                neg_indices = inbatch_negative_sampling(scores, neg_ratio=self.in_batch_neg_ratio, hard_negative=self.hard_negative, generator=self._sampler_generator)
                logits = gather_inbatch_logits(scores, neg_indices)
                if self.mode == 1:  # pair_wise
                    loss = self.criterion(logits[:, 0], logits[:, 1:], in_batch_neg=True)
                else:  # point-wise/list-wise -> cross entropy on sampled logits
                    targets = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
                    loss = self.criterion(logits, targets)
            else:
                if self.mode == 1:  # pair_wise
                    pos_score, neg_score = self.model(x_dict)
                    loss = self.criterion(pos_score, neg_score)
                else:
                    y_pred = self.model(x_dict)
                    loss = self.criterion(y_pred, y)

            # Add regularization loss
            reg_loss = self.reg_loss_fn(self.model)
            loss = loss + reg_loss

            # used for debug
            # if i == 0:
            #     print()
            #     if self.mode == 0:
            #         print('pred: ', [f'{float(each):5.2g}' for each in y_pred.detach().cpu().tolist()])
            #         print('truth:', [f'{float(each):5.2g}' for each in y.detach().cpu().tolist()])
            #     elif self.mode == 2:
            #         pred = y_pred.detach().cpu().mean(0)
            #         pred = torch.softmax(pred, dim=0).tolist()
            #         print('pred: ', [f'{float(each):4.2g}' for each in pred])
            #     elif self.mode == 1:
            #         print('pos:', [f'{float(each):5.2g}' for each in pos_score.detach().cpu().tolist()])
            #         print('neg: ', [f'{float(each):5.2g}' for each in neg_score.detach().cpu().tolist()])

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    def fit(self, train_dataloader, val_dataloader=None):
        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            self.train_one_epoch(train_dataloader)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  # update lr in epoch level by scheduler

            if val_dataloader:
                auc = self.evaluate(self.model, val_dataloader)
                print('epoch:', epoch_i, 'validation: auc:', auc)
                if self.early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f'validation: best auc: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))  # save best auc model

    def evaluate(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model(x_dict)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        return self.evaluate_fn(targets, predicts)

    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model(x_dict)
                predicts.extend(y_pred.tolist())
        return predicts

    def inference_embedding(self, model, mode, data_loader, model_path):
        # inference
        assert mode in ["user", "item"], "Invalid mode={}.".format(mode)
        model.mode = mode
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), map_location=self.device, weights_only=True))
        model = model.to(self.device)
        model.eval()
        predicts = []
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="%s inference" % (mode), smoothing=0, mininterval=1.0)
            for i, x_dict in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y_pred = model(x_dict)
                predicts.append(y_pred.data)
        return torch.cat(predicts, dim=0)
