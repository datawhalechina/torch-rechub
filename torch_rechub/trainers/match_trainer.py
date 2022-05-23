import os
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from ..basic.callback import EarlyStopper
from ..basic.loss_func import BPRLoss


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
    """

    def __init__(
        self,
        model,
        mode=0,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={
            "lr": 1e-3,
            "weight_decay": 1e-5
        },
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_patience=10,
        device="cpu",
        gpus=[],
        model_path="./",
    ):
        self.model = model  #for uniform weights save method in one gpu or multi gpu
        self.mode = mode
        if mode == 0:  #point-wise loss, binary cross_entropy
            self.criterion = torch.nn.BCELoss()  #default loss binary cross_entropy
        elif mode == 1:  #pair-wise loss
            self.criterion = BPRLoss()
        elif mode == 2:  #list-wise loss, softmax
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("mode only contain value in %s, but got %s" % ([0, 1, 2], mode))
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  #default optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.evaluate_fn = roc_auc_score  #default evaluate function
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.device = torch.device(device)  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.model_path = model_path

    def train_one_epoch(self, data_loader, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (x_dict, y) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
            y = y.to(self.device)
            if self.mode == 0:
                y = y.float()  #torch._C._nn.binary_cross_entropy expected Float
            else:
                y = y.long()  #
            if self.mode == 1:  #pair_wise
                pos_score, neg_score = self.model(x_dict)
                loss = self.criterion(pos_score, neg_score)
            else:
                y_pred = self.model(x_dict)
                loss = self.criterion(y_pred, y)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    def fit(self, train_dataloader, val_dataloader=None):
        self.model.to(self.device)
        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            self.train_one_epoch(train_dataloader)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  #update lr in epoch level by scheduler

            if val_dataloader:
                auc = self.evaluate(self.model, val_dataloader)
                print('epoch:', epoch_i, 'validation: auc:', auc)
                if self.early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f'validation: best auc: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    torch.save(self.early_stopper.best_weights, os.path.join(self.model_path, "model.pth"))  #save best auc model
                    break
            else:
                #if no val data, we save weights in the last epoch
                save_path = os.path.join(self.model_path, "model.pth")
                if len(self.gpus) > 1:
                    torch.save(self.model.module.state_dict(), save_path)  #save best auc model
                else:
                    torch.save(self.model.state_dict(), save_path)  #save best auc model

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
        #inference
        assert mode in ["user", "item"], "Invalid mode={}.".format(mode)
        model.mode = mode
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pth")))
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
