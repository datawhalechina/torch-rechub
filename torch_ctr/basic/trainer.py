import os
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from .callback import EarlyStopper


class CTRTrainer(object):

    def __init__(
        self,
        model,
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
        model_path="./model.pth",
    ):
        self.model = model  #for uniform weights save method in one gpu or multi gpu
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  #default optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.criterion = torch.nn.BCELoss()  #default loss cross_entropy
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
            y_pred = self.model(x_dict)
            loss = self.criterion(y_pred, y.float())
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    def train(self, train_dataloader, val_dataloader):
        self.model.to(self.device)
        for epoch_i in range(self.n_epoch):
            self.train_one_epoch(train_dataloader)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  #update lr in epoch level by scheduler
            auc = self.evaluate(self.model, val_dataloader)
            print('epoch:', epoch_i, 'validation: auc:', auc)
            if self.early_stopper.stop_training(auc, self.model.state_dict()):
                print(f'validation: best auc: {self.early_stopper.best_auc}')
                self.model.load_state_dict(self.early_stopper.best_weights)
                torch.save(self.early_stopper.best_weights, os.path.join(self.model_path, "model.pth"))  #save best auc model
                break

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
            for i, x_dict in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y_pred = model(x_dict)
                predicts.extend(y_pred.tolist())
        return predicts