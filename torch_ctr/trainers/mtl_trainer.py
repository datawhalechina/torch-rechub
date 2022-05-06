import imp
import os
import torch
import tqdm
import numpy as np
from ..basic.callback import EarlyStopper
from ..basic.utils import get_loss_func, get_metric_func
from ..models.multi_task import ESMM


class MTLTrainer(object):

    def __init__(
        self,
        model,
        task_types,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={
            "lr": 1e-3,
            "weight_decay": 1e-5
        },
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_taskid=0,
        earlystop_patience=10,
        device="cpu",
        gpus=[],
        model_path="./",
    ):
        super(MTLTrainer, self).__init__()

        self.model = model
        self.task_types = task_types
        self.n_task = len(task_types)
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  #default Adam optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.loss_fns = [get_loss_func(task_type) for task_type in task_types]
        self.evaluate_fns = [get_metric_func(task_type) for task_type in task_types]
        self.n_epoch = n_epoch
        self.earlystop_taskid = earlystop_taskid
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.device = torch.device(device)
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.model_path = model_path

    def train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = np.zeros(self.n_task)
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for iter_i, (x_dict, ys) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
            ys = ys.to(self.device)
            y_preds = self.model(x_dict)
            loss_list = [self.loss_fns[i](y_preds[:, i], ys[:, i].float()) for i in range(self.n_task)]
            if isinstance(self.model, ESMM):
                loss = sum(loss_list[1:])  #ESSM only compute loss for ctr and ctcvr task
            else:
                loss = sum(loss_list)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += np.array([l.item() for l in loss_list])
        log_dict = {"task_%d:" % (i): total_loss[i] / (iter_i + 1) for i in range(self.n_task)}
        print("train loss: ", log_dict)

    def train(self, train_dataloader, val_dataloader):
        self.model.to(self.device)
        for epoch_i in range(self.n_epoch):
            self.train_one_epoch(train_dataloader)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  #update lr in epoch level by scheduler
            scores = self.evaluate(self.model, val_dataloader)
            print('epoch:', epoch_i, 'validation scores: ', scores)
            if self.early_stopper.stop_training(scores[self.earlystop_taskid], self.model.state_dict()):
                print('validation best auc of main task %d: %.6f' % (self.earlystop_taskid, self.early_stopper.best_auc))
                self.model.load_state_dict(self.early_stopper.best_weights)
                torch.save(self.early_stopper.best_weights, os.path.join(self.model_path, "model.pth"))  #save best auc model
                break

    def evaluate(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, ys) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
                ys = ys.to(self.device)
                y_preds = self.model(x_dict)
                targets.extend(ys.tolist())
                predicts.extend(y_preds.tolist())
        targets, predicts = np.array(targets), np.array(predicts)
        scores = [self.evaluate_fns[i](targets[:, i], predicts[:, i]) for i in range(self.n_task)]
        return scores

    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, x_dict in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y_preds = model(x_dict)
                predicts.extend(y_preds.tolist())
        return predicts