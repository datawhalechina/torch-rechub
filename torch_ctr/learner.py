import torch
import tqdm
from sklearn.metrics import roc_auc_score
from .utils import EarlyStopper

class CTRLearner(object):
    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-5, n_epoch=10,
                       device="cpu", model_path="./model.pth", earlystop_patience=10):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay) #default optimizer
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.8)
        self.criterion = torch.nn.BCELoss() #default loss cross_entropy
        self.evaluate_func = roc_auc_score #default evaluate function
        self.n_epoch = n_epoch
        self.model_path = model_path
        self.device = device
        self.early_stopper = EarlyStopper(patience=earlystop_patience, save_path=model_path)

    
    def train_one_epoch(self, data_loader, log_interval=100):
        #print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train:", smoothing=0, mininterval=1.0)
        for i, (x_dict, y) in enumerate(tk0):
            for key in x_dict:
                x_dict[key] = x_dict[key].to(self.device)
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
        #scheduler.step()

    def train(self, train_dataloader, val_dataloader):
        self.model.to(self.device)
        for epoch_i in range(self.n_epoch):
            self.train_one_epoch(train_dataloader)
            auc = self.evaluate(self.model, val_dataloader)
            print('epoch:', epoch_i, 'validation: auc:', auc)
            if self.early_stopper.is_continuable(auc):
                self.save_model(self.model, self.model_path)
            else:
                print(f'validation: best auc: {self.early_stopper.best_auc}')
                self.model = self.load_model(self.model_path) #load best auc model
                break
    
    def evaluate(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation:",smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                for key in x_dict:
                    x_dict[key] = x_dict[key].to(self.device)
                y = y.to(self.device)
                y_pred = model(x_dict)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        return self.evaluate_func(targets, predicts)

    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict:",smoothing=0, mininterval=1.0)
            for i, x_dict in enumerate(tk0):
                for key in x_dict:
                    x_dict[key] = x_dict[key].to(self.device)
                y_pred = model(x_dict)
                predicts.extend(y_pred.tolist())
        return predicts

    def save_model(self, model, model_path):
        torch.save(model, model_path)
    
    def load_model(self, model_path):
        return torch.load(model_path)