import os

import numpy as np
import torch
import torch.nn as nn
import tqdm

from ..basic.callback import EarlyStopper
from ..basic.loss_func import RegularizationLoss
from ..models.multi_task import ESMM
from ..utils.data import get_loss_func, get_metric_func
from ..utils.mtl import MetaBalance, gradnorm, shared_task_layers


class MTLTrainer(object):
    """A trainer for multi task learning.

    Args:
        model (nn.Module): any multi task learning model.
        task_types (list): types of tasks, only support ["classfication", "regression"].
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler): optional scheduler called with step() once per epoch, e.g. StepLR.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        adaptive_params (dict): parameters of adaptive loss weight method. Now only support `{"method" : "uwl"}`.
        n_epoch (int): epoch number of training.
        earlystop_taskid (int): task id of earlystop metrics relies between multi task (default = 0).
        earlystop_patience (int): how long to wait after the validation metric last improved (default = 10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
        earlystop_mode (str): ``"max"`` when a larger validation score is better,
            or ``"min"`` when a smaller score is better (default = ``"max"``).
    """

    def __init__(
        self,
        model,
        task_types,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        regularization_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        adaptive_params=None,
        n_epoch=10,
        earlystop_taskid=0,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
        model_path="./",
        model_logger=None,
        earlystop_mode="max",
    ):
        self.model = model
        if gpus is None:
            gpus = []
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        if regularization_params is None:
            regularization_params = {"embedding_l1": 0.0, "embedding_l2": 0.0, "dense_l1": 0.0, "dense_l2": 0.0}
        self.task_types = task_types
        self.n_task = len(task_types)
        self.loss_weight = None
        self.adaptive_method = None
        if adaptive_params is not None:
            if adaptive_params["method"] == "uwl":
                self.adaptive_method = "uwl"
                self.loss_weight = nn.ParameterList(nn.Parameter(torch.zeros(1)) for _ in range(self.n_task))
                self.model.add_module("loss weight", self.loss_weight)
            elif adaptive_params["method"] == "metabalance":
                self.adaptive_method = "metabalance"
                share_layers, task_layers = shared_task_layers(self.model)
                self.meta_optimizer = MetaBalance(share_layers)
                self.share_optimizer = optimizer_fn(share_layers, **optimizer_params)
                self.task_optimizer = optimizer_fn(task_layers, **optimizer_params)
            elif adaptive_params["method"] == "gradnorm":
                self.adaptive_method = "gradnorm"
                self.alpha = adaptive_params.get("alpha", 0.16)
                share_layers = shared_task_layers(self.model)[0]
                # gradnorm calculate the gradients of each loss on the last
                # fully connected shared layer weight(dimension is 2)
                for i in range(len(share_layers)):
                    if share_layers[-i].ndim == 2:
                        self.last_share_layer = share_layers[-i]
                        break
                self.initial_task_loss = None
                self.loss_weight = nn.ParameterList(nn.Parameter(torch.ones(1)) for _ in range(self.n_task))
                self.model.add_module("loss weight", self.loss_weight)
        if self.adaptive_method != "metabalance":
            self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default Adam optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **(scheduler_params or {}))
        self.loss_fns = [get_loss_func(task_type) for task_type in task_types]
        self.evaluate_fns = [get_metric_func(task_type) for task_type in task_types]
        self.n_epoch = n_epoch
        self.earlystop_taskid = earlystop_taskid
        self.early_stopper = EarlyStopper(patience=earlystop_patience, mode=earlystop_mode)

        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model_path = model_path
        # Initialize regularization loss
        self.reg_loss_fn = RegularizationLoss(**regularization_params)
        self.model_logger = model_logger

    def train_one_epoch(self, data_loader, log_interval=10):
        """Train one epoch; log_interval is a positive integer, defaulting to 10."""
        self.model.train()
        total_loss = np.zeros(self.n_task)
        interval_loss = np.zeros(self.n_task)
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for iter_i, (x_dict, ys) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  # tensor to GPU
            ys = ys.to(self.device)
            y_preds = self.model(x_dict)
            loss_list = [self.loss_fns[i](y_preds[:, i], ys[:, i].float()) for i in range(self.n_task)]
            if isinstance(self.model, ESMM):
                # ESSM only compute loss for ctr and ctcvr task
                loss = sum(loss_list[1:])
            else:
                if self.adaptive_method is not None:
                    if self.adaptive_method == "uwl":
                        loss = 0
                        for loss_i, w_i in zip(loss_list, self.loss_weight):
                            w_i = torch.clamp(w_i, min=0)
                            loss += 2 * loss_i * torch.exp(-w_i) + w_i
                else:
                    loss = sum(loss_list) / self.n_task

            # Add regularization loss
            reg_loss = self.reg_loss_fn(self.model)
            loss = loss + reg_loss
            if self.adaptive_method == 'metabalance':
                self.share_optimizer.zero_grad()
                self.task_optimizer.zero_grad()
                self.meta_optimizer.step(loss_list)
                self.share_optimizer.step()
                self.task_optimizer.step()
            elif self.adaptive_method == "gradnorm":
                self.optimizer.zero_grad()
                if self.initial_task_loss is None:
                    self.initial_task_loss = [l.item() for l in loss_list]
                gradnorm(loss_list, self.loss_weight, self.last_share_layer, self.initial_task_loss, self.alpha)
                self.optimizer.step()
                # renormalize
                loss_weight_sum = sum([w.item() for w in self.loss_weight])
                normalize_coeff = len(self.loss_weight) / loss_weight_sum
                for w in self.loss_weight:
                    w.data = w.data * normalize_coeff
            else:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
            loss_values = np.array([l.item() for l in loss_list])
            total_loss += loss_values
            interval_loss += loss_values
            if (iter_i + 1) % log_interval == 0:
                tk0.set_postfix({f'task_{i}_loss': interval_loss[i] / log_interval for i in range(self.n_task)}, lr=self._current_lr())
                interval_loss.fill(0)
        log_dict = {"task_%d:" % (i): total_loss[i] / (iter_i + 1) for i in range(self.n_task)}
        loss_list = [total_loss[i] / (iter_i + 1) for i in range(self.n_task)]
        print("train loss: ", log_dict)
        if self.loss_weight:
            print("loss weight: ", [w.item() for w in self.loss_weight])

        return loss_list

    def fit(self, train_dataloader, val_dataloader, mode='base', seed=0, log_interval=10):
        """Train with progress updates every log_interval batches.

        Use a positive integer for log_interval (default: 10).
        Set log_interval=1 to show each batch's task losses and learning rate.
        """
        model_file = os.path.join(self.model_path, "model_{}_{}.pth".format(mode, seed))
        total_log = []

        # Log hyperparameters once
        for logger in self._iter_loggers():
            logger.log_hyperparams({'n_epoch': self.n_epoch, 'learning_rate': self._current_lr(), 'adaptive_method': self.adaptive_method})

        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            _log_per_epoch = self.train_one_epoch(train_dataloader, log_interval=log_interval)

            # Collect metrics
            logs = {f'train/task_{task_id}_loss': loss_val for task_id, loss_val in enumerate(_log_per_epoch)}
            lr_value = self._current_lr()
            if lr_value is not None:
                logs['learning_rate'] = lr_value

            print("Current lr : {}".format(lr_value))
            if self.scheduler is not None:
                self.scheduler.step()  # update lr in epoch level by scheduler

            scores = self.evaluate(self.model, val_dataloader)
            print('epoch:', epoch_i, 'validation scores: ', scores)

            score = scores[self.earlystop_taskid]
            is_best = score > self.early_stopper.best_auc if self.early_stopper.mode == "max" else score < self.early_stopper.best_auc
            stop_training = self.early_stopper.stop_training(score, self.model.state_dict())
            if is_best:
                torch.save(self.early_stopper.best_weights, model_file)

            for task_id, score in enumerate(scores):
                logs[f'val/task_{task_id}_score'] = score
                _log_per_epoch.append(score)
            logs['auc'] = scores[self.earlystop_taskid]

            if self.loss_weight:
                for task_id, weight in enumerate(self.loss_weight):
                    logs[f'loss_weight/task_{task_id}'] = weight.item()

            total_log.append(_log_per_epoch)

            # Log metrics once per epoch
            for logger in self._iter_loggers():
                logger.log_metrics(logs, step=epoch_i)

            if stop_training:
                print('validation best score of main task %d: %.6f' % (self.earlystop_taskid, self.early_stopper.best_auc))
                break

        if self.early_stopper.best_weights is not None:
            self.model.load_state_dict(self.early_stopper.best_weights)
        torch.save(self.model.state_dict(), model_file)

        for logger in self._iter_loggers():
            logger.finish()

        return total_log

    def _iter_loggers(self):
        """Return logger instances as a list.

        Returns
        -------
        list
            Active logger instances. Empty when ``model_logger`` is ``None``.
        """
        if self.model_logger is None:
            return []
        if isinstance(self.model_logger, (list, tuple)):
            return list(self.model_logger)
        return [self.model_logger]

    def _current_lr(self):
        """Fetch current learning rate regardless of adaptive method."""
        if self.adaptive_method == "metabalance":
            return self.share_optimizer.param_groups[0]['lr'] if hasattr(self, 'share_optimizer') else None
        if hasattr(self, 'optimizer'):
            return self.optimizer.param_groups[0]['lr']
        return None

    def evaluate(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, ys) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  # tensor to GPU
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

    def export_onnx(self, output_path, dummy_input=None, batch_size=2, seq_length=10, opset_version=14, dynamic_batch=True, device=None, verbose=False, onnx_export_kwargs=None):
        """Export the trained multi-task model to ONNX format.

        This method exports multi-task learning models (e.g., MMOE, PLE, ESMM, SharedBottom)
        to ONNX format for deployment. The exported model will have multiple outputs
        corresponding to each task.

        Note:
            The ONNX model will output a tensor of shape [batch_size, n_task] where
            n_task is the number of tasks in the multi-task model.

        Args:
            output_path (str): Path to save the ONNX model file.
            dummy_input (dict, optional): Example input dict {feature_name: tensor}.
                If not provided, dummy inputs will be generated automatically.
            batch_size (int): Batch size for auto-generated dummy input (default: 2).
            seq_length (int): Sequence length for SequenceFeature (default: 10).
            opset_version (int): ONNX opset version (default: 14).
            dynamic_batch (bool): Enable dynamic batch size (default: True).
            device (str, optional): Device for export ('cpu', 'cuda', etc.).
                If None, defaults to 'cpu' for maximum compatibility.
            verbose (bool): Print export details (default: False).
            onnx_export_kwargs (dict, optional): Extra kwargs forwarded to ``torch.onnx.export``.
                The exporter tries the dynamo path first and falls back to the
                legacy exporter automatically. Pass ``{\"dynamo\": False}`` here
                to force legacy export when needed.

        Returns:
            bool: True if export succeeded, False otherwise.

        Example:
            >>> trainer = MTLTrainer(mmoe_model, task_types=["classification", "classification"], ...)
            >>> trainer.fit(train_dl, val_dl)
            >>> trainer.export_onnx("mmoe.onnx")

            >>> # Export on specific device
            >>> trainer.export_onnx("mmoe.onnx", device="cpu")
        """
        from ..utils.onnx_export import ONNXExporter

        # Handle DataParallel wrapped model
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Use provided device or default to 'cpu'
        export_device = device if device is not None else 'cpu'

        exporter = ONNXExporter(model, device=export_device)
        return exporter.export(
            output_path=output_path,
            dummy_input=dummy_input,
            batch_size=batch_size,
            seq_length=seq_length,
            opset_version=opset_version,
            dynamic_batch=dynamic_batch,
            verbose=verbose,
            onnx_export_kwargs=onnx_export_kwargs,
        )

    def visualization(self, input_data=None, batch_size=2, seq_length=10, depth=3, show_shapes=True, expand_nested=True, save_path=None, graph_name="model", device=None, dpi=300, **kwargs):
        """Draw the model; see utils.visualization.visualize_model for options.

        Inputs are generated from model features unless input_data is provided.
        Example: trainer.visualization(save_path="model.png", depth=3).
        """
        from ..utils.visualization import visualize_model

        return visualize_model(
            self.model,
            input_data=input_data,
            batch_size=batch_size,
            seq_length=seq_length,
            depth=depth,
            show_shapes=show_shapes,
            expand_nested=expand_nested,
            save_path=save_path,
            graph_name=graph_name,
            device=device,
            dpi=dpi,
            **kwargs
        )
